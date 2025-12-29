import streamlit as st
import fitz  # PyMuPDF
import google.generativeai as genai
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import subprocess
import sys
import re
import os
import glob

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="LMI Solver (Stable Edition)", layout="wide")
st.title("🧪 PhD LMI Solver: Stable Legacy Version")

with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input("Gemini API Key", type="password")
    if api_key:
        genai.configure(api_key=api_key)
        models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        selected_model = st.selectbox("Model", models, index=0)

# --- 2. CLASSIC PROMPT ---
# This prompt is designed for CVXPY 1.4.x which is more flexible with shapes.
LEGACY_PROMPT = """
Write a Python 3.11 script using CVXPY 1.4.3.
System: {problem_text}

STRICT ARCHITECTURE:
1. Define matrices A, b, F, EA, Eb using np.array.
2. Loop p in np.arange(0, 2.1, 0.1).
3. Inside loop:
   - Define x = cp.Variable((2,1)), nu = cp.Variable(nonneg=True).
   - Fix lam = 1.0 (to avoid DCP Variable*Variable error).
   - Use cp.bmat([[...], [...]]) for the 4x4 LMI.
   - Task 2: Calculate v_x0 and Omega* in NumPy using results from Task 1.
   - Solve a second 2x2 LMI for nu_task2.
4. Output: print(p, nu_task1, nu_task2) and plot Gamma vs p.
"""

# --- 3. RUNTIME ENGINE ---
def run_classic_sandbox(code):
    patch = "import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\nimport uuid\ndef mock_show():\n  plt.savefig(f'plot_{uuid.uuid4().hex[:5]}.png')\n  plt.close()\nplt.show = mock_show\n"
    try:
        proc = subprocess.run([sys.executable, "-c", patch + code], capture_output=True, text=True, timeout=90)
        plots = []
        for f in glob.glob("plot_*.png"):
            with open(f, "rb") as img: plots.append(img.read())
            os.remove(f)
        return proc.stdout, proc.stderr, plots
    except Exception as e:
        return "", str(e), []

# --- 4. UI ---
st.subheader("1. Problem Description")
user_input = st.text_area("Paste LMI matrices and Task 1/2 requirements:", height=200)

if st.button("🚀 Run Stable Solver", type="primary"):
    if not api_key:
        st.error("Missing API Key")
    else:
        model = genai.GenerativeModel(selected_model)
        with st.spinner("Generating stable code..."):
            response = model.generate_content(LEGACY_PROMPT.format(problem_text=user_input))
            code_match = re.search(r"```python\n(.*?)```", response.text, re.DOTALL)
            code = code_match.group(1) if code_match else response.text
            
        with st.spinner("Executing Solver..."):
            out, err, plots = run_classic_sandbox(code)
            if err: st.error(err)
            if out: st.code(out)
            for p in plots: st.image(p)
            
        with st.expander("📂 View Script"):
            st.code(code)