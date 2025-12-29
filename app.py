import streamlit as st
import google.generativeai as genai
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import sys
import re
import glob

# --- CONFIG ---
st.set_page_config(page_title="LMI Stable Solver", layout="wide")
st.title("🧪 PhD LMI Solver: Stable Environment 3.11")

with st.sidebar:
    api_key = st.text_input("Gemini API Key", type="password")
    if api_key:
        genai.configure(api_key=api_key)
        model_choice = st.selectbox("Model", ["gemini-1.5-pro", "gemini-1.5-flash"])

# --- PROMPT ---
# We use a simpler prompt because CVXPY 1.4.3 handles shapes automatically
STABLE_PROMPT = """
Write a Python 3.11 script to solve this LMI problem: {problem}

RULES:
1. Define A, b, F, EA, Eb as numpy arrays.
2. Loop p from 0 to 2 with step 0.1.
3. Use cp.Variable((2,1)) for x and cp.Variable(nonneg=True) for nu.
4. Set lam = 1.0 as a constant.
5. Task 1: Solve 4x4 LMI using cp.bmat().
6. Task 2: Calculate Omega* in NumPy, then solve a 2x2 LMI for nu_new.
7. Plot results using plt.show().
"""

# --- RUNTIME ---
def run_stable_sandbox(code):
    patch = "import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\nimport uuid\ndef mock_show():\n  plt.savefig(f'p_{uuid.uuid4().hex[:5]}.png')\n  plt.close()\nplt.show = mock_show\n"
    try:
        proc = subprocess.run([sys.executable, "-c", patch + code], capture_output=True, text=True, timeout=90)
        plots = []
        for f in glob.glob("p_*.png"):
            with open(f, "rb") as img: plots.append(img.read())
            import os; os.remove(f)
        return proc.stdout, proc.stderr, plots
    except Exception as e:
        return "", str(e), []

# --- UI ---
user_input = st.text_area("Describe your LMI matrices (A, b, F, etc.):", height=200)

if st.button("🚀 Run Experiment", type="primary"):
    if not api_key:
        st.error("Enter API Key")
    else:
        model = genai.GenerativeModel(model_choice)
        with st.spinner("Generating stable code..."):
            response = model.generate_content(STABLE_PROMPT.format(problem=user_input))
            code = re.search(r"```python\n(.*?)```", response.text, re.DOTALL).group(1)
            
        with st.spinner("Executing..."):
            out, err, plots = run_stable_sandbox(code)
            if err: st.error(err)
            if out: st.code(out)
            for p in plots: st.image(p)