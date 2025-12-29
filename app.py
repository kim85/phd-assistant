import streamlit as st
import google.generativeai as genai
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import sys
import re
import glob
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="LMI Stable Solver", layout="wide")
st.title("🧪 PhD LMI Solver: Stable 3.11 Edition")

with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("Gemini API Key", type="password")
    if api_key:
        genai.configure(api_key=api_key)
        model_choice = st.selectbox("Select Model", ["gemini-1.5-pro", "gemini-1.5-flash"])

# --- AGENTIC PROMPT ---
# Optimized for CVXPY 1.4.3 (more forgiving with cp.bmat shapes)
STABLE_PROMPT = """
Write a Python 3.11 script using CVXPY 1.4.3 and NumPy 1.24.3.
Problem: {problem}

RULES:
1. Define matrices A, b, F, EA, Eb using np.array.
2. Loop p from 0 to 2 with step 0.1.
3. Solve Task 1: 
   - Variables: x = cp.Variable((2,1)), nu = cp.Variable(nonneg=True), lam = 1.0 (constant).
   - Use cp.bmat([[...], [...]]) for the 4x4 LMI.
4. Solve Task 2: 
   - Calculate v_x0 and Omega* in NumPy using Task 1 results.
   - Solve 2x2 LMI for nu_new.
5. Print nu_task1, nu_task2 for each p and generate Gamma vs p plots.
"""

# --- RUNTIME ENGINE ---
def run_stable_sandbox(code):
    # Patch to save plots without a display
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

# --- UI ---
st.subheader("1. Problem Description")
user_input = st.text_area("Describe your LMI matrices and Task 1/2 requirements:", height=250)

if st.button("🚀 Run Stable Experiment", type="primary"):
    if not api_key:
        st.error("Please enter your API Key in the sidebar.")
    else:
        model = genai.GenerativeModel(model_choice)
        with st.spinner("Generating stable code..."):
            response = model.generate_content(STABLE_PROMPT.format(problem=user_input))
            code_match = re.search(r"```python\n(.*?)```", response.text, re.DOTALL)
            code = code_match.group(1) if code_match else response.text
            
        with st.spinner("Executing in Stable Environment..."):
            out, err, plots = run_stable_sandbox(code)
            
            if err:
                st.error("Execution Error:")
                st.code(err)
            
            if out:
                st.info("Solver Output:")
                st.code(out)
            
            for p in plots:
                st.image(p)

with st.expander("🛠️ View Generated Code"):
    st.code(code if 'code' in locals() else "# Run experiment to see code", language="python")