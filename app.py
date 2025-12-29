import streamlit as st
import fitz  # PyMuPDF
import google.generativeai as genai
from PIL import Image
import io
import re
import subprocess
import sys
import glob
import os
import json
import numpy as np

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="PhD Assistant: LMI Specialist", layout="wide")
st.title("PhD Assistant: Sledgehammer LMI Agent 🧪")

with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("Enter Google API Key", type="password")
    if api_key:
        genai.configure(api_key=api_key)
        try:
            models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
            selected_model = st.selectbox("Select Model", models, index=0)
        except: st.error("Invalid API Key")

# --- 2. AGENTIC FUNCTIONS ---

def extract_code(text):
    match = re.search(r"```(python|py)?\n(.*?)```", text, re.DOTALL)
    if match:
        code = match.group(2).strip()
        return code.replace("<<=", "<<").replace(">>=", ">>")
    return None

def run_architect_agent(problem, model_name):
    model = genai.GenerativeModel(model_name)
    prompt = f"""
    Analyze this LMI problem: {problem}
    TASK: Create a JSON 'Dimension Ledger'. 
    RETURN ONLY JSON.
    Example:
    {{
      "dimensions": {{"n": 2, "m": 2, "l": 2}},
      "ledger": {{
          "Row1_Height": "n", "Row2_Height": "m", "Row3_Height": 1, "Row4_Height": "l",
          "Col1_Width": "n", "Col2_Width": "m", "Col3_Width": 1, "Col4_Width": "l"
      }},
      "variables": {{"x": "n x 1", "Lambda": "scalar", "nu": "scalar"}}
    }}
    """
    response = model.generate_content(prompt)
    json_match = re.search(r"\{.*\}", response.text, re.DOTALL)
    return json_match.group(0) if json_match else response.text

def run_coder_agent(ledger_json, model_name):
    model = genai.GenerativeModel(model_name)
    prompt = f"""
    Using this JSON Ledger: {ledger_json}
    
    TASK: Generate a full Python script to solve the LMI over p in [0, 2] with step 0.1.
    
    STRICT RULES:
    1. Loop through p = np.arange(0, 2.1, 0.1).
    2. Define x = cp.Variable((n, 1)), lam = 1.0 (fixed for DCP), nu = cp.Variable(nonneg=True).
    3. Use cp.bmat([]) to construct the 4x4 LMI.
    4. SLEDGEHAMMER RULE: Wrap EVERY block in cp.reshape(block, (h, w), order='C').
    5. TASK 2: After solving Task 1, calculate v_x0 and Omega* using NumPy. 
    6. Formulate A* and b* and solve a second LMI for nu_new.
    7. Generate 2 plots: (1) Gamma vs p for Curve 1 and 2, (2) Trajectory of optimal x.
    8. Print a table comparing nu from Task 1 and Task 2.
    """
    return extract_code(model.generate_content(prompt).text)

def run_code_backend(code):
    patch = "import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\nimport uuid\ndef mock_show():\n  plt.savefig(f'r_{uuid.uuid4().hex[:5]}.png')\n  plt.close()\nplt.show = mock_show\n"
    try:
        proc = subprocess.run([sys.executable, "-c", patch + code], capture_output=True, text=True, timeout=90)
        plots = []
        for f in glob.glob("r_*.png"):
            with open(f, "rb") as img: plots.append(img.read())
            os.remove(f)
        return proc.stdout, proc.stderr, plots
    except Exception as e:
        return "", str(e), []

# --- 3. UI LAYOUT ---

if "architect_plan" not in st.session_state: st.session_state.architect_plan = ""
if "active_code" not in st.session_state: st.session_state.active_code = ""

tab1, tab2 = st.tabs(["📐 Phase 1: Planning", "🚀 Phase 2: Execution"])

with tab1:
    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("Source Analysis")
        uploaded_file = st.file_uploader("Upload Problem PDF", type="pdf")
        if uploaded_file and st.button("🔍 Extract Verbatim"):
            doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            imgs = [Image.open(io.BytesIO(p.get_pixmap(dpi=150).tobytes("png"))) for p in doc]
            m = genai.GenerativeModel(selected_model)
            st.session_state.extracted_text = m.generate_content(["Extract LMI matrices exactly."] + imgs).text
        
        prob_desc = st.text_area("Problem Metadata", value=st.session_state.get("extracted_text", ""), height=300)

    with col_right:
        st.subheader("📋 The Ledger")
        if st.button("🏗️ Build Dimension Ledger"):
            st.session_state.architect_plan = run_architect_agent(prob_desc, selected_model)
        
        plan_input = st.text_area("JSON Ledger", value=st.session_state.architect_plan, height=300)

with tab2:
    if st.button("🚀 Generate & Run Experiment", type="primary"):
        with st.spinner("Writing Sledgehammer Code..."):
            st.session_state.active_code = run_coder_agent(plan_input, selected_model)
        
        with st.spinner("Solving LMIs & Generating Plots..."):
            out, err, plots = run_code_backend(st.session_state.active_code)
            
            if err: st.error(err)
            if out: st.code(out)
            for p in plots: st.image(p)

    st.subheader("Generated Python Script")
    st.code(st.session_state.active_code, language="python")