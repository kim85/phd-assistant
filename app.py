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
import base64

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="PhD Assistant: LMI Specialist", layout="wide")
st.title("PhD Assistant: Zero-Defect LMI Agent 🧪")

# --- 2. SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("Enter Google API Key", type="password")
    
    selected_model_name = None
    if api_key:
        try:
            genai.configure(api_key=api_key)
            models = list(genai.list_models())
            valid_models = [m.name for m in models if 'generateContent' in m.supported_generation_methods]
            valid_models.sort(key=lambda x: (not '1.5-pro' in x, not 'flash' in x))
            if valid_models:
                selected_model_name = st.selectbox("Select Model", valid_models, index=0)
        except Exception as e:
            st.error(f"API Key Error: {e}")

    st.divider()
    st.header("💾 Case Management")
    HISTORY_FILE = "experiment_history.json"
    if "history" not in st.session_state:
        if os.path.exists(HISTORY_FILE):
            try:
                with open(HISTORY_FILE, "r") as f: st.session_state.history = json.load(f)
            except: st.session_state.history = {}
        else: st.session_state.history = {}

    save_name = st.text_input("New Case Name", placeholder="e.g. Experiment_v1")
    if st.button("💾 Create/Load Case"):
        if save_name:
            st.session_state.current_case_name = save_name
            st.rerun()

# --- 3. AGENTIC FUNCTIONS ---

def extract_code(text):
    match = re.search(r"```(python|py)?\n(.*?)```", text, re.DOTALL)
    if match:
        code = match.group(2).strip()
        return code.replace("<<=", "<<").replace(">>=", ">>")
    return None

def run_architect_agent(problem, model_name):
    """Step 1: Returns a JSON ledger of dimensions."""
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
      "variables": {{"x": "n x 1", "lam": "scalar", "nu": "scalar"}}
    }}
    """
    response = model.generate_content(prompt)
    json_match = re.search(r"\{.*\}", response.text, re.DOTALL)
    return json_match.group(0) if json_match else response.text

def run_coder_agent(ledger_json, template, model_name):
    """Step 2: Uses the Sledgehammer Fix (cp.reshape) for every block."""
    model = genai.GenerativeModel(model_name)
    prompt = f"""
    Using this JSON Ledger: {ledger_json}
    Reference Style: {template}
    
    THE SLEDGEHAMMER RULE:
    1. For the LMI `cp.bmat`, you MUST wrap EVERY SINGLE BLOCK in `cp.reshape(block, (rows, cols))`.
    2. Use the heights and widths from the Ledger for the reshape dimensions.
    3. Scalars (lam, nu) MUST be reshaped to (1, 1).
    4. Vectors (Ax-b) MUST be reshaped to (n, 1) or (1, n).
    5. No exceptions. Every entry in the bmat list-of-lists must be a 2D CVXPY expression.
    6. Use '*' for scalars (lam, nu) and '@' for matrix products.
    """
    code_text = model.generate_content(prompt).text
    return extract_code(code_text)

def run_code_backend(code):
    patch = "import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\nimport uuid\ndef mock_show():\n  plt.savefig(f'r_{uuid.uuid4().hex[:5]}.png')\n  plt.close()\nplt.show = mock_show\n"
    try:
        proc = subprocess.run([sys.executable, "-c", patch + code], capture_output=True, text=True, timeout=60)
        plots = []
        for f in glob.glob("r_*.png"):
            with open(f, "rb") as img: plots.append(img.read())
            os.remove(f)
        return proc.stdout, proc.stderr, plots
    except Exception as e:
        return "", str(e), []

# --- 4. UI ---

if "active_code" not in st.session_state: st.session_state.active_code = ""
if "architect_plan" not in st.session_state: st.session_state.architect_plan = ""

t1, t2 = st.tabs(["📐 1. Theory & Ledger", "🚀 2. Sledgehammer Workbench"])

with t1:
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Source PDF Analysis")
        uploaded_file = st.file_uploader("Upload Problem PDF", type="pdf")
        if uploaded_file and st.button("🔍 Extract Verbatim"):
            doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            imgs = [Image.open(io.BytesIO(p.get_pixmap(dpi=150).tobytes("png"))) for p in doc]
            m = genai.GenerativeModel(selected_model_name)
            st.session_state.extracted_text = m.generate_content(["Extract LMI matrices verbatim."] + imgs).text
        
        prob_desc = st.text_area("Problem Metadata", value=st.session_state.get("extracted_text", ""), height=350)

    with col_b:
        st.subheader("📋 Step 1: The Ledger")
        if st.button("🏗️ Build Dimension Ledger"):
            st.session_state.architect_plan = run_architect_agent(prob_desc, selected_model_name)
        
        plan_input = st.text_area("JSON Ledger", value=st.session_state.architect_plan, height=250)
        try:
            st.json(json.loads(plan_input))
        except: st.error("Invalid JSON")

with t2:
    if st.button("🚀 Write Sledgehammer Code", type="primary"):
        # We pass a strict template that enforces cp.reshape
        template = "Use cp.CLARABEL. Ensure all bmat blocks are 2D reshaped."
        st.session_state.active_code = run_coder_agent(plan_input, template, selected_model_name)
    
    code_final = st.text_area("Final Script", value=st.session_state.active_code, height=450)
    
    if st.button("▶️ Run Experiment"):
        with st.spinner("Executing..."):
            out, err, plots = run_code_backend(code_final)
            if err: st.error(res['err'] if 'res' in locals() else err)
            if out: st.code(out)
            for p in plots: st.image(p)