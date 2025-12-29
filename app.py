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
st.title("PhD Assistant: Agentic LMI Specialist 🧪")

# --- 2. SIDEBAR CONFIGURATION ---
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
    if st.button("💾 Create Case"):
        if save_name:
            st.session_state.history[save_name] = {"extracted_text": "", "plan": "", "code": ""}
            st.session_state.current_case_name = save_name
            with open(HISTORY_FILE, "w") as f: json.dump(st.session_state.history, f)
            st.rerun()

    if st.session_state.history:
        load_name = st.selectbox("Load Case", list(st.session_state.history.keys()))
        if st.button("📂 Load Selected"):
            data = st.session_state.history[load_name]
            st.session_state.current_case_name = load_name
            st.session_state.extracted_text = data.get("extracted_text", "")
            st.session_state.architect_plan = data.get("plan", "")
            st.session_state.active_code = data.get("code", "")
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
    Analyze this Control Theory problem: {problem}
    TASK: Create a JSON 'Dimension Ledger'. 
    YOU MUST RETURN ONLY A JSON OBJECT.
    Example Format:
    {{
      "dimensions": {{"n": 2, "m": 2, "l": 2}},
      "matrices": {{"A": "n x n", "b": "n x 1"}},
      "variables": {{"x": "n x 1", "lam": "scalar", "nu": "scalar"}},
      "multiplication_rules": "Use * for scalars (lam, nu) and @ for matrices (A, F)"
    }}
    """
    response = model.generate_content(prompt)
    json_match = re.search(r"\{.*\}", response.text, re.DOTALL)
    return json_match.group(0) if json_match else response.text

def run_coder_agent(ledger_json, template, model_name):
    """Step 2: Uses the Ledger to write safe CVXPY code."""
    model = genai.GenerativeModel(model_name)
    prompt = f"""
    Using this JSON Ledger: {ledger_json}
    Reference Style: {template}
    
    RULES:
    1. Check 'variables' in Ledger. If 'scalar', use '*' operator ONLY. NEVER use '@'.
    2. Check 'dimensions'. Use `np.zeros((rows, cols))` for all zero blocks.
    3. Ensure cp.bmat alignment matches the Ledger's row heights.
    4. Implement a loop trying [cp.CLARABEL, cp.SCS] for stability.
    Return ONLY Python code.
    """
    code_text = model.generate_content(prompt).text
    return extract_code(code_text)

def run_code_backend(code):
    patch = "import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\nimport uuid\ndef mock_show():\n  plt.savefig(f'res_{uuid.uuid4().hex[:5]}.png')\n  plt.close()\nplt.show = mock_show\n"
    try:
        proc = subprocess.run([sys.executable, "-c", patch + code], capture_output=True, text=True, timeout=60)
        plots = []
        for f in glob.glob("res_*.png"):
            with open(f, "rb") as img: plots.append(img.read())
            os.remove(f)
        return proc.stdout, proc.stderr, plots
    except Exception as e:
        return "", str(e), []

# --- 4. MAIN UI ---

if "active_code" not in st.session_state: st.session_state.active_code = ""
if "architect_plan" not in st.session_state: st.session_state.architect_plan = ""

t1, t2 = st.tabs(["📐 1. Theory & Ledger", "🚀 2. Code Workbench"])

with t1:
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Source PDF Analysis")
        uploaded_file = st.file_uploader("Upload Notes", type="pdf")
        if uploaded_file and st.button("🔍 Extract Verbatim"):
            doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            imgs = [Image.open(io.BytesIO(p.get_pixmap(dpi=150).tobytes("png"))) for p in doc]
            m = genai.GenerativeModel(selected_model_name)
            st.session_state.extracted_text = m.generate_content(["Extract LMI matrices and dimensions verbatim."] + imgs).text
        
        prob_desc = st.text_area("Problem Metadata", value=st.session_state.get("extracted_text", ""), height=350)

    with col_b:
        st.subheader("📋 The Dimension Ledger")
        if st.button("🏗️ Build Ledger"):
            st.session_state.architect_plan = run_architect_agent(prob_desc, selected_model_name)
        
        plan_input = st.text_area("Review JSON Ledger (Edit if wrong!)", value=st.session_state.architect_plan, height=200)
        
        try:
            ledger_data = json.loads(plan_input)
            st.json(ledger_data)
            st.success("Ledger parsed successfully. The Coder will follow these dimensions.")
        except:
            st.warning("Ledger is not valid JSON. Correct it before proceeding to Step 2.")

with t2:
    with st.expander("⭐ Golden Template Style"):
        SETTINGS_FILE = "settings.json"
        saved_t = ""
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, "r") as f: saved_t = json.load(f).get("golden_code", "")
        curr_t = st.text_area("Style Reference", value=saved_t, height=150)
        if st.button("💾 Save Style"):
            with open(SETTINGS_FILE, "w") as f: json.dump({"golden_code": curr_t}, f)

    if st.button("🚀 Write Code from Ledger", type="primary"):
        st.session_state.active_code = run_coder_agent(plan_input, curr_t, selected_model_name)
    
    code_final = st.text_area("Final Python Script", value=st.session_state.active_code, height=450)
    
    if st.button("▶️ Run Experiment"):
        with st.spinner("Executing..."):
            out, err, plots = run_code_backend(code_final)
            if err: st.error(err)
            if out: st.code(out)
            for p in plots: st.image(p)
            
            # Auto-save results to current case
            if "current_case_name" in st.session_state:
                name = st.session_state.current_case_name
                st.session_state.history[name]["plan"] = plan_input
                st.session_state.history[name]["code"] = code_final
                with open(HISTORY_FILE, "w") as f: json.dump(st.session_state.history, f)
                st.toast("Progress saved to History.")