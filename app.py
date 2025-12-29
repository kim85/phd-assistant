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

# Robust Import for Word generation
try:
    from docx import Document
    from docx.shared import Inches
except ImportError:
    Document = None
    Inches = None

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="PhD Agent: LMI Specialist", layout="wide")
st.title("PhD Assistant: Zero-Defect LMI Agent 🧪")

# --- 2. SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("1. Enter Google API Key", type="password")
    
    selected_model_name = None
    if api_key:
        try:
            genai.configure(api_key=api_key)
            models = list(genai.list_models())
            # Filtering for Flash models (Fast/Cheap) or Pro (Powerful)
            valid_models = [m.name for m in models if 'generateContent' in m.supported_generation_methods]
            valid_models.sort(key=lambda x: (not '1.5' in x, not 'pro' in x))
            if valid_models:
                selected_model_name = st.selectbox("Select Reasoning Model", valid_models, index=0)
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

    save_name = st.text_input("Case Name", placeholder="e.g. Robust_Stability_v1")
    if st.button("💾 Create/Update Case"):
        if save_name:
            st.session_state.history[save_name] = {
                "extracted_text": st.session_state.get("extracted_text", ""),
                "plan": st.session_state.get("architect_plan", ""),
                "code": st.session_state.get("active_code", "")
            }
            st.session_state.current_case_name = save_name
            with open(HISTORY_FILE, "w") as f: json.dump(st.session_state.history, f)
            st.success(f"Case '{save_name}' synchronized!")

    if st.session_state.history:
        load_name = st.selectbox("Existing Cases", list(st.session_state.history.keys()))
        if st.button("📂 Load Case"):
            data = st.session_state.history[load_name]
            st.session_state.current_case_name = load_name
            st.session_state.extracted_text = data.get("extracted_text", "")
            st.session_state.architect_plan = data.get("plan", "")
            st.session_state.active_code = data.get("active_code", "")
            st.rerun()

# --- 3. AGENTIC CORE LOGIC ---

def extract_code(text):
    """Regex to pull code blocks and sanitize common AI syntax errors."""
    match = re.search(r"```(python|py)?\n(.*?)```", text, re.DOTALL)
    if match:
        code = match.group(2).strip()
        # Fix the most common hallucinated operators
        code = code.replace("<<=", "<<").replace(">>=", ">>")
        return code
    return None

def run_architect_agent(problem, model_name):
    """Phase 1: Generates the Dimensional Ledger."""
    model = genai.GenerativeModel(model_name)
    prompt = f"""
    Analyze this Control Theory / LMI problem: {problem}
    TASK:
    1. Define specific dimensions for all matrices (n, m, l).
    2. List every block in the LMI and its specific (rows x cols) size.
    3. Identify which variables are SCALARS (e.g., lambda, nu) and which are MATRICES (e.g., P, K).
    4. Dimension Rule: All blocks in Row X must have the same height. 
    Return a structural plan. No code.
    """
    return model.generate_content(prompt).text

def run_coder_agent(plan, template, model_name):
    """Phase 2: Translates the Ledger into CVXPY code with strict operator enforcement."""
    model = genai.GenerativeModel(model_name)
    prompt = f"""
    Math Plan: {plan}
    Reference Style: {template}
    
    CRITICAL SYNTAX RULES:
    1. SCALAR OPERATOR: If multiplying a scalar variable (like lambda or nu) by a matrix (like F or I), 
       YOU MUST USE `*`. Example: `F * lam`. 
       DO NOT use `@` with scalars (e.g., `F @ lam` is a ValueError).
    2. MATRIX OPERATOR: Use `@` ONLY for Matrix-Matrix or Matrix-Vector products.
    3. BMAT SAFETY: Use `np.zeros((rows, cols))` for all zero blocks. NEVER use scalar 0.
    4. LATEX SAFETY: Use raw strings `r"..."` for all plot labels/titles containing backslashes or LaTeX.
    
    Return ONLY the python code.
    """
    code_text = model.generate_content(prompt).text
    return extract_code(code_text)

def run_code_backend(code):
    """Executes code with an injected non-interactive plotting patch."""
    patch = (
        "import matplotlib\n"
        "matplotlib.use('Agg')\n"
        "import matplotlib.pyplot as plt\n"
        "import uuid\n"
        "def mock_show():\n"
        "  plt.savefig(f'res_{uuid.uuid4().hex[:5]}.png')\n"
        "  plt.close()\n"
        "plt.show = mock_show\n"
    )
    try:
        proc = subprocess.run([sys.executable, "-c", patch + code], capture_output=True, text=True, timeout=60)
        plots = []
        for f in glob.glob("res_*.png"):
            with open(f, "rb") as img: plots.append(img.read())
            os.remove(f)
        return proc.stdout, proc.stderr, plots
    except Exception as e:
        return "", f"System Error: {str(e)}", []

# --- 4. STREAMLIT UI ---

if "active_code" not in st.session_state: st.session_state.active_code = ""
if "architect_plan" not in st.session_state: st.session_state.architect_plan = ""

tab_theory, tab_workbench = st.tabs(["📐 1. Theory & Dimensional Plan", "🚀 2. Workbench & Execution"])

with tab_theory:
    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("Source PDF Analysis")
        uploaded_file = st.file_uploader("Upload Problem PDF", type="pdf")
        if uploaded_file:
            doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            imgs = [Image.open(io.BytesIO(p.get_pixmap(dpi=150).tobytes("png"))) for p in doc]
            if st.button("🔍 Extract Verbatim Math"):
                with st.spinner("Extracting..."):
                    m = genai.GenerativeModel(selected_model_name)
                    st.session_state.extracted_text = m.generate_content(["Extract LMI matrices and dimensions exactly."] + imgs).text
        
        prob_desc = st.text_area("Problem Metadata", value=st.session_state.get("extracted_text", ""), height=350)

    with col_right:
        st.subheader("The Architect's Ledger")
        if st.button("🏗️ Build Dimensional Plan"):
            with st.spinner("Analyzing math structure..."):
                st.session_state.architect_plan = run_architect_agent(prob_desc, selected_model_name)
        
        plan_input = st.text_area("Ledger (Verify dimensions here)", value=st.session_state.architect_plan, height=350)

with tab_workbench:
    # Golden Template Management
    with st.expander("⭐ Edit Reference Style (Golden Template)", expanded=False):
        SETTINGS_FILE = "settings.json"
        saved_t = "import cvxpy as cp\nimport numpy as np\n# Logic..."
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, "r") as f: saved_t = json.load(f).get("golden_code", saved_t)
        curr_t = st.text_area("Style Guide", value=saved_t, height=200)
        if st.button("💾 Save Template"):
            with open(SETTINGS_FILE, "w") as f: json.dump({"golden_code": curr_t}, f)

    if st.button("🚀 Write Code from Plan", type="primary"):
        with st.spinner("Enforcing syntax rules..."):
            st.session_state.active_code = run_coder_agent(plan_input, curr_t, selected_model_name)
    
    code_final = st.text_area("Python Script", value=st.session_state.active_code, height=450)
    
    col_run, col_report = st.columns([1,1])
    with col_run:
        if st.button("▶️ Execute Experiment"):
            with st.spinner("Running Solver..."):
                out, err, plots = run_code_backend(code_final)
                st.session_state.run_results = {"out": out, "err": err, "plots": plots}

    if "run_results" in st.session_state:
        res = st.session_state.run_results
        if res['err']:
            st.error("Error Detected (Likely Dimension or Operator Error):")
            st.code(res['err'])
        
        if res['out']:
            st.info("Solver Output:")
            st.code(res['out'])
        
        for p in res['plots']:
            st.image(p)

        with col_report:
            if Document and st.button("📄 Export to Word"):
                doc = Document()
                doc.add_heading("Experiment Report", 0)
                doc.add_heading("Generated Code", level=1)
                doc.add_paragraph(code_final)
                if res['plots']:
                    doc.add_heading("Plots", level=1)
                    for i, p_bytes in enumerate(res['plots']):
                        tmp = f"fig_{i}.png"
                        with open(tmp, "wb") as f: f.write(p_bytes)
                        doc.add_picture(tmp, width=Inches(5))
                        os.remove(tmp)
                
                doc_io = io.BytesIO()
                doc.save(doc_io)
                doc_io.seek(0)
                st.download_button("📥 Download Report", doc_io, "PhD_Report.docx")