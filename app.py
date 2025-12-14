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
import time

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="PhD Assistant", layout="wide")
st.title("PhD Assistant: Experiment Designer 🧪")

# --- 2. SIDEBAR (Setup) ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # A. API KEY (Bring Your Own Key)
    api_key = st.text_input("1. Enter Google API Key", type="password")
    
    # B. DYNAMIC MODEL SELECTOR
    selected_model_name = None
    if api_key:
        try:
            genai.configure(api_key=api_key)
            model_list = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
            if model_list:
                st.success(f"Key Valid! Found {len(model_list)} models.")
                selected_model_name = st.selectbox("2. Select AI Model", model_list, index=0)
            else:
                st.error("Key valid, but no models found.")
        except Exception as e:
            st.error(f"API Key Error: {e}")

    st.divider()
    
    # --- C. CASE MANAGEMENT ---
    st.header("💾 Case Management")
    HISTORY_FILE = "experiment_history.json"
    
    # Load History
    if "history" not in st.session_state:
        if os.path.exists(HISTORY_FILE):
            try:
                with open(HISTORY_FILE, "r") as f:
                    st.session_state.history = json.load(f)
            except:
                st.session_state.history = {}
        else:
            st.session_state.history = {}
    
    if "current_case_name" not in st.session_state:
        st.session_state.current_case_name = None

    # 1. Create New Case
    save_name = st.text_input("New Case Name", placeholder="e.g. Test 1", label_visibility="collapsed")
    if st.button("💾 Create New Case"):
        if save_name:
            st.session_state.history[save_name] = {"extracted_text": "", "active_code": ""}
            st.session_state.current_case_name = save_name
            with open(HISTORY_FILE, "w") as f: json.dump(st.session_state.history, f)
            st.success(f"Created '{save_name}'!")
            st.rerun()

    # 2. Update/Save Current Case
    if st.session_state.current_case_name:
        st.divider()
        st.info(f"Editing: **{st.session_state.current_case_name}**")
        if st.button("💾 Save Changes to Case", type="primary"):
            st.session_state.history[st.session_state.current_case_name] = {
                "extracted_text": st.session_state.get("extracted_text", ""),
                "active_code": st.session_state.get("active_code", "")
            }
            with open(HISTORY_FILE, "w") as f: json.dump(st.session_state.history, f)
            st.toast(f"Saved to '{st.session_state.current_case_name}'!")

    # 3. Load Existing Case
    if st.session_state.history:
        st.divider()
        load_name = st.selectbox("Select Case to Load", list(st.session_state.history.keys()))
        if st.button("📂 Load Selected Case"):
            data = st.session_state.history[load_name]
            st.session_state.extracted_text = data.get("extracted_text", "")
            st.session_state.active_code = data.get("active_code", "")
            st.session_state.current_case_name = load_name
            st.session_state.show_review = True
            st.session_state.run_output = None
            st.success(f"Loaded '{load_name}'!")
            st.rerun()
            
        # Download Button (Backup)
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, "rb") as f:
                st.download_button(
                    label="📥 Download History (Backup)",
                    data=f,
                    file_name="experiment_history.json",
                    mime="application/json"
                )

    st.divider()
    st.header("📂 Input")
    uploaded_file = st.file_uploader("3. Upload Notes (PDF)", type="pdf")

# --- 3. SESSION STATE ---
if "messages" not in st.session_state: st.session_state.messages = []
if "pdf_images" not in st.session_state: st.session_state.pdf_images = []
if "active_code" not in st.session_state: st.session_state.active_code = ""
if "run_output" not in st.session_state: st.session_state.run_output = None
if "extracted_text" not in st.session_state: st.session_state.extracted_text = ""
if "show_review" not in st.session_state: st.session_state.show_review = False

# --- 4. FUNCTIONS ---

def extract_code(text):
    match = re.search(r"```(python|py)?\n(.*?)```", text, re.DOTALL)
    if match: return match.group(2).strip()
    return None

def validate_code_safety(code):
    forbidden = ["os.system", "subprocess.Popen", "shutil.rmtree", "open(", "remove(", "exec("]
    for term in forbidden:
        if term in code: return False, f"Security Risk: Code contains forbidden term '{term}'."
    return True, ""

def generate_with_retry(model, inputs, retries=3):
    for i in range(retries):
        try: return model.generate_content(inputs)
        except Exception as e:
            if i == retries - 1: raise e
            time.sleep(1)

def run_code_backend(code):
    try:
        import cvxpy
        import matplotlib
    except ImportError as e: return "", f"Missing library {e.name}.", []

    is_safe, msg = validate_code_safety(code)
    if not is_safe: return "", msg, []

    patched_code = "import matplotlib\nmatplotlib.use('Agg')\n" + code
    try:
        proc = subprocess.run([sys.executable, "-c", patched_code], capture_output=True, text=True, timeout=15)
    except subprocess.TimeoutExpired:
        return "", "Error: Code execution timed out (limit: 15s).", []

    plots = []
    for f in glob.glob("*.png"):
        try:
            with open(f, "rb") as img: plots.append(img.read())
            os.remove(f)
        except: pass
    return proc.stdout, proc.stderr, plots

def fix_code_ai(bad_code, error, model_name):
    model = genai.GenerativeModel(model_name)
    prompt = f"Fix this Python code error.\nERROR: {error}\nCODE: {bad_code}\nReturn ONLY code in ```python``` blocks."
    resp = generate_with_retry(model, prompt)
    return extract_code(resp.text)

# --- 5. MAIN LOGIC ---

if st.session_state.current_case_name:
    st.info(f"📂 Active Case: **{st.session_state.current_case_name}**")

# Step A: Processing PDF
if uploaded_file and ("last_file" not in st.session_state or st.session_state.last_file != uploaded_file.name):
    st.session_state.last_file = uploaded_file.name
    with st.spinner("Processing PDF images..."):
        try:
            doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            st.session_state.pdf_images = [Image.open(io.BytesIO(page.get_pixmap(dpi=150).tobytes("png"))) for page in doc]
            st.session_state.messages = [{"role": "assistant", "content": f"✅ Uploaded! I see {len(st.session_state.pdf_images)} pages."}]
            st.session_state.extracted_text = ""
            st.session_state.show_review = False
            st.session_state.active_code = ""
        except Exception as e: st.error(f"Error reading PDF: {e}")
    st.rerun()

# Step B: Workflow
if (uploaded_file or st.session_state.show_review) and selected_model_name:
    
    # 1. ANALYZE
    if uploaded_file and not st.session_state.show_review:
        st.info("Step 1: Extract text and formulas.")
        if st.button("🔍 Analyze & Extract Text", type="primary"):
            with st.spinner(f"Analyzing with {selected_model_name}..."):
                try:
                    model = genai.GenerativeModel(selected_model_name)
                    inputs = ["Analyze this PDF. Provide a detailed summary in Russian, explicitly listing all matrices (A, b, etc.) and the objective function. Do NOT write code yet."] + st.session_state.pdf_images
                    response = generate_with_retry(model, inputs)
                    st.session_state.extracted_text = response.text
                    st.session_state.show_review = True
                    st.rerun()
                except Exception as e: st.error(f"Analysis failed: {e}")

    # 2. REVIEW & GENERATE
    if st.session_state.show_review:
        st.success("Step 1 Complete. Review below.")
        live_problem_description = st.text_area("📝 Edit Experiment Details:", key="extracted_text", height=300)
        
        col1, col2 = st.columns([1, 4])
        with col1:
             if st.button("Start Over"):
                 st.session_state.show_review = False
                 st.session_state.extracted_text = ""
                 st.rerun()
        with col2:
            if st.button("🚀 Generate Python Code (Step 2)", type="primary"):
                with st.spinner("Generating Code..."):
                    try:
                        model = genai.GenerativeModel(selected_model_name)
                        
                        CVXPY_TEMPLATE = """
                        import cvxpy as cp
                        import numpy as np
                        import matplotlib.pyplot as plt
                        
                        # 1. INPUT DATA (Load from description ONLY)
                        # n = ...
                        # A = np.array(...) 
                        
                        # 2. VARIABLES
                        # x = cp.Variable((n, 1)) 
                        
                        # 3. CONSTRAINTS
                        # constraints = []
                        
                        # 4. SOLVE
                        # prob = cp.Problem(cp.Minimize(0), constraints)
                        # prob.solve(solver=cp.SCS)
                        """

                        prompt = f"""
                        You are a strict coder. Implement the experiment EXACTLY as described.
                        
                        REFERENCE TEMPLATE (Structure Only - DO NOT COPY VALUES):
                        {CVXPY_TEMPLATE}

                        DESCRIPTION (SOURCE OF TRUTH):
                        {live_problem_description}
                        
                        STANDARD RULES:
                        1. Use `cp.SCS` for cvxpy solver.
                        2. Use `angle=` for Ellipse.
                        3. Save to 'plot.png'. DO NOT use plt.show().
                        4. CRITICAL: If b is (n,1), x MUST be cp.Variable((n,1)).
                        5. Use `*` for scalar mult, `<< 0` for LMI (PSD).
                        6. SYNTAX SAFETY:
                           - Use `cp.reshape(..., order='F')` to silence warnings.
                           - Use `np.sqrt(max(val, 0))` for square roots to handle solver noise.
                           - NEVER use variable name `lambda` (reserved). Use `lam`.
                           - Always check `if prob.status in ["optimal", ...]` before accessing .value
                           - Use `plt.close()` after saving plots.
                        7. DATA INTEGRITY:
                           - ONLY define matrices (A, b, F, etc.) that are explicitly listed in the DESCRIPTION.
                           - Do NOT invent new parameters (like rho, alpha) unless they are in the text.
                        """
                        response = generate_with_retry(model, prompt)
                        st.session_state.messages.append({"role": "assistant", "content": response.text})
                        found_code = extract_code(response.text)
                        if found_code:
                            st.session_state.active_code = found_code
                            st.session_state.run_output = None
                        else: st.warning("No code found.")
                        st.rerun()
                    except Exception as e: st.error(f"Generation failed: {e}")

# Step C: Chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

# Step D: Workbench
if st.session_state.active_code:
    st.divider()
    st.header("🛠️ Workbench")
    edited_code = st.text_area("Python Code", value=st.session_state.active_code, height=350)
    st.session_state.active_code = edited_code
    
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("▶️ Run Code"):
            with st.spinner("Running..."):
                out, err, plots = run_code_backend(edited_code)
                st.session_state.run_output = {"out": out, "err": err, "plots": plots}
    
    if st.session_state.run_output:
        res = st.session_state.run_output
        t1, t2, t3 = st.tabs(["Plots", "Output", "Errors"])
        with t1:
            for p in res['plots']: st.image(p)
        with t2: st.code(res['out'] if res['out'] else "No text output.")
        with t3:
            if res['err']:
                st.error(res['err'])
                if st.button("🚑 Fix This Error"):
                    with st.spinner("Fixing..."):
                        fixed = fix_code_ai(edited_code, res['err'], selected_model_name)
                        if fixed:
                            st.session_state.active_code = fixed
                            st.session_state.run_output = None
                            st.success("Fixed!")
                            st.rerun()
            else: st.success("No errors.")