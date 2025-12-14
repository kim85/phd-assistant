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
    
    # A. API KEY
    try:
        default_key = st.secrets["GOOGLE_AI_API_KEY"]
    except:
        default_key = ""
    
    api_key = st.text_input("1. Enter Google API Key", value=default_key, type="password")
    
    # B. DYNAMIC MODEL SELECTOR
    selected_model_name = None
    if api_key:
        try:
            genai.configure(api_key=api_key)
            
            # Fetch valid models for this key
            model_list = []
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    model_list.append(m.name)
            
            if model_list:
                st.success(f"Key Valid! Found {len(model_list)} models.")
                selected_model_name = st.selectbox("2. Select AI Model", model_list, index=0)
            else:
                st.error("Key valid, but no models found. Check Google AI Studio permissions.")
                
        except Exception as e:
            st.error(f"API Key Error: {e}")

    st.divider()
    
    # --- C. KNOWLEDGE BASE (THE BRAIN) ---
    st.header("🧠 Knowledge Base")
    st.info("The model learns from your corrections here.")
    
    KB_FILE = "knowledge_base.json"
    
    # Load KB
    if "knowledge_base" not in st.session_state:
        if os.path.exists(KB_FILE):
            with open(KB_FILE, "r") as f:
                st.session_state.knowledge_base = json.load(f)
        else:
            st.session_state.knowledge_base = []

    # Display Rules
    if st.session_state.knowledge_base:
        with st.expander("View Learned Rules", expanded=False):
            for i, rule in enumerate(st.session_state.knowledge_base):
                col_rule, col_del = st.columns([4, 1])
                col_rule.write(f"{i+1}. {rule}")
                if col_del.button("❌", key=f"del_{i}"):
                    st.session_state.knowledge_base.pop(i)
                    with open(KB_FILE, "w") as f:
                        json.dump(st.session_state.knowledge_base, f)
                    st.rerun()
    else:
        st.caption("No rules learned yet.")

    st.divider()
    
    # --- D. CASE MANAGEMENT ---
    st.header("💾 Case Management")
    HISTORY_FILE = "experiment_history.json"
    
    if "history" not in st.session_state:
        if os.path.exists(HISTORY_FILE):
            try:
                with open(HISTORY_FILE, "r") as f: st.session_state.history = json.load(f)
            except: st.session_state.history = {}
        else: st.session_state.history = {}
    
    if "current_case_name" not in st.session_state: st.session_state.current_case_name = None

    save_name = st.text_input("Case Name", placeholder="e.g. Matrix Test 1", label_visibility="collapsed")
    if st.button("💾 Create New Case"):
        if save_name:
            st.session_state.history[save_name] = {"extracted_text": "", "active_code": ""}
            st.session_state.current_case_name = save_name
            with open(HISTORY_FILE, "w") as f: json.dump(st.session_state.history, f)
            st.success(f"Created '{save_name}'!")
            st.rerun()

    if st.session_state.current_case_name:
        st.divider()
        st.subheader(f"Editing: {st.session_state.current_case_name}")
        if st.button("💾 Update/Save Changes", type="primary"):
            st.session_state.history[st.session_state.current_case_name] = {
                "extracted_text": st.session_state.get("extracted_text", ""),
                "active_code": st.session_state.get("active_code", "")
            }
            with open(HISTORY_FILE, "w") as f: json.dump(st.session_state.history, f)
            st.toast(f"Saved extracted text & code to '{st.session_state.current_case_name}'!")

    if st.session_state.history:
        st.divider()
        load_name = st.selectbox("Select Case", list(st.session_state.history.keys()), label_visibility="collapsed")
        if st.button("📂 Load Case"):
            data = st.session_state.history[load_name]
            st.session_state.extracted_text = data.get("extracted_text", "")
            st.session_state.active_code = data.get("active_code", "")
            st.session_state.current_case_name = load_name
            st.session_state.show_review = True
            st.session_state.run_output = None
            st.success(f"Loaded '{load_name}'!")
            st.rerun()

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

def learn_from_fix(original_error, fixed_code, model_name):
    """Analyzes a fix and extracts a generic rule."""
    model = genai.GenerativeModel(model_name)
    prompt = f"""
    You are a Teacher. 
    A student encountered this error: "{original_error}"
    They fixed it with this code:
    {fixed_code}
    
    Extract ONE short, generic rule (1 sentence) to prevent this mistake in the future.
    Example: "Always use cp.Variable((n,1)) for vectors."
    """
    resp = generate_with_retry(model, prompt)
    return resp.text.strip()

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
            # Reset states
            st.session_state.extracted_text = ""
            st.session_state.show_review = False
            st.session_state.active_code = ""
        except Exception as e: st.error(f"Error reading PDF: {e}")
    st.rerun()

# Step B: Workflow
if (uploaded_file or st.session_state.show_review) and selected_model_name:
    
    # 1. ANALYZE BUTTON (Requires file)
    if uploaded_file and not st.session_state.show_review:
        st.info("Step 1: Extract text and formulas from the PDF.")
        if st.button("🔍 Analyze & Extract Text", type="primary"):
            with st.spinner(f"Analyzing with {selected_model_name}..."):
                try:
                    model = genai.GenerativeModel(selected_model_name)
                    # Ask specifically for text description first
                    inputs = ["Analyze this PDF. Provide a detailed summary in Russian, explicitly listing all matrices (A, b, etc.) and the objective function. Do NOT write code yet."] + st.session_state.pdf_images
                    
                    response = generate_with_retry(model, inputs)
                    st.session_state.extracted_text = response.text
                    st.session_state.show_review = True
                    st.rerun()
                except Exception as e: st.error(f"Analysis failed: {e}")

    # 2. REVIEW & EDIT SECTION
    if st.session_state.show_review:
        st.success("Step 1 Complete. Review or Edit the problem description below.")
        
        # Capture the current live text from the widget
        live_problem_description = st.text_area("📝 Edit Experiment Details (Correct any math typos here):", 
                     key="extracted_text", # Binds directly to st.session_state.extracted_text
                     height=300)
        
        col1, col2 = st.columns([1, 4])
        with col1:
             if st.button("Start Over"):
                 st.session_state.show_review = False
                 st.session_state.extracted_text = ""
                 st.rerun()
        with col2:
            # 3. GENERATE CODE BUTTON
            if st.button("🚀 Generate Python Code (Step 2)", type="primary"):
                
                with st.spinner("Generating Code from your corrected text..."):
                    try:
                        model = genai.GenerativeModel(selected_model_name)
                        
                        # PREPARE LEARNED RULES
                        learned_rules_str = ""
                        if st.session_state.knowledge_base:
                            learned_rules_str = "USER-DEFINED KNOWLEDGE BASE (STRICTLY FOLLOW THESE RULES):\n" + "\n".join([f"- {r}" for r in st.session_state.knowledge_base])

                        # IMPROVED: Empty Template to prevent hallucinations
                        CVXPY_TEMPLATE = """
                        import cvxpy as cp
                        import numpy as np
                        import matplotlib.pyplot as plt
                        
                        # 1. INPUT DATA 
                        # !!! CRITICAL: POPULATE THIS SECTION ONLY WITH DATA FROM THE DESCRIPTION !!!
                        # Do NOT use random numbers. Do NOT use identity matrices unless specified.
                        
                        # Example of extraction (Replace with ACTUAL data):
                        # n = ...
                        # A = np.array(...) 
                        
                        # 2. VARIABLES
                        # Use (n, 1) for column vectors
                        # x = cp.Variable((n, 1)) 
                        
                        # 3. CONSTRAINTS
                        # constraints = []
                        
                        # 4. SOLVE
                        # prob = cp.Problem(cp.Minimize(0), constraints)
                        # try:
                        #     prob.solve(solver=cp.SCS)
                        # except:
                        #     prob.solve()
                            
                        # 5. OUTPUT
                        # if prob.status in ["optimal", "optimal_inaccurate"]:
                        #     print("Result found.")
                        #     plt.close()
                        """

                        prompt = f"""
                        You are a strict coder. Implement the experiment EXACTLY as described in the DESCRIPTION.
                        
                        REFERENCE TEMPLATE (Structure Only - DO NOT COPY VALUES):
                        {CVXPY_TEMPLATE}

                        {learned_rules_str}

                        DESCRIPTION (SOURCE OF TRUTH - FOLLOW STRICTLY):
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
                        7. DATA INTEGRITY (ZERO TOLERANCE):
                           - You are a TRANSLATOR, not a designer.
                           - Translate the math in the DESCRIPTION to Python code 1:1.
                           - If the description defines A as [[1, 3], [2, 4]], your code MUST have A = np.array([[1, 3], [2, 4]]).
                           - Do NOT invent new parameters (like rho, alpha) unless they are in the text.
                           - Do NOT assume matrices are Identity or Zero unless explicitly stated.
                           - If a value is missing, insert a comment: # ERROR: MISSING VALUE FOR [VARIABLE]
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
                
                col_fix, col_learn = st.columns(2)
                with col_fix:
                    if st.button("🚑 Fix This Error"):
                        with st.spinner("Fixing..."):
                            fixed = fix_code_ai(edited_code, res['err'], selected_model_name)
                            if fixed:
                                st.session_state.active_code = fixed
                                st.session_state.run_output = None
                                st.success("Fixed!")
                                st.rerun()
                
                # LEARNING BUTTON
                with col_learn:
                    # We can only learn if we have a successful fix. 
                    # But often we want to learn from the ERROR itself to avoid it next time.
                    # Let's allow learning from the error context.
                    if st.button("🧠 Learn from this Error"):
                        with st.spinner("Analyzing error to create a rule..."):
                            rule = learn_from_fix(res['err'], edited_code, selected_model_name)
                            if rule:
                                st.session_state.knowledge_base.append(rule)
                                with open("knowledge_base.json", "w") as f:
                                    json.dump(st.session_state.knowledge_base, f)
                                st.success(f"Learned Rule: {rule}")
                                time.sleep(2)
                                st.rerun()
            else: st.success("No errors.")