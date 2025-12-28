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
import base64

# Robust Import for Word generation
try:
    from docx import Document
    from docx.shared import Inches
except ImportError:
    Document = None
    Inches = None

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
            # Verify key validity by making a lightweight call
            list(genai.list_models()) 
            
            st.success("Key Valid!")
            
            # STRICT LIMIT: Force only this model
            selected_model_name = "models/gemini-2.5-flash-lite"
            st.info(f"🔒 Using required model:\n**{selected_model_name}**")
            
        except Exception as e:
            st.error(f"API Key Error: {e}")

    st.divider()
    
    # --- C. KNOWLEDGE BASE (RAG Feedback Loop) ---
    st.header("🧠 Knowledge Base")
    st.info("Rules learned from previous errors.")
    
    KB_FILE = "knowledge_base.json"
    
    if "knowledge_base" not in st.session_state:
        if os.path.exists(KB_FILE):
            try:
                with open(KB_FILE, "r") as f: st.session_state.knowledge_base = json.load(f)
            except: st.session_state.knowledge_base = []
        else: st.session_state.knowledge_base = []

    if st.session_state.knowledge_base:
        with st.expander("View Rules", expanded=False):
            for i, rule in enumerate(st.session_state.knowledge_base):
                col1, col2 = st.columns([4, 1])
                col1.write(f"- {rule}")
                if col2.button("❌", key=f"del_{i}"):
                    st.session_state.knowledge_base.pop(i)
                    with open(KB_FILE, "w") as f: json.dump(st.session_state.knowledge_base, f)
                    st.rerun()
        if st.button("🗑️ Clear All Rules"):
            st.session_state.knowledge_base = []
            with open(KB_FILE, "w") as f: json.dump([], f)
            st.rerun()

    st.divider()
    
    # --- D. CASE MANAGEMENT (Saves Output + Plots) ---
    st.header("💾 Case Management")
    HISTORY_FILE = "experiment_history.json"
    
    if "history" not in st.session_state:
        if os.path.exists(HISTORY_FILE):
            try:
                with open(HISTORY_FILE, "r") as f: st.session_state.history = json.load(f)
            except: st.session_state.history = {}
        else: st.session_state.history = {}
    
    if "current_case_name" not in st.session_state: st.session_state.current_case_name = None

    # New Case
    save_name = st.text_input("New Case Name", placeholder="e.g. Test 1", label_visibility="collapsed")
    if st.button("💾 Create New Case"):
        if save_name:
            # Capture current state to preserve work if valid
            st.session_state.history[save_name] = {
                "extracted_text": st.session_state.get("extracted_text", ""),
                "active_code": st.session_state.get("active_code", ""),
                "run_output": st.session_state.get("run_output", None)
            }
            st.session_state.current_case_name = save_name
            with open(HISTORY_FILE, "w") as f: json.dump(st.session_state.history, f)
            st.success(f"Created '{save_name}'!")
            st.rerun()

    # Update Existing Case
    if st.session_state.current_case_name:
        st.divider()
        st.info(f"Editing: **{st.session_state.current_case_name}**")
        if st.button("💾 Save All (Code + Output)", type="primary"):
            # Serialize run output (including plots as base64)
            serializable_output = None
            if st.session_state.run_output:
                serializable_output = {
                    "out": st.session_state.run_output["out"],
                    "err": st.session_state.run_output["err"],
                    "plots": [base64.b64encode(p).decode('utf-8') for p in st.session_state.run_output["plots"]]
                }

            st.session_state.history[st.session_state.current_case_name] = {
                "extracted_text": st.session_state.get("extracted_text", ""),
                "active_code": st.session_state.get("active_code", ""),
                "run_output": serializable_output
            }
            with open(HISTORY_FILE, "w") as f: json.dump(st.session_state.history, f)
            st.toast(f"Saved '{st.session_state.current_case_name}' with outputs!")

    # Load Case
    if st.session_state.history:
        st.divider()
        load_name = st.selectbox("Select Case to Load", list(st.session_state.history.keys()))
        if st.button("📂 Load Case"):
            data = st.session_state.history[load_name]
            st.session_state.extracted_text = data.get("extracted_text", "")
            st.session_state.active_code = data.get("active_code", "")
            
            # Deserialize outputs
            run_data = data.get("run_output")
            if run_data:
                st.session_state.run_output = {
                    "out": run_data["out"],
                    "err": run_data["err"],
                    "plots": [base64.b64decode(p) for p in run_data["plots"]]
                }
            else:
                st.session_state.run_output = None

            st.session_state.current_case_name = load_name
            st.session_state.show_review = True
            st.success(f"Loaded '{load_name}'!")
            st.rerun()

    st.divider()
    st.header("📂 Input")
    uploaded_file = st.file_uploader("3. Upload Notes (PDF)", type="pdf")

# --- 3. SESSION STATE INIT ---
if "messages" not in st.session_state: st.session_state.messages = []
if "pdf_images" not in st.session_state: st.session_state.pdf_images = []
if "active_code" not in st.session_state: st.session_state.active_code = ""
if "run_output" not in st.session_state: st.session_state.run_output = None
if "extracted_text" not in st.session_state: st.session_state.extracted_text = ""
if "show_review" not in st.session_state: st.session_state.show_review = False
if "validation_result" not in st.session_state: st.session_state.validation_result = None

# --- 4. FUNCTIONS ---

def extract_code(text):
    match = re.search(r"```(python|py)?\n(.*?)```", text, re.DOTALL)
    if match: return match.group(2).strip()
    return None

def validate_code_safety(code):
    forbidden = ["os.system", "subprocess.Popen", "shutil.rmtree", "exec("]
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
    try: import cvxpy, matplotlib, docx
    except ImportError as e: return "", f"Missing library {e.name}. Install it.", []

    is_safe, msg = validate_code_safety(code)
    if not is_safe: return "", msg, []

    patched_code = "import matplotlib\nmatplotlib.use('Agg')\n" + code
    try:
        proc = subprocess.run([sys.executable, "-c", patched_code], capture_output=True, text=True, timeout=20)
    except subprocess.TimeoutExpired:
        return "", "Error: Timeout (20s). Infinite loop likely.", []

    plots = []
    for f in glob.glob("*.png"):
        try:
            with open(f, "rb") as img: plots.append(img.read())
            # os.remove(f) # Keep files momentarily for PDF generation if needed
        except: pass
    return proc.stdout, proc.stderr, plots

def fix_code_ai(bad_code, error, model_name, problem_desc):
    model = genai.GenerativeModel(model_name)
    # RAG: Inject learned rules into fix prompt at the END for recency bias
    kb_text = ""
    if st.session_state.knowledge_base:
        kb_text = "CRITICAL USER RULES (YOU MUST APPLY THESE):\n" + "\n".join([f"- {r}" for r in st.session_state.knowledge_base])
    
    prompt = f"""
    You are an expert Python debugger. Fix the error in the code below.
    
    PROBLEM DESCRIPTION (SOURCE OF TRUTH - DO NOT CHANGE MATRICES):
    {problem_desc}
    
    ERROR: {error}
    
    BROKEN CODE:
    {bad_code}
    
    INSTRUCTIONS:
    1. Fix the syntax/logic error.
    2. Do NOT change matrix values from the description.
    3. If the error is 'broadcasting', use .reshape(n,1) on vectors.
    4. Ensure solvers are called in a try-except loop (CLARABEL, SCS, ECOS).
    
    {kb_text}
    
    Return ONLY the complete fixed code.
    """
    resp = generate_with_retry(model, prompt)
    return extract_code(resp.text)

def learn_from_fix(original_error, fixed_code, model_name):
    model = genai.GenerativeModel(model_name)
    prompt = f"""
    Analyze this error and the fix provided.
    
    Error: "{original_error}"
    
    Fixed Code Snippet:
    {fixed_code}
    
    TASK: Extract ONE specific, technical rule to prevent this exact mistake in the future.
    Bad Rule: "Fix dimensions."
    Good Rule: "Always use cp.Variable((n,1)) for vectors instead of (n,)."
    Good Rule: "Use * for scalar multiplication, not @."
    """
    resp = generate_with_retry(model, prompt)
    return resp.text.strip()

def validate_problem_logic(text, model_name):
    model = genai.GenerativeModel(model_name)
    prompt = f"""
    Act as a Math Logic Validator. Analyze the following problem description.
    Check for:
    1. Dimension mismatches (e.g. Matrix A is 2x2 but b is 3x1).
    2. Missing variables (e.g. used 'rho' but never defined it).
    3. Logical inconsistencies.
    
    PROBLEM:
    {text}
    
    Output Format:
    - If valid: "VALID"
    - If issues found: List them as bullet points. Suggest resolutions.
    """
    resp = generate_with_retry(model, prompt)
    return resp.text

def create_word_report(problem, code, output, plots, knowledge_base=None):
    if Document is None:
        return None
        
    doc = Document()
    doc.add_heading('Experiment Report', 0)
    
    # 1. Problem
    doc.add_heading('1. Problem Description', level=1)
    doc.add_paragraph(problem)
    
    # 2. Code
    doc.add_heading('2. Python Code', level=1)
    doc.add_paragraph(code, style='Quote')
    
    # 3. Output
    doc.add_heading('3. Execution Output', level=1)
    out_text = output['out'] or "[No Standard Output]"
    doc.add_paragraph(out_text, style='Quote')

    # 4. Errors (Explicit section)
    if output['err']:
        doc.add_heading('4. Errors Detected', level=1)
        doc.add_paragraph(output['err'], style='Quote')

    # 5. Learned Rules (Knowledge Base)
    if knowledge_base:
        doc.add_heading('5. Learned Rules (Knowledge Base)', level=1)
        if not knowledge_base:
             doc.add_paragraph("No specific rules learned for this case yet.")
        else:
            for rule in knowledge_base:
                doc.add_paragraph(rule, style='List Bullet')
    
    # 6. Plots
    if plots:
        doc.add_heading('6. Generated Plots', level=1)
        for i, plot_bytes in enumerate(plots):
            # Save temp file for FPDF
            temp_name = f"temp_plot_{i}.png"
            with open(temp_name, "wb") as f:
                f.write(plot_bytes)
            doc.add_picture(temp_name, width=Inches(6))
            os.remove(temp_name)
            
    # Save to memory stream
    doc_io = io.BytesIO()
    doc.save(doc_io)
    doc_io.seek(0)
    return doc_io

# --- 5. MAIN LOGIC ---

if st.session_state.current_case_name:
    st.info(f"📂 Active Case: **{st.session_state.current_case_name}**")

# Step A: Processing
if uploaded_file and ("last_file" not in st.session_state or st.session_state.last_file != uploaded_file.name):
    st.session_state.last_file = uploaded_file.name
    with st.spinner("Processing PDF images..."):
        try:
            doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            st.session_state.pdf_images = [Image.open(io.BytesIO(page.get_pixmap(dpi=150).tobytes("png"))) for page in doc]
            st.session_state.messages = [{"role": "assistant", "content": f"✅ Uploaded! {len(st.session_state.pdf_images)} pages ready."}]
            st.session_state.extracted_text = ""
            st.session_state.show_review = False
            st.session_state.active_code = ""
        except Exception as e: st.error(f"Error reading PDF: {e}")
    st.rerun()

# Step B: Workflow
if (uploaded_file or st.session_state.show_review) and selected_model_name:
    
    # 1. ANALYZE (Raw Extraction)
    if uploaded_file and not st.session_state.show_review:
        st.info("Step 1: Extract text and formulas.")
        if st.button("🔍 Analyze & Extract Text", type="primary"):
            with st.spinner(f"Analyzing with {selected_model_name}..."):
                try:
                    model = genai.GenerativeModel(selected_model_name)
                    # VERBATIM PROMPT
                    inputs = ["Extract the problem statement VERBATIM from these images. Correct obvious OCR typos (e.g. 'O' vs '0', 'l' vs '1') but keep matrix values strictly as written. Do not generate code yet."] + st.session_state.pdf_images
                    response = generate_with_retry(model, inputs)
                    st.session_state.extracted_text = response.text
                    st.session_state.show_review = True
                    st.rerun()
                except Exception as e: st.error(f"Analysis failed: {e}")

    # 2. REVIEW & VALIDATE & GENERATE
    if st.session_state.show_review:
        st.success("Step 1 Complete. Review extracted text below.")
        
        # Live Editing
        live_problem_description = st.text_area("📝 Problem Description (Source of Truth):", key="extracted_text", height=300)
        
        col_val, col_gen = st.columns([1, 1])
        
        # VALIDATION FEATURE
        with col_val:
            if st.button("🕵️ Validate Problem Statement"):
                with st.spinner("Checking logic and dimensions..."):
                    val_report = validate_problem_logic(live_problem_description, selected_model_name)
                    st.session_state.validation_result = val_report
        
        if st.session_state.validation_result:
            if "VALID" in st.session_state.validation_result:
                st.success("✅ Logic Check Passed")
            else:
                st.warning("⚠️ Potential Issues Found:")
                st.write(st.session_state.validation_result)

        with col_gen:
            if st.button("🚀 Generate Python Code (Step 2)", type="primary"):
                with st.spinner("Generating Code..."):
                    try:
                        model = genai.GenerativeModel(selected_model_name)
                        
                        # RAG: Inject Knowledge Base
                        kb_str = ""
                        if st.session_state.knowledge_base:
                            kb_str = "CRITICAL USER OVERRIDES (YOU MUST FOLLOW THESE RULES):\n" + "\n".join([f"- {r}" for r in st.session_state.knowledge_base])
                            st.toast(f"Applying {len(st.session_state.knowledge_base)} user rules...")

                        # SANITIZED & DEFENSIVE TEMPLATE
                        CVXPY_TEMPLATE = """
                        import cvxpy as cp
                        import numpy as np
                        import matplotlib.pyplot as plt
                        
                        # 1. INPUT DATA (Load from description ONLY)
                        # n = ...
                        
                        # DEFENSIVE: Explicitly reshape inputs to avoid broadcasting errors
                        # Use dtype=float to prevent integer math issues
                        # A = np.array(..., dtype=float).reshape(n, n)
                        # b = np.array(..., dtype=float).reshape(n, 1) # Force column vector
                        
                        # 2. VARIABLES
                        # x = cp.Variable((n, 1)) # Explicit column vector
                        
                        # 3. CONSTRAINTS
                        # constraints = []
                        
                        # 4. SOLVE (Robust Fallback)
                        prob = cp.Problem(cp.Minimize(0), constraints)
                        
                        # Try multiple solvers in case one fails on the platform
                        solved = False
                        for solver in [cp.CLARABEL, cp.SCS, cp.ECOS]:
                            try:
                                prob.solve(solver=solver)
                                if prob.status in ["optimal", "optimal_inaccurate"]:
                                    solved = True
                                    break
                            except:
                                continue
                        
                        if not solved:
                            print("Warning: All solvers failed or problem is infeasible.")
                        """

                        prompt = f"""
                        You are a strict coder. Implement the experiment EXACTLY as described.
                        
                        REFERENCE TEMPLATE:
                        {CVXPY_TEMPLATE}

                        DESCRIPTION (SOURCE OF TRUTH):
                        {live_problem_description}
                        
                        STANDARD RULES:
                        1. Use a SOLVER LOOP (CLARABEL, SCS, ECOS) to ensure robustness.
                        2. Use `angle=` for Ellipse.
                        3. Save to 'plot.png'. DO NOT use plt.show().
                        4. CRITICAL DIMENSIONS:
                           - If `b` is (n,1), `x` MUST be `cp.Variable((n,1))` to avoid broadcasting errors.
                           - `A @ x - b` requires `x` to be a column vector.
                        5. Use `*` for scalar mult, `<< 0` for LMI (PSD).
                        6. SYNTAX SAFETY:
                           - Ensure all np.array definitions use `dtype=float`.
                           - Use `cp.reshape(..., order='F')` to silence warnings.
                           - Use `np.sqrt(max(val, 0))` for square roots.
                           - NEVER use variable name `lambda`. Use `lam`.
                        7. DATA INTEGRITY:
                           - ONLY define matrices listed in DESCRIPTION.
                           - Do NOT invent parameters.
                           
                        {kb_str}
                        """
                        response = generate_with_retry(model, prompt)
                        found_code = extract_code(response.text)
                        if found_code:
                            st.session_state.active_code = found_code
                            st.session_state.run_output = None
                        else: st.warning("No code found.")
                        st.rerun()
                    except Exception as e: st.error(f"Generation failed: {e}")

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
                
                if st.button("🚑 Auto-Fix & Learn"):
                    with st.spinner("Fixing and Learning..."):
                        # 1. Get the fixed code (passing description to ensure no invention)
                        fixed_code = fix_code_ai(
                            edited_code, 
                            res['err'], 
                            selected_model_name,
                            st.session_state.extracted_text # Pass source truth
                        )
                        
                        if fixed_code:
                            # 2. Extract rule by comparing broken vs fixed
                            new_rule = learn_from_fix(res['err'], fixed_code, selected_model_name)
                            
                            # 3. Save rule
                            if new_rule:
                                st.session_state.knowledge_base.append(new_rule)
                                with open("knowledge_base.json", "w") as f:
                                    json.dump(st.session_state.knowledge_base, f)
                                st.toast(f"Learned: {new_rule}")
                            
                            # 4. Update code
                            st.session_state.active_code = fixed_code
                            st.session_state.run_output = None
                            st.rerun()
                        else:
                            st.error("AI could not fix the code.")
            else:
                st.success("No errors.")
        
        # Word Report Button
        st.divider()
        if st.button("📄 Download Word Report (.docx)"):
            if Document is not None:
                try:
                    word_bytes = create_word_report(
                        st.session_state.extracted_text,
                        edited_code,
                        res,
                        res['plots'],
                        st.session_state.knowledge_base
                    )
                    st.download_button(
                        label="Click to Download DOCX",
                        data=word_bytes,
                        file_name=f"{st.session_state.current_case_name or 'report'}.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                    )
                except Exception as e:
                    st.error(f"Report Generation Error: {e}")
            else:
                st.error("Missing 'python-docx' library. Please update requirements.txt.")