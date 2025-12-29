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
import uuid

# Robust Import for Word generation
try:
    from docx import Document
    from docx.shared import Inches
except ImportError:
    Document = None
    Inches = None

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="PhD Agent: Experiment Designer", layout="wide")
st.title("PhD Assistant: Agentic Experiment Designer 🧪")

# --- 2. SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("1. Enter Google API Key", type="password")
    
    # --- DEBUGGING INFO (New) ---
    with st.expander("ℹ️ System Debug Info"):
        try:
            import cvxpy
            import numpy
            st.caption(f"Python: {sys.version.split()[0]}")
            st.caption(f"CVXPY: {cvxpy.__version__}")
            st.caption(f"Numpy: {numpy.__version__}")
            st.caption(f"OS: {sys.platform}")
        except ImportError:
            st.error("Libraries missing!")

    selected_model_name = None
    if api_key:
        try:
            genai.configure(api_key=api_key)
            models = list(genai.list_models())
            
            # Filter specifically for Flash or Flash-Lite models
            valid_models = [
                m.name for m in models 
                if 'generateContent' in m.supported_generation_methods 
                and 'flash' in m.name.lower()
            ]
            
            # Sort to likely show newest versions first
            valid_models.sort(reverse=True)
            
            if valid_models:
                selected_model_name = st.selectbox("Select Model", valid_models, index=0)
            else:
                st.warning("No Flash models found. Using fallback.")
                selected_model_name = "models/gemini-2.5-flash-lite" # Fallback
                
        except Exception as e:
            st.error(f"API Key Error: {e}")

    st.divider()
    
    # --- CASE MANAGEMENT ---
    st.header("💾 Case Management")
    st.info("⚠️ Note: On Streamlit Cloud, data resets when the app reboots.")
    HISTORY_FILE = "experiment_history.json"
    
    # Init History
    if "history" not in st.session_state:
        if os.path.exists(HISTORY_FILE):
            try:
                with open(HISTORY_FILE, "r") as f: st.session_state.history = json.load(f)
            except: st.session_state.history = {}
        else: st.session_state.history = {}
    
    if "current_case_name" not in st.session_state:
        st.session_state.current_case_name = None

    # Helper Functions for Serialization
    def serialize_run_output(run_output):
        if not run_output: return None
        return {
            "out": run_output["out"],
            "err": run_output["err"],
            "plots": [base64.b64encode(p).decode('utf-8') for p in run_output["plots"]]
        }

    def deserialize_run_output(run_data):
        if not run_data: return None
        return {
            "out": run_data.get("out", ""),
            "err": run_data.get("err", ""),
            "plots": [base64.b64decode(p) for p in run_data.get("plots", [])]
        }

    # 1. Create New Case
    save_name = st.text_input("New Case Name", placeholder="e.g. Test 1", label_visibility="collapsed")
    if st.button("💾 Create New Case"):
        if save_name:
            st.session_state.history[save_name] = {
                "extracted_text": st.session_state.get("extracted_text", ""),
                "architect_plan": st.session_state.get("architect_plan", ""),
                "active_code": st.session_state.get("active_code", ""),
                "run_output": serialize_run_output(st.session_state.get("run_output", None))
            }
            st.session_state.current_case_name = save_name
            with open(HISTORY_FILE, "w") as f: json.dump(st.session_state.history, f)
            st.success(f"Created '{save_name}'!")
            st.rerun()

    # 2. Update Existing Case
    if st.session_state.current_case_name:
        st.caption(f"Editing: **{st.session_state.current_case_name}**")
        if st.button("💾 Save All (Sidebar)", type="secondary"):
            st.session_state.history[st.session_state.current_case_name] = {
                "extracted_text": st.session_state.get("extracted_text", ""),
                "architect_plan": st.session_state.get("architect_plan", ""),
                "active_code": st.session_state.get("active_code", ""),
                "run_output": serialize_run_output(st.session_state.get("run_output", None))
            }
            with open(HISTORY_FILE, "w") as f: json.dump(st.session_state.history, f)
            st.toast(f"Saved '{st.session_state.current_case_name}'!")

    # 3. Load Case
    if st.session_state.history:
        load_name = st.selectbox("Load Case", list(st.session_state.history.keys()))
        if st.button("📂 Load Selected"):
            data = st.session_state.history[load_name]
            st.session_state.current_case_name = load_name
            st.session_state.extracted_text = data.get("extracted_text", "")
            st.session_state.architect_plan = data.get("architect_plan", "")
            st.session_state.active_code = data.get("active_code", "")
            st.session_state.run_output = deserialize_run_output(data.get("run_output"))
            st.session_state.show_review = True
            st.rerun()

    st.divider()
    uploaded_file = st.file_uploader("3. Upload Notes (PDF)", type="pdf")

# --- 3. THE AGENTIC ENGINE ---

def extract_code(text):
    if not text: return None
    # Improved regex to handle optional language tag and varying whitespace (more robust)
    match = re.search(r"