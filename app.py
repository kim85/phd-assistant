# Add this to your Workbench tab (Tab 2)
if "run_results" in st.session_state:
    res = st.session_state.run_results
    if res['err']:
        st.error(f"Execution Error: {res['err']}")
        
        # THE FIXER BUTTON
        if st.button("🚑 Auto-Fix This Error"):
            with st.spinner("Analyzing the Traceback..."):
                fixer_prompt = f"""
                The following CVXPY code failed:
                {st.session_state.active_code}
                
                ERROR MESSAGE:
                {res['err']}
                
                TASK:
                Identify if this is a Scalar Multiplication error (needs *) 
                or a BMAT Dimension error (needs np.zeros). 
                Rewrite the code to fix ONLY this error.
                """
                m = genai.GenerativeModel(selected_model_name)
                fixed_code = extract_code(m.generate_content(fixer_prompt).text)
                st.session_state.active_code = fixed_code
                st.rerun()