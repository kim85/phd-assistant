import streamlit as st
import google.generativeai as genai
import fitz # PyMuPDF

st.title("🔍 System Diagnostic Tool")

# --- TEST 1: BUTTON CLICK ---
st.subheader("Test 1: Button Interaction")
if "counter" not in st.session_state:
    st.session_state.counter = 0

if st.button("Click Me"):
    st.session_state.counter += 1

st.write(f"Button Click Count: **{st.session_state.counter}**")
st.info("👉 If this number does not go up when you click, Streamlit is broken on your machine.")

# --- TEST 2: API CONNECTION ---
st.subheader("Test 2: Google API Connection")
# Try to get key from secrets, else ask
try:
    default_key = st.secrets["GOOGLE_AI_API_KEY"]
except:
    default_key = ""

api_key = st.text_input("Enter API Key to Test", value=default_key, type="password")

if st.button("Test API Connection"):
    if not api_key:
        st.error("Enter a key first.")
    else:
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content("Say 'Hello' if you can hear me.")
            st.success(f"✅ Success! The AI said: {response.text}")
        except Exception as e:
            st.error(f"❌ API Failed: {e}")

# --- TEST 3: PDF READABILITY ---
st.subheader("Test 3: PDF Reading")
uploaded_file = st.file_uploader("Upload PDF to Test", type="pdf")

if uploaded_file:
    try:
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        
        st.write(f"Pages found: {len(doc)}")
        st.write(f"Characters found: {len(text)}")
        
        if len(text) < 100:
            st.warning("⚠️ Very little text found. This PDF might be an image/scan.")
        else:
            st.success("✅ PDF Text extracted successfully.")
    except Exception as e:
        st.error(f"❌ PDF Error: {e}")