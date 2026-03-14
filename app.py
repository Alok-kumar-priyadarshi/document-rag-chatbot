import streamlit as st
from pypdf import PdfReader
from rag_pipeline import process_document, ask_question

st.title("Document Chatbot (RAG System)")

uploaded_file = st.file_uploader("Upload Document", type=["pdf", "txt"])

if uploaded_file:

    text = ""

    if uploaded_file.type == "application/pdf":
        
        reader = PdfReader(uploaded_file)
        
        for page in reader.pages:
            text += page.extract_text()

    else:
        text = uploaded_file.read().decode()

    st.success("Document Loaded")

    chunks, index = process_document(text)

    question = st.text_input("Ask a question")

    if question:

        answer, retrieved = ask_question(question, chunks, index)

        st.write("### Answer")
        st.write(answer)

        st.write("### Retrieved Context")

        for chunk in retrieved:
            st.write(chunk)