import streamlit as st
import requests
from pypdf import PdfReader
from docx import Document
import io


MODEL_URL = "http://llama-server:8080/completion"


def read_txt_file(uploaded_file):
    return uploaded_file.read().decode("utf-8", errors="ignore")


def read_pdf_file(uploaded_file):
    text = ""

    pdf_reader = PdfReader(uploaded_file)

    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    return text


def read_docx_file(uploaded_file):
    text = ""

    file_bytes = uploaded_file.read()
    document = Document(io.BytesIO(file_bytes))

    for paragraph in document.paragraphs:
        text += paragraph.text + "\n"

    return text


def extract_file_text(uploaded_file):
    if uploaded_file is None:
        return ""

    file_name = uploaded_file.name.lower()

    if file_name.endswith(".txt"):
        return read_txt_file(uploaded_file)

    if file_name.endswith(".pdf"):
        return read_pdf_file(uploaded_file)

    if file_name.endswith(".docx"):
        return read_docx_file(uploaded_file)

    return "Unsupported file type."


st.set_page_config(page_title="LLM Chat Interface", page_icon="💬")

st.title("LLM Chat Interface")

st.write("Type a message, upload a document, or do both.")

user_message = st.text_area("Your message")

uploaded_file = st.file_uploader(
    "Upload a document",
    type=["txt", "pdf", "docx"]
)

if uploaded_file is not None:
    st.success(f"Uploaded file: {uploaded_file.name}")

if st.button("Send"):
    file_text = ""

    if uploaded_file is not None:
        with st.spinner("Reading document..."):
            file_text = extract_file_text(uploaded_file)

    if user_message.strip() == "" and file_text.strip() == "":
        st.warning("Please type a message or upload a document.")
    else:
        final_prompt = ""

        if user_message.strip():
            final_prompt += "User message:\n"
            final_prompt += user_message.strip()
            final_prompt += "\n\n"

        if file_text.strip():
            final_prompt += "Document content:\n"
            final_prompt += file_text.strip()
            final_prompt += "\n\n"

        with st.spinner("Sending to the model..."):
            try:
                response = requests.post(
                    MODEL_URL,
                    json={"prompt": final_prompt},
                    timeout=120
                )

                st.subheader("Response")
                st.write(response.text)

            except Exception as error:
                st.error("Could not connect to the model server.")
                st.write(error)
