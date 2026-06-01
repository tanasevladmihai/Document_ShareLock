import streamlit as st
import requests
from pypdf import PdfReader
from docx import Document
import io


OLLAMA_API_URL = "http://llama-server:8080/completion"
MODEL_URL = "http://llama-server:8080/completion"
MAX_OUTPUT_TOKENS = 2048

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
        user_prompt_content = "<instruction>\n"
        user_prompt_content += "You are a precise AI assistant. Answer the user's request using ONLY the provided user's request AND document context below. "
        user_prompt_content += "If the document context is empty, respond directly to the user's message. Do not invent any instructions.\n"
        user_prompt_content += "</instruction>\n\n"

        if user_message.strip():
            user_prompt_content += f"<user_request>\n{user_message.strip()}\n</user_request>\n\n"

        if file_text.strip():
            user_prompt_content += f"<document_context>\n{file_text.strip()}\n</document_context>\n\n"

        final_prompt = (
            f"<start_of_turn>user\n{user_prompt_content.strip()}<end_of_turn>\n"
            "<start_of_turn>model\n"
        )

        with st.spinner("Sending to the model..."):
            try:
                response = requests.post(
                    MODEL_URL,
                    json={
                        "prompt": final_prompt,
                        "stop": ["<end_of_turn>", "<start_of_turn>"],
                        "n_predict": MAX_OUTPUT_TOKENS,
                        "temperature": 0.5,
                        "top_p": 0.9,
                        "cache_prompt": True
                    },
                    timeout=300
                )
                st.subheader("Response")
                
                response_json = response.json()
                
                if "content" in response_json:
                    st.write(response_json["content"].strip())
                else:
                    choices_data = response_json.get("choices", [{}])
                    generated_output = choices_data[0].get("text", "").strip() if choices_data else ""
                    
                    if generated_output:
                        st.write(generated_output)
                    else:
                        st.write("No message content found.")

            except Exception as error:
                st.error("Could not connect to the model server.")
                st.write(error)
