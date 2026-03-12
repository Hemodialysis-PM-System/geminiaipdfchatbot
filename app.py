import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from google import genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains.question_answering import load_qa_chain
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import time

load_dotenv()


# read all pdf files and return text
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


# split text into chunks
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_text(text)
    return chunks  # list of strings


# get embeddings for each chunk
def get_vector_store(text_chunks):
    # Pass the key directly from Streamlit's secrets
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001", 
        google_api_key=st.secrets["GOOGLE_API_KEY"],
        task_type="retrieval_document"
    )
    batch_size = 5 
    for i in range(0, len(text_chunks), batch_size):
        batch = text_chunks[i:i + batch_size]
        try:
            if i == 0:
                vector_store = FAISS.from_texts(batch, embedding=embeddings)
            else:
                vector_store.add_texts(batch)
            # Small pause between batches to respect the 2026 Rate Limits
            time.sleep(2) 
        except Exception as e:
            st.error(f"Quota hit at chunk {i}. Please wait a moment...")
            time.sleep(15) # Wait for the token bucket to refill
    vector_store.save_local("/tmp/faiss_index")


def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        google_api_key=st.secrets["GOOGLE_API_KEY"],
        temperature=0.3
    )
    prompt = PromptTemplate(template=prompt_template,
                            input_variables=["context", "question"])
    chain = load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)
    return chain


def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "upload some pdfs and ask me a question"}]


def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=st.secrets["GOOGLE_API_KEY"],
        task_type="retrieval_query"
    )  # type: ignore

    # 2. Load the vector database from the Streamlit /tmp/ directory
    try:
        new_db = FAISS.load_local("/tmp/faiss_index", embeddings, allow_dangerous_deserialization=True)
        # Limit to 2 docs to stay within Free Tier token limits
        docs = new_db.similarity_search(user_question, k=2)
    except Exception as e:
        st.error(f"Vector Database Error: {e}")
        return {"output_text": "Please upload and process a PDF first."}

    # 3. Get the chain and attempt to generate a response
    chain = get_conversational_chain()

    try:
        response = chain(
            {"input_documents": docs, "question": user_question}, 
            return_only_outputs=True
        )
    except Exception as e:
        if "429" in str(e):
            st.warning("Rate limit hit. Retrying in 15 seconds...")
            time.sleep(15) # Wait for the quota to reset
            return user_input(user_question) # Optional: automatic retry
        else:
            st.error(f"Gemini API error: {e}")
            return {"output_text": "An error occurred."}

    return response


def main():
    st.set_page_config(
        page_title="Google Gemini Pro PDF Chatbot",
        page_icon="🐬"
    )

    # Sidebar for uploading PDF files
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                client = genai.Client(api_key=st.secrets["GOOGLE_API_KEY"])
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

    # Main content area for displaying chat messages
    st.title("Chat with PDF files using Gemini🐬")
    st.write("Welcome to the chat!")
    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

    # Chat input
    # Placeholder for chat messages

    if "messages" not in st.session_state.keys():
        st.session_state.messages = [
            {"role": "assistant", "content": "upload some pdfs and ask me a question"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    # Display chat messages and bot response
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = user_input(prompt)
                full_response = response.get('output_text', "No answer found.")
                st.markdown(full_response)
        if response is not None:
            message = {"role": "assistant", "content": full_response}
            st.session_state.messages.append(message)


if __name__ == "__main__":
    main()
