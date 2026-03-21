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


from langchain_core.documents import Document

# Import at the top: from langchain_core.documents import Document

def get_pdf_text(pdf_docs):
    documents = []
    for pdf_path in pdf_docs:
        # This gets the actual filename (e.g., Fresenius 5008 Service manual.pdf)
        file_name = os.path.basename(pdf_path) 
        pdf_reader = PdfReader(pdf_path)
        
        for i, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            if page_text:
                # We save the text AND the real page number (i + 1)
                new_doc = Document(
                    page_content=page_text,
                    metadata={"source": file_name, "page": i + 1}
                )
                documents.append(new_doc)
    return documents


# split text into chunks
def get_text_chunks(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=100
    )
    # Change 'split_text' to 'split_documents' to keep the metadata
    chunks = splitter.split_documents(documents)
    return chunks


# get embeddings for each chunk
def get_vector_store(text_chunks):
    # Pass the key directly from Streamlit's secrets
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001", 
        google_api_key=st.secrets["GOOGLE_API_KEY"],
        task_type="retrieval_document"
    )
    # Process in larger batches of 50 for much faster speed
    batch_size = 50 
    vector_store = None
    
    for i in range(0, len(text_chunks), batch_size):
        batch = text_chunks[i:i + batch_size]
        try:
            if vector_store is None:
                vector_store = FAISS.from_documents(batch, embedding=embeddings)
            else:
                vector_store.add_documents(batch)
            # Reduced sleep time for better speed
            time.sleep(1) 
        except Exception as e:
            st.error(f"Quota reached. Waiting 10 seconds...")
            time.sleep(10)
            
    vector_store.save_local("/tmp/faiss_index")


def get_conversational_chain():
    prompt_template = """
    Answer the question as accurately as possible using the provided context.
    
    CRITICAL REQUIREMENT: For every point in your answer, you MUST state the Source PDF name and the Page Number.
    Example: "Check PT07 temperature (Source: Fresenius 5008 Service manual.pdf, Page 45)"
    
    Context:\n {context}?\n
    Question: \n{question}\n
    
    Answer (Include Source and Page):
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
        docs = new_db.similarity_search(user_question, k=5)
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

def auto_ingest_data():
    """Checks for the /data folder and processes PDFs automatically."""
    data_dir = "data"
    index_path = "/tmp/faiss_index"

    # Only process if the vector index doesn't already exist in the temporary folder
    if not os.path.exists(index_path):
        if not os.path.exists(data_dir):
            st.error(f"Folder '{data_dir}' not found. Please create it and add your PDFs.")
            return

        # Identify all PDF files in the folder
        pdf_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.pdf')]
        
        if pdf_files:
            with st.spinner("🚀 System is auto-loading troubleshooting manuals..."):
                raw_text = get_pdf_text(pdf_files)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.toast("Manuals loaded automatically!", icon="✅")
        else:
            st.warning("No PDFs found in the /data folder.")

def main():
    st.set_page_config(
        page_title="HoloLens PDF Assistant",
        page_icon="🏥"
    )

    # --- EDIT 1: Run the auto-loader immediately ---
    auto_ingest_data()

    # --- EDIT 2: Simplified Sidebar ---
    with st.sidebar:
        st.title("System Status")
        st.success("Manuals: Pre-loaded")
        # Keep the clear history button for testing
        if st.sidebar.button('Clear Chat History'):
            clear_chat_history()
            st.rerun()

    # Main content area
    st.title("Biomedical Troubleshooting AI 🏥")
    st.write("System ready for HoloLens troubleshooting.")

    # --- EDIT 3: Update Initial Message ---
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [
            {"role": "assistant", "content": "I have loaded the dialysis manuals. How can I help you troubleshoot?"}]

    # (Keep your existing chat display and input logic below)
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input("Ask a troubleshooting question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Searching manuals..."):
                response = user_input(prompt)
                full_response = response.get('output_text', "No answer found.")
                st.markdown(full_response)
        if response is not None:
            message = {"role": "assistant", "content": full_response}
            st.session_state.messages.append(message)


if __name__ == "__main__":
    main()
