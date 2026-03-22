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
        file_name = os.path.basename(pdf_path) 
        pdf_reader = PdfReader(pdf_path)
        
        for i, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            if page_text:
                # ADD THIS LINE: Explicitly label the digital page inside the text
                labeled_text = f"SOURCE_FILE: {file_name} | DIGITAL_PAGE: {i + 1}\n{page_text}"
                
                new_doc = Document(
                    page_content=labeled_text,
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
    prompt_template = """
    You are a Senior Biomedical Engineer. Use the provided context to answer the troubleshooting question.
    
    CRITICAL HIERARCHY RULES:
    1. PRIMARY SOURCES: 'Fresenius 5008 Service manual.pdf' and 'PM Form_Haemodialysis Unit_FMC-5008_2025.pdf'. Use these for technical steps and maintenance.
    2. SECONDARY SOURCE: 'Fresenius 5008 User manual.pdf'. Use this ONLY for general operation or if the answer is missing from the Service Manual.
    3. If there is a conflict, the Service Manual is ALWAYS correct.

    INSTRUCTIONS FOR CITATIONS:
    - At the end of every answer, clearly list the source file name and the Digital Page number.
    - Use the 'DIGITAL_PAGE' label found at the top of the context snippets.                                                                                                                  
                                                                                                                    
    Context:
    {context}

    Question: 
    {question}

    Answer (Include Filename and Digital Page):
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
        
        # --- NEW: WEIGHTED SEARCH ---
        # We add priority terms to the search to favor the Service Manual and PM Form
        priority_terms = "Technical Safety Check TSC Maintenance Procedure Service Manual PM Form"
        enhanced_query = f"{priority_terms} {user_question}"
        
        # Increased k to 10 so we capture snippets from all 3 PDFs
        docs = new_db.similarity_search(enhanced_query, k=10) 
    except Exception as e:
        st.error(f"Vector Database Error: {e}")
        return {"output_text": "System not ready. Please wait for auto-load to finish."}

    chain = get_conversational_chain()

    try:
        # Pass the docs and the original question to the chain
        response = chain(
            {"input_documents": docs, "question": user_question}, 
            return_only_outputs=True
        )
    except Exception as e:
        if "429" in str(e):
            st.warning("Rate limit hit. Retrying in 15 seconds...")
            time.sleep(15)
            return user_input(user_question)
        else:
            st.error(f"Gemini API error: {e}")
            return {"output_text": "An error occurred."}

    return response
    
def auto_ingest_data():
    """Checks for the /data folder and processes PDFs automatically."""
    data_dir = "data"
    index_path = "/tmp/faiss_index"

    if not os.path.exists(index_path):
        if not os.path.exists(data_dir):
            st.error(f"Folder '{data_dir}' not found. Please create it in your GitHub repo.")
            return

        pdf_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.pdf')]
        
        if pdf_files:
            with st.spinner("🚀 Pre-loading Technical Manuals..."):
                # Use the updated functions that handle Document objects
                documents = get_pdf_text(pdf_files)
                text_chunks = get_text_chunks(documents)
                get_vector_store(text_chunks)
                st.toast("Manuals loaded!", icon="✅")
        else:
            st.warning("No PDFs found in the /data folder.")

def main():
    st.set_page_config(
        page_title="HoloLens PDF Assistant",
        page_icon="🏥"
    )

    # 1. INITIALIZE HISTORY (Must be at the very top of main)
    # This prevents the history from disappearing on refresh
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "I have loaded the dialysis manuals. How can I help you troubleshoot?"}
        ]

    # 2. RUN AUTO-LOADER
    auto_ingest_data()

    # Sidebar
    with st.sidebar:
        st.title("System Status")
        st.success("Manuals: Pre-loaded")
        if st.button('Clear Chat History'):
            # This clears the session state correctly
            st.session_state.messages = [
                {"role": "assistant", "content": "Chat cleared. How can I help?"}
            ]
            st.rerun()

    st.title("Biomedical Troubleshooting AI 🏥")
    st.write("System ready for HoloLens troubleshooting.")

    # 3. DISPLAY CHAT HISTORY
    # We loop through the session_state every time the page refreshes
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 4. CHAT INPUT & RESPONSE LOGIC
    if prompt := st.chat_input("Ask a troubleshooting question..."):
        # Save and display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and save assistant message
        with st.chat_message("assistant"):
            with st.spinner("Searching manuals..."):
                # user_input now uses the k=5 and metadata logic we discussed
                response = user_input(prompt)
                full_response = response.get('output_text', "No answer found.")
                st.markdown(full_response)
        
        # SAVE THE RESPONSE to history so it survives a refresh
        st.session_state.messages.append({"role": "assistant", "content": full_response})


if __name__ == "__main__":
    main()
