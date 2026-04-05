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
                # NEW: We prefix the text with the Digital Page number.
                # This ensures the AI sees the number 119 right next to the "Remove Lines" text.
                labeled_text = f"SOURCE_FILE: {file_name} | DIGITAL_PAGE: {i + 1}\n{page_text}"
                
                new_doc = Document(
                    page_content=labeled_text,
                    metadata={"source": file_name, "page": i + 1}
                )
                documents.append(new_doc)
    return documents


def get_text_chunks(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600, 
        chunk_overlap=50
    )
    
    final_chunks = []
    for doc in documents:
        # Split each page individually
        chunks = splitter.split_text(doc.page_content)
        for chunk in chunks:
            # Re-attach the metadata to the text content of EVERY chunk
            # This ensures the AI sees the source/page even in the middle of a paragraph
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "Unknown")
            
            labeled_chunk = f"SOURCE: {source} | DIGITAL_PAGE: {page}\n{chunk}"
            
            final_chunks.append(Document(
                page_content=labeled_chunk,
                metadata=doc.metadata
            ))
    return final_chunks


# get embeddings for each chunk
def get_vector_store(text_chunks):
    # Pass the key directly from Streamlit's secrets
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001", 
        google_api_key=st.secrets["GOOGLE_API_KEY"],
        task_type="retrieval_document"
    )
    # make it slower but more "patient" so it doesn't crash
    batch_size = 10 # sends the pages in small groups of 10
    vector_store = None
    
    for i in range(0, len(text_chunks), batch_size):
        batch = text_chunks[i:i + batch_size]
        try:
            if vector_store is None:
                vector_store = FAISS.from_documents(batch, embedding=embeddings)
            else:
                vector_store.add_documents(batch)
            # Reduced sleep time for better speed
            time.sleep(2) # the system wait 2 seconds between batches to avoid hitting the free-tier limit
        except Exception as e:
            st.error(f"Quota reached. Waiting 10 seconds...") # displays an error message
            time.sleep(10) # waits 10 seconds before trying again automatically
            
    vector_store.save_local("/tmp/faiss_index")


def get_conversational_chain():
    prompt_template = """
    You are a Senior Biomedical Engineer. Answer the troubleshooting question using the provided context.
    
    STRICT CITATION RULES:
    1. For every instruction, look at the 'DIGITAL_PAGE' label at the top of the specific text block.
    2. The 'DIGITAL_PAGE' is the absolute index of the PDF (1 to 168).
    3. IGNORE all other numbers that look like pages (e.g., ignore footers like 6-36, 4-2).
    4. Format: [Filename | Digital Page: X;]
    
    Context:
    {context}
    
    Question: 
    {question}
    
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
    )

    try:
        # 1. Load the database (which will now only contain Service Manual & PM Form)
        new_db = FAISS.load_local("/tmp/faiss_index", embeddings, allow_dangerous_deserialization=True)
        
        # 2. Perform a single, deep search. 
        # Since these are technical documents, k=10 is ideal to catch 
        # details spread across multiple pages (e.g., the TSC checklist).
        docs = new_db.similarity_search(user_question, k=20)
        # This tells the AI to look at 20 different parts of the manual instead of just 10
                
    except Exception as e:
        st.error(f"Vector Database Error: {e}")
        return {"output_text": "System not ready. Please ensure manuals are pre-loaded."}

    chain = get_conversational_chain()

    try:
        # 3. Generate the response using only the technical snippets
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
            return {"output_text": "An error occurred during response generation."}

    return response
    
def auto_ingest_data():
    # Checks for the /data folder and processes PDFs automatically.
    data_dir = "data"
    index_path = "/tmp/faiss_index"

    if not os.path.exists(index_path):
        if not os.path.exists(data_dir):
            st.error(f"Folder '{data_dir}' not found. Please create it in your GitHub repo.")
            return

        pdf_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.pdf')]
        
        if pdf_files:
            with st.spinner("🚀 Pre-loading Technical Manuals..."): # Displays a message while the heavy processing happens
                # Use the updated functions that handle Document objects
                documents = get_pdf_text(pdf_files) # Extracts raw text while preserving digital page numbers
                text_chunks = get_text_chunks(documents) # Breaks the 500+ pages into character segments for better AI focus
                get_vector_store(text_chunks) # Converts these segments into vectors and saves them to the local FAISS database.
                st.toast("Manuals loaded!", icon="✅") # Shows a quick notification once the system is ready for questions
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
            {"role": "assistant", "content": "I have loaded the manuals. How can I help you troubleshoot?"}
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
