import streamlit as st
import os
import PyPDF2
import docx
import pytesseract
import pandas as pd
from PIL import Image
import warnings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_pinecone import PineconeVectorStore
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from google.generativeai.types.safety_types import HarmBlockThreshold, HarmCategory
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

warnings.filterwarnings('ignore')

# Set Streamlit UI with custom styling
st.set_page_config(page_title="AI Chatbot", layout="wide")
background_image = "https://source.unsplash.com/1600x900/?technology,ai"
st.markdown(f"""
    <style>
    .stApp {{
        background-image: url({background_image});
        background-size: cover;
    }}
    </style>
""", unsafe_allow_html=True)

st.title("🤖 AI-Powered File Q&A Chatbot")

# Load API Keys
GOOGLE_API_KEY = "AIzaSyC8RV_8JK9bvHLjwCEvfqcw1gsA8Y6lxm4"
PINECONE_API_KEY = "pcsk_NTbou_HfFeKsZnWcLR2QpsSUov8XwpoV8Yuvm6jLLVXQzoZnd9PkJAZkZCJLCLwjUTfFx"
PINECONE_INDEX = "langchain-test-index"

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# Function to extract text from different file types
def extract_text(file, file_type):
    text = ""
    if file_type == "pdf":
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    elif file_type == "docx":
        doc = docx.Document(file)
        for para in doc.paragraphs:
            text += para.text + "\n"
    elif file_type == "txt":
        text = file.read().decode("utf-8")
    elif file_type in ["png", "jpg", "jpeg"]:
        image = Image.open(file)
        text = pytesseract.image_to_string(image)
    elif file_type in ["csv"]:
        df = pd.read_csv(file)
        text = df.to_string()
    elif file_type in ["xlsx", "xls"]:
        df = pd.read_excel(file)
        text = df.to_string()
    return text

# Initialize QA System
def initialize_qa_system():
    embeddings = GoogleGenerativeAIEmbeddings(
        model='models/embedding-001',
        google_api_key=GOOGLE_API_KEY,
        task_type="retrieval_query"
    )
    index_name = PINECONE_INDEX

    prompt_template = """
    ## AI Response:
    Context: \n {context}
    Question: \n {question}
    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])
    
    safety_settings = {
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    }
    
    chat_model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.3,
        safety_settings=safety_settings
    )
    
    retriever_from_llm = MultiQueryRetriever.from_llm(
        retriever=PineconeVectorStore(index_name=index_name, embedding=embeddings).as_retriever(search_kwargs={"k": 5}),
        llm=chat_model
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=chat_model,
        retriever=retriever_from_llm,
        return_source_documents=True,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt}
    )
    
    return qa_chain

qa_chain = initialize_qa_system()

# Sidebar for File Upload
st.sidebar.header("📂 Upload File")
uploaded_file = st.sidebar.file_uploader("Choose a file", type=["pdf", "docx", "txt", "png", "jpg", "jpeg", "csv", "xlsx", "xls"])
file_text = ""
file_type = ""

if uploaded_file is not None:
    file_type = uploaded_file.name.split(".")[-1]
    file_text = extract_text(uploaded_file, file_type)
    st.sidebar.success("File uploaded successfully!")
    st.text_area("📄 Extracted Text", file_text, height=200)

# User Input
st.subheader("💬 Ask a Question Based on the File:")
user_question = st.text_input("Type your question here...")

if st.button("Get Answer"):
    if user_question:
        response = qa_chain.invoke({"query": user_question, "context": file_text})
        bot_response = response.get("result", "I'm sorry, I couldn't find an answer.")
        source_documents = [doc.page_content for doc in response.get("source_documents", [])]
        
        st.subheader("🤖 AI Response:")
        st.write(bot_response)
        
        if source_documents:
            st.subheader("📚 Sources:")
            for doc in source_documents:
                st.write(doc)
