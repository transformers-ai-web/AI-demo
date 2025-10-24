from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import uvicorn
import llm
from pydantic import BaseModel

from fastapi import UploadFile, File
import shutil, os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader

import os
from openai import OpenAI

token = os.environ["GITHUB_TOKEN"]
endpoint = "https://models.github.ai/inference"
model = "openai/gpt-4.1-mini"

client = OpenAI(
    base_url=endpoint,
    api_key=token,
)



app = FastAPI()




from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    user_message = request.message
    response = llm.get_llm_response(user_message)
    print(response)
    return {"response": response}

# @app.get("/")
# def read_root():
#     response = llm.get_llm_response("where do you live?")
#     return {"response": response}


# ==============================
# RAG IMPLEMENTATION
# ==============================

UPLOAD_DIR = "uploaded_docs"
CHROMA_DIR = "chroma_storage"

os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize / load Chroma DB
vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

# --- Upload document endpoint ---
@app.post("/rag/upload")
async def upload_doc(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Load and process
    if file.filename.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path)
    documents = loader.load()

    # Split into chunks and store in Chroma
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    vectorstore.add_documents(chunks)
    vectorstore.persist()

    return {"status": "success", "filename": file.filename}


class RAGQuery(BaseModel):
    query: str


@app.post("/rag/query")
async def rag_query(request: RAGQuery):
    query = request.query

    # Retrieve top 3 relevant chunks
    results = vectorstore.similarity_search(query, k=3)
    context = "\n\n".join([doc.page_content for doc in results])

    # Combine context + user query
    prompt = f"""
You are a helpful assistant that answers based on retrieved context.

Context:
{context}

User query:
{query}

Answer concisely using only the above context.
"""

    # Use same GitHub LLM call
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
    )

    return {"response": response.choices[0].message.content}
