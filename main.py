from fastapi import FastAPI, UploadFile, File, Query
import uvicorn
from chat_api import llm
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings
import os
import tempfile

from openai import OpenAI

client = OpenAI(
    base_url="https://models.github.ai/inference",
    api_key=os.environ["GITHUB_TOKEN"]
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str


class RAGRequest(BaseModel):
    query: str

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    user_message = request.message
    response = llm.get_llm_response(user_message)
    print(response)
    return {"response": response}

retriever = None

@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global retriever

    # Save temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(await file.read())
        temp_path = tmp_file.name

    # Load & Split PDF
    loader = PyPDFLoader(temp_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    all_splits = text_splitter.split_documents(docs)

    # --- 🔄 Replace HuggingFace with GitHub Models API ---

    # Using LangChain’s OpenAIEmbeddings but pointing to GitHub endpoint
    embedding_function = OpenAIEmbeddings(
        model="text-embedding-3-small",  # You can also try text-embedding-3-large
        openai_api_key=os.environ["GITHUB_TOKEN"],
        openai_api_base="https://models.github.ai/inference"
    )

    # Build Chroma vectorstore
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=embedding_function)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

    # Cleanup
    os.remove(temp_path)

    return {"message": "PDF uploaded and processed successfully!"}

@app.post("/api/askdoc")
async def ask_doc_endpoint(request: RAGRequest):
    query = request.query
    docs = retriever.get_relevant_documents(query)
    context = "\n\n".join([d.page_content for d in docs])

    # Call GitHub Models API for completion
    completion = client.chat.completions.create(
        model="gpt-4.1-mini",  # or any GitHub-hosted model you prefer
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers based on provided context."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"}
        ],
    )

    answer = completion.choices[0].message.content
    return {"response": answer}



