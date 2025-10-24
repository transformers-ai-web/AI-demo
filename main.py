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


