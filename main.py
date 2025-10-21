from fastapi import FastAPI
import uvicorn
import llm


app = FastAPI()

@app.get("/")
def read_root():
    response = llm.get_llm_response("where do you live?")
    return {"response": response}