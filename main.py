from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse, Response
from pydantic import BaseModel
from agent import run_financial_advisor

class FinancialRequest(BaseModel):
    query: str

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.get("/api")
def test():
    return {"message": "Hello, World!"}

@app.post("/api/chat/financial-advice")
def get_financial_advice(data: FinancialRequest):

    result = run_financial_advisor(data.query)
    
    return StreamingResponse(content=result, media_type="text/markdown")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)