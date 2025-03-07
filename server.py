from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse, Response
from pydantic import BaseModel
from agent import run_financial_advisor
import os
import json

class FinancialRequest(BaseModel):
    query: str

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.post("/api/chat/financial-advice")
def get_financial_advice(data: FinancialRequest):

    # chart_path = "portfolio_chart.png"
    # if os.path.exists(chart_path):
    #     return FileResponse(chart_path, media_type="image/png")
    # return StreamingResponse(run_financial_advisor(data.query), media_type="application/json")
    result = run_financial_advisor(data.query)
    
    
    return StreamingResponse(content=result, media_type="text/plain")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)