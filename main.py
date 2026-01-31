#!/usr/bin/env python3
import os, json, logging, requests
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("friday-geo")

app = FastAPI(title="Friday GEO API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalyzeRequest(BaseModel):
    url: str
    competitors: str
    question: str

def get_content(url: str):
    # Firecrawl Logic
    try:
        from firecrawl import Firecrawl
        fc = Firecrawl(api_key=os.getenv("FIRECRAWL_API_KEY"))
        return fc.scrape(url, formats=['markdown']).markdown
    except:
        # Fallback to Requests
        res = requests.get(url, timeout=10)
        return res.text[:5000]

@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    try:
        from google import genai
        from google.genai import types
        
        content = get_content(req.url)
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        
        sys_instr = "Return ONLY JSON with: 'answer' (str), 'entities' (list), 'comparison' (list of {factor, you, competitor}), 'roadmap' (list of {title, desc})."
        
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=f"Analyze {req.url} vs {req.competitors} for: {req.question}. Content: {content[:6000]}",
            config=types.GenerateContentConfig(system_instruction=sys_instr, response_mime_type="application/json")
        )
        return json.loads(response.text)
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
