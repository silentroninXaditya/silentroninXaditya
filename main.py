#!/usr/bin/env python3
import os, json, logging, requests, uvicorn
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# REQUIRED FOR THE 404 FIX
from google import genai
from google.genai import types
from google.genai.types import HttpOptions

# Initialize Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("friday-geo")

app = FastAPI(title="Friday GEO API")

# Perfect CORS for Frontend Connection
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

# --- ADDED ROOT ROUTE HERE ---
@app.get("/")
async def root():
    return {"message": "Friday GEO API is active. Send POST requests to /analyze"}
# -----------------------------

def get_content(url: str):
    """Robust scraper with Firecrawl and Requests fallback."""
    try:
        from firecrawl import Firecrawl
        api_key = os.getenv("FIRECRAWL_API_KEY")
        if not api_key:
            raise ValueError("No Firecrawl Key")
        fc = Firecrawl(api_key=api_key)
        # Fixed: Simplified call to avoid 'params' error in your logs
        result = fc.scrape(url)
        return result.get('markdown', '') or str(result)
    except Exception as e:
        logger.warning(f"Firecrawl fallback triggered: {e}")
        try:
            res = requests.get(url, timeout=10, headers={"User-Agent": "Friday/1.0"})
            return res.text[:5000]
        except:
            return "Could not retrieve site content."

@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    try:
        logger.info(f"Target: {req.url} | Competitors: {req.competitors}")
        
        # 1. Scrape Content
        content = get_content(req.url)
        
        # 2. Initialize Gemini with STABLE API VERSION (Fixes 404)
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="GEMINI_API_KEY missing.")
        
        # This specific config forces the v1 stable endpoint
        client = genai.Client(
            api_key=api_key,
            http_options=HttpOptions(api_version="v1")
        )
        
        # 3. System Instructions
        sys_instr = (
            "You are Friday, a GEO specialist. Return ONLY JSON with: "
            "'answer' (string), 'entities' (list of strings), "
            "'comparison' (list of {factor, you, competitor}), "
            "'roadmap' (list of {title, desc})."
        )
        
        # 4. Generate Analysis
        response = client.models.generate_content(
            model='gemini-1.5-flash',
            contents=f"User Query: {req.question}\nTarget: {req.url}\nCompetitors: {req.competitors}\nContent: {content[:8000]}",
            config=types.GenerateContentConfig(
                system_instruction=sys_instr, 
                response_mime_type="application/json"
            )
        )
        
        return json.loads(response.text)

    except Exception as e:
        logger.error(f"CRITICAL ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
