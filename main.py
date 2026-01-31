#!/usr/bin/env python3
import os, json, logging, requests, uvicorn
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# REQUIRED FOR STABLE v1 HANDSHAKE
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

# ROOT ROUTE - Supports GET and HEAD for Render Health Checks
@app.api_route("/", methods=["GET", "HEAD"])
async def root():
    return {"message": "Friday GEO API is active. Send POST requests to /analyze"}

def get_content(url: str):
    """Robust scraper with Firecrawl and Requests fallback."""
    try:
        from firecrawl import Firecrawl
        api_key = os.getenv("FIRECRAWL_API_KEY")
        if api_key:
            fc = Firecrawl(api_key=api_key)
            result = fc.scrape(url)
            return result.get('markdown', '') or str(result)
    except Exception as e:
        logger.warning(f"Firecrawl fallback triggered: {e}")
    
    try:
        res = requests.get(url, timeout=10, headers={"User-Agent": "Friday/1.0"})
        return res.text[:8000]
    except:
        return "Could not retrieve site content."

@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    try:
        logger.info(f"Target: {req.url} | Competitors: {req.competitors}")
        
        content = get_content(req.url)
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="GEMINI_API_KEY missing.")
        
        # 1. FORCE STABLE v1 VERSION (Safely)
        client = genai.Client(
            api_key=api_key,
            http_options=HttpOptions(api_version="v1")
        )
        
        # 2. PRO ADDON: Strict Response Schema
        # This tells Gemini 1.5 EXACTLY what the JSON should look like.
        response_schema = {
            "type": "OBJECT",
            "properties": {
                "answer": {"type": "STRING"},
                "entities": {"type": "ARRAY", "items": {"type": "STRING"}},
                "comparison": {
                    "type": "ARRAY", 
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "factor": {"type": "STRING"},
                            "you": {"type": "STRING"},
                            "competitor": {"type": "STRING"}
                        }
                    }
                },
                "roadmap": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "title": {"type": "STRING"},
                            "desc": {"type": "STRING"}
                        }
                    }
                }
            }
        }
        
        sys_instr = "You are Friday, a GEO specialist. Analyze the content and return findings in the requested JSON schema."
        
        # 3. GENERATE
        response = client.models.generate_content(
            model='gemini-1.5-flash',
            contents=f"User Query: {req.question}\nTarget: {req.url}\nCompetitors: {req.competitors}\nContent: {content[:10000]}",
            config=types.GenerateContentConfig(
                system_instruction=sys_instr, 
                response_mime_type="application/json",
                response_schema=response_schema, # Ensures perfect JSON every time
                temperature=0.1
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
