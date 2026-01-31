#!/usr/bin/env python3
import os, json, logging, requests, uvicorn
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Using the modern GenAI SDK for maximum free-tier stability
from google import genai
from google.genai import types

# Initialize Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("friday-geo")

app = FastAPI(title="Friday GEO API")

# CORS Setup - Essential for your GitHub frontend to communicate with Render
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

@app.api_route("/", methods=["GET", "HEAD"])
async def root():
    return {"message": "Friday GEO API is active. Ready for free-tier analysis."}

def get_content(url: str):
    """Scraper: Uses Firecrawl with a robust Requests fallback."""
    try:
        from firecrawl import Firecrawl
        api_key = os.getenv("FIRECRAWL_API_KEY")
        if api_key:
            fc = Firecrawl(api_key=api_key)
            result = fc.scrape(url)
            # Handle both dict and object returns from Firecrawl
            return result.get('markdown', '') if isinstance(result, dict) else str(result)
    except Exception as e:
        logger.warning(f"Firecrawl fallback: {e}")
    
    try:
        res = requests.get(url, timeout=10, headers={"User-Agent": "Friday/1.0"})
        return res.text[:10000] # Limit context to save tokens
    except:
        return "Content unavailable."

@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    try:
        logger.info(f"Processing: {req.url}")
        content = get_content(req.url)
        
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="GEMINI_API_KEY missing.")
        
        # STABILITY FIX: Initialize standard client without forcing internal versioning
        # This prevents the 400 'systemInstruction' naming error.
        client = genai.Client(api_key=api_key)
        
        # PRO SCHEMA: Hard-coding the output format ensures Gemini never breaks your UI
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
                            "factor": {"type": "STRING"}, "you": {"type": "STRING"}, "competitor": {"type": "STRING"}
                        }
                    }
                },
                "roadmap": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "title": {"type": "STRING"}, "desc": {"type": "STRING"}
                        }
                    }
                }
            },
            "required": ["answer", "entities", "comparison", "roadmap"]
        }

        # The Prompt: Feeding everything into the Flash model (Fast & Free)
        response = client.models.generate_content(
            model='gemini-1.5-flash',
            contents=f"Analyze this for GEO. Query: {req.question}\nSite: {req.url}\nCompetitors: {req.competitors}\nData: {content[:15000]}",
            config=types.GenerateContentConfig(
                system_instruction="You are Friday, a GEO specialist. Extract insights and return valid JSON.",
                response_mime_type="application/json",
                response_schema=response_schema,
                temperature=0.2
            )
        )
        
        return json.loads(response.text)

    except Exception as e:
        logger.error(f"Friday Analysis Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "alive"}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
