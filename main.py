#!/usr/bin/env python3
import os, json, logging, httpx, uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Modern GenAI SDK
from google import genai
from google.genai import types

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("friday-geo")

app = FastAPI(title="Friday GEO API v3.0")

# CORS remains essential for your GitHub Pages frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalyzeRequest(BaseModel):
    url: str
    competitors: str
    question: str

async def get_jina_content(url: str):
    """
    Better & Free Scraper: Jina Reader.
    Converts any URL to LLM-ready Markdown instantly.
    """
    try:
        # We prepend r.jina.ai to the target URL
        jina_url = f"https://r.jina.ai/{url}"
        async with httpx.AsyncClient() as client:
            # No API key needed for basic usage (20 req/min)
            response = await client.get(jina_url, timeout=20)
            if response.status_code == 200:
                return response.text[:15000] # Clean Markdown content
            return "Scraper returned an error."
    except Exception as e:
        logger.error(f"Jina Scraping Error: {e}")
        return "Content unavailable."

@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    try:
        logger.info(f"Friday is analyzing: {req.url}")
        
        # 1. Scrape using Jina (Better than Firecrawl for free tier)
        content = await get_jina_content(req.url)
        
        api_key = os.getenv("GEMINI_API_KEY")
        client = genai.Client(api_key=api_key)
        
        # 2. Structured Output Schema (Ensures your UI never breaks)
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
            },
            "required": ["answer", "entities", "comparison", "roadmap"]
        }

        # 3. Use Gemini 2.5 Flash-Lite (Best Free Limits: 1000 requests/day)
        response = client.models.generate_content(
            model='gemini-2.5-flash-lite',
            contents=f"Perform GEO Audit.\nQuery: {req.question}\nSite Content: {content}",
            config=types.GenerateContentConfig(
                system_instruction="You are Friday, a GEO specialist. Return analysis in strict JSON.",
                response_mime_type="application/json",
                response_schema=response_schema,
                temperature=0.1
            )
        )
        
        return json.loads(response.text)

    except Exception as e:
        logger.error(f"Friday Analysis Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "Friday 3.0 is Online"}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
