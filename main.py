import os, json, httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from firecrawl import Firecrawl
from google import genai
from google.genai import types

app = FastAPI()

# Enable CORS so your index.html can talk to this server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Clients
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
firecrawl = Firecrawl(api_key=os.getenv("FIRECRAWL_API_KEY"))

class AnalysisRequest(BaseModel):
    url: str
    competitors: str
    question: str

@app.post("/analyze")
async def analyze_geo(request: AnalysisRequest):
    try:
        # 1. Scrape content
        scrape_result = firecrawl.scrape(request.url, formats=['markdown'])
        markdown_content = getattr(scrape_result, 'markdown', "") or ""

        # 2. GEO Analysis Prompt (Strict JSON for index.html)
        system_instruction = (
            "Return ONLY a JSON object with these keys: "
            "'answer' (string), 'entities' (list), "
            "'comparison' (list of {factor, you, competitor}), "
            "'roadmap' (list of {title, desc})."
        )

        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=f"Analyze {request.url} for query: {request.question}. Content: {markdown_content[:6000]}",
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json"
            )
        )
        return json.loads(response.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
