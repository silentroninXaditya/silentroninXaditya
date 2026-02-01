import os, json, logging, httpx, uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import AsyncGroq # Ensure 'groq' is in requirements.txt

# Logging Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("friday-geo-v3")

app = FastAPI(title="Friday GEO API v3.0 (Groq Edition)")

# CORS: Configured for your GitHub Pages frontend
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

async def get_jina_content(url: str):
    """
    Scraper: Jina Reader API.
    Converts any URL to LLM-ready Markdown.
    """
    jina_key = os.getenv("JINA_API_KEY")
    headers = {"X-Return-Format": "markdown"}
    if jina_key:
        headers["Authorization"] = f"Bearer {jina_key}"
        
    try:
        jina_url = f"https://r.jina.ai/{url}"
        async with httpx.AsyncClient() as client:
            # 20s timeout is safer for heavy pages in 2026
            response = await client.get(jina_url, headers=headers, timeout=20.0)
            if response.status_code == 200:
                return response.text[:15000] # Trim to stay within token limits
            return f"Error: Scraper returned status {response.status_code}"
    except Exception as e:
        logger.error(f"Scrape failed: {e}")
        return "Scraping failed due to connection error."

@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    try:
        content = await get_jina_content(req.url)
        groq_key = os.getenv("GROQ_API_KEY")
        if not groq_key:
            raise HTTPException(status_code=500, detail="GROQ_API_KEY missing")
        
        client = AsyncGroq(api_key=groq_key)
        
        # System instruction tuned for 100-150 line expert analysis
        system_msg = (
            "You are Friday, a GEO Intelligence Expert. Analyze content deeply. "
            "You MUST return a JSON object with a field 'big_chat' containing a 100-150 line expert report "
            "using Markdown (bolding, lists). Also fill 'answer', 'entities', 'comparison', and 'roadmap'."
        )

        chat_completion = await client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": f"URL: {req.url}\nCompetitors: {req.competitors}\nQuestion: {req.question}\n\nContent: {content}"}
            ],
            model="llama-3.3-70b-versatile",
            response_format={"type": "json_object"},
            max_tokens=4000, # Large overhead for 150 lines
            temperature=0.5
        )
        
        return json.loads(chat_completion.choices[0].message.content)

    except Exception as e:
        logger.error(f"Friday Analysis Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "Friday 3.0 is Online", "model": "Llama 3.3 70B"}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
