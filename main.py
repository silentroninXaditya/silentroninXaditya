import os, json, logging, httpx, uvicorn, asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import AsyncGroq 

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
    jina_key = os.getenv("JINA_API_KEY")
    headers = {"X-Return-Format": "markdown"}
    if jina_key:
        headers["Authorization"] = f"Bearer {jina_key}"
        
    try:
        # FIX: URL Preprocessing Node
        clean_url = url.strip().replace(" ", "")
        if not clean_url.startswith(("http://", "https://")):
            clean_url = f"https://{clean_url}"
            
        jina_url = f"https://r.jina.ai/{clean_url}"
        async with httpx.AsyncClient() as client:
            # FIX: Timeout Node (15s to stay well under Render's 30s limit)
            response = await client.get(jina_url, headers=headers, timeout=15.0)
            if response.status_code == 200:
                # FIX: Context Window Node (Trimming to 8000 chars)
                return response.text[:8000] 
            return "" 
    except Exception as e:
        logger.error(f"Scrape failed for {url}: {e}")
        return ""

@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    try:
        # Step 1: Parallel Scraping (Target + Competitors)
        # This ensures we actually have data on the competition
        comp_list = [c.strip() for c in req.competitors.split(",") if c.strip()][:2]
        urls_to_scrape = [req.url] + comp_list
        
        # Gather all content simultaneously
        scraping_results = await asyncio.gather(*[get_jina_content(u) for u in urls_to_scrape])
        
        target_content = scraping_results[0]
        competitor_context = "\n\n".join(scraping_results[1:])

        groq_key = os.getenv("GROQ_API_KEY")
        if not groq_key:
            raise HTTPException(status_code=500, detail="GROQ_API_KEY missing")
        
        client = AsyncGroq(api_key=groq_key)
        
        # Step 2: Expert Intelligence Prompt
        system_msg = (
            "You are Friday, a GEO Intelligence Expert. Analyze content deeply. "
            "You MUST return a JSON object with: "
            "1. 'big_chat': A 100-150 line expert report using Markdown (bolding, lists). "
            "2. 'answer': A concise summary. "
            "3. 'entities': A list of key SEO/GEO entities identified. "
            "4. 'comparison': A list of {factor, you, competitor} objects. "
            "5. 'roadmap': A list of {title, desc} objects."
        )

        user_content = (
            f"TARGET SITE: {req.url}\nCONTENT: {target_content}\n\n"
            f"COMPETITOR DATA: {competitor_context}\n\n"
            f"USER QUERY: {req.question}"
        )

        chat_completion = await client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_content}
            ],
            model="llama-3.3-70b-versatile",
            response_format={"type": "json_object"},
            max_tokens=4000,
            temperature=0.4 # Lower temperature for more stable JSON formatting
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
