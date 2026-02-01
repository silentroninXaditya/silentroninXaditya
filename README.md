# 🤖 Friday 3.0: The GEO Intelligence Engine

Friday 3.0 is a specialized **Generative Engine Optimization (GEO)** tool designed to bridge the gap between static web content and AI-driven search visibility. Built with a high-performance FastAPI backend and an adaptive Llama 3.3/8B intelligence node.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-05998b.svg)](https://fastapi.tiangolo.com/)

---

## 🚀 Unique Selling Points (USPs)

* **Resilient Scraping Node:** Bypasses bot-detection on premium sites (like Tesla and Lucid) using custom User-Agent injection and Jina AI integration.
* **Dual-Model Intelligence:** Dynamically switches between **Llama 3.3 70B** (for deep expert analysis) and **Llama 3.1 8B** (for rapid high-frequency testing).
* **Context Optimization:** Implements aggressive token-saving strategies, truncating context windows to 5,000 characters to stay within free-tier limits without sacrificing quality.
* **Zero-Protocol Handling:** Automatically sanitizes and repairs malformed URLs (adding missing `https://`) before scraping.

---

## 🏗️ System Architecture

Friday 3.0 operates as a distributed system:
1.  **Frontend:** A responsive UI hosted on **GitHub Pages**.
2.  **Backend:** A production-ready **FastAPI** server on **Render**.
3.  **Extraction:** **Jina AI Reader** converts raw HTML into LLM-ready Markdown.
4.  **Inference:** **Groq LPU** (Language Processing Unit) generates insights in sub-second speeds using Llama 3.x.



---

## 🛠️ Technical Logic & Hurdles

### 1. Overcoming the "Token Wall"
Initially, the 70B model's strict rate limits (100k TPD) caused system hangs. We optimized the prompt logic and reduced the `context_window` to **5,000 characters**, allowing for 40% more daily analyses on the same API quota.

### 2. The "Scraper War"
To solve the `403 Forbidden` errors from high-security sites, we engineered a custom header stack:
```python
headers = {
    "X-Return-Format": "markdown",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ..."
}
