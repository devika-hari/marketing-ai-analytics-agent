# Marketing AI Analytics Agent (RAG + LLM Tools)

A lightweight AI-powered analytics assistant that answers marketing performance questions using campaign data.

The project demonstrates how modern marketing data platforms can combine **data engineering, vector search, and LLM agents** to generate insights and recommendations.

---

## Features

- Retrieval-Augmented Generation (RAG) for campaign information
- Vector Database: Chroma; Embeddings: BAAI/bge-small-en-v1.5
- LLM agent powered by **Groq (free tier)** using `llama-3.3-70b-versatile`
- Automated marketing strategy recommendations
- KPI calculations such as ROI, CPA, and Conversion Rate

---

## Example Questions

```
# Analytics
which campaign has the highest roi?
which campaign has the worst cpa?
show me conversion rates
how much did each campaign spend?
give me a full summary

# Search
tell me about the Holiday Offer campaign
what campaigns ran on email?
show me Google Ads campaigns

# Strategy
what are your marketing recommendations?
how should i optimize my campaigns?
where should i increase spend?
```

---

## Project Architecture

<img width="445" height="380" alt="image" src="https://github.com/user-attachments/assets/9db16feb-8c64-40c4-ac86-4a1980bb93f1" />

---

## Dataset

The project includes a synthetic dataset of marketing campaigns across multiple channels:

- Google Ads
- Facebook Ads
- Instagram Ads
- Email Marketing

Input metrics:

- Spend, Clicks, Conversions, Revenue

Derived KPIs:

- ROI (Revenue / Spend)
- CPA (Spend / Conversions)
- Conversion Rate (Conversions / Clicks)

Sample `data/campaign_data.csv`:

```csv
campaign,channel,spend,clicks,conversions,revenue
Spring Sale,Google Ads,5000,1200,90,15000
```

---

## Installation

```bash
pip install pandas langchain langchain-groq langchain-chroma langchain-community langchain-huggingface fastembed chromadb langgraph python-dotenv
```

---

## Setup

This project requires **only one API key** — Groq. Everything else (embeddings, vector store, data processing) runs locally with no keys needed.

| Component | Key needed? |
|---|---|
| Groq LLM | ✅ Groq key (free, no credit card) |
| FastEmbed embeddings | ❌ runs locally |
| Chroma vector store | ❌ runs locally |
| Pandas / data processing | ❌ no key needed |

1. Get a free API key from [console.groq.com](https://console.groq.com) (no credit card required)

2. Create a `.env` file in the project root:

```
GROQ_API_KEY=gsk_...
```

3. Add your campaign data to `data/campaign_data.csv`

---

## Run the Agent

```bash
python agent.py
```

You can then interact with the AI assistant through the terminal.

---

## Example Output

```
Your question: which campaign has the highest roi?

AI Insight:
The campaign with the highest ROI is Email Blast with a ROI of 12.00.

```

---

## Tech Stack

| Component | Tool |
|---|---|
| LLM | Groq (llama-3.3-70b-versatile) — free |
| Embeddings | BAAI/bge-small-en-v1.5 via FastEmbed — local, free |
| Vector Store | Chroma — local |
| Agent Framework | LangGraph |
| Data | Pandas |

---

## Future Improvements

- Connect to real marketing APIs (Google Ads / Meta Ads)
- Deploy as a Streamlit or FastAPI application
- Add real-time campaign monitoring dashboards
- Expand RAG with marketing documentation or campaign briefs
- Add multi-turn conversation memory

---

## Author

Devika Hari
Data Engineer
