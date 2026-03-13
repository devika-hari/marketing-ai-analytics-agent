import os
import pandas as pd
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings  # ← changed
from langchain_core.tools import Tool
from langgraph.prebuilt import create_react_agent
from langchain_chroma import Chroma
from langchain_core.documents import Document

load_dotenv() # ← loads GROQ_API_KEY from .env automatically

# Load dataset
df = pd.read_csv("data/campaign_data.csv")

# Create KPIs
df["roi"] = df["revenue"] / df["spend"]
df["cpa"] = df["spend"] / df["conversions"]
df["conversion_rate"] = df["conversions"] / df["clicks"]

# -------- RAG SETUP --------

documents = []

for _, row in df.iterrows():
    text = f"""
Campaign: {row['campaign']}
Channel: {row['channel']}
Spend: {row['spend']}
Revenue: {row['revenue']}
ROI: {row['roi']}
CPA: {row['cpa']}
Conversion Rate: {row['conversion_rate']}
"""
    documents.append(Document(page_content=text))

#print("Loading local embedding model (first run downloads ~80MB)...")
#embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

print("Loading local embedding model...")
embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")  # ← changed

vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    collection_name="campaigns"
)

retriever = vectorstore.as_retriever()



def analyze_performance(query: str) -> str:
    # Groq passes the argument as __arg1 sometimes
    query = query.lower()

    if "roi" in query:
        best = df.loc[df["roi"].idxmax()]
        return f"Best ROI campaign: {best['campaign']} with ROI {best['roi']:.2f}"

    if "cpa" in query:
        worst = df.loc[df["cpa"].idxmax()]
        return f"Worst CPA campaign: {worst['campaign']} with CPA {worst['cpa']:.2f}"

    if "conversion" in query:
        best = df.loc[df["conversion_rate"].idxmax()]
        return f"Best conversion rate: {best['campaign']} with rate {best['conversion_rate']:.2%}"

    if "spend" in query:
        return df[["campaign", "spend"]].sort_values("spend", ascending=False).to_string(index=False)

    # Default: return full summary
    summary = df[["campaign", "roi", "cpa", "conversion_rate"]].copy()
    summary["roi"] = summary["roi"].map("{:.2f}".format)
    summary["cpa"] = summary["cpa"].map("{:.2f}".format)
    summary["conversion_rate"] = summary["conversion_rate"].map("{:.2%}".format)
    return summary.to_string(index=False)


def search_campaigns(query: str) -> str:
    docs = retriever.invoke(query)
    return "\n".join([d.page_content for d in docs])


def marketing_recommendations(query: str) -> str:
    avg_roi = df["roi"].mean()
    avg_cpa = df["cpa"].mean()

    high_roi = df[df["roi"] > avg_roi]
    high_cpa = df[df["cpa"] > avg_cpa]

    recommendations = []

    if not high_roi.empty:
        best_channels = high_roi["channel"].unique()
        recommendations.append(
            f"Consider increasing spend on channels with strong ROI such as {', '.join(best_channels)}."
        )

    if not high_cpa.empty:
        weak_campaigns = high_cpa["campaign"].tolist()[:3]
        recommendations.append(
            f"Optimize or review campaigns with high acquisition costs such as {', '.join(weak_campaigns)}."
        )

    recommendations.append(
        "Email campaigns show strong conversion efficiency and could be expanded for lifecycle marketing."
    )

    return "\n".join(recommendations)



# -------- AGENT TOOLS --------

tools = [
    Tool(
        name="Campaign_Search",
        func=search_campaigns,
        description="Search campaign data by name or channel. Input: a campaign name or channel like 'email' or 'Black_Friday'."
    ),
    Tool(
        name="Campaign_Analytics",
        func=analyze_performance,
        description="Get campaign metrics. Input: one of 'roi', 'cpa', 'conversion', 'spend', or 'summary'."
    ),
    Tool(
        name="Marketing_Strategy_Advisor",
        func=marketing_recommendations,
        description="Get marketing recommendations and optimization advice. Input: any string like 'recommendations'."
    )
]
'''

**Questions to try:**
```
# Analytics
which campaign has the highest roi?
which campaign has the worst cpa?
show me conversion rates
how much did each campaign spend?
give me a full summary

# Search
tell me about the Black_Friday campaign
what campaigns ran on email?
show me social media campaigns

# Strategy
what are your marketing recommendations?
how should i optimize my campaigns?
where should i increase spend?
'''

# -------- LLM + AGENT --------

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    max_retries=2,
    request_timeout=30
)

agent = create_react_agent(llm, tools)

print("\nMarketing AI Analytics Agent (Groq + local embeddings)")
print("Ask questions about campaign performance. Type 'exit' to stop.\n")

while True:
    q = input("Your question: ").strip()

    if not q:
        continue

    if q.lower() == "exit":
        print("Goodbye!")
        break

    try:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": q}]},
            config={"recursion_limit": 10}
        )
        print("\nAI Insight:")
        print(result["messages"][-1].content)
        print("\n--------------------------\n")

    except Exception as e:
        print(f"\nError: {e}\n")