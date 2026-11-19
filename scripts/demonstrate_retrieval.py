"""Live Retrieval Demonstration Script.

Runs 5 live queries across the indexed knowledge base to demonstrate:
1. DarGlobal company info query
2. DarGlobal project query (AIDA Oman & Trump Golf)
3. Wasalt listing query (Al-Narjis apartment or Al-Mahdiyah villa)
4. Cross-source real estate query (luxury villas)
5. Out-of-domain / unanswerable query (Tokyo penthouse)
"""

import json
from pathlib import Path
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)

demo_queries = [
    {
        "category": "DarGlobal Company Information",
        "query": "Where is DarGlobal headquartered and what is its London Stock Exchange ticker?",
        "source_filter": "darglobal",
    },
    {
        "category": "DarGlobal Project",
        "query": "Tell me about AIDA in Muscat Oman and the Trump International partnership",
        "source_filter": "darglobal",
    },
    {
        "category": "Wasalt Property Listing",
        "query": "Villa for sale in Al-Mahdiyah Riyadh with land area and street width",
        "source_filter": "wasalt",
    },
    {
        "category": "Cross-Source Query",
        "query": "Show me luxury villas across Spain and Saudi Arabia",
        "source_filter": "all",
    },
    {
        "category": "Out-of-Domain / Unanswerable Query",
        "query": "What are the prices for DarGlobal luxury penthouses in Tokyo, Japan?",
        "source_filter": "all",
    },
]

print("\n" + "=" * 80)
print("LIVE RETRIEVAL DEMONSTRATION — DARGLOBAL & WASALT KNOWLEDGE ASSISTANT")
print("=" * 80 + "\n")

for idx, q_info in enumerate(demo_queries, 1):
    payload = {
        "query": q_info["query"],
        "top_k": 3,
        "source_filter": q_info["source_filter"],
        "method": "hybrid",
    }

    res = client.post("/api/v1/retrieve", json=payload)
    data = res.json()

    print(f"[{idx}] {q_info['category'].upper()}")
    print(f"Query:        '{data['query']}'")
    print(f"Filter:       source={q_info['source_filter']}")
    print(f"Latency:      {data['latency_ms']} ms | Total matches: {data['total_found']}")

    if data["items"]:
        top = data["items"][0]
        print(f"Top Result:   {top['title']} ({top['source'].upper()})")
        print(f"URL:          {top['url']}")
        print(f"Provenance:   {top['provenance']}")
        print(f"RRF Score:    {top['score']:.4f}")
        if top['semantic_similarity'] is not None:
            print(f"Similarity:   {top['semantic_similarity']:.4f}")
        print("Snippet:")
        for line in top["text_content"].strip().split("\n")[:4]:
            print(f"  > {line}")
    else:
        print("Top Result:   No matches found (out-of-domain)")
    print("-" * 80 + "\n")
