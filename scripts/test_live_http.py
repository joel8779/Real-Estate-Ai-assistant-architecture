"""Test live running HTTP endpoints for all queries."""

import json
import httpx

queries = [
    "hi",
    "hello",
    "hey",
    "good morning",
    "thanks",
    "what's up",
    "tell me about DarGlobal",
    "what is AIDA?",
]

client = httpx.Client(timeout=30.0)

print("\n--- NON-STREAMING /api/v1/chat ---")
for q in queries:
    print(f"\nQUERY: {q}")
    res = client.post("http://127.0.0.1:8000/api/v1/chat", json={"messages": [{"role": "user", "content": q}]})
    data = res.json()
    print(f"  status: {res.status_code}")
    print(f"  data_mode: {data.get('data_mode')}")
    print(f"  model: {data.get('model')}")
    print(f"  citations count: {len(data.get('citations', []))}")
    print(f"  provenance_warning: {data.get('provenance_warning')}")
    print(f"  answer: {data.get('answer', '')[:120]}")

print("\n--- STREAMING /api/v1/chat/stream ---")
for q in ["hi", "what's up", "tell me about DarGlobal"]:
    print(f"\nSTREAM QUERY: {q}")
    with client.stream("POST", "http://127.0.0.1:8000/api/v1/chat/stream", json={"messages": [{"role": "user", "content": q}]}) as res:
        events = []
        for line in res.iter_lines():
            if line.startswith("event: "):
                event_name = line[7:].strip()
            elif line.startswith("data: "):
                event_data = json.loads(line[6:].strip())
                events.append((event_name, event_data))

        event_names = [e[0] for e in events]
        print(f"  events emitted: {set(event_names)}")
        has_status = "status" in event_names
        print(f"  status event present: {has_status}")
        cit = next((d for n, d in events if n == "citations"), {})
        print(f"  citations count: {len(cit.get('citations', []))}")
        print(f"  data_mode: {cit.get('data_mode')}")
        print(f"  model: {cit.get('model')}")
        print(f"  provenance_warning: {cit.get('provenance_warning')}")
        comp = next((d for n, d in events if n == "complete"), {})
        print(f"  complete data_mode: {comp.get('data_mode')}, model: {comp.get('model')}")
