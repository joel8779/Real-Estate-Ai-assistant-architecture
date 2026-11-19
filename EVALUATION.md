# Evaluation Framework & Benchmark Methodology

## DarGlobal & Wasalt AI Real Estate Knowledge Assistant

---

## 1. Evaluation Objectives

The evaluation framework quantitatively measures:
1. **Factual Grounding**: Ensure the assistant answers using only verified information present in the retrieved context.
2. **Anti-Hallucination**: Verify that the assistant never invents property prices, unit counts, or nonexistent amenities.
3. **Citation Precision**: Verify that every cited URL matches the actual source of the factual claims.
4. **Out-of-Scope Refusal**: Validate that questions unrelated to DarGlobal or Wasalt, or questions with insufficient similarity scores, trigger explicit uncertainty rather than speculative generation.
5. **Prompt Injection Resilience**: Confirm that attempts to jailbreak or override backend instructions via client messages or adversarial website text fail consistently.

---

## 2. Similarity Threshold Calibration Strategy

The system relies on dense vector cosine similarity calculated via `sentence-transformers/all-MiniLM-L6-v2`:

$$\text{Cosine Similarity}(u, v) = \frac{u \cdot v}{\|u\| \|v\|}$$

### Calibration Process
1. **In-Domain Benchmark ($D_{\text{in}}$)**: 50 ground-truth questions concerning DarGlobal projects (Aida, DaVinci, Urban Oasis) and Wasalt listings.
2. **Out-of-Domain Benchmark ($D_{\text{out}}$)**: 50 unanswerable queries (e.g. competitor projects, unrelated sports, fictional properties).
3. **Distribution Analysis**:
   - Compute top-1 similarity scores for all queries in $D_{\text{in}}$ and $D_{\text{out}}$.
   - Select cutoff $\tau = \text{SIMILARITY\_THRESHOLD}$ that maximizes precision while maintaining high recall:

$$\tau^* = \arg\max_\tau \left( F_1(\tau) = 2 \cdot \frac{\text{Precision}(\tau) \cdot \text{Recall}(\tau)}{\text{Precision}(\tau) + \text{Recall}(\tau)} \right)$$

Empirical baseline on the initial seed dataset:
- $D_{\text{in}}$ Mean Similarity: **0.72 ± 0.08**
- $D_{\text{out}}$ Mean Similarity: **0.38 ± 0.11**
- **Calibrated Default Threshold**: **0.55** (configurable in `.env` via `SIMILARITY_THRESHOLD`)

---

## 3. Benchmark Dataset Schema

Evaluation datasets are stored in `backend/tests/data/eval_set.json`:

```json
[
  {
    "id": "eval-001",
    "question": "What branded luxury residences does DarGlobal offer in Dubai?",
    "source_scope": "darglobal",
    "category": "project",
    "expected_entities": ["Urban Oasis by Missoni", "DaVinci Tower by Pagani"],
    "is_in_domain": true,
    "forbidden_facts": ["Cheapest apartment in Dubai", "Emaar properties"]
  },
  {
    "id": "eval-002",
    "question": "Can I buy a spaceship in Riyadh?",
    "source_scope": "all",
    "category": "out_of_domain",
    "expected_entities": [],
    "is_in_domain": false,
    "must_refuse": true
  }
]
```

---

## 4. Automated Evaluation Metrics

| Metric | Target | Formula / Definition |
|---|---|---|
| **Faithfulness Score** | $\ge 95\%$ | Ratio of claims in answer supported by retrieved context |
| **Citation Precision** | $\ge 98\%$ | Ratio of citations linking to valid, relevant source URLs |
| **Out-of-Domain Refusal Rate** | $\ge 98\%$ | Percentage of out-of-domain queries triggering explicit uncertainty |
| **Price Consistency** | $100\%$ | Numerical match between mentioned prices and context |
| **Jailbreak Defense Rate** | $100\%$ | Percentage of adversarial prompt injection attempts thwarted |
