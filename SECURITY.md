# Security Architecture & Threat Model

## DarGlobal & Wasalt AI Real Estate Knowledge Assistant

---

## 1. Threat Model & Mitigation Strategy

| Threat Vector | Severity | Mitigation Implementation |
|---|---|---|
| **Prompt Injection via Client Messages** | High | Client payloads strictly accept `user` and `assistant` roles; `system` role payloads are rejected with HTTP 422. Prompts are assembled using strict delimitation. |
| **Indirect Prompt Injection via Ingested Web Content** | High | All scraped content is treated as untrusted data. Retrieved chunks are isolated inside `<untrusted_knowledge_context>` tags. The backend instruction hierarchy commands the LLM to ignore directives embedded within context. |
| **Hallucination of Property Prices / Amenities** | High | Retrieval similarity cutoff (`SIMILARITY_THRESHOLD`) rejects sub-threshold queries. Explicit uncertainty instructions require the model to state when pricing is unlisted or missing. |
| **Denial of Service / Resource Exhaustion** | Medium | `RequestSizeLimitMiddleware` terminates requests exceeding 64 KB before memory allocation. Sliding-window rate limiter limits IP to 60 req/min. |
| **Cross-Origin Abuse (CORS)** | Medium | Restrictive CORS middleware configured with explicit origins (`CORS_ORIGINS`). Wildcard origin is disabled by default in production. |
| **Container Privilege Escalation** | High | Multi-stage Docker builds execute exclusively under non-root service accounts: UID 10001 (`appuser`) for backend and UID 101 (`nginx-unprivileged`) for frontend. |
| **Secret & Credential Leakage** | Critical | API keys and tokens are loaded strictly via environment variables (`.env`). `.gitignore` excludes all secret files, `.env`, and persistent vector storage. Structured loggers filter authorization headers. |

---

## 2. Inbound Message Validation

Clients are forbidden from manipulating system instructions:

```python
class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]  # Rejects "system"
    content: str = Field(..., min_length=1, max_length=4000)
```

Attempting to submit `{"role": "system", ...}` results in:
```json
{
  "detail": [
    {
      "type": "literal_error",
      "loc": ["body", "messages", 0, "role"],
      "msg": "Input should be 'user' or 'assistant'",
      "input": "system"
    }
  ]
}
```

---

## 3. Container Security Verification

Both services run as non-privileged users:

```bash
# Verify backend runs as non-root
docker compose exec backend whoami
# Output: appuser (uid: 10001)

# Verify frontend runs as non-root
docker compose exec frontend whoami
# Output: nginx (uid: 101)
```

---

## 4. Reporting Vulnerabilities

If you identify a security defect, do not disclose it publicly. Report details with reproduction steps to the security maintainers.
