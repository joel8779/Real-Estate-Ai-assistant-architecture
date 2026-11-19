"""Lightweight Intent Classification Module.

Provides deterministic, rule-based intent routing for incoming queries before
retrieval (ChromaDB / BM25) or LLM synthesis occurs.

Categories:
1. GREETING: Pure conversational greetings (e.g. "hi", "hello", "hey", "good morning").
2. ACKNOWLEDGEMENT: Conversational appreciation (e.g. "thanks", "thank you", "appreciate it").
3. OUT_OF_SCOPE: Explicit non-real-estate inquiries (e.g. cooking, programming, medical, trivia).
4. PROPERTY_QUESTION: Genuine property, listing, development, or corporate real estate inquiries.
"""

import re
from enum import Enum
from typing import Optional, Tuple


class UserIntent(str, Enum):
    GREETING = "greeting"
    ACKNOWLEDGEMENT = "acknowledgement"
    OUT_OF_SCOPE = "out_of_scope"
    PROPERTY_QUESTION = "property_question"


# Real estate, project, developer, location, financial, and company keywords.
# If ANY of these appear in the query, it is NEVER classified as a greeting or acknowledgement.
DOMAIN_KEYWORDS = {
    # Property Types & Attributes
    "property", "properties", "real estate", "listing", "listings", "villa", "villas",
    "apartment", "apartments", "mansion", "mansions", "penthouse", "penthouses",
    "residence", "residences", "residential", "commercial", "office", "land", "plot",
    "plots", "sqft", "square feet", "sqm", "bedroom", "bedrooms", "bhk", "amenity",
    "amenities", "pool", "golf", "beach", "sea", "ocean", "view", "luxury", "branded",
    "off-plan", "off plan", "freehold", "leasehold", "mortgage", "broker", "brokerage",
    "agency", "agent", "rent", "rental", "sale", "buy", "purchase", "sell",

    # Financial & Investment Terms
    "price", "prices", "pricing", "cost", "costs", "budget", "rate", "rates",
    "payment", "payment plan", "installment", "installments", "down payment",
    "invest", "invests", "investing", "investment", "investments", "yield", "roi",

    # Brands, Projects & Partners
    "darglobal", "dar global", "wasalt", "aida", "les vagues", "urban oasis",
    "da vinci", "w residences", "w residences dubai", "w villas", "trump",
    "trump international", "pagani", "missoni", "aston martin", "fendi",
    "elie saab", "mouawad", "dar al arkan",

    # Locations & Geographies
    "oman", "muscat", "yiti", "dubai", "uae", "emirates", "saudi", "saudi arabia",
    "ksa", "riyadh", "jeddah", "dammam", "khobar", "qatar", "doha", "london", "uk",
    "spain", "marbella", "costa del sol", "malaga", "al alya", "al naseem",
    "al rabwah", "al narjis", "al malqa", "al khor",

    # Corporate & Operations
    "ceo", "founder", "chairman", "board", "director", "stock", "shares", "ticker",
    "lse", "london stock exchange", "revenue", "financials", "quarterly", "annual report",
    "headquarters", "hq", "contact", "support", "office location", "customer service"
}

# Greeting phrases (normalized: stripped of punctuation, lowercased)
EXACT_GREETINGS = {
    "hi", "hello", "hey", "hey there", "hi there", "hello there", "good morning",
    "good afternoon", "good evening", "good day", "greetings", "howdy", "salam",
    "salaam", "assalamu alaikum", "as-salamu alaykum", "asalam alaikum", "hola",
    "bonjour", "hi assistant", "hello assistant", "hey assistant", "hi bot",
    "hello bot", "hey bot", "what's up", "whats up", "sup", "how are you",
    "how are you doing", "hows it going", "how's it going", "nice to meet you"
}

# Acknowledgement phrases (normalized: stripped of punctuation, lowercased)
EXACT_ACKNOWLEDGEMENTS = {
    "thanks", "thank you", "thanks a lot", "thank you very much", "thanks so much",
    "thank you so much", "many thanks", "appreciate it", "much appreciated",
    "great thanks", "great thank you", "ok thanks", "okay thanks", "alright thanks",
    "perfect thanks", "perfect thank you", "cool thanks", "got it thanks",
    "understood thanks", "thank you for the help", "thanks for the help",
    "thank you for your help", "thanks for your help"
}

# Explicit out-of-scope intent indicators (cooking, programming, homework, medical, general trivia)
OUT_OF_SCOPE_PATTERNS = [
    r"\b(bake|recipe|recipes|cook|cooking|baking|cake|pizza|pasta|cookie|cookies|soup|ingredient|ingredients)\b",
    r"\b(write\s+(?:a\s+)?(?:python|javascript|java|c\+\+|sql|code|function|script|class|html|css))\b",
    r"\b(debug|quicksort|binary\s+search|linked\s+list|regex|algorithm|leetcode)\b",
    r"\b(write\s+(?:a\s+)?(?:poem|essay|song|lyrics|novel|story)\b)",
    r"\b(diagnose|symptoms?\s+of|prescription|headache|fever|flu|covid|medicine)\b",
    r"\b(who\s+won\s+the\s+(?:world\s+cup|super\s+bowl|champions\s+league|nba|ipl))\b",
    r"\b(weather\s+(?:today|tomorrow|in))\b",
    r"\b(capital\s+of\s+(?:france|germany|japan|italy|canada|brazil))\b",
]


def classify_intent(query: str) -> Tuple[UserIntent, Optional[str]]:
    """Determine the intent of an incoming user query.

    Returns:
        Tuple of (UserIntent, standard_response_or_None)
    """
    cleaned = query.strip()
    lower = cleaned.lower()

    # Normalize apostrophes, punctuation, and extra whitespace
    no_apostrophe = re.sub(r"['’]", "", lower)
    normalized = re.sub(r"[^\w\s]", " ", no_apostrophe)
    normalized = re.sub(r"\s+", " ", normalized).strip()


    if not normalized:
        return UserIntent.GREETING, (
            "Hello! I am the DarGlobal & Wasalt Real Estate Knowledge Assistant. "
            "How can I assist you with luxury developments, property listings, or investment opportunities today?"
        )

    # 1. Critical Safeguard: If query contains ANY domain keywords, it is ALWAYS a PROPERTY_QUESTION.
    # Short queries like "AIDA?", "Oman?", "villas?", "prices?" are guaranteed to pass to RAG.
    for kw in DOMAIN_KEYWORDS:
        if " " in kw:
            if kw in lower:
                return UserIntent.PROPERTY_QUESTION, None
        else:
            if re.search(r"\b" + re.escape(kw) + r"\b", lower):
                return UserIntent.PROPERTY_QUESTION, None

    # 2. Check for exact GREETING match
    if normalized in EXACT_GREETINGS:
        return UserIntent.GREETING, (
            "Hello! I am the DarGlobal & Wasalt Real Estate Knowledge Assistant. "
            "How can I assist you with luxury developments, property listings, or investment opportunities today?"
        )

    # Check for short multi-word greetings (e.g. "hi there assistant", "hello friend")
    greeting_words = {"hi", "hello", "hey", "howdy", "greetings", "salam"}
    tokens = normalized.split()
    if len(tokens) <= 3 and tokens[0] in greeting_words and all(
        t in greeting_words or t in {"there", "everyone", "assistant", "bot", "friend", "team"}
        for t in tokens
    ):
        return UserIntent.GREETING, (
            "Hello! I am the DarGlobal & Wasalt Real Estate Knowledge Assistant. "
            "How can I assist you with luxury developments, property listings, or investment opportunities today?"
        )

    # 3. Check for exact ACKNOWLEDGEMENT match
    if normalized in EXACT_ACKNOWLEDGEMENTS:
        return UserIntent.ACKNOWLEDGEMENT, (
            "You're very welcome! If you have any further questions regarding DarGlobal developments or Wasalt properties, please feel free to ask."
        )

    if len(tokens) <= 4 and ("thanks" in tokens or "thank" in tokens) and not any(
        t in {"question", "about", "project", "where", "how", "what", "which"} for t in tokens
    ):
        return UserIntent.ACKNOWLEDGEMENT, (
            "You're very welcome! If you have any further questions regarding DarGlobal developments or Wasalt properties, please feel free to ask."
        )

    # 4. Check for explicit OUT_OF_SCOPE patterns
    for pattern in OUT_OF_SCOPE_PATTERNS:
        if re.search(pattern, lower):
            return UserIntent.OUT_OF_SCOPE, (
                "Based on the available records, information regarding this question is not currently available in our knowledge base. "
                "I am specialized in DarGlobal luxury developments and Wasalt property listings. "
                "Please consult official representatives for assistance with other inquiries."
            )

    # 5. Default: Genuine property or corporate query forwarded to RAG retrieval pipeline
    return UserIntent.PROPERTY_QUESTION, None
