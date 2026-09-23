"""LLM Bid Advisor — local Qwen 3.5:9B via Ollama.

Provides a chat interface for asking the local LLM about GeM bid
documents. All communication with Ollama happens over its REST API
(http://localhost:11434) so no extra Python packages are needed beyond
the already-installed ``requests``.
"""

from __future__ import annotations

import json
import re
from typing import Generator
import requests

OLLAMA_BASE = "http://localhost:11434"
DEFAULT_MODEL = "qwen3.5:9b"


# ── Health check ─────────────────────────────────────────────────────

def check_ollama_available(model: str = DEFAULT_MODEL) -> dict:
    try:
        r = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
        r.raise_for_status()
        models = [m["name"] for m in r.json().get("models", [])]
        found = any(model.split(":")[0] in m for m in models)
        if found:
            return {"ok": True, "detail": f"Model `{model}` ready"}
        return {
            "ok": False,
            "detail": (
                f"Ollama is running but model `{model}` not found. "
                f"Available: {', '.join(models) or '(none)'}. "
                f"Pull it with: `ollama pull {model}`"
            ),
        }
    except requests.ConnectionError:
        return {"ok": False, "detail": "Cannot reach Ollama at localhost:11434. Is it running?"}
    except Exception as exc:
        return {"ok": False, "detail": f"Ollama check failed: {exc}"}


# ── PDF Text Extraction ──────────────────────────────────────────────

def extract_full_pdf_text(pdf_path) -> str | None:
    """Extract complete text directly from a local PDF file using PyMuPDF."""
    try:
        import fitz
        doc = fitz.open(pdf_path)
        pages_text = []
        for i, page in enumerate(doc):
            t = page.get_text().strip()
            if t:
                pages_text.append(f"--- Page {i+1} ---\n{t}")
        doc.close()
        return "\n\n".join(pages_text) if pages_text else None
    except Exception:
        return None


# ── Prompt construction ──────────────────────────────────────────────

def build_system_prompt(bid_metadata: dict, bid_chunks: list[str]) -> str:
    """Create a system prompt that gives the LLM full context of a bid."""
    meta_lines = []
    field_labels = {
        "bid_id": "Bid ID",
        "bid_title": "Title",
        "client_organization": "Client/Organisation",
        "ministry": "Ministry",
        "quantity": "Quantity",
        "emd_amount": "EMD Amount",
        "delivery_period": "Delivery Period",
        "start_date_to_submit_bid": "Bid Start Date",
        "end_date_to_submit_bid": "Bid End Date",
        "published_date": "Published Date",
    }
    for key, label in field_labels.items():
        val = bid_metadata.get(key)
        if val and str(val).lower() not in ("n/a", "none", "null", ""):
            meta_lines.append(f"  {label}: {val}")

    metadata_block = "\n".join(meta_lines) if meta_lines else "  (no metadata available)"
    chunks_block = "\n\n---\n\n".join(
        f"[Section {i+1}]\n{chunk}" for i, chunk in enumerate(bid_chunks)
    )

    return f"""You are a knowledgeable Government e-Marketplace (GeM) procurement advisor.
You have been given the complete details of a specific GeM bid document below.
Answer the user's questions about this bid accurately and helpfully.

When the user asks whether they should participate/bid:
- Analyse eligibility criteria, EMD requirements, delivery timelines, quantities
- Highlight risks and opportunities
- Give a clear recommendation (Participate / Skip / Needs More Info)
- Be practical and concise

BID METADATA:
{metadata_block}

BID DOCUMENT CONTENT:
{chunks_block}

Important rules:
- Base your answers ONLY on the bid information provided above.
- STRICT ACCURACY FOR NUMBERS & QUANTITIES: Never guess, estimate, or invent individual item quantities, prices, or dates. If individual item breakdown quantities are not explicitly written in the document text, explicitly state "Total Bid Quantity: X (Individual breakdown not specified in PDF text)" and do NOT invent numbers for individual rows.
- EXHAUSTIVE LISTING: When listing products, items, categories, schedule of requirements, or technical components, NEVER truncate, summarize, or omit items. List EVERY item mentioned in the bid document in full.
- Use clear bullet points and tables for structured data.
- When giving a recommendation, always explain your reasoning based on exact specs and clauses."""


# ── Streaming chat ───────────────────────────────────────────────────

def chat_stream(
    messages: list[dict],
    model: str = DEFAULT_MODEL,
) -> Generator[str, None, None]:
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "options": {
            "temperature": 0.2,
            "num_ctx": 16384,
            "num_predict": 4096,
        },
    }
    with requests.post(
        f"{OLLAMA_BASE}/api/chat",
        json=payload,
        stream=True,
        timeout=120,
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            data = json.loads(line)
            token = data.get("message", {}).get("content", "")
            if token:
                yield token
            if data.get("done"):
                break


def chat_no_stream(
    messages: list[dict],
    model: str = DEFAULT_MODEL,
) -> str:
    """Non-streaming fallback — returns the full response text."""
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_ctx": 16384,
            "num_predict": 4096,
        },
    }
    resp = requests.post(
        f"{OLLAMA_BASE}/api/chat",
        json=payload,
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json().get("message", {}).get("content", "")


# ── Global AI Matchmaker Prompt ───────────────────────────────────────

def build_matchmaker_prompt(company_profile: str, candidate_bids: list[dict]) -> str:
    """Build a prompt for evaluating and ranking multiple bids against a company profile."""
    bids_formatted = []
    for idx, bid in enumerate(candidate_bids, 1):
        meta = bid.get("metadata", {})
        bid_id = meta.get("bid_id", f"BID-{idx}")
        title = meta.get("bid_title", "Untitled Bid")
        org = meta.get("client_organization") or meta.get("organisation", "N/A")
        qty = meta.get("quantity", "N/A")
        emd = meta.get("emd_amount", "N/A")
        end_date = meta.get("end_date_to_submit_bid", "N/A")
        text_content = bid.get("text", "")[:1500]

        bids_formatted.append(
            f"--- CANDIDATE BID #{idx} ---\n"
            f"Bid ID: {bid_id}\n"
            f"Title: {title}\n"
            f"Organisation: {org}\n"
            f"Quantity: {qty} | EMD Amount: {emd} | Closing Date: {end_date}\n"
            f"Summary/Key Requirements:\n{text_content}\n"
        )

    bids_block = "\n".join(bids_formatted)

    return f"""You are a Strategic Procurement Advisor for GeM (Government e-Marketplace) tenders.
Your task is to analyze multiple candidate government bids against the provided Company Profile/Capabilities and rank the most suitable bids to participate in.

COMPANY PROFILE / USER REQUIREMENTS:
{company_profile}

CANDIDATE BIDS TO EVALUATE:
{bids_block}

INSTRUCTIONS & FORMAT:
1. Provide an Executive Recommendation Summary (2-3 sentences).
2. For EACH candidate bid, evaluate and rank from MOST RELEVANT to LEAST RELEVANT:
    - **Bid ID & Title**
    - **Match Score**: [High Match / Medium Match / Low Match / Skip]
    - **Why It Fits**: Direct alignment with company capabilities/products.
    - **Eligibility & Key Requirements**: Turnover, experience, MSE/MII criteria mentioned.
    - **Risks / Action Needed**: EMD requirement, tight delivery schedule, missing certifications, or potential disqualifiers.
3. Conclude with a final prioritized recommendation list (e.g. "Top Bids to Pursue Immediately").

STRICT RULES:
- Base analysis ONLY on the provided bid details.
- Never guess or invent quantities or eligibility thresholds not in the text.
- Be objective, clear, and actionable."""


# ── Local Synonym Dictionary ──────────────────────────────────────────

# Instant, Ollama-free expansion for common government procurement terms.
# Keys are lowercased. Values are lists of alternative search terms.
PROCUREMENT_SYNONYMS: dict[str, list[str]] = {
    # Electronics & Electrical
    "pcb": ["printed circuit board", "PCBA", "circuit board assembly", "PCB assembly"],
    "pcba": ["printed circuit board assembly", "PCB", "circuit board"],
    "smps": ["switched mode power supply", "SMPS power supply", "power supply unit"],
    "fpga": ["field programmable gate array", "programmable logic device"],
    "ups": ["uninterruptible power supply", "UPS system", "power backup"],
    "led": ["light emitting diode", "LED light", "LED luminaire", "LED fitting"],
    "lcd": ["liquid crystal display", "LCD monitor", "LCD screen"],
    "cctv": ["closed circuit television", "surveillance camera", "CCTV camera", "video surveillance"],
    "ac": ["air conditioner", "air conditioning", "split AC", "AC unit"],
    "inverter": ["power inverter", "solar inverter", "inverter battery"],
    "transformer": ["power transformer", "distribution transformer", "electrical transformer"],
    "cable": ["electrical cable", "power cable", "cable wire", "conductor cable"],
    "battery": ["battery cell", "lead acid battery", "lithium battery", "rechargeable battery"],
    "charger": ["battery charger", "SMPS charger", "charging unit"],
    "harness": ["wiring harness", "cable harness", "wire assembly"],

    # Vehicles & Defence
    "brv": ["bullet resistant vehicle", "bulletproof vehicle", "armored vehicle", "armoured vehicle"],
    "lmv": ["light motor vehicle", "light vehicle", "LMV vehicle"],
    "vehicle": ["motor vehicle", "transport vehicle", "utility vehicle"],
    "armored": ["armoured", "bullet resistant", "bulletproof", "ballistic protection"],
    "armoured": ["armored", "bullet resistant", "bulletproof", "ballistic protection"],

    # IT & Software
    "it": ["information technology", "IT services", "IT infrastructure"],
    "software": ["software application", "software solution", "custom software", "software development"],
    "server": ["rack server", "tower server", "blade server", "server hardware"],
    "computer": ["desktop computer", "personal computer", "PC", "workstation"],
    "laptop": ["notebook computer", "portable computer", "laptop computer"],
    "printer": ["laser printer", "inkjet printer", "multifunction printer", "MFP"],
    "router": ["network router", "wifi router", "wireless router"],
    "switch": ["network switch", "ethernet switch", "managed switch"],
    "firewall": ["network firewall", "firewall appliance", "UTM firewall"],

    # Office & Furniture
    "furniture": ["office furniture", "steel furniture", "modular furniture"],
    "chair": ["office chair", "revolving chair", "executive chair"],
    "table": ["office table", "work table", "conference table"],

    # Medical
    "ppe": ["personal protective equipment", "PPE kit", "safety equipment"],
    "ventilator": ["medical ventilator", "ICU ventilator", "breathing apparatus"],
    "monitor": ["patient monitor", "vital signs monitor", "multi-parameter monitor"],

    # General Procurement
    "amc": ["annual maintenance contract", "AMC service", "maintenance contract"],
    "manpower": ["manpower supply", "outsourced manpower", "contractual manpower", "human resource supply"],
    "stationery": ["office stationery", "stationery items", "office supplies"],
    "uniform": ["uniform supply", "uniform stitching", "livery"],
    "cleaning": ["cleaning service", "housekeeping", "janitorial service", "sanitation"],
    "security": ["security guard", "security service", "security manpower", "guarding service"],
    "catering": ["catering service", "mess service", "food supply"],
    "solar": ["solar panel", "solar power plant", "solar module", "photovoltaic"],
    "pump": ["water pump", "submersible pump", "centrifugal pump", "motor pump"],
    "pipe": ["GI pipe", "PVC pipe", "HDPE pipe", "pipeline"],
    "cement": ["OPC cement", "PPC cement", "portland cement"],
    "steel": ["mild steel", "TMT bar", "structural steel", "MS steel"],
    "tyre": ["tyre", "tire", "vehicle tyre", "rubber tyre"],
    "tire": ["tyre", "tire", "vehicle tyre", "rubber tyre"],
    "oil": ["lubricant oil", "engine oil", "hydraulic oil", "lubricating oil"],
    "paint": ["wall paint", "emulsion paint", "enamel paint", "industrial paint"],
}


def clean_conversational_query(query: str) -> str:
    """Strip generic conversational prefixes and filler phrases to isolate the core technical subject."""
    cleaned = query.strip()
    # Strip common conversational patterns at start
    patterns = [
        r"^(?:give\s+me|show\s+me|find|search|get|list|display|i\s+need|i\s+want|looking\s+for)\s+",
        r"^(?:all\s+)?(?:bids?|tenders?|contracts?)\s+(?:related\s+to|for|about|on|of|with)\s+",
        r"^(?:related\s+to|for|about)\s+",
    ]
    for pat in patterns:
        cleaned = re.sub(pat, "", cleaned, flags=re.IGNORECASE).strip()
    return cleaned if cleaned else query.strip()


# Reverse index: full phrase -> abbreviation key(s), so that searching the
# full form also finds bids that only use the short form.
# e.g. "printed circuit board" -> "pcb", "bullet resistant vehicle" -> "brv".
_REVERSE_SYNONYMS: dict[str, list[str]] = {}
for _key, _syns in PROCUREMENT_SYNONYMS.items():
    for _syn in _syns:
        _norm = _syn.lower().strip()
        # Only index multi-word phrases and meaningful single words;
        # single generic words would cause false-positive expansions.
        if len(_norm) < 3:
            continue
        _REVERSE_SYNONYMS.setdefault(_norm, []).append(_key)


def local_expand_query(query: str) -> list[str]:
    """Expand a query using the local synonym dictionary. Instant, no LLM needed."""
    cleaned_query = clean_conversational_query(query)
    query_lower = cleaned_query.lower()
    # Use the cleaned query as primary — the raw conversational form
    # ("show me bids related to X") shifts dense embeddings toward generic
    # bid language and duplicates work. Only fall back to raw if cleaning
    # yields nothing new.
    terms = [cleaned_query if cleaned_query else query]

    def _add(term: str) -> None:
        if term.lower() not in {t.lower() for t in terms}:
            terms.append(term)

    # Exact match on the cleaned query
    if query_lower in PROCUREMENT_SYNONYMS:
        for syn in PROCUREMENT_SYNONYMS[query_lower]:
            if syn.lower() != query_lower:
                terms.append(syn)

    # Also check individual words in multi-word queries
    words = re.findall(r"[A-Za-z0-9]+", cleaned_query)
    for word in words:
        word_lower = word.lower()
        if word_lower != query_lower and word_lower in PROCUREMENT_SYNONYMS:
            for syn in PROCUREMENT_SYNONYMS[word_lower]:
                _add(syn)

    # Reverse lookup: if the query contains a known full phrase, add its
    # abbreviation key plus sibling synonyms.
    # Handles "printed circuit board" -> "PCB"/"PCBA", etc.
    # Only the LONGEST matching phrases trigger, on word boundaries, so that
    # "printed circuit board" does not also trigger via its substring
    # "circuit board" (which would pull in generic circuit/breaker/board junk).
    # Standalone generic fragments are never emitted as search terms.
    _GENERIC_FRAGMENTS = {"circuit", "board", "circuit board"}
    candidates: list[str] = []
    for phrase in _REVERSE_SYNONYMS:
        if phrase in _GENERIC_FRAGMENTS:
            continue
        if re.search(rf"\b{re.escape(phrase)}\b", query_lower):
            candidates.append(phrase)
    # Drop any candidate that is a substring of a longer matched phrase.
    candidates.sort(key=len, reverse=True)
    longest_only: list[str] = []
    for cand in candidates:
        if not any(cand != longer and cand in longer for longer in longest_only):
            longest_only.append(cand)
    for phrase in longest_only:
        for key in _REVERSE_SYNONYMS[phrase]:
            _add(key)
            for sib in PROCUREMENT_SYNONYMS.get(key, []):
                if sib.lower() in _GENERIC_FRAGMENTS:
                    continue
                # Never add back a term that is itself a mere substring of the
                # query's matched long phrase — e.g. don't add lone
                # "circuit board" when the query already was
                # "printed circuit board".
                if sib.lower() in query_lower and len(sib) < len(phrase):
                    continue
                _add(sib)

    # Drop standalone generic fragments that match half the corpus
    # ("board" hits PVC/sun boards, "circuit" hits breakers).
    terms = [t for t in terms if t.lower() not in _GENERIC_FRAGMENTS]
    return terms[:8]  # cap at 8 terms to avoid search explosion


# ── Query Expansion (Hybrid: local + LLM) ────────────────────────────

def expand_query(query: str, model: str = DEFAULT_MODEL) -> list[str]:
    """Expand a query using local synonyms first, then optionally enhance with LLM.

    The local dictionary provides instant, reliable expansion for common
    government procurement terms (PCB → printed circuit board, etc.).
    The LLM can add domain-specific terms the dictionary doesn't cover.
    """
    # Step 1: Always start with local synonyms (instant, no Ollama needed)
    base_terms = local_expand_query(query)

    # Step 2: Try LLM enhancement if available
    prompt = f"""You are a procurement search expert for Indian government tenders (GeM portal).

Expand the following search query into 2-4 related technical terms or phrases
that describe the SAME product, item, or specification as the original query.

Rules:
- Always include the original term
- Only add terms for the product/item itself: full forms of abbreviations
  (PCB -> printed circuit board), synonyms, spec variants, and closely related
  product names
- Do NOT add generic administrative, financial, or contractual terms that
  appear in almost every bid's terms & conditions — e.g. EMD, PBG, bid
  guarantee, turnover, MSME, delivery period, warranty period — unless one of
  those terms is literally what the original query is asking about
- Do NOT add broad category words on their own (e.g. "equipment", "supply",
  "goods") — every added term must be specific enough to only match bids
  about this particular item
- Do NOT repeat these already-known terms: {json.dumps(base_terms)}
- Respond with ONLY a JSON object of the form {{"terms": ["...", "..."]}}.
  No explanation, no markdown, no extra text.

Query: "{query}"
/no_think"""

    content = ""

    try:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0.1,
                "num_ctx": 2048,
                "num_predict": 400,
            },
        }

        resp = requests.post(
            f"{OLLAMA_BASE}/api/chat",
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()

        content = resp.json().get("message", {}).get("content", "").strip()

        # Strip <think>...</think> blocks that Qwen 3.x may return
        content = re.sub(
            r"<think>.*?</think>",
            "",
            content,
            flags=re.DOTALL,
        ).strip()

        parsed = json.loads(content)

        # Expected format:
        # {"terms": ["term1", "term2"]}
        if isinstance(parsed, dict):
            llm_terms = parsed.get("terms", [])
        elif isinstance(parsed, list):
            # Defensive support for models returning a raw JSON array
            llm_terms = parsed
        else:
            llm_terms = []

        if isinstance(llm_terms, list):
            seen = {t.lower() for t in base_terms}

            for term in llm_terms:
                term_str = str(term).strip()

                if term_str and term_str.lower() not in seen:
                    seen.add(term_str.lower())
                    base_terms.append(term_str)

    except requests.ConnectionError:
        # Ollama not running — local synonyms are still sufficient
        pass

    except Exception as exc:
        print(
            f"[QueryExpansion] LLM enhancement failed "
            f"(using local synonyms): {exc!r}"
        )

    return base_terms[:8]