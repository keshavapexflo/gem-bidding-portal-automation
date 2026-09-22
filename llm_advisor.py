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


# ── Query Expansion ───────────────────────────────────────────────────

def expand_query(query: str, model: str = DEFAULT_MODEL) -> list[str]:
    """Use Qwen to expand a short query into related procurement terms."""
    prompt = f"""You are a procurement search expert for Indian government tenders (GeM portal).
Expand the following search query into 4-6 related technical terms or phrases that would appear in government bid documents.

Rules:
- Always include the original term
- Include full forms (PCB → printed circuit board, FPGA → field programmable gate array)
- Include common abbreviations and variants used in Indian government tenders
- Include closely related procurement terms
- Return ONLY a JSON array of strings, no explanation, no markdown

Query: "{query}"

Output (JSON array only):"""

    try:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_ctx": 2048,
                "num_predict": 200,
            },
        }
        resp = requests.post(
            f"{OLLAMA_BASE}/api/chat",
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        content = resp.json().get("message", {}).get("content", "").strip()

        match = re.search(r'\[.*?\]', content, re.DOTALL)
        if match:
            terms = json.loads(match.group())
            if isinstance(terms, list) and terms:
                seen: set[str] = set()
                result: list[str] = []
                for t in [query] + [str(x).strip() for x in terms]:
                    if t and t.lower() not in seen:
                        seen.add(t.lower())
                        result.append(t)
                return result[:6]
    except Exception:
        pass

    return [query]