import csv
import hashlib
import io
import json
from datetime import datetime, timezone
from html import escape
from pathlib import Path
import re
import traceback

from gem_hybrid_retrieval import (
    DEFAULT_CHROMA_PATH,
    DEFAULT_COLLECTION,
    HybridRetriever,
    SearchResult,
)
from gem_live_status import check_bids_for_extension
from llm_advisor import (
    check_ollama_available,
    build_system_prompt,
    chat_stream,
    extract_full_pdf_text,
    build_matchmaker_prompt,
    expand_query,
    local_expand_query,
)
import streamlit as st

NEW_BID_WINDOW_DAYS = 7
PROJECT_DIR = Path(__file__).resolve().parent
BIDS_DIR = PROJECT_DIR / "downloads" / "bids"
STATIC_DIR = PROJECT_DIR / "static"


def is_new_bid(metadata) -> bool:
  added_at = metadata.get("added_at")
  if not added_at:
    return False
  try:
    parsed = datetime.fromisoformat(str(added_at).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
      parsed = parsed.replace(tzinfo=timezone.utc)
    age_seconds = (datetime.now(timezone.utc) - parsed.astimezone(timezone.utc)).total_seconds()
    return 0 <= age_seconds < NEW_BID_WINDOW_DAYS * 24 * 60 * 60
  except (TypeError, ValueError):
    return False


def find_pdf_file(metadata) -> Path | None:
  pdf_path_val = metadata.get("pdf_path") or metadata.get("pdf_link")
  if pdf_path_val and not str(pdf_path_val).lower().startswith(("http://", "https://")):
    root = BIDS_DIR.resolve()
    p = (root / str(pdf_path_val)).resolve()
    try:
      p.relative_to(root)
    except ValueError:
      p = None
    if p and p.is_file():
      return p
  bid_id = metadata.get("bid_id")
  if bid_id:
    digits_match = re.findall(r"\d+", bid_id)
    if digits_match:
      for digit_seq in reversed(digits_match):
        if len(digit_seq) >= 5:
          matches = list(BIDS_DIR.rglob(f"*{digit_seq}*.pdf"))
          if matches:
            return matches[0]
  return None


def search_by_bid_id(retriever: HybridRetriever, query: str, limit: int = 15) -> list[SearchResult]:
    import sqlite3
    from contextlib import closing
    query = query.strip()
    results: list[SearchResult] = []
    seen_ids: set[str] = set()
    try:
        source = retriever.chroma_path / "chroma.sqlite3"
        uri = f"file:{source.resolve().as_posix()}?mode=ro"
        with closing(sqlite3.connect(uri, uri=True)) as db:
            db.row_factory = sqlite3.Row
            rows = db.execute(
                """
                SELECT e.embedding_id, d.string_value AS text
                FROM embedding_metadata bid
                JOIN embeddings e ON e.id = bid.id
                JOIN embedding_metadata d ON d.id = e.id AND d.key = 'chroma:document'
                JOIN segments s ON s.id = e.segment_id
                JOIN collections c ON c.id = s.collection
                WHERE bid.key = 'bid_id'
                  AND lower(bid.string_value) = lower(?)
                  AND c.name = ?
                  AND s.scope = 'METADATA'
                LIMIT ?
                """,
                (query, retriever.collection_name, limit),
            ).fetchall()
            if rows:
                chunk_ids = [r["embedding_id"] for r in rows]
                text_map = {r["embedding_id"]: r["text"] or "" for r in rows}
                placeholders = ",".join("?" for _ in chunk_ids)
                meta_rows = db.execute(
                    f"""
                    SELECT e.embedding_id, m.key,
                           m.string_value, m.int_value, m.float_value, m.bool_value
                    FROM embeddings e
                    JOIN embedding_metadata m ON m.id = e.id
                    JOIN segments s ON s.id = e.segment_id
                    JOIN collections c ON c.id = s.collection
                    WHERE e.embedding_id IN ({placeholders})
                      AND c.name = ?
                      AND s.scope = 'METADATA'
                      AND m.key != 'chroma:document'
                    """,
                    (*chunk_ids, retriever.collection_name),
                ).fetchall()
                meta_by_id: dict[str, dict] = {cid: {} for cid in chunk_ids}
                for mr in meta_rows:
                    val = (
                        mr["string_value"] if mr["string_value"] is not None
                        else mr["int_value"] if mr["int_value"] is not None
                        else mr["float_value"] if mr["float_value"] is not None
                        else bool(mr["bool_value"]) if mr["bool_value"] is not None
                        else None
                    )
                    if val is not None:
                        meta_by_id[mr["embedding_id"]][mr["key"]] = val
                for cid in chunk_ids:
                    results.append(SearchResult(chunk_id=cid, text=text_map[cid], metadata=meta_by_id[cid], score=1.0))
                    seen_ids.add(cid)
    except Exception:
        pass
    if len(results) >= limit:
        return results[:limit]
    digits = re.sub(r"[^0-9]", "", query)
    if not digits or len(digits) < 4:
        return results
    prefix_pattern = f"%/{digits}%"
    try:
        source = retriever.chroma_path / "chroma.sqlite3"
        uri = f"file:{source.resolve().as_posix()}?mode=ro"
        with closing(sqlite3.connect(uri, uri=True)) as db:
            db.row_factory = sqlite3.Row
            rows = db.execute(
                """
                SELECT e.embedding_id, d.string_value AS text, bid.string_value AS bid_id_val
                FROM embedding_metadata bid
                JOIN embeddings e ON e.id = bid.id
                JOIN embedding_metadata d ON d.id = e.id AND d.key = 'chroma:document'
                JOIN segments s ON s.id = e.segment_id
                JOIN collections c ON c.id = s.collection
                WHERE bid.key = 'bid_id'
                  AND bid.string_value LIKE ?
                  AND c.name = ?
                  AND s.scope = 'METADATA'
                ORDER BY bid.string_value
                LIMIT ?
                """,
                (prefix_pattern, retriever.collection_name, limit - len(results)),
            ).fetchall()
            if rows:
                chunk_ids = [r["embedding_id"] for r in rows if r["embedding_id"] not in seen_ids]
                text_map = {r["embedding_id"]: r["text"] or "" for r in rows}
                bid_id_map = {r["embedding_id"]: r["bid_id_val"] for r in rows}
                if chunk_ids:
                    placeholders = ",".join("?" for _ in chunk_ids)
                    meta_rows = db.execute(
                        f"""
                        SELECT e.embedding_id, m.key,
                               m.string_value, m.int_value, m.float_value, m.bool_value
                        FROM embeddings e
                        JOIN embedding_metadata m ON m.id = e.id
                        JOIN segments s ON s.id = e.segment_id
                        JOIN collections c ON c.id = s.collection
                        WHERE e.embedding_id IN ({placeholders})
                          AND c.name = ?
                          AND s.scope = 'METADATA'
                          AND m.key != 'chroma:document'
                        """,
                        (*chunk_ids, retriever.collection_name),
                    ).fetchall()
                    meta_by_id = {cid: {} for cid in chunk_ids}
                    for mr in meta_rows:
                        val = (
                            mr["string_value"] if mr["string_value"] is not None
                            else mr["int_value"] if mr["int_value"] is not None
                            else mr["float_value"] if mr["float_value"] is not None
                            else bool(mr["bool_value"]) if mr["bool_value"] is not None
                            else None
                        )
                        if val is not None:
                            meta_by_id[mr["embedding_id"]][mr["key"]] = val
                    def sort_key(cid):
                        bid_val = bid_id_map.get(cid, "")
                        last_seg = bid_val.rsplit("/", 1)[-1]
                        return (0 if last_seg.startswith(digits) else 1, bid_val)
                    for cid in sorted(chunk_ids, key=sort_key):
                        results.append(SearchResult(chunk_id=cid, text=text_map[cid], metadata=meta_by_id[cid], score=0.9))
                        seen_ids.add(cid)
    except Exception:
        pass
    return results[:limit]


CSV_COLUMNS = [
    "Date", "Portal", "Bid ID", "Client / Organization", "Bid Title",
    "Quantity", "EMD Amount", "Quoted Bid", "Delivery period",
    "Start date to submit bid", "End date to submit bid", "PDF Link",
]
CSV_COLUMNS_WITH_STATUS = CSV_COLUMNS + ["Live End Date (GeM)", "Extended?", "Live Status"]


def build_export_rows(results):
  seen_bid_ids = set()
  rows = []
  for res in results:
    meta = res.metadata
    bid_id = meta.get("bid_id", "N/A")
    if bid_id in seen_bid_ids:
      continue
    seen_bid_ids.add(bid_id)
    rows.append({
        "Date": meta.get("date", ""),
        "Portal": meta.get("portal", "GeM"),
        "Bid ID": bid_id,
        "Client / Organization": meta.get("client_organization") or meta.get("organisation", ""),
        "Bid Title": meta.get("bid_title", ""),
        "Quantity": meta.get("quantity", ""),
        "EMD Amount": meta.get("emd_amount", ""),
        "Quoted Bid": "",
        "Delivery period": meta.get("delivery_period", ""),
        "Start date to submit bid": meta.get("start_date_to_submit_bid") or meta.get("published_date", ""),
        "End date to submit bid": meta.get("end_date_to_submit_bid", ""),
        "PDF Link": meta.get("pdf_link") or meta.get("pdf_path", ""),
    })
  return rows


def rows_to_csv_bytes(rows, fieldnames=CSV_COLUMNS) -> bytes:
  buffer = io.StringIO()
  writer = csv.DictWriter(buffer, fieldnames=fieldnames)
  writer.writeheader()
  writer.writerows(rows)
  return buffer.getvalue().encode("utf-8-sig")


st.set_page_config(page_title="Lets Bid | GeM Intelligence", page_icon="LB", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .reportview-container { background: #0f1116; }
    .main { background-color: #0f1116; color: #e2e8f0; }
    .card { background-color: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 20px; margin-bottom: 16px; }
    .card-title { color: #f8fafc; font-size: 1.15rem; font-weight: 600; margin-bottom: 8px; }
    .card-subtitle { color: #94a3b8; font-size: 0.85rem; margin-bottom: 12px; }
    .card-text { color: #cbd5e1; font-size: 0.95rem; line-height: 1.6; background-color: #0f172a; padding: 12px; border-radius: 8px; border-left: 4px solid #3b82f6; }
    .badge { display: inline-block; padding: 4px 8px; border-radius: 6px; font-size: 0.75rem; font-weight: 600; margin-right: 8px; }
    .badge-high { background-color: #064e3b; color: #6ee7b7; border: 1px solid #047857; }
    .badge-med { background-color: #78350f; color: #fde047; border: 1px solid #b45309; }
    .badge-low { background-color: #451a03; color: #f97316; border: 1px solid #c2410c; }
    .badge-none { background-color: #1e293b; color: #94a3b8; border: 1px solid #475569; }
    .badge-info { background-color: #1e3a8a; color: #93c5fd; border: 1px solid #1d4ed8; }
    .badge-new { background-color: #14532d; color: #bbf7d0; border: 1px solid #22c55e; }
    :root { --ink: #e8edf5; --muted: #9ca9bd; --accent: #5b8cff; }
    .stApp { background: #0b1020; color: var(--ink); }
    [data-testid="stAppViewContainer"] > .main { background: radial-gradient(circle at 78% -20%, #1a315b 0, transparent 34%), #0b1020; }
    [data-testid="stMainBlockContainer"] { max-width: 1280px; padding-top: 2.4rem; }
    [data-testid="stSidebar"] { background: #101827; border-right: 1px solid #26344b; }
    h1, h2, h3 { letter-spacing: -0.02em; }
    .hero { padding: 2rem 2.15rem; border: 1px solid #30446a; border-radius: 18px; background: linear-gradient(115deg, #152544 0%, #152039 55%, #102b40 100%); box-shadow: 0 18px 38px rgba(0,0,0,0.22); margin: 0 0 2rem 0; }
    .eyebrow { color: #8fb1ff; font-size: 0.74rem; font-weight: 750; letter-spacing: 0.14em; margin-bottom: 0.5rem; }
    .hero h1 { font-size: 2.15rem; margin: 0; color: #f7f9ff; }
    .hero p { max-width: 700px; color: #c5d1e7; font-size: 1rem; line-height: 1.6; margin: 0.65rem 0 0; }
    .section-label { color: #91a9d9; font-size: 0.72rem; font-weight: 750; letter-spacing: 0.12em; margin-bottom: 0.2rem; }
    .stButton>button { background: linear-gradient(135deg, #4f7ff1, #3566d9); color: white; border-radius: 9px; border: 1px solid #6d98ff; padding: 0.55rem 1.05rem; font-weight: 650; transition: all 0.3s ease; }
    .stButton>button:hover { background: #3d71e9; transform: translateY(-1px); box-shadow: 0 7px 16px rgba(59,130,246,0.28); }
    .card { background: linear-gradient(145deg, #172238, #131d2f); border: 1px solid #30405b; border-radius: 15px; padding: 1.35rem; margin: 1.1rem 0 0.8rem; box-shadow: 0 10px 24px rgba(0,0,0,0.16); }
    .card-title { color: #f5f8ff; font-size: 1.12rem; font-weight: 700; margin-bottom: 0.7rem; }
    .card-subtitle { color: #9daec8; font-size: 0.85rem; line-height: 2.25; margin-bottom: 0.9rem; }
    .card-text { color: #d5deed; font-size: 0.95rem; line-height: 1.6; background: #0d1422; padding: 0.9rem 1rem; border-radius: 9px; border-left: 3px solid #5c8fff; }
    .badge { border-radius: 999px; font-weight: 650; margin: 0 0.35rem 0.2rem 0; }
    [data-testid="stTextInput"] input, [data-testid="stSelectbox"] div[data-baseweb="select"] > div, [data-testid="stDateInput"] input {
        border-radius: 9px !important; border-color: #3a4964 !important; background: #121b2a !important;
        color: #f1f5f9 !important; -webkit-text-fill-color: #f1f5f9 !important; caret-color: #8fb1ff !important; opacity: 1 !important;
    }
    [data-testid="stTextInput"] input::placeholder, [data-testid="stDateInput"] input::placeholder {
        color: #94a3b8 !important; -webkit-text-fill-color: #94a3b8 !important; opacity: 1 !important;
    }
    [data-testid="stWidgetLabel"] p, [data-testid="stTextInput"] label p, [data-testid="stDateInput"] label p,
    [data-testid="stSlider"] label p, [data-testid="stCheckbox"] label p { color: #cbd5e1 !important; }
    [data-baseweb="select"] *, [data-testid="stSlider"] [data-testid="stMarkdownContainer"] p { color: #e2e8f0 !important; }
    [data-testid="stExpander"] { border: 1px solid #2d3d56; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
    <section class="hero">
      <div class="eyebrow">GOVERNMENT PROCUREMENT INTELLIGENCE</div>
      <h1>Lets Bid</h1>
      <p>Search GeM bid documents with semantic relevance and exact keyword matching in one focused workspace.</p>
    </section>
""", unsafe_allow_html=True)


@st.cache_resource
def get_retriever(chroma_path, collection_name, lexical_db_path=None):
  ret = HybridRetriever(chroma_path=chroma_path, collection_name=collection_name, lexical_db_path=lexical_db_path)
  ret.warmup()
  return ret


st.sidebar.header("Workspace")
chroma_path_str = st.sidebar.text_input("Chroma Path", str(DEFAULT_CHROMA_PATH))
collection_name = st.sidebar.text_input("Collection Name", DEFAULT_COLLECTION)
lexical_db = st.sidebar.text_input("Lexical SQLite Path (Optional)", "")

try:
  retriever = get_retriever(chroma_path=chroma_path_str, collection_name=collection_name, lexical_db_path=Path(lexical_db) if lexical_db else None)
except Exception as e:
  st.sidebar.error(f"Error initializing retriever: {e}")
  retriever = None

st.sidebar.markdown("---")
st.sidebar.subheader("Index status")
if retriever:
  try:
    state = retriever.lexical.state()
    chunk_count = state.get("chunk_count", 0)
    index_is_current = state.get("collection") == collection_name and chunk_count > 0
    if index_is_current:
      st.sidebar.success("Index is present & current.")
      st.sidebar.write(f"**Collection:** `{state.get('collection')}`")
      st.sidebar.write(f"**Chunks:** `{chunk_count:,}`")
      st.sidebar.write(f"**Last Indexed:** `{state.get('indexed_at', 'Unknown')}`")
    elif state:
      st.sidebar.warning(f"Lexical index is incomplete ({chunk_count:,} indexed).")
    else:
      st.sidebar.warning("Lexical index not built yet (using Chroma native fallback).")
  except Exception as e:
    st.sidebar.error(f"Could not read index state: {e}")

  if st.sidebar.button("Rebuild Lexical Index"):
    with st.sidebar.status("Building lexical index...") as status:
      try:
        indexed = retriever.build_lexical_index(rebuild=True)
        status.update(label=f"Built index for {indexed:,} chunks!", state="complete")
        st.cache_resource.clear()
        st.rerun()
      except Exception as e:
        status.update(label=f"Failed: {e}", state="error")
        st.sidebar.code(traceback.format_exc())

st.sidebar.markdown("---")
st.sidebar.subheader("Retrieval tuning")
rrf_k = st.sidebar.slider("RRF K parameter", min_value=1, max_value=200, value=60)
dense_weight = st.sidebar.slider("Dense Weight", min_value=0.0, max_value=5.0, value=1.0, step=0.1)
lexical_weight = st.sidebar.slider("Lexical Weight", min_value=0.0, max_value=5.0, value=1.0, step=0.1)
exclude_boilerplate = st.sidebar.checkbox("Exclude Boilerplate Documents", value=True)
use_reranker = st.sidebar.checkbox("Enable Cross-Encoder Reranker", value=True, help="Uses cross-encoder/ms-marco-MiniLM-L-6-v2.")
use_query_expansion = st.sidebar.checkbox(
    "🧠 AI Query Expansion", value=True,
    help="Uses Qwen to expand your query (e.g. 'pcb' → 'printed circuit board, PCBA'). Requires Ollama.",
)

st.sidebar.markdown("---")
st.sidebar.subheader("🤖 AI Advisor")
ollama_status = check_ollama_available()
if ollama_status["ok"]:
  st.sidebar.success(ollama_status["detail"])
else:
  st.sidebar.error(ollama_status["detail"])

tab_search, tab_matchmaker = st.tabs(["🔍 Search & Single-Bid Advisor", "🎯 Global AI Matchmaker (Rank Bids)"])

with tab_search:
  st.markdown('<div class="section-label">DISCOVER OPPORTUNITIES</div>', unsafe_allow_html=True)
  st.subheader("Search bid intelligence")
  st.caption("Use natural language, precise product terms, or a known bid ID.")
  query = st.text_input("Enter your query manually", placeholder="e.g., bullet resistant vehicle for police force")

  st.markdown("##### Refine results")
  col_date1, col_date2 = st.columns(2)
  with col_date1:
    bid_start_date = st.date_input("Published/Start Date (On or After)", value=None)
  with col_date2:
    bid_end_date = st.date_input("Bid Closing Date (On or Before)", value=None)

  limit = st.slider("Result Limit", min_value=1, max_value=100, value=10)
  search_button = st.button("Search")

  if "search_results" not in st.session_state:
    st.session_state.search_results = None

  # Always initialize these so Streamlit reruns (e.g. from other widgets)
  # never hit a NameError before the search block below runs.
  results = st.session_state.search_results or []
  export_rows = build_export_rows(results) if results else []

  if search_button:
    if not query.strip():
      st.warning("Please enter a non-empty search query.")
    elif not retriever:
      st.error("Retriever is not initialized.")
    else:
      where_filter = None
      with st.spinner("Searching and ranking results..."):
        try:
          fetch_limit = limit * 3 if (use_reranker or bid_start_date or bid_end_date) else limit

          # ── Bid ID Detection ──────────────────────────────────────────
          is_bid_id_query = bool(re.fullmatch(r"GEM/\d{4}/[BR]/\d+", query.strip(), flags=re.IGNORECASE))
          if is_bid_id_query:
            results = search_by_bid_id(retriever, query, limit=fetch_limit)
            if results:
              st.caption(f"🔎 **Bid ID detected** — showing direct metadata match for `{query.strip()}`")
          else:
            # ── Query Expansion ──────────────────────────────────────────
            search_terms = [query]
            if use_query_expansion:
              if ollama_status["ok"]:
                with st.spinner("🧠 Expanding query with AI..."):
                  search_terms = expand_query(query)
              else:
                # LLM unavailable — use local synonym dictionary only
                search_terms = local_expand_query(query)
              if len(search_terms) > 1:
                st.caption(f"🧠 **Query expanded to:** {', '.join(f'`{t}`' for t in search_terms)}")

            # Run search for each term and merge
            seen_chunk_ids: set[str] = set()
            merged_results = []
            for term in search_terms:
              term_results = retriever.search(term, limit=fetch_limit, where=where_filter, exclude_boilerplate=exclude_boilerplate, rrf_k=rrf_k, dense_weight=dense_weight, lexical_weight=lexical_weight)
              for r in term_results:
                if r.chunk_id not in seen_chunk_ids:
                  seen_chunk_ids.add(r.chunk_id)
                  merged_results.append(r)
            merged_results.sort(key=lambda r: r.score, reverse=True)

            # Bid-level deduplication: keep best-scoring chunk per bid
            seen_bid_ids_dedup: dict[str, SearchResult] = {}
            deduped_results: list[SearchResult] = []
            for r in merged_results:
              bid_id = r.metadata.get("bid_id", r.chunk_id)
              if bid_id not in seen_bid_ids_dedup:
                seen_bid_ids_dedup[bid_id] = r
                deduped_results.append(r)
              elif r.score > seen_bid_ids_dedup[bid_id].score:
                # Replace with higher-scoring chunk for same bid
                deduped_results = [r if x is seen_bid_ids_dedup[bid_id] else x for x in deduped_results]
                seen_bid_ids_dedup[bid_id] = r
            results = deduped_results

            if use_reranker and results:
              results = retriever.rerank(query, results, top_k=limit, min_score=0.0)

          # ── Date Filtering ───────────────────────────────────────────
          if results and (bid_start_date or bid_end_date):
            filtered_results = []
            for res in results:
              keep = True
              meta = res.metadata
              pub_date_raw = meta.get("published_date") or meta.get("bid_start_date") or meta.get("date")
              if bid_start_date and pub_date_raw:
                try:
                  clean_start = str(pub_date_raw).split("T")[0].split(" ")[0]
                  # Handle both YYYY-MM-DD and DD-MM-YYYY formats
                  if len(clean_start.split("-")[0]) == 4:
                    doc_start = datetime.strptime(clean_start, "%Y-%m-%d").date()
                  else:
                    doc_start = datetime.strptime(clean_start, "%d-%m-%Y").date()
                  if doc_start < bid_start_date:
                    keep = False
                except Exception:
                  pass
              end_date_raw = meta.get("end_date_to_submit_bid") or meta.get("bid_end_date")
              if bid_end_date and end_date_raw:
                try:
                  clean_end = str(end_date_raw).split(" ")[0]
                  fmt = "%Y-%m-%d" if len(clean_end.split("-")[0]) == 4 else "%d-%m-%Y"
                  doc_end = datetime.strptime(clean_end, fmt).date()
                  if doc_end > bid_end_date:
                    keep = False
                except Exception:
                  pass
              if keep:
                filtered_results.append(res)
            results = filtered_results[:limit]

          st.session_state.search_results = results
          export_rows = build_export_rows(results) if results else []
          if not results:
            st.info("No matching results found.")
        except Exception as search_err:
          st.error(f"Search failed: {search_err}")
          st.code(traceback.format_exc())

  if results:
    st.success(f"Showing {len(results)} matching chunks.")

    csv_bytes = rows_to_csv_bytes(export_rows)
    st.download_button(label=f"📊 Export {len(export_rows)} bid(s) to CSV", data=csv_bytes, file_name=f"gem_bids_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv", key="export_csv_button")
    st.caption(f"{len(results)} chunk(s) → {len(export_rows)} unique bid(s) in export.")

  st.markdown("##### 🔄 Check for Extensions on GeM (Live)")
  st.caption("Queries the live GeM portal for each bid and compares its current end date.")
  if st.button(
      f"Check {len(export_rows)} bid(s) for extensions",
      key="check_extensions_button",
      disabled=not export_rows,
  ):
    bid_stored_end_dates = {row["Bid ID"]: row["End date to submit bid"] for row in export_rows if row["Bid ID"] and row["Bid ID"] != "N/A"}
    progress_bar = st.progress(0, text="Starting live check...")
    def _update_progress(i, total, bid_number):
      progress_bar.progress(i / total, text=f"Checking {i}/{total}: {bid_number}")
    with st.spinner("Checking live status on GeM..."):
      try:
        status_results = check_bids_for_extension(bid_stored_end_dates, progress_callback=_update_progress)
      except Exception as check_err:
        st.error(f"Live check failed: {check_err}")
        st.code(traceback.format_exc())
        status_results = None
    progress_bar.empty()
    if status_results:
      augmented_rows = []
      extended_count = 0
      not_found_count = 0
      for row in export_rows:
        bid_id = row["Bid ID"]
        status = status_results.get(bid_id, {})
        extended = status.get("extended")
        live_status = status.get("status", "unknown")
        if extended is True:
          extended_label = "YES"; extended_count += 1
        elif live_status == "not_found":
          extended_label = "Unknown (not in ongoing listing)"
        elif extended is False:
          extended_label = "No"
        else:
          extended_label = "Unknown (could not parse dates)"
        if live_status == "not_found":
          not_found_count += 1
        augmented_row = dict(row)
        augmented_row["Live End Date (GeM)"] = status.get("live_end_date", "")
        augmented_row["Extended?"] = extended_label
        augmented_row["Live Status"] = live_status
        augmented_rows.append(augmented_row)
      st.success(f"Checked {len(augmented_rows)} bid(s): {extended_count} extended, {not_found_count} not found.")
      augmented_csv_bytes = rows_to_csv_bytes(augmented_rows, fieldnames=CSV_COLUMNS_WITH_STATUS)
      st.download_button(label=f"📥 Download extension-checked CSV ({extended_count} extended)", data=augmented_csv_bytes, file_name=f"gem_bids_extension_check_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv", key="export_csv_with_status_button")
      if extended_count > 0:
        st.markdown("**Bids with changed end dates:**")
        for row in augmented_rows:
          if row["Extended?"] == "YES":
            st.markdown(f"- **{row['Bid ID']}** — was `{row['End date to submit bid']}`, now `{row['Live End Date (GeM)']}`")

  for idx, res in enumerate(results, 1):
    score = res.score
    dense_rank = res.dense_rank
    lexical_rank = res.lexical_rank
    if dense_rank is not None and dense_rank <= 3:
      conf_class = "badge-high"; conf_lbl = "High Match (Semantic)"
    elif lexical_rank is not None and lexical_rank <= 3:
      conf_class = "badge-high"; conf_lbl = "High Match (Lexical)"
    elif dense_rank is not None or lexical_rank is not None:
      conf_class = "badge-med"; conf_lbl = "Medium Match"
    else:
      conf_class = "badge-low"; conf_lbl = "Low Match"
    bid_title = res.metadata.get("bid_title", "N/A")
    bid_id = res.metadata.get("bid_id", "N/A")
    chunk_type = res.metadata.get("chunk_type", "N/A")
    ministry = res.metadata.get("ministry", "N/A")
    start_date_str = res.metadata.get("published_date") or res.metadata.get("bid_start_date") or "N/A"
    end_date_str = res.metadata.get("end_date_to_submit_bid") or res.metadata.get("bid_end_date") or "N/A"
    badges = [
        f'<span class="badge {conf_class}">{escape(conf_lbl)}</span>',
        '<span class="badge badge-new">NEW</span>' if is_new_bid(res.metadata) else "",
        f'<span class="badge badge-info">Bid ID: {escape(str(bid_id))}</span>',
        f'<span class="badge badge-none">Type: {escape(str(chunk_type))}</span>',
        f'<span class="badge badge-none">Ministry: {escape(str(ministry))}</span>',
        f'<span class="badge badge-none">Published: {escape(str(start_date_str))}</span>',
        f'<span class="badge badge-none">End: {escape(str(end_date_str))}</span>',
        f'<span class="badge badge-none">{"Relevance Score (reranked)" if use_reranker else "Fused Score"}: {score:.5f}</span>',
        f'<span class="badge badge-none">Dense Rank: {dense_rank or "N/A"}</span>',
        f'<span class="badge badge-none">Lexical Rank: {lexical_rank or "N/A"}</span>',
    ]
    card_html = (
        f'<div class="card"><div class="card-title">#{idx} | {escape(str(bid_title))}</div>'
        f'<div class="card-subtitle">{"".join(badges)}</div>'
        f'<div class="card-text">{escape(str(res.text))}</div></div>'
    )
    st.markdown(card_html, unsafe_allow_html=True)
    col_meta, col_dl = st.columns([4, 1])
    with col_meta:
      with st.expander(f"Inspect metadata for chunk: {res.chunk_id}"):
        st.json(res.metadata)
    with col_dl:
      pdf_file = find_pdf_file(res.metadata)
      if pdf_file:
        try:
          STATIC_DIR.mkdir(exist_ok=True)
          path_tag = hashlib.sha256(str(pdf_file.resolve()).encode("utf-8")).hexdigest()[:12]
          target_static_path = STATIC_DIR / f"{path_tag}_{pdf_file.name}"
          if not target_static_path.exists():
            import shutil
            shutil.copy2(pdf_file, target_static_path)
          with open(pdf_file, "rb") as f:
            pdf_bytes = f.read()
          st.download_button(label="📥 Download PDF", data=pdf_bytes, file_name=pdf_file.name, mime="application/pdf", key=f"dl_{res.chunk_id}_{idx}", use_container_width=True)
          st.markdown(f'<a href="/static/{target_static_path.name}" target="_blank" style="text-decoration: none;"><button style="background-color: #1e293b; color: #3b82f6; border: 1px solid #3b82f6; border-radius: 8px; padding: 6px 12px; width: 100%; cursor: pointer; margin-top: 4px; font-weight: 500;">📄 Open PDF in Tab</button></a>', unsafe_allow_html=True)
        except Exception:
          st.error("Error loading PDF")
      else:
        st.caption("PDF not found locally")

  # ── AI CHATBOT SECTION ────────────────────────────────────────────────────
  st.markdown("---")
  st.markdown('<div class="section-label">AI BID ADVISOR</div>', unsafe_allow_html=True)
  st.subheader("🤖 Chat with AI about a bid")
  st.caption("Select a bid from your search results to discuss with the local LLM (Qwen 3.5:9B).")

  selected_bid_id = None
  selected_metadata = {}
  selected_chunks = []

  seen_ids_chat = {}
  for res in results:
    bid_id = res.metadata.get("bid_id", "N/A")
    if bid_id not in seen_ids_chat:
      seen_ids_chat[bid_id] = res.metadata.get("bid_title", "Untitled")
  bid_options = {f"{bid_id}  —  {title}": bid_id for bid_id, title in seen_ids_chat.items()}
  if not bid_options:
    st.info("No bids available from current search results.")
  else:
    selected_label = st.selectbox("Select a bid to discuss", options=list(bid_options.keys()), key="chat_bid_selector")
    selected_bid_id = bid_options[selected_label]
    for res in results:
      if res.metadata.get("bid_id") == selected_bid_id:
        selected_chunks.append(res.text)
        if not selected_metadata:
          selected_metadata = res.metadata

  if selected_bid_id:
    pdf_file = find_pdf_file(selected_metadata) if selected_metadata else None
    full_pdf_text = extract_full_pdf_text(pdf_file) if pdf_file else None
    if full_pdf_text:
      selected_chunks = [full_pdf_text]
      st.caption(f"📄 **Full Local PDF Loaded**: `{pdf_file.name}` ({len(full_pdf_text):,} characters).")
    else:
      st.caption(f"🧩 **Search Chunks Loaded**: {len(selected_chunks)} section(s).")

    if "chat_histories" not in st.session_state:
      st.session_state.chat_histories = {}
    if selected_bid_id not in st.session_state.chat_histories:
      st.session_state.chat_histories[selected_bid_id] = []
    chat_history = st.session_state.chat_histories[selected_bid_id]

    chat_container = st.container()
    with chat_container:
      for msg in chat_history:
        with st.chat_message(msg["role"], avatar="🤖" if msg["role"] == "assistant" else "👤"):
          st.markdown(msg["content"])

    user_input = st.chat_input("Ask about this bid... e.g. 'Should I participate?'", key="chat_input")
    if user_input:
      if not ollama_status["ok"]:
        st.error("⚠️ Ollama is not available. Please start Ollama and ensure qwen3.5:9b is pulled.")
      else:
        chat_history.append({"role": "user", "content": user_input})
        system_prompt = build_system_prompt(selected_metadata, selected_chunks)
        llm_messages = [{"role": "system", "content": system_prompt}] + chat_history
        with chat_container:
          with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)
        with chat_container:
          with st.chat_message("assistant", avatar="🤖"):
            try:
              response_text = st.write_stream(chat_stream(llm_messages))
              chat_history.append({"role": "assistant", "content": response_text})
            except Exception as chat_err:
              error_msg = f"LLM request failed: {chat_err}"
              st.error(error_msg)
              chat_history.append({"role": "assistant", "content": f"⚠️ {error_msg}"})
        st.session_state.chat_histories[selected_bid_id] = chat_history

    if chat_history:
      if st.button("🗑️ Clear chat for this bid", key="clear_chat"):
        st.session_state.chat_histories[selected_bid_id] = []
        st.rerun()

# ── TAB 2: GLOBAL AI MATCHMAKER ──────────────────────────────────────────────
with tab_matchmaker:
  st.markdown('<div class="section-label">AI TENDER MATCHMAKER</div>', unsafe_allow_html=True)
  st.subheader("🎯 Rank all bids for your company")
  st.caption("Describe your company's products, turnover, and certifications. Qwen 3.5:9B will rank candidate bids by suitability.")

  col_p1, col_p2, col_p3 = st.columns(3)
  preset_text = ""
  if col_p1.button("🚓 Preset: Defence & Armored Vehicles"):
    preset_text = "Company Profile: Manufacturer of bullet-resistant tactical vehicles, light motor vehicles (LMV), and vehicle armoring.\nTurnover: 5 Crore/year\nExperience: 5 years supplying state police and defence forces.\nCertifications: ISO 9001, MII Class-1 Local Supplier."
  if col_p2.button("⚡ Preset: Electrical & Power Supplies"):
    preset_text = "Company Profile: Supplier of SMPS battery chargers, PCBs, harnesses, and electrical control cards.\nTurnover: 50 Lakhs/year\nExperience: 3 years in aerospace & electronics.\nCertifications: MSE / MSME Registered."
  if col_p3.button("💻 Preset: IT & Software Solutions"):
    preset_text = "Company Profile: Provider of custom software, display cards, and embedded systems.\nTurnover: 1 Crore/year\nExperience: 4 years IT consulting.\nCertifications: CMMI Level 3, MSME Registered."

  company_profile_input = st.text_area("Enter your Company Profile & Capabilities", value=preset_text, placeholder="e.g. We manufacture bulletproof vehicles, turnover 5 Cr, MSE registered...", height=140, key="matchmaker_profile_input")
  col_m1, col_m2 = st.columns([3, 1])
  with col_m1:
    domain_filter = st.text_input("Domain / Category Filter (Optional)", placeholder="e.g. vehicle, electrical, pcb, software", key="matchmaker_domain_filter")
  with col_m2:
    pool_size = st.slider("Bids to Evaluate", min_value=3, max_value=20, value=8, key="matchmaker_pool_size")

  matchmaker_btn = st.button("🚀 Find & Rank Best Bids for Me", key="run_matchmaker_btn")
  if matchmaker_btn:
    if not company_profile_input.strip():
      st.warning("Please enter your company profile or select a preset.")
    elif not retriever:
      st.error("Retriever is not initialized.")
    elif not ollama_status["ok"]:
      st.error("⚠️ Ollama is not available. Please start Ollama with qwen3.5:9b.")
    else:
      with st.spinner("Searching candidate bids across database..."):
        try:
          search_query = domain_filter.strip() if domain_filter.strip() else company_profile_input.split("\n")[0]

          # Apply query expansion to matchmaker search too
          matchmaker_search_terms = [search_query]
          if use_query_expansion and domain_filter.strip():
            if ollama_status["ok"]:
              matchmaker_search_terms = expand_query(search_query)
            else:
              matchmaker_search_terms = local_expand_query(search_query)
            if len(matchmaker_search_terms) > 1:
              st.caption(f"🧠 **Domain expanded to:** {', '.join(f'`{t}`' for t in matchmaker_search_terms)}")

          # Search each expanded term and merge
          seen_match_chunks: set[str] = set()
          all_match_results = []
          for mterm in matchmaker_search_terms:
            mterm_results = retriever.search(mterm, limit=pool_size * 3, exclude_boilerplate=True)
            for r in mterm_results:
              if r.chunk_id not in seen_match_chunks:
                seen_match_chunks.add(r.chunk_id)
                all_match_results.append(r)
          all_match_results.sort(key=lambda r: r.score, reverse=True)

          seen_match_bids = set()
          candidate_bids = []
          for res in all_match_results:
            bid_id = res.metadata.get("bid_id", "N/A")
            if bid_id not in seen_match_bids:
              seen_match_bids.add(bid_id)
              candidate_bids.append({"metadata": res.metadata, "text": res.text})
              if len(candidate_bids) >= pool_size * 2:
                break
          if use_reranker and candidate_bids:
            sr_candidates = [SearchResult(chunk_id=b["metadata"].get("bid_id", f"bid-{i}"), text=b["text"], metadata=b["metadata"], score=0.0) for i, b in enumerate(candidate_bids)]
            reranked_sr = retriever.rerank(search_query, sr_candidates, top_k=pool_size, min_score=0.0)
            candidate_bids = [{"metadata": r.metadata, "text": r.text} for r in reranked_sr]
          else:
            candidate_bids = candidate_bids[:pool_size]
          st.info(f"Loaded {len(candidate_bids)} candidate bid(s). Analyzing with Qwen 3.5:9B...")
          match_prompt = build_matchmaker_prompt(company_profile_input, candidate_bids)
          match_messages = [
              {"role": "system", "content": match_prompt},
              {"role": "user", "content": "Please analyze and rank these bids for my company now."},
          ]
          st.markdown("### 📊 AI Tender Matchmaker Report")
          st.write_stream(chat_stream(match_messages))
        except Exception as match_err:
          st.error(f"Matchmaker failed: {match_err}")
          st.code(traceback.format_exc())