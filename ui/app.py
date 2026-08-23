"""Streamlit frontend: question box, grounded answer, trust score, and
cited sources -- calls the FastAPI backend in api/main.py.

Usage:
    # in one terminal:
    uvicorn api.main:app --port 8000
    # in another:
    streamlit run ui/app.py
"""

import sys
from pathlib import Path

import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from common import REPO_ROOT  # noqa: E402

API_URL = "http://localhost:8000"
DEMO_CACHE_PATH = REPO_ROOT / "data" / "eval" / "demo_cache.json"

st.set_page_config(page_title="HalluciNet", page_icon="📜")
st.title("HalluciNet")
st.caption("Grounded Q&A on the Revolt of 1857 -- graph + vector retrieval, with a trust score for every answer.")

mode = st.sidebar.selectbox("Retrieval mode", ["hybrid", "vector", "graph"], index=0)
k = st.sidebar.slider("Vector top-k", min_value=1, max_value=10, value=5)
st.sidebar.markdown("---")
st.sidebar.caption(
    "If the API isn't reachable (e.g. during a live demo with Ollama running slow), "
    "answers fall back to the pre-cached results in data/eval/demo_cache.json."
)


def query_api(question: str, mode: str, k: int) -> dict:
    resp = requests.post(f"{API_URL}/ask", json={"question": question, "mode": mode, "k": k}, timeout=180)
    resp.raise_for_status()
    return resp.json()


def query_demo_cache(question: str) -> dict | None:
    import json

    if not DEMO_CACHE_PATH.exists():
        return None
    cache = json.loads(DEMO_CACHE_PATH.read_text(encoding="utf-8"))
    for entry in cache:
        if entry["question"].strip().lower() == question.strip().lower():
            return entry
    return None


question = st.text_input("Ask a question about the 1857 Revolt", placeholder="Who led the resistance at Jhansi?")

if st.button("Ask") and question:
    result = None
    used_cache = False
    try:
        with st.spinner("Retrieving context and generating a grounded answer..."):
            result = query_api(question, mode, k)
    except requests.RequestException as exc:
        st.warning(f"API not reachable ({exc}) -- checking the demo cache instead.")
        result = query_demo_cache(question)
        used_cache = result is not None

    if result is None:
        st.error("Could not get an answer -- API is down and this question isn't in the demo cache.")
    else:
        if used_cache:
            st.info("Showing a pre-cached answer (live API unavailable).")

        st.subheader("Answer")
        st.write(result["answer"])

        trust_score = result.get("trust_score")
        if trust_score is None:
            st.caption("Trust score: N/A (no checkable factual claims in this answer)")
        else:
            st.metric("Trust score", f"{trust_score:.0%}", help=f"{result.get('n_entailed', '?')}/{result.get('n_claims', '?')} claims entailed by the retrieved evidence")

        seed_entities = result.get("seed_entities") or []
        if seed_entities:
            st.caption(f"Entities recognized in your question: {', '.join(seed_entities)}")

        sources = result.get("sources") or []
        if sources:
            st.subheader("Sources")
            for s in sources:
                st.markdown(f"- [{s['title']}]({s['url']})")
