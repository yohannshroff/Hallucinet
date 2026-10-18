"""Streamlit frontend for HalluciNet: ask a question, get a grounded answer
with a trust score and the evidence behind it; compare retrieval modes;
browse what's in the knowledge base. Calls the FastAPI backend in api/main.py.

Usage:
    # in one terminal:
    uvicorn api.main:app --port 8000
    # in another:
    streamlit run ui/app.py
"""

import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from common import REPO_ROOT  # noqa: E402

API_URL = os.environ.get("HALLUCINET_API_URL", "http://localhost:8000")
DEMO_CACHE_PATH = REPO_ROOT / "data" / "eval" / "demo_cache.json"
SAMPLE_QUESTIONS_CSV = REPO_ROOT / "data" / "eval" / "sample_questions.csv"

MODES = {
    "hybrid": "Hybrid: knowledge graph + book passages (recommended)",
    "graph": "Graph only: structured facts",
    "vector": "Passages only: book text search",
}
MODE_SHORT = {"hybrid": "Hybrid", "graph": "Graph only", "vector": "Passages only"}
VERDICT_ICON = {"entailed": "✅", "contradicted": "❌", "neutral": "⚠️", "no_evidence": "⚠️"}
VERDICT_TEXT = {
    "entailed": "Supported by the evidence",
    "contradicted": "Contradicted by the evidence",
    "neutral": "Not found in the evidence",
    "no_evidence": "No evidence was retrieved",
}

st.set_page_config(page_title="HalluciNet", page_icon="📜", layout="wide")

# --------------------------------------------------------------------------
# Backend access
# --------------------------------------------------------------------------


@st.cache_data(ttl=20, show_spinner=False)
def api_get(path: str):
    """GET from the backend; None if it's unreachable. Cached briefly so
    Streamlit's frequent reruns don't hammer the API."""
    try:
        resp = requests.get(f"{API_URL}{path}", timeout=8)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException:
        return None


def query_api(question: str, mode: str, k: int) -> dict:
    resp = requests.post(f"{API_URL}/ask", json={"question": question, "mode": mode, "k": k}, timeout=240)
    resp.raise_for_status()
    return resp.json()


@st.cache_data(show_spinner=False)
def load_demo_cache() -> list:
    if not DEMO_CACHE_PATH.exists():
        return []
    return json.loads(DEMO_CACHE_PATH.read_text(encoding="utf-8"))


def query_demo_cache(question: str):
    for entry in load_demo_cache():
        if entry["question"].strip().lower() == question.strip().lower():
            return entry
    return None


@st.cache_data(show_spinner=False)
def example_questions(limit: int = 6) -> list:
    if not SAMPLE_QUESTIONS_CSV.exists():
        return []
    df = pd.read_csv(SAMPLE_QUESTIONS_CSV)
    df = df[df["category"] != "out_of_scope"]
    return df["question"].head(limit).tolist()


def answer_question(question: str, mode: str, k: int):
    """Returns (result_dict | None, from_cache, error_text)."""
    try:
        started = time.perf_counter()
        result = query_api(question, mode, k)
        result["elapsed_s"] = time.perf_counter() - started
        return result, False, None
    except requests.RequestException as exc:
        cached = query_demo_cache(question)
        if cached is not None:
            return cached, True, None
        return None, False, str(exc)


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def trust_banner(result: dict):
    score = result.get("trust_score")
    n_claims, n_ok = result.get("n_claims"), result.get("n_entailed")
    if score is None:
        st.info("**Trust score: not applicable.** The answer made no factual claims to check (for example, it said the sources don't cover this).")
        return
    detail = f"{n_ok} of {n_claims} claims in this answer are backed by the retrieved evidence."
    if score >= 0.75:
        st.success(f"**Trust score {score:.0%} — well supported.** {detail}")
    elif score >= 0.4:
        st.warning(f"**Trust score {score:.0%} — partly supported.** {detail} Check the sources below.")
    else:
        st.error(f"**Trust score {score:.0%} — weakly supported.** {detail} Treat this answer with caution.")
    st.progress(min(max(score, 0.0), 1.0))


def render_result(result: dict, from_cache: bool = False):
    if from_cache:
        st.info("A live answer couldn't be generated (the language model may be switched off), so this is a saved answer from the demo cache.")

    with st.container(border=True):
        st.markdown(result["answer"])

    trust_banner(result)

    meta = []
    if result.get("mode"):
        meta.append(MODE_SHORT.get(result["mode"], result["mode"]))
    timing = result.get("timing_ms") or {}
    if timing:
        meta.append(f"answered in {(timing.get('answer', 0) + timing.get('trust_check', 0)) / 1000:.1f}s")
    if result.get("seed_entities"):
        meta.append("recognised: " + ", ".join(result["seed_entities"]))
    if meta:
        st.caption(" · ".join(meta))

    graph_facts = result.get("graph_facts") or []
    passages = result.get("passages") or []
    claims = result.get("claims") or []
    legacy_sources = result.get("sources") or []

    tab_sources, tab_claims = st.tabs(["Sources", "Claim-by-claim check"])

    with tab_sources:
        if not (graph_facts or passages or legacy_sources):
            st.write("No sources were retrieved for this question.")
        if graph_facts:
            st.markdown("**Knowledge graph facts**")
            for fact in graph_facts:
                link = f" — [source]({fact['source']})" if str(fact.get("source", "")).startswith("http") else ""
                st.markdown(f"- {fact['text']}{link}")
        if passages:
            st.markdown("**Book passages**")
            for p in passages:
                with st.expander(f"{p['title']} · {p['score']:.0%} match"):
                    st.markdown(f"> {p['snippet']}")
        if legacy_sources and not (graph_facts or passages):
            # demo-cache entries from before the richer response format
            for s in legacy_sources:
                st.markdown(f"- [{s['title']}]({s['url']})" if s.get("url") else f"- {s['title']}")

    with tab_claims:
        if not claims:
            st.write("No per-claim breakdown is available for this answer.")
        for c in claims:
            icon = VERDICT_ICON.get(c["verdict"], "•")
            st.markdown(f"{icon} **{c['claim']}**  \n{VERDICT_TEXT.get(c['verdict'], c['verdict'])} (confidence {c['score']:.0%})")
            if c.get("best_evidence"):
                st.caption(f"Closest evidence: {c['best_evidence']}")


def counts_frame(counts: dict) -> pd.DataFrame:
    return pd.DataFrame({"type": list(counts), "count": list(counts.values())})


def status_dot(ok) -> str:
    return "🟢" if ok else "🔴"


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------

st.session_state.setdefault("history", [])
st.session_state.setdefault("view", None)

with st.sidebar:
    st.header("Settings")
    mode = st.radio("How should it search?", list(MODES), format_func=MODES.get, index=0)
    k = st.slider("Book passages to retrieve", 1, 10, 5, disabled=(mode == "graph"))

    st.divider()
    health = api_get("/health")
    st.subheader("System")
    if health is None:
        st.markdown("🔴 Backend not reachable")
        st.caption(f"Start it with `uvicorn api.main:app --port 8000` (looking at {API_URL}).")
    else:
        st.markdown(f"{status_dot(health.get('ollama_reachable'))} Language model (Ollama)")
        st.markdown(f"{status_dot(health.get('neo4j_reachable'))} Knowledge graph (Neo4j)")
        st.markdown(f"{status_dot(health.get('index_ready'))} Book search index")
        if not health.get("ollama_reachable"):
            st.caption("Without the language model, only saved demo answers are available.")
    if st.button("Refresh status", width="stretch"):
        api_get.clear()
        st.rerun()

    if st.session_state.history:
        st.divider()
        st.subheader("Recent questions")
        for i, entry in enumerate(reversed(st.session_state.history[-6:])):
            if st.button(entry["question"], key=f"hist_{i}", width="stretch"):
                st.session_state.view = entry

# --------------------------------------------------------------------------
# Main page
# --------------------------------------------------------------------------

st.title("HalluciNet")
st.caption("Questions about Indian history, answered only from the loaded sources — each answer comes with a trust score and its evidence.")

tab_ask, tab_compare, tab_kb, tab_how = st.tabs(["Ask", "Compare search modes", "Knowledge base", "How it works"])

# ---- Ask -----------------------------------------------------------------
with tab_ask:

    def set_question(text: str):
        st.session_state["question"] = text
        st.session_state["run_question"] = text

    def submit_question():
        st.session_state["run_question"] = st.session_state.get("question", "").strip()

    def pick_example():
        text = st.session_state.get("example_pick")
        if text:
            set_question(text)
            st.session_state["example_pick"] = None  # so the same example can be picked again

    examples = example_questions()
    if examples:
        st.pills("Try one of these:", examples, key="example_pick", on_change=pick_example)

    with st.form("ask_form", border=False):
        st.text_input(
            "Ask a question about Indian history",
            key="question",
            placeholder="Who led the resistance at Jhansi?",
        )
        st.form_submit_button("Ask", type="primary", on_click=submit_question)

    to_run = st.session_state.pop("run_question", None)
    if to_run:
        with st.spinner("Searching the sources and writing an answer… (the first question can take a bit longer while the model loads)"):
            result, from_cache, error = answer_question(to_run, mode, k)
        if result is None:
            st.error(f"Couldn't get an answer: a live answer isn't available right now and this question isn't in the demo cache.\n\n`{error}`")
        else:
            result.setdefault("question", to_run)
            entry = {**result, "from_cache": from_cache}
            st.session_state.history.append(entry)
            st.session_state.view = entry

    view = st.session_state.view
    if view:
        st.subheader(view["question"])
        render_result(view, from_cache=view.get("from_cache", False))
    else:
        st.caption("Pick an example above or type your own question.")

# ---- Compare -------------------------------------------------------------
with tab_compare:
    st.markdown(
        "The same question, answered three ways. This is the heart of the project: does combining the "
        "knowledge graph with book search give more trustworthy answers than either alone?"
    )
    st.caption("This asks the language model three times, so it takes roughly three times as long.")
    cmp_question = st.text_input("Question to compare", value="Who led the resistance at Jhansi?", key="cmp_question")
    if st.button("Compare all three", type="primary", disabled=not cmp_question.strip()):
        results = {}
        progress = st.progress(0.0, text="Starting…")
        for i, m in enumerate(MODES):
            progress.progress(i / len(MODES), text=f"Searching: {MODE_SHORT[m]}…")
            res, _, err = answer_question(cmp_question.strip(), m, k)
            results[m] = res if res else {"error": err}
        progress.empty()
        st.session_state["compare_results"] = {"question": cmp_question.strip(), "results": results}

    cmp = st.session_state.get("compare_results")
    if cmp:
        st.subheader(cmp["question"])
        cols = st.columns(len(MODES))
        for col, m in zip(cols, MODES):
            res = cmp["results"].get(m, {})
            with col:
                st.markdown(f"**{MODE_SHORT[m]}**")
                if "error" in res:
                    st.error("No answer (backend unreachable).")
                    continue
                score = res.get("trust_score")
                st.metric("Trust score", "n/a" if score is None else f"{score:.0%}")
                with st.container(border=True):
                    st.markdown(res["answer"])
                st.caption(f"{res.get('n_entailed', '?')} of {res.get('n_claims', '?')} claims supported")

# ---- Knowledge base ------------------------------------------------------
with tab_kb:
    stats = api_get("/stats")
    kg = api_get("/kg")
    if stats is None or kg is None:
        st.warning("The backend isn't reachable, so the knowledge base can't be shown. Start it with `uvicorn api.main:app --port 8000`.")
    else:
        graph, corpus = stats["graph"], stats["corpus"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Graph nodes", graph["nodes"])
        m2.metric("Relationships", graph["relationships"])
        m3.metric("Passages", f"{corpus['chunks']:,}")
        m4.metric("Words", f"{corpus['words']:,}")
        st.caption(
            ("Live counts from the Neo4j graph database." if stats["neo4j_live"] else "Neo4j isn't reachable — showing counts from the seed files.")
            + f" Period(s) loaded: {', '.join(stats['periods'])}. More events and periods will be added over time."
        )

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**What the graph contains**")
            st.bar_chart(counts_frame(graph["node_types"]), x="type", y="count", horizontal=True, sort="-count")
        with c2:
            st.markdown("**How things are connected**")
            st.bar_chart(counts_frame(graph["relation_types"]), x="type", y="count", horizontal=True, sort="-count")

        if corpus["sources"]:
            st.markdown("**Source texts**")
            for s in corpus["sources"]:
                st.markdown(f"- {s}")

        st.divider()
        entities = pd.DataFrame(kg["entities"])
        relationships = pd.DataFrame(kg["relationships"])

        view_choice = st.segmented_control("Browse", ["Entities", "Relationships"], default="Entities")
        search = st.text_input("Filter", placeholder="Type a name, e.g. Lakshmibai", label_visibility="collapsed")

        if view_choice == "Relationships":
            table = relationships[["subject", "relation", "object", "source", "notes"]]
        else:
            table = entities[["name", "entity_type", "aliases", "description", "period_tag"]]
            types = st.multiselect("Entity type", sorted(entities["entity_type"].unique()))
            if types:
                table = table[table["entity_type"].isin(types)]
        if search.strip():
            mask = table.apply(lambda col: col.astype(str).str.contains(search.strip(), case=False, regex=False)).any(axis=1)
            table = table[mask]
        st.dataframe(
            table,
            hide_index=True,
            width="stretch",
            column_config={"source": st.column_config.LinkColumn("source", display_text="open")} if view_choice == "Relationships" else None,
        )
        st.caption(f"{len(table)} rows")

# ---- How it works --------------------------------------------------------
with tab_how:
    st.markdown(
        """
**1. Find the evidence.** Your question is matched against two stores: a small *knowledge graph* of people, places and
events and how they connect, and the full text of the source books, split into short passages.

**2. Write an answer from that evidence only.** A language model running on this computer is told to use nothing
except what was retrieved — and to say so when the sources don't cover the question.

**3. Check the answer.** The answer is split into individual claims. A separate checking model asks, for each claim:
*does any piece of the retrieved evidence actually back this up?*

**4. Report a trust score.** The score is the share of claims that are backed by the evidence. It is **not** a
guarantee that the answer is historically true — only that it's supported by *these* sources.
        """
    )
    st.markdown("**Good to know**")
    st.markdown(
        "- A low score doesn't always mean a wrong answer; it can mean the sources word things differently.\n"
        "- A high score can't catch an error that is in the sources themselves.\n"
        "- The knowledge graph is small and hand-built, and currently covers one event (the 1857 Revolt) as the first case study."
    )
