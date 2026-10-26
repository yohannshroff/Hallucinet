"""Streamlit frontend for HalluciNet: ask questions in a chat, get grounded
answers with a trust score and the evidence behind them; compare retrieval
modes; browse what's in the knowledge base. Calls the FastAPI backend in
api/main.py.

Styling: base colours and fonts in .streamlit/config.toml, everything else in
ui/styles.css (injected once below). Layout code here only adds class names.

Usage:
    # in one terminal:
    uvicorn api.main:app --port 8000
    # in another:
    streamlit run ui/app.py
"""

import html
import json
import os
import re
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
STYLES_CSS = Path(__file__).with_name("styles.css")
ASK_TIMEOUT = 240  # seconds per question on the Ask page
COMPARE_TIMEOUT = 600  # per mode on Compare: three answers in a row, and a slow laptop can take minutes each
TIMEOUT_LEAD = "The backend took longer than"

MODES = {
    "hybrid": "Hybrid: knowledge graph + book passages (recommended)",
    "graph": "Graph only: structured facts",
    "vector": "Passages only: book text search",
}
MODE_SHORT = {"hybrid": "Hybrid", "graph": "Graph only", "vector": "Passages only"}
VERDICT_TEXT = {
    "entailed": "Supported",
    "contradicted": "Contradicted",
    "contradiction": "Contradicted by the evidence",  # the NLI model's own label
    "neutral": "Not found in the evidence",
    "no_evidence": "No evidence retrieved",
}
PAGES = {"chat": "Ask", "compare": "Compare modes", "kb": "Knowledge base", "how": "How it works"}
K_MIN, K_MAX = 1, 10

st.set_page_config(page_title="HalluciNet", layout="centered", initial_sidebar_state="expanded")
st.markdown(f"<style>{STYLES_CSS.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

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


def query_api(question: str, mode: str, k: int, timeout: int = ASK_TIMEOUT) -> dict:
    resp = requests.post(f"{API_URL}/ask", json={"question": question, "mode": mode, "k": k}, timeout=timeout)
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
def example_questions(limit: int = 3) -> list:
    if not SAMPLE_QUESTIONS_CSV.exists():
        return []
    df = pd.read_csv(SAMPLE_QUESTIONS_CSV)
    df = df[df["category"] != "out_of_scope"]
    return df["question"].head(limit).tolist()


def error_text(exc: requests.RequestException, timeout: int) -> str:
    """Why a request failed, in plain words. A timed-out question keeps running on
    the backend, so the next one has to wait for it."""
    if isinstance(exc, requests.Timeout):
        return (f"{TIMEOUT_LEAD} {timeout // 60} minutes. It may still be finishing an earlier "
                "question; try again in a minute.")
    if isinstance(exc, requests.ConnectionError):
        return "The backend isn't running. Start it with: uvicorn api.main:app --port 8000"
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        return f"The backend hit an error (HTTP {exc.response.status_code}). Check the uvicorn terminal."
    return str(exc)


def answer_question(question: str, mode: str, k: int, timeout: int = ASK_TIMEOUT):
    """Returns (result_dict | None, from_cache, error_text)."""
    try:
        started = time.perf_counter()
        result = query_api(question, mode, k, timeout)
        result["elapsed_s"] = time.perf_counter() - started
        return result, False, None
    except requests.RequestException as exc:
        cached = query_demo_cache(question)
        if cached is not None:
            return cached, True, None
        return None, False, error_text(exc, timeout)


# --------------------------------------------------------------------------
# Rendering helpers
# --------------------------------------------------------------------------


def esc(text) -> str:
    return html.escape(str(text), quote=False)


def label(text: str):
    st.markdown(f'<div class="hn-label">{esc(text)}</div>', unsafe_allow_html=True)


def secondary(text: str):
    st.markdown(f'<div class="hn-secondary">{text}</div>', unsafe_allow_html=True)


def rule():
    st.markdown('<hr class="hn-rule">', unsafe_allow_html=True)


def header_bar(left: str, right: str = ""):
    st.markdown(
        f'<div class="hn-header"><span class="hn-header-q">{esc(left)}</span>'
        f'<span class="hn-header-meta">{esc(right)}</span></div>',
        unsafe_allow_html=True,
    )


FACT_ARROW_RE = re.compile(r"\s*--([\w-]+)-->\s*")


def readable_fact(text: str) -> str:
    """'Hodson --captured--> Bahadur Shah' -> 'Hodson captured Bahadur Shah'."""
    return FACT_ARROW_RE.sub(lambda m: f" {m.group(1).replace('_', ' ')} ", text)


PAGES_RE = r"\(?\bpp?\.\s*(\d+)\s*[-–]\s*(\d+)\)?"
TITLE_PAGES_RE = re.compile(r"^(.*?)(?: - [^()]*)?\s*\(pp\. (\d+)-(\d+)\)$")
PAGE_ANCHOR_RE = re.compile(r"#page=(\d+)")


def short_title(title: str) -> str:
    """'The Great Mutiny - India 1857 by Christopher Hibbert (pp. 401-404)' -> 'The Great Mutiny, pp. 401–404'."""
    m = TITLE_PAGES_RE.match(title)
    return f"{m.group(1)}, pp. {m.group(2)}–{m.group(3)}" if m else title


def page_range(source: dict):
    """(first, last) book page a source covers, or None."""
    m = re.search(PAGES_RE, source.get("full_title", ""))
    if m:
        return int(m.group(1)), int(m.group(2))
    m = PAGE_ANCHOR_RE.search(str(source.get("url", "")))
    return (int(m.group(1)), int(m.group(1))) if m else None


def build_sources(result: dict) -> list:
    """One numbered list for the answer's citations: graph facts, then book passages
    -- the order and numbering the LLM saw. Old demo-cache entries only have `sources`."""
    out = []
    for f in result.get("graph_facts") or []:
        title = f.get("sentence") or readable_fact(f["text"])
        m = PAGE_ANCHOR_RE.search(f.get("source", ""))
        if m:
            title = f"{title.rstrip('.')} · p. {m.group(1)}"
        out.append({"title": title, "kind": "Knowledge graph", "url": f.get("source", ""), "snippet": ""})
    for p in result.get("passages") or []:
        out.append({
            "title": short_title(p["title"]), "full_title": p["title"], "kind": "Book passage",
            "url": p.get("source", ""), "snippet": p.get("snippet", ""), "passage": True,
        })
    if not out:
        for s in result.get("sources") or []:
            out.append({"title": s["title"], "kind": "Source", "url": s.get("url", ""), "snippet": ""})
    for n, s in enumerate(out, start=1):
        s["n"] = n
    return out


def dedupe_after_stop(m, mark: str) -> str:
    """'[3][6]. [6][7]' -> '[3][6][7].': markers repeated after the full stop join the ones before it."""
    before = re.findall(mark, m.group(1))
    extra = [x for x in re.findall(mark, m.group(2)) if x not in before]
    return "".join(before + extra) + "."


# "data/raw/The Great Mutiny - India 1857 by Christopher Hibbert.pdf#page=393", bare or in brackets
PDF_PATH_RE = r"[\[(]?\s*(?:\S*/)?[^()\[\]/\n<>]*\.pdf(?:#page=(?P<page>\d+))?\s*[\])]?"


def cite(answer: str, sources: list) -> str:
    """Escape the answer, then replace the LLM's source numbers ([2] = source 2), inline
    URLs, file paths, passage titles and page ranges with [n] markers that match the
    source list. A file path that matches no source is dropped; anything else is left."""
    passages = [s for s in sources if s.get("passage")]
    by_url = {s["url"]: s["n"] for s in sources if str(s["url"]).startswith("http")}
    by_title = {s["full_title"]: s["n"] for s in passages}
    by_pages = {}
    for s in passages:
        m = re.search(PAGES_RE, s["full_title"])
        if m:
            by_pages.setdefault(f"{m.group(1)}-{m.group(2)}", s["n"])

    def on_page(page: int):
        hits = [s["n"] for s in sources if (r := page_range(s)) and r[0] <= page <= r[1]]
        return hits[0] if hits else None

    alternatives = [rf"(?P<path>{PDF_PATH_RE})", r"(?P<url>https?://[^\s)\]]*[^\s)\].,;:])"]  # trailing punctuation stays in the sentence
    if by_title:
        alternatives.append("(?P<title>" + "|".join(re.escape(esc(t)) for t in sorted(by_title, key=len, reverse=True)) + ")")
    alternatives += [f"(?P<pages>{PAGES_RE})", r"\[(?P<idx>\d+(?:\s*,\s*\d+)*)\]"]  # [2] or [3, 6]
    pattern = re.compile("|".join(alternatives))
    by_esc_title = {esc(t): n for t, n in by_title.items()}

    def marker(n: int) -> str:
        return f'<span class="hn-cite">[{n}]</span>'

    def replace(m):
        if m.group("path"):
            n = on_page(int(m.group("page"))) if m.group("page") else None
            return marker(n) if n else ""
        if m.group("url"):
            n = by_url.get(m.group("url"))
        elif by_title and m.group("title"):
            n = by_esc_title[m.group("title")]
        elif m.group("pages"):
            nums = re.findall(r"\d+", m.group("pages"))
            n = by_pages.get(f"{nums[0]}-{nums[1]}")
        else:
            # a number outside the source list points at nothing, so it is dropped
            return "".join(marker(i) for i in map(int, re.findall(r"\d+", m.group("idx"))) if 1 <= i <= len(sources))
        return marker(n) if n else m.group(0)

    mark = r"<span class=\"hn-cite\">\[\d+\]</span>"
    text = pattern.sub(replace, esc(answer))
    text = re.sub(rf"({mark})(\s*\1)+", r"\1", text)  # "[2] [2]" -> "[2]"
    text = re.sub(rf"({mark})\s*,\s*(?={mark})", r"\1", text)  # "[2], [4]" -> "[2][4]"
    text = re.sub(rf"\(\s*(?:[Ss]ources?:?\s*)?((?:{mark}[,;\s]*)+)\)", r"\1", text)  # "(Source: [2])" -> "[2]"
    text = re.sub(rf"\b[Ss]ources?:\s*((?:{mark}[,;\s]*)+)", r"\1", text)  # "Source: [2]" -> "[2]"
    text = re.sub(r"\(\s*(?:[Ss]ources?:?)?[\s,;]*\)|\b[Ss]ources?:[\s,;]*$", "", text)  # empty leftovers
    text = re.sub(r" +([.,;])", r"\1", text)
    text = re.sub(rf"((?:{mark})+)\.\s*((?:{mark})+)", lambda m: dedupe_after_stop(m, mark), text)  # "[6]. [6]" -> "[6]."
    return text.strip()


def trust_block(result: dict):
    score = result.get("trust_score")
    n_claims, n_ok = result.get("n_claims"), result.get("n_entailed")
    label("Trust score")
    if score is None:
        st.markdown('<div class="hn-trust-value">n/a</div>', unsafe_allow_html=True)
        secondary("Not applicable: the answer made no checkable claims (for example, it said the sources don't cover this).")
        return
    width = min(max(score, 0.0), 1.0) * 100
    noun = "claim" if n_claims == 1 else "claims"
    st.markdown(
        f'<div class="hn-trust-value">{score:.0%}</div>'
        f'<div class="hn-bar"><div style="width:{width:.1f}%"></div></div>',
        unsafe_allow_html=True,
    )
    secondary(f"{n_ok} of {n_claims} {noun} supported by the evidence")


def claims_block(result: dict):
    claims = result.get("claims") or []
    with st.expander("Claim-by-claim check"):
        if not claims:
            st.markdown("No per-claim breakdown is available for this answer.")
        for c in claims:
            # HTML, not markdown: an asterisk left in a claim would otherwise turn into bold/italics
            verdict = VERDICT_TEXT.get(c["verdict"], c["verdict"].capitalize())
            evidence = f'<div class="hn-claim-ev">Closest evidence: {esc(c["best_evidence"])}</div>' if c.get("best_evidence") else ""
            st.markdown(
                f'<div class="hn-claim"><div class="hn-claim-text">{esc(c["claim"])}</div>'
                f'<div class="hn-claim-verdict">{esc(verdict)} · support {c["score"]:.0%}</div>{evidence}</div>',
                unsafe_allow_html=True,
            )


def sources_block(sources: list):
    label("Sources")
    if not sources:
        secondary("No sources were retrieved for this question.")
        return
    rows = []
    for s in sources:
        row = (
            f'<span class="hn-src-n">[{s["n"]}]</span><span class="hn-src-title">{esc(s["title"])}</span>'
            f'<span class="hn-src-kind">{s["kind"]}</span>'
        )
        if s["snippet"]:
            rows.append(f'<details class="hn-src"><summary>{row}</summary><blockquote>{esc(s["snippet"])}</blockquote></details>')
        else:
            rows.append(f'<div class="hn-src"><div class="hn-src-row">{row}</div></div>')
    st.markdown(f'<div class="hn-sources">{"".join(rows)}</div>', unsafe_allow_html=True)


def meta_line(result: dict):
    parts = []
    if result.get("seed_entities"):
        parts.append("Recognised: " + esc(", ".join(result["seed_entities"])))
    if result.get("mode"):
        parts.append(MODE_SHORT.get(result["mode"], result["mode"]))
    timing = result.get("timing_ms") or {}
    if timing:
        parts.append(f"{(timing.get('answer', 0) + timing.get('trust_check', 0)) / 1000:.1f} s")
    if parts:
        secondary(" · ".join(parts))


def render_answer(entry: dict, idx: int):
    st.markdown('<div class="hn-label">HalluciNet</div>', unsafe_allow_html=True)
    result = entry.get("result")
    if result is None:
        secondary(
            "Couldn't get an answer: a live answer isn't available right now and this question isn't in the demo cache."
            f"<br>{esc(entry.get('error', ''))}"
        )
        return
    sources = build_sources(result)
    with st.container(key=f"answer_{idx}"):
        st.markdown(cite(result["answer"], sources), unsafe_allow_html=True)
    if entry.get("from_cache"):
        secondary("Saved answer from the demo cache: a live answer couldn't be generated (the language model may be switched off).")
    rule()
    trust_block(result)
    claims_block(result)
    rule()
    sources_block(sources)
    rule()
    meta_line(result)


# --------------------------------------------------------------------------
# State and actions
# --------------------------------------------------------------------------

st.session_state.setdefault("messages", [])  # [{question, result, from_cache, error}]
st.session_state.setdefault("selected", None)
st.session_state.setdefault("page", "chat")
st.session_state.setdefault("k", 5)
st.session_state.setdefault("mode", "hybrid")


def new_question():
    st.session_state.messages = []
    st.session_state.selected = None
    st.session_state.page = "chat"


def select_message(i: int):
    st.session_state.selected = i
    st.session_state.page = "chat"
    st.session_state.scroll_to = i


def go_to(page: str):
    st.session_state.page = page


def ask(text: str):
    st.session_state.run_question = text
    st.session_state.page = "chat"


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------

with st.sidebar:
    st.markdown('<div class="hn-wordmark">HalluciNet</div>', unsafe_allow_html=True)
    st.button("New question", key="new_question", on_click=new_question, width="stretch")

    messages = st.session_state.messages
    if messages:
        label("Recent")
        with st.container(key="recent"):
            for i in reversed(range(len(messages))):
                st.button(
                    messages[i]["question"], key=f"recent_{i}", width="stretch",
                    type="primary" if i == st.session_state.selected else "secondary",
                    on_click=select_message, args=(i,),
                )

    with st.container(key="nav"):
        for page, page_label in PAGES.items():
            if page != st.session_state.page:
                st.button(page_label, key=f"nav_{page}", type="tertiary", on_click=go_to, args=(page,))

    with st.container(key="settings"):
        label("Settings")
        mode = st.selectbox("Retrieval mode", list(MODES), format_func=MODE_SHORT.get, key="mode", help=None)
        secondary(esc(MODES[mode]))
        st.markdown(
            f'<div class="hn-slider-head"><span>Vector top-k</span><span>{st.session_state.k}</span></div>',
            unsafe_allow_html=True,
        )
        k = st.slider("Vector top-k", K_MIN, K_MAX, key="k", label_visibility="collapsed", disabled=(mode == "graph"))
        st.markdown(f'<div class="hn-slider-range"><span>{K_MIN}</span><span>{K_MAX}</span></div>', unsafe_allow_html=True)

        health = api_get("/health")
        if health is None:
            status = f"Backend not reachable at {esc(API_URL)}. Start it with <code>uvicorn api.main:app --port 8000</code>."
        else:
            on_off = lambda ok: "on" if ok else "off"  # noqa: E731
            status = (
                f"Language model {on_off(health.get('ollama_reachable'))} · Graph {on_off(health.get('neo4j_reachable'))}"
                f" · Search index {on_off(health.get('index_ready'))}"
            )
        secondary(status + "<br>If the backend or the language model is unavailable, saved demo answers are shown instead.")
        with st.container(key="refresh"):
            if st.button("Refresh status", type="tertiary"):
                api_get.clear()
                st.rerun()

settings_meta = f"{MODE_SHORT[mode]} · top-k {k}" if mode != "graph" else MODE_SHORT[mode]

# --------------------------------------------------------------------------
# Chat page
# --------------------------------------------------------------------------


def chat_page():
    prompt = st.chat_input("Ask about Indian history")
    if prompt and prompt.strip():
        st.session_state.run_question = prompt.strip()
    pending = st.session_state.pop("run_question", None)
    messages = st.session_state.messages

    if not messages and not pending:
        with st.container(key="empty"):
            st.markdown(
                '<p class="hn-title">HalluciNet</p>'
                '<p class="hn-subtitle">Questions about Indian history, answered only from the loaded sources.</p>',
                unsafe_allow_html=True,
            )
            with st.container(key="examples"):
                for i, q in enumerate(example_questions()):
                    st.button(q, key=f"example_{i}", type="tertiary", on_click=ask, args=(q,))
        return

    sel = st.session_state.selected
    current = pending or (messages[sel]["question"] if sel is not None and sel < len(messages) else messages[-1]["question"])
    header_bar(current, settings_meta)

    for i, entry in enumerate(messages):
        st.markdown(f'<div id="hn-msg-{i}"></div>', unsafe_allow_html=True)
        with st.chat_message("user"):
            st.markdown(esc(entry["question"]))
        with st.chat_message("assistant"):
            render_answer(entry, i)

    if pending:
        with st.chat_message("user"):
            st.markdown(esc(pending))
        with st.chat_message("assistant"):
            with st.spinner("Searching the sources and writing an answer… (the first question can take longer while the model loads)"):
                result, from_cache, error = answer_question(pending, mode, k)
        if result is not None:
            result.setdefault("question", pending)
        messages.append({"question": pending, "result": result, "from_cache": from_cache, "error": error})
        st.session_state.selected = len(messages) - 1
        st.rerun()

    target = st.session_state.pop("scroll_to", None)
    if target is not None:  # target is a message index (int), never user text
        with st.container(key="scroller"):
            st.iframe(
                f"<script>window.parent.document.getElementById('hn-msg-{int(target)}')?.scrollIntoView({{behavior: 'smooth'}});</script>",
                height=1,
            )


# --------------------------------------------------------------------------
# Other pages
# --------------------------------------------------------------------------


def compare_page():
    header_bar("Compare search modes", settings_meta)
    st.markdown(
        "The same question, answered three ways. This is the heart of the project: does combining the "
        "knowledge graph with book search give more trustworthy answers than either alone?"
    )
    secondary("This asks the language model three times in a row, so it can take several minutes.")
    cmp_question = st.text_input("Question to compare", value="Who led the resistance at Jhansi?", key="cmp_question")
    if st.button("Compare all three", type="primary", disabled=not cmp_question.strip()):
        results = {}
        progress = st.progress(0.0, text="Starting…")
        busy = False
        for i, m in enumerate(MODES):
            if busy:  # don't queue more work behind a question the backend is still answering
                results[m] = {"error": "Skipped: the backend is still busy with the previous mode."}
                continue
            progress.progress(i / len(MODES), text=f"Asking: {MODE_SHORT[m]} ({i} of {len(MODES)} done)…")
            res, _, err = answer_question(cmp_question.strip(), m, k, COMPARE_TIMEOUT)
            results[m] = res if res else {"error": err}
            busy = bool(err and err.startswith(TIMEOUT_LEAD))
        progress.empty()
        st.session_state["compare_results"] = {"question": cmp_question.strip(), "results": results}

    cmp = st.session_state.get("compare_results")
    if not cmp:
        return
    rule()
    label(cmp["question"])
    cols = st.columns(len(MODES))
    for col, m in zip(cols, MODES):
        res = cmp["results"].get(m, {})
        with col:
            label(MODE_SHORT[m])
            if "error" in res:
                secondary("No answer. " + esc(res["error"] or ""))
                continue
            trust_block(res)
            rule()
            with st.container(key=f"answer_cmp_{m}"):
                st.markdown(cite(res["answer"], build_sources(res)), unsafe_allow_html=True)


def counts_frame(counts: dict) -> pd.DataFrame:
    return pd.DataFrame({"type": list(counts), "count": list(counts.values())})


def kb_page():
    header_bar("Knowledge base")
    stats = api_get("/stats")
    kg = api_get("/kg")
    if stats is None or kg is None:
        secondary("The backend isn't reachable, so the knowledge base can't be shown. Start it with <code>uvicorn api.main:app --port 8000</code>.")
        return
    graph, corpus = stats["graph"], stats["corpus"]
    numbers = [("Graph nodes", graph["nodes"]), ("Relationships", graph["relationships"]), ("Passages", f"{corpus['chunks']:,}"), ("Words", f"{corpus['words']:,}")]
    st.markdown(
        '<div class="hn-stats">'
        + "".join(f'<div><div class="hn-stat-value">{v}</div><div class="hn-stat-label">{esc(n)}</div></div>' for n, v in numbers)
        + "</div>",
        unsafe_allow_html=True,
    )
    secondary(
        ("Live counts from the Neo4j graph database." if stats["neo4j_live"] else "Neo4j isn't reachable — showing counts from the seed files.")
        + f" Period(s) loaded: {esc(', '.join(stats['periods']))}. More events and periods will be added over time."
    )
    rule()
    c1, c2 = st.columns(2)
    with c1:
        label("What the graph contains")
        st.bar_chart(counts_frame(graph["node_types"]), x="type", y="count", horizontal=True, sort="-count", color="#173846")
    with c2:
        label("How things are connected")
        st.bar_chart(counts_frame(graph["relation_types"]), x="type", y="count", horizontal=True, sort="-count", color="#173846")

    if corpus["sources"]:
        label("Source texts")
        for s in corpus["sources"]:
            st.markdown(f"- {s}")

    rule()
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
    secondary(f"{len(table)} rows")


def how_page():
    header_bar("How it works")
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
    label("Good to know")
    st.markdown(
        "- A low score doesn't always mean a wrong answer; it can mean the sources word things differently.\n"
        "- A high score can't catch an error that is in the sources themselves.\n"
        "- The knowledge graph is small and built from the source book, and currently covers one event (the 1857 Revolt) as the first case study."
    )


{"chat": chat_page, "compare": compare_page, "kb": kb_page, "how": how_page}[st.session_state.page]()
