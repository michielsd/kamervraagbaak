import gzip
import json
import math
from pathlib import Path
from typing import Any

import streamlit as st
from openai import OpenAI


EMBEDDING_FILES = {
    "Binnenlandse zaken": [
        "Data/bzk.json.gz",
    ],
    "Financiën": [
        "Data/fin.json.gz",
    ],
}


def _load_json_or_json_gz(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            data = json.load(f)
    else:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"Expected JSON object in {path}, got {type(data).__name__}.")
    return data


def _resolve_source_path(source_name: str) -> Path:
    candidates = EMBEDDING_FILES.get(source_name) or []
    for p in candidates:
        if p.exists():
            return p
    names = ", ".join(str(p) for p in candidates) or "(geen kandidaten)"
    raise FileNotFoundError(f"Geen datafile gevonden voor '{source_name}'. Geprobeerd: {names}")


@st.cache_resource(show_spinner=False)
def _load_flat_chunks(source_name: str) -> list[dict[str, Any]]:
    """Load chunks with embeddings from selected data file."""
    source_path = _resolve_source_path(source_name)
    data = _load_json_or_json_gz(source_path)
    docs = data.get("documents")
    if not isinstance(docs, list):
        return []
    out: list[dict[str, Any]] = []
    for doc in docs:
        if not isinstance(doc, dict):
            continue
        chunks = doc.get("chunks")
        if not isinstance(chunks, list):
            continue
        for chunk in chunks:
            if not isinstance(chunk, dict):
                continue
            emb = chunk.get("embedding")
            if not isinstance(emb, list) or not emb:
                continue
            if not all(isinstance(x, (int, float)) for x in emb):
                continue
            out.append(chunk)
    return out


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError("Embedding dimensions do not match.")
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        xf = float(x)
        yf = float(y)
        dot += xf * yf
        norm_a += xf * xf
        norm_b += yf * yf
    denom = math.sqrt(norm_a) * math.sqrt(norm_b)
    if denom == 0:
        return 0.0
    return dot / denom


def query_local_embeddings(query: str, source_name: str, top_k: int = 10) -> dict[str, Any] | None:
    openai_api_key = st.secrets.get("OPENAI_API_KEY", "")
    if not openai_api_key:
        st.error("Configureer `OPENAI_API_KEY` in Streamlit secrets om te zoeken.")
        return None

    chunks = _load_flat_chunks(source_name)
    if not chunks:
        st.warning("Geen chunks met embeddings gevonden in de gekozen databron.")
        return {"results": []}

    try:
        client = OpenAI(api_key=openai_api_key)
        resp = client.embeddings.create(
            model="text-embedding-3-small",
            input=query,
        )
        query_embedding = resp.data[0].embedding
    except Exception as e:
        st.error(f"Fout bij maken query-embedding: {e}")
        return None

    scored: list[dict[str, Any]] = []
    for chunk in chunks:
        emb = chunk.get("embedding")
        if not isinstance(emb, list):
            continue
        try:
            sim = _cosine_similarity(query_embedding, emb)
        except Exception:
            continue
        scored.append(
            {
                "chunk_id": chunk.get("chunk_id"),
                "content": chunk.get("content", ""),
                "metadata": chunk.get("metadata") or {},
                "cosine_similarity": sim,
            }
        )

    scored.sort(key=lambda x: -(x.get("cosine_similarity") or 0.0))
    return {"results": scored[:top_k]}


# Load secrets from Streamlit
openai_api_key = st.secrets["OPENAI_API_KEY"]
password = st.secrets["PASSWORD"]

pwd = st.sidebar.text_input("Wachtwoord:", value="", type="password")
if pwd != password:
    st.error("Voer het wachtwoord in om toegang te krijgen tot de applicatie")
    st.stop()

def _deduplicated_rows(results: list[dict]) -> list[dict]:
    """Deduplicate by metadata.titel (keep highest cosine_similarity per title), sorted by similarity."""
    if not results:
        return []
    by_title: dict[str, dict] = {}
    for r in results:
        meta = r.get("metadata") or {}
        titel = meta.get("titel") or "(geen titel)"
        sim = r.get("cosine_similarity") or 0.0
        if titel not in by_title or (by_title[titel].get("cosine_similarity") or 0) < sim:
            by_title[titel] = r
    return sorted(by_title.values(), key=lambda x: -(x.get("cosine_similarity") or 0))


def build_results_table(indexed_rows: list[tuple[int, dict]], key_prefix: str = "") -> None:
    """
    Show a table: hyperlinked title | datum | similarity | checkbox.
    Expects rows already deduplicated and indexed with their original position.
    """
    if not indexed_rows:
        st.info("Geen resultaten.")
        return

    # Header
    c0, c1, c2, c3 = st.columns([3, 1.2, 0.48, 0.5])
    c0.markdown("**Titel**")
    c1.markdown("**Datum**")
    c2.markdown("**Match**")
    c3.markdown("**Select**")
    st.divider()

    for original_idx, r in indexed_rows:
        meta = r.get("metadata") or {}
        titel = meta.get("titel") or "(geen titel)"
        link = meta.get("kamerstuk_link") or ""
        datum = meta.get("datum") or "—"
        sim = r.get("cosine_similarity")
        col0, col1, col2, col3 = st.columns([3, 1.2, 0.48, 0.5])
        with col0:
            if link:
                st.markdown(f"[{titel}]({link})")
            else:
                st.write(titel)
        col1.write(datum)
        with col2:
            if sim is not None:
                # Map cosine similarity from [-1, 1] to progress [0, 1]
                progress = (float(sim) + 1) / 2
                st.progress(min(1.0, max(0.0, progress)))
            else:
                st.write("—")
        with col3:
            st.checkbox("", key=f"{key_prefix}result_cb_{original_idx}", label_visibility="collapsed")


st.title("📑Kamervraagbaak")

# Search form: pressing Enter in the search box submits the form
with st.form("search_form"):
    source_name = st.selectbox(
        "Bron",
        options=["Binnenlandse zaken", "Financiën"],
        index=0,
    )
    search_query = st.text_input("Zoeken", placeholder="Typ je zoekvraag...")
    top_k = st.number_input("Aantal resultaten (top_k)", min_value=1, max_value=100, value=10)
    submitted = st.form_submit_button("Zoeken")
if submitted:
    if search_query and search_query.strip():
        with st.spinner("Bezig met zoeken..."):
            data = query_local_embeddings(search_query.strip(), source_name=source_name, top_k=top_k)
        if data is not None:
            if "search_run" not in st.session_state:
                st.session_state["search_run"] = 0
            st.session_state["search_run"] += 1
            st.session_state["search_results"] = data.get("results") or []
            st.session_state["search_source_name"] = source_name
            key_prefix = f"run_{st.session_state['search_run']}_"
            st.session_state["search_key_prefix"] = key_prefix
            # Initialize all row checkboxes to True (avoids conflict with session state later)
            for i, _ in enumerate(_deduplicated_rows(data.get("results") or [])):
                st.session_state[f"{key_prefix}result_cb_{i}"] = True
    else:
        st.warning("Voer een zoekvraag in.")

# Always show table from session state so it persists when checkboxes are toggled
if st.session_state.get("search_results"):
    st.subheader("Resultaten")
    key_prefix = st.session_state.get("search_key_prefix", "run_0_")
    rows = _deduplicated_rows(st.session_state["search_results"])
    commissies = sorted(
        {
            (r.get("metadata") or {}).get("commissie")
            for r in rows
            if (r.get("metadata") or {}).get("commissie")
        }
    )
    selected_commissie = st.selectbox(
        "Filter op commissie",
        ["Alle"] + commissies,
        key=f"{key_prefix}commissie_filter",
    )
    if selected_commissie == "Alle":
        filtered_indexed_rows = list(enumerate(rows))
    else:
        filtered_indexed_rows = [
            (i, r)
            for i, r in enumerate(rows)
            if ((r.get("metadata") or {}).get("commissie") == selected_commissie)
        ]
    n_rows = len(filtered_indexed_rows)
    # Select all / Deselect all buttons
    sel_col, desel_col, _ = st.columns([1, 1, 4])
    with sel_col:
        if st.button("Select all", key=f"{key_prefix}select_all"):
            for original_idx, _ in filtered_indexed_rows:
                st.session_state[f"{key_prefix}result_cb_{original_idx}"] = True
            st.rerun()
    with desel_col:
        if st.button("Deselect all", key=f"{key_prefix}deselect_all"):
            for original_idx, _ in filtered_indexed_rows:
                st.session_state[f"{key_prefix}result_cb_{original_idx}"] = False
            st.rerun()
    build_results_table(
        filtered_indexed_rows,
        key_prefix=key_prefix,
    )


# --- Chatbot: uses selected table rows as context ---
if "chat_messages" not in st.session_state:
    st.session_state["chat_messages"] = []

# Determine if chat input should be enabled: need a search and at least one doc selected
has_search = bool(st.session_state.get("search_results"))
selected_chunks: list[dict] = []
if has_search:
    key_prefix = st.session_state.get("search_key_prefix", "run_0_")
    rows = _deduplicated_rows(st.session_state["search_results"])
    n_rows = len(rows)
    selected_indices = [i for i in range(n_rows) if st.session_state.get(f"{key_prefix}result_cb_{i}", False)]
    selected_chunks = [rows[i] for i in selected_indices]
chat_enabled = has_search and len(selected_chunks) > 0

st.subheader("Chat")
for msg in st.session_state["chat_messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if not chat_enabled:
    if not has_search:
        st.caption("Voer eerst een zoekopdracht uit om te chatten.")
    else:
        st.caption("Selecteer minimaal één document in de tabel om te chatten.")

prompt = st.chat_input("Stel je vraag", disabled=not chat_enabled)
if prompt and chat_enabled:
    st.session_state["chat_messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    context = "\n\n---\n\n".join(c.get("content", "") for c in selected_chunks)
    system_content = (
        "Je bent een assistent die vragen beantwoordt op basis van de gegeven Kamerbrieven. "
        "Antwoord alleen op basis van de onderstaande context. Verwijs waar mogelijk naar de bronnen.\n\n"
        "Context (gekozen documenten):\n\n"
        f"{context}"
    )
    try:
        openai_api_key = st.secrets.get("OPENAI_API_KEY", "")
        if not openai_api_key:
            answer = "Configureer `OPENAI_API_KEY` in Streamlit secrets om de chatbot te gebruiken."
            with st.chat_message("assistant"):
                st.markdown(answer)
        else:
            from openai import OpenAI
            client = OpenAI(api_key=openai_api_key)
            stream = client.chat.completions.create(
                model=st.session_state.get("openai_model", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": system_content},
                    *[{"role": m["role"], "content": m["content"]} for m in st.session_state["chat_messages"]],
                ],
                stream=True,
            )
            with st.chat_message("assistant"):
                answer = st.write_stream(
                    (chunk.choices[0].delta.content or "" for chunk in stream if chunk.choices)
                )
        st.session_state["chat_messages"].append({"role": "assistant", "content": answer})
    except Exception as e:
        st.session_state["chat_messages"].pop()  # remove the user message so they can retry
        with st.chat_message("assistant"):
            st.error(f"Fout: {e}")
        st.rerun()