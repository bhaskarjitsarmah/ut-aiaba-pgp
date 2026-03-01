"""
Vector Database Internals Explorer
------------------------------------
A Streamlit app that visually explains how ChromaDB works:
  - Text Chunks  →  Embeddings  →  SQLite + HNSW Index  →  Similarity Search

Install dependencies:
    pip install streamlit chromadb langchain_openai scikit-learn numpy pandas plotly

Run:
    streamlit run vector_db_explorer.py
"""

import json
import os
import sqlite3

import chromadb
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from langchain_openai import OpenAIEmbeddings
from sklearn.decomposition import PCA

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Vector DB Internals Explorer",
    page_icon="🗄️",
    layout="wide",
)

# ─── Constants ────────────────────────────────────────────────────────────────
DB_PATH = "./demo_medical_db"

# ─── Load OpenAI credentials from notebook's config.json ──────────────────────
_config_path = os.path.join(os.path.dirname(__file__), "week 1", "config.json")
with open(_config_path) as _f:
    _config = json.load(_f)
OPENAI_API_KEY  = _config["OPENAI_API_KEY"]
OPENAI_API_BASE = _config["OPENAI_API_BASE"]

CHUNKS = [
    {"id": "c01", "text": "Diabetes mellitus is characterized by high blood sugar levels over a prolonged period of time.", "topic": "Diabetes", "page": 1},
    {"id": "c02", "text": "Symptoms of diabetes include frequent urination, increased thirst, and increased hunger.", "topic": "Diabetes", "page": 2},
    {"id": "c03", "text": "Type 1 diabetes results from the immune system attacking insulin-producing beta cells in the pancreas.", "topic": "Diabetes", "page": 3},
    {"id": "c04", "text": "Type 2 diabetes is characterized by insulin resistance and relative insulin deficiency in the body.", "topic": "Diabetes", "page": 4},
    {"id": "c05", "text": "Hypertension is a long-term medical condition in which blood pressure is persistently elevated.", "topic": "Hypertension", "page": 1},
    {"id": "c06", "text": "Blood pressure is measured using two numbers: systolic pressure over diastolic pressure in mmHg.", "topic": "Hypertension", "page": 2},
    {"id": "c07", "text": "Treatment for hypertension includes lifestyle changes and medications such as ACE inhibitors.", "topic": "Hypertension", "page": 3},
    {"id": "c08", "text": "Pulmonary embolism is a blockage of the main artery of the lung caused by a blood clot.", "topic": "Pulmonary Embolism", "page": 1},
    {"id": "c09", "text": "Symptoms of pulmonary embolism include shortness of breath, chest pain, and hemoptysis.", "topic": "Pulmonary Embolism", "page": 2},
    {"id": "c10", "text": "Anticoagulants such as heparin and warfarin are the primary treatment for pulmonary embolism.", "topic": "Pulmonary Embolism", "page": 3},
]

TOPIC_COLORS = {
    "Diabetes": "#4A90D9",
    "Hypertension": "#27AE60",
    "Pulmonary Embolism": "#E74C3C",
}


# ─── Cached Resources ─────────────────────────────────────────────────────────
@st.cache_resource
def build_embedder():
    """OpenAI embedding model — same as used in the notebook."""
    model = OpenAIEmbeddings(
        openai_api_key=OPENAI_API_KEY,
        openai_api_base=OPENAI_API_BASE,
    )
    texts = [c["text"] for c in CHUNKS]
    vectors = model.embed_documents(texts)       # list of 1536-dim vectors
    return model, np.array(vectors)


@st.cache_resource
def build_chromadb(_embeddings):
    """Create a persistent ChromaDB and populate it with chunks + embeddings."""
    client = chromadb.PersistentClient(path=DB_PATH)
    try:
        client.delete_collection("medical")
    except Exception:
        pass
    collection = client.create_collection("medical")
    collection.add(
        ids=[c["id"] for c in CHUNKS],
        documents=[c["text"] for c in CHUNKS],
        embeddings=_embeddings.tolist(),
        metadatas=[{"topic": c["topic"], "page": c["page"]} for c in CHUNKS],
    )
    return client, collection


@st.cache_data
def compute_2d_coords(_embeddings):
    """Reduce embeddings to 2D using PCA for visualization."""
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(_embeddings)
    return coords, pca


# ─── Helpers ──────────────────────────────────────────────────────────────────
def embed_query(model, text):
    """Embed a single query string using OpenAI — returns a (1, 1536) numpy array."""
    return np.array([model.embed_query(text)])


def get_sqlite_tables():
    db_file = os.path.join(DB_PATH, "chroma.sqlite3")
    if not os.path.exists(db_file):
        return []
    conn = sqlite3.connect(db_file)
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [row[0] for row in cur.fetchall()]
    conn.close()
    return tables


def read_sqlite_table(table, limit=20):
    db_file = os.path.join(DB_PATH, "chroma.sqlite3")
    conn = sqlite3.connect(db_file)
    try:
        df = pd.read_sql(f"SELECT * FROM [{table}] LIMIT {limit}", conn)
    except Exception as e:
        df = pd.DataFrame({"error": [str(e)]})
    conn.close()
    return df


def get_db_files():
    rows = []
    purpose_map = {
        "chroma.sqlite3": "Main DB — text, metadata, chunk IDs",
        "data_level0.bin": "HNSW vectors (raw float arrays)",
        "header.bin": "HNSW index header / config",
        "length.bin": "Vector magnitudes",
        "link_lists.bin": "HNSW graph neighbour links",
    }
    for root, _, files in os.walk(DB_PATH):
        for f in files:
            fpath = os.path.join(root, f)
            size = os.path.getsize(fpath)
            rows.append({
                "File": os.path.relpath(fpath, DB_PATH),
                "Size": f"{size:,} bytes",
                "Layer": "SQLite" if f.endswith(".sqlite3") else "HNSW Index",
                "Purpose": purpose_map.get(f, "HNSW index data"),
            })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ─── Initialise ───────────────────────────────────────────────────────────────
embedding_model, embeddings = build_embedder()
chroma_client, collection = build_chromadb(embeddings)
coords_2d, pca_model = compute_2d_coords(embeddings)


# ═══════════════════════════════════════════════════════════════════════════════
# TITLE
# ═══════════════════════════════════════════════════════════════════════════════
st.title("🗄️ Vector Database Internals Explorer")
st.caption(
    "Understand what's really happening inside ChromaDB — "
    "from raw text to lightning-fast similarity search."
)
st.info(
    "Uses **10 sample medical chunks** (simulating the 4,000-page Merck Manual) "
    "and **OpenAI `text-embedding-ada-002`** — the same embedding model as the notebook (1,536 dimensions).",
    icon="ℹ️",
)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Pipeline Overview",
    "📄 Text Chunks",
    "🔢 Embeddings",
    "🗄️ ChromaDB Internals",
    "🔍 Live Similarity Search",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Pipeline Overview
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("The Full RAG Pipeline")

    # ── Flowchart ──────────────────────────────────────────────────────────────
    fig = go.Figure()

    box_nodes = [
        dict(x=0.05, y=0.6, label="📄 PDF\n4,000+ pages",        color="#4A90D9"),
        dict(x=0.22, y=0.6, label="✂️ Text Chunks\n256 tokens",   color="#7B68EE"),
        dict(x=0.39, y=0.6, label="🔢 Embedding\nModel",           color="#9B59B6"),
        dict(x=0.56, y=0.6, label="🗄️ ChromaDB\n(on disk)",        color="#27AE60"),
        dict(x=0.73, y=0.6, label="🔍 Top-K\nChunks",              color="#E67E22"),
        dict(x=0.90, y=0.6, label="💬 LLM\nAnswer",                color="#16A085"),
        dict(x=0.56, y=0.15, label="❓ User\nQuery",               color="#E74C3C"),
    ]

    for n in box_nodes:
        fig.add_shape(
            type="rect",
            x0=n["x"] - 0.07, y0=n["y"] - 0.13,
            x1=n["x"] + 0.07, y1=n["y"] + 0.13,
            fillcolor=n["color"], line_color="white", line_width=2, opacity=0.9,
        )
        fig.add_annotation(
            x=n["x"], y=n["y"], text=n["label"],
            showarrow=False, font=dict(color="white", size=12), align="center",
        )

    # horizontal arrows (left → right)
    for i in range(5):
        sx, ex = box_nodes[i]["x"] + 0.07, box_nodes[i + 1]["x"] - 0.07
        fig.add_annotation(
            x=ex, y=0.6, ax=sx, ay=0.6,
            xref="x", yref="y", axref="x", ayref="y",
            arrowhead=3, arrowsize=1.2, arrowcolor="#666", showarrow=True, text="",
        )

    # query arrow (up into ChromaDB)
    fig.add_annotation(
        x=0.56, y=0.6 - 0.13, ax=0.56, ay=0.15 + 0.13,
        xref="x", yref="y", axref="x", ayref="y",
        arrowhead=3, arrowsize=1.2, arrowcolor="#E74C3C", showarrow=True, text="",
    )

    fig.update_layout(
        xaxis=dict(visible=False, range=[0, 1]),
        yaxis=dict(visible=False, range=[0, 0.85]),
        height=260, plot_bgcolor="white",
        margin=dict(l=5, r=5, t=10, b=5),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Two-column breakdown ──────────────────────────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### ChromaDB on disk = two layers")
        st.code(
            "demo_medical_db/\n"
            "├── chroma.sqlite3        ← SQLite layer\n"
            "│   ├── text chunks\n"
            "│   ├── metadata (source, page)\n"
            "│   └── chunk IDs\n"
            "└── <uuid>/\n"
            "    ├── data_level0.bin   ← HNSW layer\n"
            "    ├── header.bin\n"
            "    ├── length.bin\n"
            "    └── link_lists.bin",
            language="text",
        )
    with c2:
        st.markdown("#### Search flow at query time")
        st.code(
            "User question\n"
            "     │\n"
            "     ▼\n"
            "Embedding model  →  query vector\n"
            "     │\n"
            "     ▼\n"
            "HNSW index  →  top-K chunk IDs\n"
            "     │\n"
            "     ▼\n"
            "SQLite  →  fetch text by IDs\n"
            "     │\n"
            "     ▼\n"
            "LLM  →  final answer",
            language="text",
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Text Chunks
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Text Chunks Stored in ChromaDB")
    st.caption(
        "Each row is one document stored in the vector DB. "
        "In the notebook these come from splitting the 4,000-page PDF into 256-token chunks."
    )

    df_chunks = pd.DataFrame([{
        "ID": c["id"],
        "Topic": c["topic"],
        "Page": c["page"],
        "Text Chunk": c["text"],
    } for c in CHUNKS])

    st.dataframe(
        df_chunks, use_container_width=True, height=400,
        column_config={
            "ID":         st.column_config.TextColumn(width="small"),
            "Topic":      st.column_config.TextColumn(width="medium"),
            "Page":       st.column_config.NumberColumn(width="small"),
            "Text Chunk": st.column_config.TextColumn(width="large"),
        },
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total chunks", len(CHUNKS))
    m2.metric("Diabetes",          sum(1 for c in CHUNKS if c["topic"] == "Diabetes"))
    m3.metric("Hypertension",      sum(1 for c in CHUNKS if c["topic"] == "Hypertension"))
    m4.metric("Pulmonary Embolism",sum(1 for c in CHUNKS if c["topic"] == "Pulmonary Embolism"))


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Embeddings
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("What an Embedding Looks Like")
    st.markdown(
        "An **embedding** converts text into a list of numbers. "
        "Similar texts produce **similar vectors** — that's the foundation of semantic search. "
        "This app uses **OpenAI `text-embedding-ada-002`** — the same model as the notebook — "
        "producing **1,536-dimensional** vectors."
    )

    col_left, col_right = st.columns([1, 2])

    with col_left:
        idx = st.selectbox(
            "Select a chunk to inspect:",
            options=range(len(CHUNKS)),
            format_func=lambda i: f"{CHUNKS[i]['id']} — {CHUNKS[i]['text'][:45]}…",
        )
        st.markdown(f"**Topic:** `{CHUNKS[idx]['topic']}`")
        st.markdown(f"**Dimensions:** `{len(embeddings[idx])}`")
        st.markdown(f"**First 5 values:** `{np.round(embeddings[idx][:5], 4).tolist()}`")

    with col_right:
        fig = px.bar(
            x=list(range(len(embeddings[idx]))),
            y=embeddings[idx],
            labels={"x": "Dimension index", "y": "Value"},
            title=f"Embedding vector — {CHUNKS[idx]['id']}",
            color=embeddings[idx],
            color_continuous_scale="RdBu",
        )
        fig.update_layout(height=280, showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── 2-D scatter (PCA) ─────────────────────────────────────────────────────
    st.subheader("All Chunks in 2D Vector Space (PCA)")
    st.caption(
        "PCA reduces 1,536 dimensions → 2 so we can see them. "
        "Chunks about the same topic cluster together — that's why similarity search works."
    )

    df_2d = pd.DataFrame({
        "x":     coords_2d[:, 0],
        "y":     coords_2d[:, 1],
        "topic": [c["topic"] for c in CHUNKS],
        "id":    [c["id"] for c in CHUNKS],
        "text":  [c["text"][:60] + "…" for c in CHUNKS],
    })

    fig2 = px.scatter(
        df_2d, x="x", y="y", color="topic", text="id",
        hover_data={"text": True, "x": False, "y": False},
        title="Medical Chunks in 2D Vector Space",
        color_discrete_map=TOPIC_COLORS,
        height=430,
    )
    fig2.update_traces(marker=dict(size=14), textposition="top center")
    fig2.update_layout(xaxis_title="PCA Dim 1", yaxis_title="PCA Dim 2")
    st.plotly_chart(fig2, use_container_width=True)

    st.success(
        "📌 Chunks from the same medical topic cluster together. "
        "When you ask 'What are symptoms of high blood sugar?', "
        "your query vector lands near the Diabetes cluster — and those chunks get returned."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — ChromaDB Internals
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("What's Inside ChromaDB on Disk")

    col_sql, col_hnsw = st.columns(2)

    # ── SQLite side ───────────────────────────────────────────────────────────
    with col_sql:
        st.markdown("### 🗂️ SQLite Layer — `chroma.sqlite3`")
        st.caption("Stores text chunks, metadata, and IDs — the human-readable side.")

        tables = get_sqlite_tables()
        if tables:
            selected_table = st.selectbox("Browse a table:", tables)
            df_sql = read_sqlite_table(selected_table)
            if not df_sql.empty:
                st.dataframe(df_sql, use_container_width=True, height=320)
                st.caption(f"`{selected_table}` — up to 20 rows shown")
            else:
                st.info("Table is empty.")
        else:
            st.warning("SQLite file not found.")

    # ── HNSW side ─────────────────────────────────────────────────────────────
    with col_hnsw:
        st.markdown("### 📦 HNSW Index Layer — `*.bin` files")
        st.caption("Stores vectors and the graph structure for fast nearest-neighbour search.")

        df_files = get_db_files()
        if not df_files.empty:
            st.dataframe(
                df_files, use_container_width=True, height=200,
                column_config={
                    "Layer": st.column_config.TextColumn(width="small"),
                },
            )
        else:
            st.info("No files found.")

        st.markdown("""
**HNSW = Hierarchical Navigable Small World**
- Each embedding vector is a **node** in a multi-layer graph
- Nodes connect to their **nearest neighbours**
- Search starts at the top layer (coarse), drills down to find the closest match
- Result: finds the nearest vectors among thousands in **milliseconds**
        """)

    st.divider()

    # ── How they cooperate ────────────────────────────────────────────────────
    st.subheader("How SQLite and HNSW Work Together")
    st.markdown("""
| Step | Layer | What happens |
|------|-------|--------------|
| 1 | **HNSW Index** | Receives the query vector; traverses the graph to find the top-K closest vector IDs |
| 2 | **SQLite** | Receives those IDs; returns the original text + metadata |
| 3 | **Application** | Wraps results into LangChain `Document` objects and sends them to the LLM |
    """)

    # ── Visual: SQLite table linked to HNSW node ──────────────────────────────
    fig_link = go.Figure()

    # SQLite "table" box
    fig_link.add_shape(type="rect", x0=0.05, y0=0.25, x1=0.42, y1=0.75,
                       fillcolor="#4A90D9", opacity=0.15, line_color="#4A90D9", line_width=2)
    fig_link.add_annotation(x=0.235, y=0.82, text="SQLite (chroma.sqlite3)",
                             showarrow=False, font=dict(size=13, color="#4A90D9"))

    for i, row in enumerate([("c01", "Diabetes p.1", "0.82, 0.11, ..."),
                               ("c02", "Diabetes p.2", "0.67, 0.34, ..."),
                               ("c03", "Diabetes p.3", "0.55, 0.28, ...")]):
        y = 0.65 - i * 0.18
        fig_link.add_annotation(x=0.235, y=y,
                                 text=f"ID: {row[0]} | {row[1]} | emb: [{row[2]}]",
                                 showarrow=False, font=dict(size=11), bgcolor="white",
                                 bordercolor="#4A90D9", borderwidth=1)

    # Arrow
    fig_link.add_annotation(x=0.62, y=0.5, ax=0.42, ay=0.5,
                             xref="x", yref="y", axref="x", ayref="y",
                             arrowhead=3, arrowsize=1.5, arrowcolor="#555",
                             showarrow=True, text="IDs link\nboth layers",
                             font=dict(size=11), align="center")

    # HNSW "graph" box
    fig_link.add_shape(type="rect", x0=0.62, y0=0.25, x1=0.95, y1=0.75,
                       fillcolor="#27AE60", opacity=0.15, line_color="#27AE60", line_width=2)
    fig_link.add_annotation(x=0.785, y=0.82, text="HNSW Index (*.bin)",
                             showarrow=False, font=dict(size=13, color="#27AE60"))

    node_positions = [(0.70, 0.62), (0.85, 0.62), (0.77, 0.45), (0.68, 0.32), (0.88, 0.35)]
    for (nx, ny) in node_positions:
        fig_link.add_shape(type="circle", x0=nx - 0.03, y0=ny - 0.05,
                           x1=nx + 0.03, y1=ny + 0.05,
                           fillcolor="#27AE60", opacity=0.7, line_color="white")
    edges = [(0, 1), (0, 2), (1, 2), (2, 3), (2, 4), (3, 4)]
    for s, e in edges:
        sx, sy = node_positions[s]
        ex, ey = node_positions[e]
        fig_link.add_shape(type="line", x0=sx, y0=sy, x1=ex, y1=ey,
                           line=dict(color="#27AE60", width=1.5, dash="dot"))

    fig_link.update_layout(
        xaxis=dict(visible=False, range=[0, 1]),
        yaxis=dict(visible=False, range=[0.1, 1.0]),
        height=280, plot_bgcolor="white",
        margin=dict(l=5, r=5, t=10, b=5),
    )
    st.plotly_chart(fig_link, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Live Similarity Search
# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("Live Similarity Search")
    st.caption("Type a medical question and watch ChromaDB find the most relevant chunks in real time.")

    query_text = st.text_input(
        "Your question:",
        value="What are the symptoms of diabetes?",
        placeholder="e.g. How is blood pressure measured?",
    )
    k = st.slider("Number of results to retrieve (k)", min_value=1, max_value=5, value=3)

    if query_text.strip():
        query_vec = embed_query(embedding_model, query_text)
        query_2d  = pca_model.transform(query_vec)[0]

        results = collection.query(
            query_embeddings=query_vec.tolist(),
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        col_res, col_vis = st.columns([1, 1])

        # ── Results ───────────────────────────────────────────────────────────
        with col_res:
            st.markdown("### Results from ChromaDB")
            for i, (doc, meta, dist) in enumerate(zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )):
                similarity = max(0.0, 1.0 - dist)
                with st.container(border=True):
                    st.markdown(f"**#{i+1}** — `{meta['topic']}` · Page {meta['page']}")
                    st.markdown(f"> {doc}")
                    st.progress(float(similarity), text=f"Similarity score: {similarity:.3f}")

        # ── Vector space plot ─────────────────────────────────────────────────
        with col_vis:
            st.markdown("### Vector Space")
            result_ids = set(results["ids"][0])

            fig_search = go.Figure()

            # Non-result chunks
            for i, c in enumerate(CHUNKS):
                if c["id"] not in result_ids:
                    fig_search.add_trace(go.Scatter(
                        x=[coords_2d[i, 0]], y=[coords_2d[i, 1]],
                        mode="markers+text",
                        text=[c["id"]], textposition="top center",
                        marker=dict(size=12, color="lightgray",
                                    line=dict(color="#aaa", width=1)),
                        hovertext=c["text"][:60], showlegend=False,
                    ))

            # Result chunks
            for i, c in enumerate(CHUNKS):
                if c["id"] in result_ids:
                    fig_search.add_trace(go.Scatter(
                        x=[coords_2d[i, 0]], y=[coords_2d[i, 1]],
                        mode="markers+text",
                        text=[c["id"]], textposition="top center",
                        marker=dict(size=18, color=TOPIC_COLORS[c["topic"]],
                                    symbol="star",
                                    line=dict(color="white", width=2)),
                        hovertext=c["text"][:60],
                        name=f"{c['id']} (matched)",
                    ))
                    # Dotted line from query to result
                    fig_search.add_shape(
                        type="line",
                        x0=query_2d[0], y0=query_2d[1],
                        x1=coords_2d[i, 0], y1=coords_2d[i, 1],
                        line=dict(color="#E74C3C", width=1.5, dash="dot"),
                    )

            # Query point
            fig_search.add_trace(go.Scatter(
                x=[query_2d[0]], y=[query_2d[1]],
                mode="markers+text",
                text=["❓ Query"], textposition="top center",
                marker=dict(size=20, color="#E74C3C", symbol="diamond",
                            line=dict(color="darkred", width=2)),
                name="Your Query",
            ))

            fig_search.update_layout(
                height=430,
                title="Query vs. Document Vectors (2D PCA)",
                xaxis_title="PCA Dim 1",
                yaxis_title="PCA Dim 2",
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig_search, use_container_width=True)

        # ── Explanation ───────────────────────────────────────────────────────
        st.divider()
        st.markdown("### What just happened internally?")
        st.markdown(f"""
1. Your query *"{query_text}"* was converted to a **{query_vec.shape[1]}-dimensional OpenAI embedding**
   using `text-embedding-ada-002` — the same model as the notebook
2. The **HNSW index** traversed its graph to find the **{k} closest vectors** in milliseconds
3. **SQLite** fetched the text + metadata for those vector IDs
4. ⭐ = matched chunks (nearest neighbours in vector space)
5. In the real notebook, these chunks are passed to **GPT-4o-mini** as context to generate a final answer
        """)
