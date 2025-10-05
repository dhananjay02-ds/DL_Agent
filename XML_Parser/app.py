"""
app.py — Streamlit UI for Semantic Lineage (direct tool mode)
--------------------------------------------------------------
Upload XML ZIP → build graph → visualize → query → inspect evidence.
Now compatible with all NetworkX versions (2.x–3.4+)
"""

import os
import streamlit as st
import tempfile
import networkx as nx
from lineage_builder import ingest_zip_or_dir
from lineage_agent import graph_query, get_evidence
from pyvis.network import Network


st.set_page_config(page_title="Informatica Lineage Tool", layout="wide")
st.title("Informatica Lineage Tool")

GRAPH_PATH = "outputs/lineage_graph.gpickle"
os.makedirs("outputs", exist_ok=True)


# ---------------------------------------------------------------------
# Helper: Safe graph loading across NetworkX versions
# ---------------------------------------------------------------------
def load_graph_safe(path: str):
    """Load graph with compatibility for all NetworkX versions."""
    try:
        if hasattr(nx, "read_gpickle"):
            return nx.read_gpickle(path)
        else:
            from networkx.readwrite import gpickle
            return gpickle.read_gpickle(path)
    except Exception as e:
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)


# ---------------------------------------------------------------------
# Helper: Render NetworkX graph using PyVis
# ---------------------------------------------------------------------
def render_graph(G, height="700px", limit_nodes=250):
    """Render an interactive PyVis graph inside Streamlit."""
    if len(G.nodes) > limit_nodes:
        st.warning(
            f"Graph has {len(G.nodes):,} nodes — showing only first {limit_nodes:,} for performance."
        )
        H = G.subgraph(list(G.nodes)[:limit_nodes])
    else:
        H = G

    net = Network(height=height, bgcolor="#ffffff", directed=True)
    net.toggle_physics(True)

    for node, data in H.nodes(data=True):
        color = "#007acc" if data.get("type") == "dataset" else "#f39c12"
        title = f"<b>{node}</b><br>Type: {data.get('type')}<br>File: {data.get('file', '-')}"
        net.add_node(node, label=node.split(".")[-1], title=title, color=color)

    for u, v, data in H.edges(data=True):
        etype = data.get("type", "relation")
        net.add_edge(u, v, label=etype, color="#888888")

    tmp_html = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
    net.save_graph(tmp_html.name)
    with open(tmp_html.name, "r", encoding="utf-8") as f:
        html = f.read()
    st.components.v1.html(html, height=750, scrolling=True)


# ---------------------------------------------------------------------
# Upload or reuse graph
# ---------------------------------------------------------------------
st.header("📦 Upload XML ZIP / Folder")

uploaded = st.file_uploader("Upload a ZIP containing XML files", type=["zip"])
if uploaded:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
        tmp.write(uploaded.read())
        zip_path = tmp.name
    st.info("Building knowledge graph… please wait.")
    gpath, n_nodes, n_edges = ingest_zip_or_dir(zip_path, GRAPH_PATH)
    st.success(f"✅ Graph built with {n_nodes:,} nodes and {n_edges:,} edges.")
    st.session_state["graph_path"] = gpath
elif os.path.exists(GRAPH_PATH):
    st.info("Using existing lineage graph.")
    st.session_state["graph_path"] = GRAPH_PATH
else:
    st.warning("Please upload a ZIP file to build the graph.")


# ---------------------------------------------------------------------
# Graph Visualization Section
# ---------------------------------------------------------------------
if "graph_path" in st.session_state:
    st.divider()
    st.header("🌐 Knowledge Graph Visualization")

    if st.button("Show Graph"):
        try:
            G = load_graph_safe(st.session_state["graph_path"])
            render_graph(G)
        except Exception as e:
            st.error(f"❌ Could not render graph: {e}")


# ---------------------------------------------------------------------
# Query interface
# ---------------------------------------------------------------------
if "graph_path" in st.session_state:
    st.divider()
    st.header("💬 Ask a Question")

    st.markdown(
        """
    Example queries:
    - "Where did `projects.name` come from?"
    - "Which datasets contain a column named `customer`?"
    - "What will be impacted if I change `project.id`?"
    - "List all datasets linked by `customerid`"
    """
    )

    q = st.text_input("Enter query:")
    if st.button("Run Query") and q:
        with st.spinner("Running query..."):
            try:
                res = graph_query(q, st.session_state["graph_path"])
                st.json(res)
            except Exception as e:
                st.error(f"❌ Query failed: {e}")


# ---------------------------------------------------------------------
# Node evidence section
# ---------------------------------------------------------------------
    st.divider()
    st.header("🔍 Inspect Node Evidence")
    node = st.text_input("Enter full node name (e.g., project_import::project.id):")
    if st.button("Get Evidence") and node:
        try:
            G = load_graph_safe(st.session_state["graph_path"])
            evidence = get_evidence(G, node)
            if not evidence:
                st.info("No evidence found for this node.")
            else:
                for ev in evidence:
                    st.markdown(f"**File:** {ev.get('file','-')}")
                    st.code(ev.get("snippet", ""), language="xml")
        except Exception as e:
            st.error(f"❌ Evidence lookup failed: {e}")
