"""
app.py
------
Streamlit frontend for building, enriching, visualizing, and querying lineage graphs.
Includes:
✅ PyVis-based interactive visualization
✅ Side-by-side Base vs Enriched graph comparison
✅ Pre-enrichment snapshot (raw)
✅ Robust fallbacks for all NetworkX read/write variations
✅ Integrated LLM-powered querying via lineage_agent
"""

import os
import json
import pickle
import streamlit as st
import networkx as nx
from lineage_builder import ingest_zip_or_dir
from ai_enricher import enrich_graph
from lineage_agent import graph_query
from pyvis.network import Network
import streamlit.components.v1 as components

# ---------------------------------------------------------------------
# Paths & Config
# ---------------------------------------------------------------------
os.makedirs("outputs", exist_ok=True)
GRAPH_PATH = "outputs/lineage_graph.gpickle"
RAW_ENRICHED_PATH = "outputs/lineage_graph.enriched.raw.gpickle"
ENRICHED_GRAPH_PATH = "outputs/lineage_graph.enriched.gpickle"

st.set_page_config(page_title="Informatica Lineage Tool", layout="wide")
st.title("Informatica Lineage Tool")

# ---------------------------------------------------------------------
# Utility: Safe Graph I/O
# ---------------------------------------------------------------------
def safe_read_gpickle(path: str):
    """Cross-version safe graph load."""
    try:
        if hasattr(nx, "read_gpickle"):
            return nx.read_gpickle(path)
        else:
            import pickle
            with open(path, "rb") as f:
                return pickle.load(f)
    except Exception as e:
        print(f"[WARN] read_gpickle failed: {e}, using pickle fallback")
        with open(path, "rb") as f:
            return pickle.load(f)


def safe_write_gpickle(G, path: str):
    """Cross-version safe graph save."""
    try:
        if hasattr(nx, "write_gpickle"):
            nx.write_gpickle(G, path)
        else:
            import pickle
            with open(path, "wb") as f:
                pickle.dump(G, f)
    except Exception as e:
        print(f"[WARN] write_gpickle failed: {e}, using pickle fallback")
        with open(path, "wb") as f:
            pickle.dump(G, f)


# ---------------------------------------------------------------------
# Step 1: Upload ZIP → Build Graph
# ---------------------------------------------------------------------
uploaded_file = st.file_uploader("📦 Upload ZIP of XML datasets", type=["zip"])

if uploaded_file:
    os.makedirs("uploads", exist_ok=True)
    zip_path = os.path.join("uploads", uploaded_file.name)
    with open(zip_path, "wb") as f:
        f.write(uploaded_file.read())

    st.info("⏳ Building base knowledge graph...")
    try:
        graph_path, nodes, edges = ingest_zip_or_dir(zip_path, GRAPH_PATH)
        st.success(f"✅ Base graph built: {nodes:,} nodes, {edges:,} edges")
    except Exception as e:
        st.error(f"❌ Graph build failed: {e}")
        st.stop()

    # -----------------------------------------------------------------
    # Step 2: Save raw version before enrichment
    # -----------------------------------------------------------------
    try:
        G = safe_read_gpickle(graph_path)
        safe_write_gpickle(G, RAW_ENRICHED_PATH)
        st.info(f"🗂️ Raw graph snapshot saved → {RAW_ENRICHED_PATH}")
    except Exception as e:
        st.warning(f"[WARN] Could not save raw snapshot: {e}")
        G = None

    # -----------------------------------------------------------------
    # Step 3: Enrich Graph
    # -----------------------------------------------------------------
    if G is not None:
        try:
            G_enriched = enrich_graph(G)
            safe_write_gpickle(G_enriched, ENRICHED_GRAPH_PATH)
            st.success(
                f"✨ Enriched graph saved: {len(G_enriched.nodes)} nodes, {len(G_enriched.edges)} edges"
            )
        except Exception as e:
            st.error(f"❌ Graph enrichment failed: {e}")
            st.stop()
    else:
        st.error("❌ No base graph available for enrichment.")
        st.stop()

    # -----------------------------------------------------------------
    # Step 4: Interactive Visualization
    # -----------------------------------------------------------------
    st.subheader("🌐 Graph Visualization — Base vs Enriched")

    def render_pyvis_graph(G: nx.DiGraph, title: str):
        """Render interactive PyVis graph with color-coding."""
        net = Network(
            height="650px", width="100%", directed=True, bgcolor="#f8f9fa", font_color="black"
        )

        color_map = {
            "dataset": "#8ecae6",
            "column": "#ffb703",
            "table": "#219ebc",
            "report": "#b5179e",
            "unknown": "#adb5bd",
        }

        for node, data in G.nodes(data=True):
            node_type = data.get("node_type", data.get("type", "unknown"))
            label = data.get("label", node)
            color = color_map.get(node_type, "#cccccc")
            net.add_node(node, label=label, color=color, title=f"{node_type}")

        for u, v, data in G.edges(data=True):
            etype = data.get("type", "")
            net.add_edge(u, v, title=etype, color="#999999")

        net.repulsion(node_distance=150, spring_length=180)
        html_path = f"outputs/{title.replace(' ', '_')}.html"
        net.save_graph(html_path)
        with open(html_path, "r", encoding="utf-8") as f:
            components.html(f.read(), height=650, scrolling=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🧩 Base Graph")
        render_pyvis_graph(G, "Base_Graph")

    with col2:
        st.markdown("#### ✨ Enriched Graph")
        render_pyvis_graph(G_enriched, "Enriched_Graph")

elif os.path.exists(ENRICHED_GRAPH_PATH):
    st.info("ℹ️ Using previously built enriched graph.")
else:
    st.warning("📥 Please upload a ZIP of XML files to begin.")

# ---------------------------------------------------------------------
# Step 5: Query Interface
# ---------------------------------------------------------------------
st.divider()
st.subheader("💬 Ask a Lineage Question")

st.markdown("""
**Examples:**
- Where did `orders.amount` come from?
- What will be impacted if I change `customer.email`?
- Which datasets contain a column named `company`?
""")

query = st.text_input("Enter your question:")
if st.button("Run Query"):
    if os.path.exists(ENRICHED_GRAPH_PATH):
        try:
            result = graph_query(query, ENRICHED_GRAPH_PATH)
            st.json(json.loads(result))
        except Exception as e:
            st.error(f"❌ Query failed: {e}")
    else:
        st.error("⚠️ Please build and enrich a graph first.")

# ---------------------------------------------------------------------
# Step 6: Comparison Stats
# ---------------------------------------------------------------------
if os.path.exists(GRAPH_PATH) and os.path.exists(ENRICHED_GRAPH_PATH):
    with st.expander("📊 Compare Base vs Enriched Graph Stats"):
        try:
            G_base = safe_read_gpickle(GRAPH_PATH)
            G_enriched = safe_read_gpickle(ENRICHED_GRAPH_PATH)
            diff_nodes = len(G_enriched.nodes) - len(G_base.nodes)
            diff_edges = len(G_enriched.edges) - len(G_base.edges)

            st.write(f"**Base Graph:** {len(G_base.nodes)} nodes, {len(G_base.edges)} edges")
            st.write(f"**Enriched Graph:** {len(G_enriched.nodes)} nodes, {len(G_enriched.edges)} edges")

            if diff_nodes > 0 or diff_edges > 0:
                st.success(f"🧠 Enrichment added +{diff_nodes} nodes and +{diff_edges} edges.")
            else:
                st.info("No new nodes/edges were added by enrichment.")
        except Exception as e:
            st.error(f"Comparison failed: {e}")
