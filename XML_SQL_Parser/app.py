"""
app.py — XML + SQL Lineage Tool (LLM-Enhanced Version)
Behavior identical to the original Informatica Lineage Tool flow:
- Only builds graphs from the uploaded ZIP file
- No previous graph references, no session caching
- Stepwise UX: Upload → Build → Enrich → Visualize → Query
"""

import os
import tempfile
import io
import json
import streamlit as st
import networkx as nx
import pandas as pd

from lineage_builder import ingest_zip_or_dir, safe_read_gpickle, safe_write_gpickle
from ai_enricher import enrich_graph
from lineage_agent import graph_query
from llm_utils import get_llm

# Optional visualization
try:
    from pyvis.network import Network
    from streamlit.components.v1 import html as st_html
    PYVIS_AVAILABLE = True
except Exception:
    PYVIS_AVAILABLE = False

# Streamlit setup
st.set_page_config(layout="wide", page_title="Lineage Builder - XML + SQL")
st.title("Lineage Tool — XML + SQL")

# -------------------------------
# Helpers
# -------------------------------
def _node_label_for_display(n, attrs):
    """Return a human-friendly label for node in PyVis or export."""
    return (attrs.get("label")
            or attrs.get("table_name")
            or attrs.get("cte_name")
            or attrs.get("column_name")
            or str(n))

def _node_kind(attrs):
    """Normalize node 'kind' (use node_kind if present else type)."""
    return attrs.get("node_kind") or attrs.get("type") or "unknown"

def render_pyvis(G: nx.DiGraph, title: str):
    """
    Render a NetworkX graph using PyVis. Adds clear node labels and tooltips.
    Also shows a legend beneath the visualization.
    """
    if not PYVIS_AVAILABLE:
        st.warning("PyVis not installed — showing node counts only.")
        # Show aggregated node counts by kind
        counts = {}
        for _, a in G.nodes(data=True):
            k = _node_kind(a)
            counts[k] = counts.get(k, 0) + 1
        st.json(counts)
        return

    # Color mapping for readability (feel free to tweak)
    color_map = {
        "sql_query": "#f39c12",   # orange
        "cte": "#1abc9c",         # teal
        "table": "#2ecc71",       # green
        "column": "#3498db",      # blue
        "expression": "#9b59b6",  # purple
        "xml": "#9b59b6",         # purple-ish
        "xml_node": "#9b59b6",
        "unknown": "#95a5a6"      # gray
    }

    net = Network(height="520px", width="100%", directed=True, notebook=False)
    net.toggle_physics(True)

    # Add nodes
    for n, a in G.nodes(data=True):
        kind = _node_kind(a)
        color = color_map.get(kind, color_map["unknown"])
        label = _node_label_for_display(n, a)
        # Build a detailed title/tooltip with useful metadata
        title_parts = []
        title_parts.append(f"<b>{label}</b>")
        title_parts.append(f"<small><i>id:</i> {n}</small>")
        title_parts.append(f"<small><i>type:</i> {kind}</small>")
        if a.get("summary"):
            title_parts.append(f"<div style='margin-top:6px'>{str(a.get('summary'))[:700]}</div>")
        elif a.get("raw"):
            title_parts.append(f"<div style='margin-top:6px'>{str(a.get('raw'))[:700]}</div>")
        title_html = "<br/>".join(title_parts)
        try:
            net.add_node(str(n), label=str(label), color=color, title=title_html)
        except Exception:
            # fallback
            net.add_node(str(n), label=str(label), color=color, title=str(label))

    # Add edges
    for u, v, ed in G.edges(data=True):
        net.add_edge(str(u), str(v), label=str(ed.get("type", "")))

    # Save and embed
    tmp_html = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
    net.save_graph(tmp_html.name)
    with open(tmp_html.name, "r", encoding="utf-8") as f:
        html_data = f.read()
    st_html(html_data, height=520, scrolling=True)

    # Legend
    st.markdown("##### 🎨 Legend")
    legend_items = [
        ("sql_query", "SQL Query / Script", color_map.get("sql_query")),
        ("cte", "CTE (WITH ... AS)", color_map.get("cte")),
        ("table", "Table", color_map.get("table")),
        ("column", "Column", color_map.get("column")),
        ("expression", "Expression / Derived Column", color_map.get("expression")),
        ("xml_node", "XML element / artifact", color_map.get("xml_node")),
        ("unknown", "Other / Unknown", color_map.get("unknown")),
    ]
    cols = st.columns(3)
    for i, (key, label, color) in enumerate(legend_items):
        col = cols[i % 3]
        col.markdown(
            f"<div style='display:flex;align-items:center;margin:4px 0'>"
            f"<div style='width:14px;height:14px;background:{color};border-radius:3px;margin-right:8px;'></div>"
            f"<div style='font-size:13px'>{label} <span style='color:#666;font-size:11px'>({key})</span></div>"
            f"</div>", unsafe_allow_html=True
        )

def export_graph_to_excel_bytes(G: nx.DiGraph) -> bytes:
    """
    Export nodes and edges into an Excel workbook in memory and return bytes.
    Sheet1: Nodes
    Sheet2: Edges
    """
    node_rows = []
    for n, a in G.nodes(data=True):
        node_rows.append({
            "Node ID": n,
            "Label": _node_label_for_display(n, a),
            "Type": _node_kind(a),
            "Summary": a.get("summary", ""),
            "Title": a.get("title", ""),
            "Source path": a.get("source_path", ""),
            "Raw (truncated)": str(a.get("raw", ""))[:500],
        })

    edge_rows = []
    for u, v, ad in G.edges(data=True):
        # any extra edge attributes
        attrs = {k: v for k, v in ad.items() if k not in ("type",)}
        edge_rows.append({
            "Source": u,
            "Target": v,
            "Edge Type": ad.get("type", ""),
            "Edge Attrs": json.dumps(attrs, ensure_ascii=False)
        })

    buf = io.BytesIO()
    # Use pandas ExcelWriter to write multiple sheets
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        pd.DataFrame(node_rows).to_excel(writer, sheet_name="Nodes", index=False)
        pd.DataFrame(edge_rows).to_excel(writer, sheet_name="Edges", index=False)
    buf.seek(0)
    return buf.read()

# -------------------------------
# Upload / Build / Enrich flow
# -------------------------------
st.markdown("### 📂 Upload Your Dataset")
uploaded_file = st.file_uploader("Upload a ZIP file containing XML and/or SQL files", type=["zip"])

if uploaded_file is not None:
    st.info("Processing upload...")

    # Save uploaded zip to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    os.makedirs("graphs", exist_ok=True)
    base_graph_path = os.path.join("graphs", os.path.basename(tmp_path) + ".gpickle")
    enriched_graph_path = base_graph_path.replace(".gpickle", ".enriched.gpickle")

    # Step 1: Build Base Graph
    st.subheader("⚙️ Building Base Graph")
    try:
        gpath, n_nodes, n_edges = ingest_zip_or_dir(tmp_path, base_graph_path)
        st.success(f"✅ Base graph built successfully — {n_nodes} nodes, {n_edges} edges")
    except Exception as e:
        st.error(f"❌ Error building graph: {e}")
        st.stop()

    # Step 2: Enrich Graph
    st.subheader("✨ Enriching Graph")
    try:
        G_base = safe_read_gpickle(gpath)
        # call enrich_graph - pass llm=None to let enricher decide internally (or adjust)
        G_enriched = enrich_graph(G_base, llm=None, autolayout=True, summarize=False)
        safe_write_gpickle(G_enriched, enriched_graph_path)
        st.success(f"✅ Enriched graph saved: {enriched_graph_path}")
    except Exception as e:
        st.error(f"❌ Error during enrichment: {e}")
        st.stop()

    # Step 3: Visualize Base & Enriched
    st.subheader("🕸️ Graph Visualization — Base vs Enriched")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Base Graph")
        try:
            render_pyvis(G_base, "Base Graph")
        except Exception as e:
            st.error(f"Could not render Base Graph: {e}")

    with col2:
        st.markdown("#### Enriched Graph")
        try:
            render_pyvis(G_enriched, "Enriched Graph")
        except Exception as e:
            st.error(f"Could not render Enriched Graph: {e}")

    # Quick action: export excel
    st.markdown("---")
    st.markdown("#### 📥 Export & debug")
    gen_col, dbg_col = st.columns([2, 1])
    with gen_col:
        if st.button("📥 Generate Excel (Nodes + Edges)"):
            try:
                excel_bytes = export_graph_to_excel_bytes(G_enriched)
                st.success("Excel generated — click Download below.")
                st.download_button(
                    label="Download Excel File",
                    data=excel_bytes,
                    file_name="lineage_graph.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            except Exception as e:
                st.error(f"Error generating Excel: {e}")

    # Show small summary counts
    with gen_col:
        st.markdown("**Current graph summary**")
        counts = {}
        for _, a in G_enriched.nodes(data=True):
            kind = _node_kind(a)
            counts[kind] = counts.get(kind, 0) + 1
        st.json(counts)

    # Debug log viewer
    debug_log_path = "/tmp/lineage_intent_debug.log"
    with dbg_col:
        if os.path.exists(debug_log_path):
            if st.button("📝 View Raw LLM Debug Log"):
                try:
                    with open(debug_log_path, "r", encoding="utf-8") as lf:
                        content = lf.read()
                    st.code(content[:10000])  # show a reasonable slice
                except Exception as e:
                    st.error(f"Could not read debug log: {e}")
        else:
            st.info("No LLM debug log found (no queries run or log cleared).")

    st.markdown("---")

    # Step 4: Query Section
    st.subheader("💬 Ask a Lineage Question")
    st.markdown("Examples:")
    st.markdown("- `Where did total sales come from?`")
    st.markdown("- `What will be impacted if customers changes?`")
    st.markdown("- `Which SQL queries write to monthly_trends?`")
    st.markdown("- `Show the upstream datasets for top_customers`")

    question = st.text_input("Enter your lineage question:")

    if st.button("Run Query"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            try:
                st.info("Running lineage agent with semantic refinement...")
                llm = get_llm()  # may return None if not configured
                ans_json = graph_query(question, enriched_graph_path, prefer_enriched=True, llm=llm)
                # graph_query returns a JSON string; parse safely
                try:
                    result = json.loads(ans_json)
                except Exception:
                    # If the agent returned a non-JSON string for some reason, show raw output
                    result = {"summary": "", "reasoning": "", "raw": ans_json}

                # Concise LLM summary
                st.markdown("### 🧠 LLM Summary")
                summary_text = result.get("summary") or result.get("raw_summary") or ""
                if summary_text:
                    st.success(summary_text)
                else:
                    st.info("No clear summary generated. (Check debug log for raw LLM output.)")

                # Reasoning / trace
                st.markdown("### 🧩 Reasoning")
                reasoning_text = result.get("reasoning", "")
                if reasoning_text:
                    st.write(reasoning_text)
                else:
                    st.write("No detailed reasoning available for this query.")

                # Full JSON (expandable)
                st.markdown("---")
                with st.expander("🔍 Full Lineage JSON (Debug View)", expanded=False):
                    st.json(result)

            except Exception as e:
                st.error(f"❌ Query execution failed: {e}")

else:
    # No file uploaded yet
    st.markdown(
        """
        ---
        🪄 **How it works**
        1. Upload a ZIP containing one or more `.xml` or `.sql` files.  
        2. The tool builds a unified data lineage graph.  
        3. You can visualize base vs enriched graphs side-by-side.  
        4. Ask lineage questions in natural language (e.g., *"Where did monthly sales come from?"*).  
        5. Get both a human-friendly summary and a full reasoning JSON for traceability.
        ---
        """
    )
