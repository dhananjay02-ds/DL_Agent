"""
ai_enricher.py
--------------
Enriches a NetworkX lineage graph (built by lineage_builder) with:
- semantic node types (dataset / column / table / job / unknown)
- human-friendly labels
- auto layout (x,y positions)
- optional LLM-based summaries

This runs safely even without any LLM configured.
"""

import networkx as nx
from collections import defaultdict
from typing import Optional


# ---------------------------------------------------------------------
# 1. Safe LLM call helper
# ---------------------------------------------------------------------
def _safe_call_llm(llm, prompt: str) -> str:
    """Call an LLM object safely (compatible with get_llm() or OpenAI wrapper)."""
    if not llm:
        return ""
    try:
        resp = llm.invoke(prompt)
        return getattr(resp, "content", None) or getattr(resp, "text", None) or str(resp)
    except Exception:
        try:
            gen = getattr(llm, "generate", None) or getattr(llm, "_generate", None)
            if gen:
                out = gen([prompt])
                return str(out)
        except Exception:
            return ""
    return ""


# ---------------------------------------------------------------------
# 2. Table inference from dataset node
# ---------------------------------------------------------------------
def infer_table_id_from_dataset_node(node_id: str) -> str:
    """Derive a stable table ID from dataset-style node names."""
    safe = node_id.replace("::", "_").replace(" ", "_").replace("/", "_").lower()
    return f"table_{safe}"


# ---------------------------------------------------------------------
# 3. Graph enrichment
# ---------------------------------------------------------------------
def enrich_graph(
    G: nx.DiGraph,
    use_llm: bool = False,
    llm: Optional[object] = None,
    max_summary_nodes: int = 6
) -> nx.DiGraph:
    """
    Enrich a lineage graph in-place and return it.

    Adds:
    - node_type (dataset/column/table/job/unknown)
    - label (for UI)
    - position (x,y) for layout
    - summary (optional, via LLM)
    """

    # --- Classify nodes ---
    dataset_nodes, column_nodes, other_nodes = [], [], []
    for n, d in G.nodes(data=True):
        ntype = d.get("type", "") or d.get("node_type", "")
        ntype = str(ntype).lower()
        if "." in n or "::" in n or ntype == "column":
            column_nodes.append(n)
            G.nodes[n]["node_type"] = "column"
        elif "dataset" in ntype or "import" in n or "export" in n:
            dataset_nodes.append(n)
            G.nodes[n]["node_type"] = "dataset"
        else:
            other_nodes.append(n)
            G.nodes[n]["node_type"] = ntype or "unknown"

    # --- Create table nodes for datasets ---
    table_to_datasets = defaultdict(list)
    for n in dataset_nodes:
        table_id = infer_table_id_from_dataset_node(n)
        table_to_datasets[table_id].append(n)

    for table_id, datasets in table_to_datasets.items():
        if table_id not in G:
            G.add_node(table_id, node_type="table", label=table_id, meta={"derived_from": datasets})
        else:
            G.nodes[table_id].setdefault("node_type", "table")

    # --- Layout computation ---
    deg_in, deg_out = dict(G.in_degree()), dict(G.out_degree())
    central = {n: deg_in.get(n, 0) + deg_out.get(n, 0) for n in G.nodes}
    sorted_nodes = sorted(G.nodes, key=lambda n: (-central[n], n))

    left, mid, right = [], [], []
    for n in sorted_nodes:
        ntype = G.nodes[n].get("node_type", "")
        if ntype in ("dataset", "table", "source_table"):
            left.append(n)
        elif ntype in ("column", "unknown"):
            mid.append(n)
        else:
            right.append(n)

    def assign_positions(bucket, x_base):
        for i, n in enumerate(sorted(bucket)):
            G.nodes[n]["position"] = {"x": x_base, "y": 60 + i * 70}
            G.nodes[n]["label"] = G.nodes[n].get("label", n)

    assign_positions(left, 50)
    assign_positions(mid, 400)
    assign_positions(right, 750)

    # --- Optional LLM summaries ---
    if use_llm and llm:
        sample_nodes = sorted(sorted_nodes, key=lambda n: -central[n])[:max_summary_nodes]
        for n in sample_nodes:
            neighbors = list(G.predecessors(n))[:3] + list(G.successors(n))[:3]
            prompt = (
                f"Summarize this data object for a business glossary.\n"
                f"Node: {n}\n"
                f"Neighbors: {neighbors}\n"
                "Output one descriptive sentence."
            )
            summary = _safe_call_llm(llm, prompt)
            if summary:
                G.nodes[n]["summary"] = summary.strip().split("\n")[0]

    return G


__all__ = ["enrich_graph"]
