"""
lineage_builder.py — Advanced XML Ingestion for Knowledge Graphs
-----------------------------------------------------------------
Hybrid version combining:
✅ Earlier rich dataset–column–shared_field model (multi-cluster)
✅ Safe gpickle read/write (NetworkX version-proof)
✅ Evidence capture for each XML
✅ Compatible with Streamlit + ai_enricher + lineage_agent
"""

import os
import re
import glob
import zipfile
import tempfile
import pickle
import networkx as nx
from lxml import etree
from typing import List, Dict, Any, Optional, Tuple


# ---------------------------------------------------------------------
# Safe Graph I/O (Version-Proof)
# ---------------------------------------------------------------------
def safe_write_gpickle(G, path: str):
    """Safe write for NetworkX graphs across versions."""
    try:
        if hasattr(nx, "write_gpickle"):
            nx.write_gpickle(G, path)
        else:
            from networkx.readwrite import gpickle
            gpickle.write_gpickle(G, path)
    except Exception as e:
        print(f"[WARN] write_gpickle failed ({e}), using pickle fallback")
        with open(path, "wb") as f:
            pickle.dump(G, f)


def safe_read_gpickle(path: str):
    """Safe read for NetworkX graphs across versions."""
    try:
        if hasattr(nx, "read_gpickle"):
            return nx.read_gpickle(path)
        from networkx.readwrite import gpickle
        return gpickle.read_gpickle(path)
    except Exception as e:
        print(f"[WARN] read_gpickle failed ({e}), using pickle fallback")
        with open(path, "rb") as f:
            return pickle.load(f)


# ---------------------------------------------------------------------
# XML Parsing Utilities
# ---------------------------------------------------------------------
INVALID_XML_CHARS = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")

def sanitize_xml_text(text: str) -> str:
    """Clean illegal XML chars and BOMs."""
    clean = INVALID_XML_CHARS.sub("", text)
    return clean.replace("\ufeff", "")

def normalize_name(name: Optional[str]) -> str:
    """Normalize tag or dataset name."""
    if not name:
        return ""
    return (
        str(name)
        .strip()
        .replace("::", ".")
        .replace("__", "_")
        .replace(" ", "_")
        .lower()
    )


# ---------------------------------------------------------------------
# Generic XML Dataset Parser
# ---------------------------------------------------------------------
def parse_generic_xml_dataset(xml_text: str, source_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Parse a generic XML file into a dataset–column fragment.
    Returns dict of nodes, edges, and evidence snippets.
    """
    xml_text = sanitize_xml_text(xml_text)
    try:
        parser = etree.XMLParser(recover=True, encoding="utf-8")
        root = etree.fromstring(xml_text.encode("utf-8"), parser=parser)
    except Exception as e:
        print(f"[WARN] Failed to parse XML {source_path}: {e}")
        return {"nodes": [], "edges": [], "evidence": []}

    dataset_name = os.path.basename(source_path or "unknown").replace(".xml", "")
    nodes, edges, evidence = [], [], []

    # Identify repeating element (likely main record)
    tag_counts = {}
    for el in root.iter():
        tag_counts[el.tag] = tag_counts.get(el.tag, 0) + 1

    if not tag_counts:
        return {"nodes": [], "edges": [], "evidence": []}

    record_tag = max(tag_counts, key=tag_counts.get)
    dataset_id = normalize_name(f"{dataset_name}::{record_tag}")
    nodes.append({"id": dataset_id, "type": "dataset", "meta": {"file": source_path}})

    # Extract child columns
    for rec in root.findall(f".//{record_tag}"):
        for child in rec:
            cname = normalize_name(child.tag)
            if not cname:
                continue
            col_id = f"{dataset_id}.{cname}"
            nodes.append({"id": col_id, "type": "column", "meta": {"dataset": dataset_id, "file": source_path}})
            edges.append({"from": dataset_id, "to": col_id, "type": "has_column", "meta": {"file": source_path}})
            snippet = etree.tostring(child, encoding="unicode", pretty_print=True)[:400]
            evidence.append({"file": source_path, "node": col_id, "snippet": snippet})

    return {"nodes": nodes, "edges": edges, "evidence": evidence}


# ---------------------------------------------------------------------
# Graph Builder (Clusters + Shared Fields)
# ---------------------------------------------------------------------
def build_graph_from_fragments(fragments: List[Dict[str, Any]]) -> nx.DiGraph:
    """Combine all fragments into a unified multi-cluster knowledge graph."""
    G = nx.DiGraph()

    for frag in fragments:
        # add nodes
        for n in frag["nodes"]:
            G.add_node(n["id"], **n["meta"], type=n["type"])
        # add edges
        for e in frag["edges"]:
            G.add_edge(e["from"], e["to"], **e["meta"], type=e["type"])
        # add evidence
        for ev in frag["evidence"]:
            node = ev["node"]
            if not G.has_node(node):
                G.add_node(node, type="unknown")
            G.nodes[node].setdefault("evidence", []).append(
                {"file": ev["file"], "snippet": ev["snippet"]}
            )

    # --- Cross-link columns with same names across datasets ---
    name_index = {}
    for n in G.nodes:
        if "." in n:
            cname = n.split(".")[-1]
            name_index.setdefault(cname, []).append(n)

    for cname, cols in name_index.items():
        if len(cols) > 1:
            for i in range(len(cols) - 1):
                G.add_edge(cols[i], cols[i + 1], type="shared_field", meta={"by_name": cname})

    return G


# ---------------------------------------------------------------------
# Ingest ZIP or Folder → Build Graph
# ---------------------------------------------------------------------
def ingest_zip_or_dir(path: str, out_graph: str) -> Tuple[str, int, int]:
    """
    Build and save a knowledge graph from a ZIP or directory of XML files.
    """
    tmpdir = None
    base = path

    if zipfile.is_zipfile(path):
        tmpdir = tempfile.mkdtemp()
        with zipfile.ZipFile(path, "r") as z:
            z.extractall(tmpdir)
        base = tmpdir

    xml_paths = glob.glob(os.path.join(base, "**", "*.xml"), recursive=True)
    if not xml_paths:
        raise RuntimeError("No XML files found inside provided path.")

    fragments = []
    for i, p in enumerate(xml_paths, 1):
        try:
            text = open(p, encoding="utf-8", errors="ignore").read()
            frag = parse_generic_xml_dataset(text, p)
            if frag["nodes"]:
                fragments.append(frag)
        except Exception as e:
            print(f"[WARN] Skipping {p}: {e}")

        if i % 5 == 0 or i == len(xml_paths):
            print(f"[INFO] Parsed {i}/{len(xml_paths)} XMLs")

    if not fragments:
        raise RuntimeError("No valid XML datasets found — check structure/encoding.")

    G = build_graph_from_fragments(fragments)
    os.makedirs(os.path.dirname(out_graph) or ".", exist_ok=True)

    safe_write_gpickle(G, out_graph)
    print(f"[SUCCESS] Graph saved → {out_graph}")
    print(f"[STATS] Nodes: {len(G.nodes)}, Edges: {len(G.edges)}")

    return out_graph, len(G.nodes), len(G.edges)
