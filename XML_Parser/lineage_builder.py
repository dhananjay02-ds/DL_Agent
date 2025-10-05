"""
lineage_builder.py — Robust XML ingestion for Knowledge Graphs
---------------------------------------------------------------
Designed for any generic XML data exports (including Informatica-like, Zynk, ERP, CRM, etc.)

✅ Cleans malformed XML (removes nulls/control chars)
✅ Recovers from syntax errors (lxml.recover=True)
✅ Builds NetworkX knowledge graph:
       dataset → columns → shared field edges
✅ Works for both .zip and folder inputs
✅ Evidence snippets stored per node
"""

import os
import re
import glob
import zipfile
import tempfile
import json
from typing import List, Dict, Any, Optional, Tuple

from lxml import etree
import networkx as nx


# ---------------------------------------------------------------------
# XML sanitization utilities
# ---------------------------------------------------------------------

INVALID_XML_CHARS = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")


def sanitize_xml_text(text: str) -> str:
    """Remove illegal XML control characters and BOM."""
    clean = INVALID_XML_CHARS.sub("", text)
    clean = clean.replace("\ufeff", "")
    return clean


# ---------------------------------------------------------------------
# String utilities
# ---------------------------------------------------------------------

RE_IDENT = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)?")


def normalize_name(name: Optional[str]) -> str:
    """Normalize XML tag names and dataset identifiers."""
    if not name:
        return ""
    n = str(name).strip().replace("::", ".").replace("__", "_").replace(" ", "_")
    return n.lower()


# ---------------------------------------------------------------------
# Expression / dependency parsing (optional)
# ---------------------------------------------------------------------

def extract_expression_dependencies(expr: str) -> List[str]:
    """Extract identifiers from transformation expressions (optional)."""
    if not expr:
        return []
    expr = expr.strip()
    ids = set()
    for m in RE_IDENT.finditer(expr):
        token = m.group(0).lower()
        if re.search(re.escape(token) + r"\s*\(", expr):  # function call
            continue
        ids.add(token)
    return list(ids)


# ---------------------------------------------------------------------
# Generic XML Dataset Parser
# ---------------------------------------------------------------------

def parse_generic_xml_dataset(xml_text: str, source_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Parse a generic XML dataset file.
    Detects dataset (container) elements and creates dataset → column edges.
    """
    xml_text = sanitize_xml_text(xml_text)
    try:
        parser = etree.XMLParser(recover=True, encoding="utf-8")
        root = etree.fromstring(xml_text.encode("utf-8"), parser=parser)
    except Exception as e:
        print(f"[WARN] Failed XML parse for {source_path}: {e}")
        return {"nodes": [], "edges": [], "evidence": []}

    nodes, edges, evidence = [], [], []

    dataset_name = os.path.basename(source_path or "unknown").replace(".xml", "")

    # Heuristic: the first repeating child tag is the dataset record (e.g. <Project>)
    candidates = {}
    for el in root.iter():
        candidates[el.tag] = candidates.get(el.tag, 0) + 1

    if not candidates:
        return {"nodes": [], "edges": [], "evidence": []}

    record_tag = max(candidates, key=candidates.get)
    dataset_id = normalize_name(f"{dataset_name}::{record_tag}")
    nodes.append({"id": dataset_id, "type": "dataset", "meta": {"file": source_path}})

    # Extract columns (child tags of record_tag)
    for rec in root.findall(f".//{record_tag}"):
        for child in rec:
            colname = normalize_name(child.tag)
            if not colname:
                continue
            col_id = f"{dataset_id}.{colname}"
            nodes.append(
                {"id": col_id, "type": "column", "meta": {"dataset": dataset_id, "file": source_path}}
            )
            edges.append(
                {"from": dataset_id, "to": col_id, "type": "has_column", "meta": {"file": source_path}}
            )
            snippet = etree.tostring(child, encoding="unicode", pretty_print=True)[:400]
            evidence.append({"file": source_path, "node": col_id, "snippet": snippet})

    return {"nodes": nodes, "edges": edges, "evidence": evidence}


# ---------------------------------------------------------------------
# Graph Builder
# ---------------------------------------------------------------------

def build_graph_from_fragments(fragments: List[Dict[str, Any]]) -> nx.DiGraph:
    """Combine parsed fragments into a single NetworkX graph."""
    G = nx.DiGraph()

    for frag in fragments:
        for n in frag["nodes"]:
            G.add_node(n["id"], **n["meta"], type=n["type"])
        for e in frag["edges"]:
            G.add_edge(e["from"], e["to"], **e["meta"], type=e["type"])
        for ev in frag["evidence"]:
            node = ev["node"]
            if not G.has_node(node):
                G.add_node(node, type="unknown")
            G.nodes[node].setdefault("evidence", []).append(
                {"file": ev["file"], "snippet": ev["snippet"]}
            )

    # Optional: connect shared columns by name (schema linkage)
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
# Main entrypoint: ingest ZIP or folder of XMLs
# ---------------------------------------------------------------------

def ingest_zip_or_dir(path: str, out_graph: str) -> Tuple[str, int, int]:
    """
    Build a knowledge graph from a ZIP or directory of XML files.
    Returns the path and counts of nodes and edges.
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
        raise RuntimeError("No valid XML datasets found — check XML structure or encoding.")

    G = build_graph_from_fragments(fragments)
    os.makedirs(os.path.dirname(out_graph) or ".", exist_ok=True)

    # Handle compatibility across NetworkX versions
    try:
        if hasattr(nx, "write_gpickle"):
            nx.write_gpickle(G, out_graph)
        else:
            from networkx.readwrite import gpickle
            gpickle.write_gpickle(G, out_graph)
    except Exception as e:
        print(f"[WARN] write_gpickle failed: {e}, using pickle fallback")
        import pickle

        with open(out_graph, "wb") as f:
            pickle.dump(G, f)

    print(f"[SUCCESS] Graph saved → {out_graph}")
    print(f"[STATS] Nodes: {len(G.nodes)}, Edges: {len(G.edges)}")

    return out_graph, len(G.nodes), len(G.edges)
