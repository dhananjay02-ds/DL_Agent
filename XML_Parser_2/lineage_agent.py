"""
lineage_agent.py — Query layer for lineage graph (enriched-aware)
- loads enriched graph if available (fall back to raw)
- uses LLM for intent parsing via get_llm (optional)
- fuzzy resolution + recursive traversal
"""

import json
import re
import pickle
import networkx as nx
import traceback
from typing import Dict, List, Set
from difflib import get_close_matches
from llm_utils import get_llm
import os

DEFAULT_REL_TYPES = {"expr_dep", "shared_field", "has_column", "derived_from"}

def _load_graph_path_candidates(base_path: str):
    """
    Given base_path like outputs/lineage_graph.gpickle, return candidate enriched/raw/enriched.json paths.
    """
    base_dir = os.path.dirname(base_path) or "."
    basename = os.path.basename(base_path)
    if basename.endswith(".gpickle"):
        raw = os.path.join(base_dir, basename.replace(".gpickle", ".raw.gpickle"))
        enriched = os.path.join(base_dir, basename.replace(".gpickle", ".enriched.gpickle"))
    else:
        raw = base_path
        enriched = base_path.replace(".gpickle", ".enriched.gpickle")
    return enriched if os.path.exists(enriched) else raw

def load_graph(path: str):
    """Load a NetworkX lineage graph safely across versions. Prefer enriched if present."""
    use_path = _load_graph_path_candidates(path)
    try:
        if hasattr(nx, "read_gpickle"):
            return nx.read_gpickle(use_path)
        else:
            from networkx.readwrite import gpickle
            return gpickle.read_gpickle(use_path)
    except Exception as e:
        print(f"[WARN] Graph load fallback: {e}")
        with open(use_path, "rb") as f:
            return pickle.load(f)

def build_intent_prompt(query: str) -> str:
    return f"""
You are an expert in data lineage and dependency graphs.

Task:
Convert the user's question into a structured JSON intent plan.

Valid intents:
  - upstream (find where data came from)
  - downstream (find what is impacted)
  - search (find datasets/columns by name)
  - lookup (find direct relationships)
  - cypher (graph pattern query)
  - unknown (if unclear)

Each JSON plan must include:
  intent, subject, predicate, direction ("in" or "out"), pattern (optional)

Examples:
Q: Where did orders.amount come from?
A: {{"intent":"upstream","subject":"orders.amount","predicate":"derived_from","direction":"in"}}

Q: Which datasets contain a column named customer?
A: {{"intent":"search","subject":"customer","predicate":"has_column","direction":"out"}}

Respond **only** with valid JSON.
User query: {query}
"""

def resolve_node(G: nx.DiGraph, name: str) -> str:
    if name in G:
        return name
    lowered = {n.lower(): n for n in G.nodes}
    if name.lower() in lowered:
        return lowered[name.lower()]
    matches = get_close_matches(name.lower(), list(lowered.keys()), n=1, cutoff=0.6)
    if matches:
        return lowered[matches[0]]
    return None

def recursive_traverse(G: nx.DiGraph, node: str, direction: str = "out", rel_types: Set[str] = None, depth: int = 3) -> List[str]:
    rel_types = rel_types or DEFAULT_REL_TYPES
    node_resolved = resolve_node(G, node)
    if not node_resolved:
        return []
    visited = {node_resolved}
    queue = [(node_resolved, 0)]
    results = set()
    while queue:
        cur, d = queue.pop(0)
        if d >= depth:
            continue
        if direction == "out":
            for _, v, data in G.out_edges(cur, data=True):
                etype = (data.get("type") or "").lower()
                if (not rel_types) or (etype in rel_types) or (etype == ""):
                    if v not in visited:
                        results.add(v)
                        visited.add(v)
                        queue.append((v, d+1))
        else:
            for u, _, data in G.in_edges(cur, data=True):
                etype = (data.get("type") or "").lower()
                if (not rel_types) or (etype in rel_types) or (etype == ""):
                    if u not in visited:
                        results.add(u)
                        visited.add(u)
                        queue.append((u, d+1))
    return list(results)

def find_upstream(G, node, max_hops=3):
    return recursive_traverse(G, node, "in", DEFAULT_REL_TYPES, max_hops)

def find_downstream(G, node, max_hops=3):
    return recursive_traverse(G, node, "out", DEFAULT_REL_TYPES, max_hops)

def execute_plan(G: nx.DiGraph, plan: Dict) -> Dict:
    intent = plan.get("intent", "unknown")
    subj = plan.get("subject", "")
    pred = plan.get("predicate", "")
    direction = plan.get("direction", "out")
    resolved = resolve_node(G, subj) or subj
    if intent == "upstream":
        return {"intent": intent, "subject": subj, "resolved": resolved, "origins": find_upstream(G, subj)}
    if intent == "downstream":
        return {"intent": intent, "subject": subj, "resolved": resolved, "impact": find_downstream(G, subj)}
    if intent == "search":
        matches = [n for n, d in G.nodes(data=True) if subj.lower() in n.lower() or subj.lower() in str(d.get("node_type","")).lower() or subj.lower() in json.dumps(d).lower()]
        return {"intent": intent, "subject": subj, "datasets": matches}
    if intent == "lookup":
        results = recursive_traverse(G, subj, direction=direction, rel_types={pred})
        return {"intent": intent, "subject": subj, "results": results}
    return {"error": "Unknown or unrecognized intent", "plan": plan}

def graph_query(query: str, graph_path: str, prefer_enriched: bool = True) -> str:
    """
    Main entrypoint for app: parse intent (using LLM) => execute deterministic traversal on enriched graph.
    prefer_enriched: if True will attempt to use enriched graph (if present), else raw graph.
    """
    G = load_graph(graph_path)
    # use get_llm only to parse intent
    try:
        llm = get_llm(model="gpt-4o-mini", temperature=0)
    except Exception:
        llm = None
    debug = [f"Query: {query}"]
    try:
        prompt = build_intent_prompt(query)
        if llm:
            response = llm.invoke(prompt)
            raw = getattr(response, "content", None) or getattr(response, "text", None) or str(response)
        else:
            # fallback: very simple heuristics if no LLM available
            raw = None
        if not raw:
            # simple heuristic: keywords
            q = query.lower()
            if "where did" in q or "come from" in q:
                subj = re.findall(r"(?:where did|come from)\s+([\w\.\:\-]+)", q)
                plan = {"intent":"upstream", "subject": subj[0] if subj else q, "predicate":"derived_from", "direction":"in"}
            elif "impact" in q or "affected" in q or "affect" in q:
                subj = re.findall(r"(?:impact of|if i change)\s+([\w\.\:\-]+)", q)
                plan = {"intent":"downstream", "subject": subj[0] if subj else q, "predicate":"derived_from", "direction":"out"}
            elif "which datasets" in q or "contain" in q or "which fields" in q:
                # search
                subj = re.findall(r"(?:named|named\s+)?\s*([\w\.\-]+)", q)
                plan = {"intent":"search", "subject": subj[0] if subj else q, "predicate":"has_column", "direction":"out"}
            else:
                plan = {"intent":"unknown", "raw": q}
        else:
            raw = raw.strip()
            # try parse JSON from raw
            try:
                plan = json.loads(raw)
            except Exception:
                m = re.search(r"\{.*\}", raw, re.S)
                plan = json.loads(m.group(0)) if m else {"intent":"unknown", "raw": raw}

    except Exception as e:
        trace = traceback.format_exc()
        return json.dumps({"error": f"LLM parse failed: {e}", "trace": trace}, indent=2)
    # execute
    res = execute_plan(G, plan)
    return json.dumps(res, indent=2)

# helper exports
def find_origins(G, node, max_hops=50):
    return find_upstream(G, node, max_hops)

def impact_analysis(G, node, hops=10):
    return find_downstream(G, node, hops)

def get_evidence(G, node_id, max_snippets=3):
    if node_id not in G.nodes:
        return [{"error": f"Node '{node_id}' not found"}]
    return G.nodes[node_id].get("evidence", [])[:max_snippets]
