"""
lineage_agent.py — Advanced Data Lineage Query Engine (NetworkX-based)
----------------------------------------------------------------------
Upgrades:
✅ Recursive upstream/downstream traversal (multi-hop)
✅ Flexible edge type matching (expr_dep, shared_field, has_column, derived_from)
✅ Fuzzy node resolution for tolerant name matching
✅ Smarter dataset/column search
✅ Streamlit-safe + full debug logging
"""

import json
import re
import pickle
import networkx as nx
import traceback
from typing import Dict, List, Set
from difflib import get_close_matches
from llm_utils import get_llm


# ---------------------------------------------------------------------
# 1. Load lineage graph safely
# ---------------------------------------------------------------------
def load_graph(path: str):
    """Load a NetworkX lineage graph safely across versions."""
    try:
        if hasattr(nx, "read_gpickle"):
            return nx.read_gpickle(path)
        from networkx.readwrite import gpickle
        return gpickle.read_gpickle(path)
    except Exception as e:
        print(f"[WARN] Graph load fallback: {e}")
        with open(path, "rb") as f:
            return pickle.load(f)


# ---------------------------------------------------------------------
# 2. LLM intent parsing (raw model call)
# ---------------------------------------------------------------------
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


# ---------------------------------------------------------------------
# 3. Fuzzy node resolution
# ---------------------------------------------------------------------
def resolve_node(G: nx.DiGraph, name: str) -> str:
    """Find the closest node ID to a name (case-insensitive fuzzy match)."""
    if name in G:
        return name
    matches = get_close_matches(name.lower(), [n.lower() for n in G.nodes], n=3, cutoff=0.6)
    if matches:
        # Return the original node name with correct casing
        for n in G.nodes:
            if n.lower() == matches[0]:
                return n
    return None


# ---------------------------------------------------------------------
# 4. Recursive graph traversal utilities
# ---------------------------------------------------------------------
DEFAULT_REL_TYPES = {"expr_dep", "shared_field", "has_column", "derived_from"}

def recursive_traverse(
    G: nx.DiGraph,
    node: str,
    direction: str = "out",
    rel_types: Set[str] = None,
    depth: int = 3
) -> List[str]:
    """Recursively traverse graph up to given depth following lineage edges."""
    rel_types = rel_types or DEFAULT_REL_TYPES
    node = resolve_node(G, node)
    if not node:
        return []

    visited = {node}
    queue = [(node, 0)]
    results = set()

    while queue:
        cur, d = queue.pop(0)
        if d >= depth:
            continue

        neighbors = (
            G.successors(cur) if direction == "out" else G.predecessors(cur)
        )
        for n in neighbors:
            edge = G[cur][n] if direction == "out" else G[n][cur]
            if edge.get("type") in rel_types and n not in visited:
                results.add(n)
                visited.add(n)
                queue.append((n, d + 1))

    return list(results)


def find_upstream(G, node, max_hops=3):
    return recursive_traverse(G, node, "in", DEFAULT_REL_TYPES, max_hops)


def find_downstream(G, node, max_hops=3):
    return recursive_traverse(G, node, "out", DEFAULT_REL_TYPES, max_hops)


# ---------------------------------------------------------------------
# 5. Execute structured plan
# ---------------------------------------------------------------------
def execute_plan(G: nx.DiGraph, plan: Dict) -> Dict:
    intent = plan.get("intent", "unknown")
    subj = plan.get("subject", "")
    pred = plan.get("predicate", "")
    direction = plan.get("direction", "out")

    # Resolve node early
    resolved = resolve_node(G, subj) or subj

    if intent == "upstream":
        return {"intent": intent, "subject": subj, "resolved": resolved,
                "origins": find_upstream(G, subj)}

    if intent == "downstream":
        return {"intent": intent, "subject": subj, "resolved": resolved,
                "impact": find_downstream(G, subj)}

    if intent == "search":
        matches = [
            n for n, d in G.nodes(data=True)
            if subj.lower() in n.lower()
            or subj.lower() in d.get("type", "").lower()
            or subj.lower() in json.dumps(d).lower()
        ]
        return {"intent": intent, "subject": subj, "datasets": matches}

    if intent == "lookup":
        results = recursive_traverse(G, subj, direction=direction, rel_types={pred})
        return {"intent": intent, "subject": subj, "results": results}

    if intent == "cypher":
        return {"intent": intent, "note": "Cypher-style traversal not yet implemented"}

    return {"error": "Unknown or unrecognized intent", "plan": plan}


# ---------------------------------------------------------------------
# 6. Main query pipeline (direct LLM + fallback)
# ---------------------------------------------------------------------
def graph_query(query: str, graph_path: str) -> str:
    """LLM → plan → traversal → JSON output (Streamlit-safe)."""
    G = load_graph(graph_path)
    llm = get_llm(model="gpt-4o-mini", temperature=0)

    debug_log = [f"🔍 Query: {query}"]
    try:
        prompt = build_intent_prompt(query)
        response = llm.invoke(prompt)
        raw = getattr(response, "content", None) or getattr(response, "text", None) or str(response)
        raw = raw.strip()
        debug_log.append(f"🧠 LLM Raw Output:\n{raw}")
    except Exception as e:
        trace = traceback.format_exc()
        debug_log.append(f"❌ LLM call failed: {e}\n{trace}")
        print("\n".join(debug_log))
        return json.dumps({"error": f"LLM call failed: {e}", "trace": trace}, indent=2)

    # --- Parse JSON safely ---
    try:
        plan = json.loads(raw)
    except Exception:
        m = re.search(r"\{.*\}", raw, re.S)
        plan = json.loads(m.group(0)) if m else {"intent": "unknown", "raw": raw}

    debug_log.append(f"✅ Parsed Plan: {plan}")

    # --- Execute plan ---
    result = execute_plan(G, plan)
    debug_log.append(f"📊 Execution Result: {result}")

    print("\n".join(debug_log))
    return json.dumps(result, indent=2)


# ---------------------------------------------------------------------
# 7. Helper functions
# ---------------------------------------------------------------------
def find_origins(G, node, max_hops=50):
    return find_upstream(G, node, max_hops)

def impact_analysis(G, node, hops=10):
    return find_downstream(G, node, hops)

def get_evidence(G, node_id, max_snippets=3):
    if node_id not in G.nodes:
        return [{"error": f"Node '{node_id}' not found"}]
    return G.nodes[node_id].get("evidence", [])[:max_snippets]
