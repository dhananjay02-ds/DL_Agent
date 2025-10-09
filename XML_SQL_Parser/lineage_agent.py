"""
lineage_agent.py — LLM-driven, semantic, reasoning lineage agent (Progressive refinement + debug logging)

Enhancements:
- Robust JSON + regex fallback for intent extraction
- Automatic semantic alias resolution (e.g., "total sales" → total_sales)
- Debug logging of raw LLM output (/tmp/lineage_intent_debug.log)
- Graceful fallback reasoning and guaranteed summary
"""

import os, re, json, pickle
from typing import Any, Optional, List, Tuple, Dict, Set
import networkx as nx

# ---------- Optional LLM helper ----------
try:
    from llm_utils import get_llm
except Exception:
    get_llm = None


# ---------- IO ----------
def safe_read_gpickle(path: str) -> nx.DiGraph:
    if not os.path.exists(path):
        return nx.DiGraph()
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------- Safe LLM Call ----------
def _safe_call_llm(llm: Any, prompt: str) -> str:
    """
    Universal LLM invocation wrapper.
    Handles LangChain ChatModels, Azure/OpenAI clients, or simple callable LLMs.
    Returns plain text output, always safe.
    """
    if not llm:
        print("[DEBUG] No LLM instance provided.")
        return ""

    try:
        # Case 1: LangChain ChatModels
        if hasattr(llm, "invoke"):
            out = llm.invoke(prompt)
        # Case 2: Predict (common with LangChain wrappers)
        elif hasattr(llm, "predict"):
            out = llm.predict(prompt)
        # Case 3: Direct callable or local function
        elif callable(llm):
            out = llm(prompt)
        # Case 4: Legacy .generate() or .complete()
        elif hasattr(llm, "generate"):
            out = llm.generate(prompt)
        elif hasattr(llm, "complete"):
            out = llm.complete(prompt)
        else:
            out = llm(prompt)

        # --- Normalize output ---
        # Handle LangChain AIMessage or list of them
        if hasattr(out, "content"):
            return str(out.content).strip()

        if isinstance(out, list):
            texts = []
            for o in out:
                if hasattr(o, "content"):
                    texts.append(o.content)
                elif isinstance(o, dict):
                    texts.append(o.get("text") or o.get("content", ""))
                else:
                    texts.append(str(o))
            return "\n".join([t for t in texts if t]).strip()

        # Handle dict responses
        if isinstance(out, dict):
            for key in ["text", "response", "content", "output"]:
                if key in out and isinstance(out[key], str):
                    return out[key].strip()
            return json.dumps(out, indent=2)

        # Raw string
        if isinstance(out, str):
            return out.strip()

        # Anything else (tuple, None, etc.)
        return str(out).strip()

    except Exception as e:
        print(f"[ERROR] LLM call failed: {type(e).__name__}: {e}")
        # Log the failure in case it's silent
        debug_path = "/tmp/lineage_intent_debug.log"
        with open(debug_path, "a", encoding="utf-8") as f:
            f.write(f"[ERROR] LLM call failed for prompt:\n{prompt}\n{e}\n\n")
        return ""



# ---------- Intent Extraction ----------
_INTENT_EXAMPLES = [
    ("Where did total_sales come from?", {"intent": "upstream", "targets": ["total_sales"]}),
    ("How is average_order_value computed?", {"intent": "derivation", "targets": ["average_order_value"]}),
    ("What is impacted if sales changes?", {"intent": "downstream", "targets": ["sales"]}),
    ("Compare total_sales and total_spend", {"intent": "compare", "targets": ["total_sales", "total_spend"]}),
]


def llm_extract_intent_and_targets(llm: Any, question: str) -> Dict[str, Any]:
    """
    Robustly extract lineage intent and targets using LLM, with fallback heuristics and debug logging.
    """
    examples = "\n".join([f"Q: {q}\nA: {json.dumps(a)}" for q, a in _INTENT_EXAMPLES])
    prompt = (
        "You are an assistant that extracts INTENT and TARGET ENTITIES from natural language lineage questions.\n"
        "Return a valid JSON object with keys: 'intent' and 'targets'.\n"
        "Intent ∈ {upstream, downstream, derivation, compare, neighbors, meta, unknown}.\n"
        "Targets should be a list of identifiers (e.g., ['total_sales']).\n\n"
        f"{examples}\n\nQ: {question}\nA:"
    )

    out = _safe_call_llm(llm, prompt)

    # ---------- Debug logging ----------
    debug_path = "/tmp/lineage_intent_debug.log"
    with open(debug_path, "a", encoding="utf-8") as f:
        f.write("\n" + "=" * 100 + "\n")
        f.write(f"QUESTION: {question}\n")
        f.write(f"LLM RAW OUTPUT:\n{out}\n")
        f.write("=" * 100 + "\n")

    print(f"[DEBUG] Intent extraction for '{question}': Raw LLM output:\n{out[:400]}...")

    if not out:
        return {"intent": "unknown", "targets": []}

    text = out.strip()
    json_match = re.search(r"\{.*\}", text, re.DOTALL)
    if json_match:
        text = json_match.group(0)

    # Try parsing JSON first
    try:
        parsed = json.loads(text)
        intent = str(parsed.get("intent", "unknown")).lower()
        targets = [str(t).strip().replace(" ", "_") for t in parsed.get("targets", []) if str(t).strip()]
        if intent not in ["upstream", "downstream", "compare", "derivation", "neighbors", "meta"]:
            intent = "unknown"
        return {"intent": intent, "targets": targets}
    except Exception:
        pass

    # Regex fallback for non-JSON responses
    intent_match = re.search(r"(upstream|downstream|compare|derivation|neighbors|meta)", out, re.I)
    intent = intent_match.group(1).lower() if intent_match else "unknown"

    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", question + " " + out)
    stop = {"intent", "targets", "the", "and", "from", "what", "which", "compare", "how", "is", "are", "of", "where", "did", "does", "do", "will", "be"}
    targets = [t for t in tokens if t.lower() not in stop]
    targets = list(dict.fromkeys(targets))  # dedupe

    return {"intent": intent, "targets": targets[:3]}


# ---------- Semantic alias expansion ----------
def _important_nodes(G: nx.DiGraph, limit: int = 80) -> List[str]:
    ranked = sorted(
        G.nodes(),
        key=lambda n: len(list(G.predecessors(n))) + len(list(G.successors(n))),
        reverse=True,
    )
    return [n for n in ranked if not str(n).startswith("expr:")][:limit]


def llm_expand_semantic_targets(llm, query: str, extracted_targets: List[str], G: nx.DiGraph) -> List[str]:
    if not llm:
        return extracted_targets

    all_nodes = list(G.nodes())
    important = _important_nodes(G, limit=60 if len(all_nodes) > 300 else 100)

    prompt1 = f"""
Map query tokens to actual node names in this lineage graph.
Question: "{query}"
Extracted targets: {extracted_targets}
Candidate nodes: {important}

Return JSON list of matching or related node names.
"""
    out1 = _safe_call_llm(llm, prompt1)
    try:
        primary = json.loads(out1.strip())
        if isinstance(primary, list) and len(primary) > 0:
            return list(dict.fromkeys(primary))
    except Exception:
        pass

    subset = [n for n in all_nodes if any(tok.lower() in n.lower() for tok in extracted_targets)]
    subset = subset[:120] if len(subset) > 120 else subset

    prompt2 = f"""
Refine mapping of user targets to lineage nodes.
Question: "{query}"
Tokens: {extracted_targets}
Subset of related nodes: {subset}

Return JSON list of best matches.
"""
    out2 = _safe_call_llm(llm, prompt2)
    try:
        refined = json.loads(out2.strip())
        if isinstance(refined, list) and len(refined) > 0:
            return list(dict.fromkeys(refined))
    except Exception:
        pass

    return extracted_targets


# ---------- Matching ----------
def _normalize(s): return re.sub(r"[^a-z0-9_\.]", "", s.lower()) if s else ""
def _token_set(s): return set(t for t in re.split(r"[_\.]", _normalize(s)) if t)


def match_score(candidate, target_norm):
    c = _normalize(candidate)
    if not c or not target_norm: return 0
    if c == target_norm: return 200
    if c.startswith(target_norm): return 180
    if target_norm in c: return 160
    ct, tt = _token_set(c), _token_set(target_norm)
    return int(100 * len(ct & tt) / max(len(tt), 1)) if ct & tt else 0


def node_candidate_strings(nid, attrs):
    out = [nid]
    for k in ("label", "table_name", "cte_name", "column_name", "expr"):
        if attrs.get(k): out.append(str(attrs[k]))
    return out


def fuzzy_find_nodes_with_scores(G, token, top_n=8):
    norm = _normalize(token)
    if not norm: return []
    scored = []
    for n, a in G.nodes(data=True):
        best = 0
        for cand in node_candidate_strings(n, a):
            best = max(best, match_score(cand, norm))
        if best > 0: scored.append((n, best))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_n]


# ---------- Traversal ----------
def collect_semantic_upstream(G, nid, max_depth=3):
    if nid not in G.nodes: return []
    visited, setfront = set(), [nid]
    for _ in range(max_depth):
        nxt = []
        for f in setfront:
            preds = list(G.predecessors(f))
            for p in preds:
                if p not in visited:
                    visited.add(p)
                    nxt.append(p)
        setfront = nxt
    return list(visited)


def collect_downstream(G, nid, max_nodes=500):
    if nid not in G.nodes: return []
    try: return list(nx.descendants(G, nid))[:max_nodes]
    except Exception: return []


# ---------- Node brief ----------
def node_brief(G, nid):
    if nid not in G.nodes:
        return {"id": nid, "type": None, "label": nid}
    a = dict(G.nodes[nid])
    ntype = a.get("type", "")
    if ntype in ("expression", "expr"):
        expr = a.get("expr") or a.get("text") or ""
        label = expr.strip()[:80] + ("..." if len(expr.strip()) > 80 else "")
        return {"id": nid, "type": ntype, "label": label, "expr": expr}
    label = a.get("label") or a.get("table_name") or a.get("cte_name") or a.get("column_name") or str(nid)
    return {"id": nid, "type": ntype, "label": label, "summary": (a.get("summary") or "")[:400]}


# ---------- Main API ----------
def graph_query(query: str, graph_path: str, prefer_enriched=True, llm=None) -> str:
    if prefer_enriched and graph_path.endswith(".gpickle"):
        enriched = graph_path.replace(".gpickle", ".enriched.gpickle")
        if os.path.exists(enriched): graph_path = enriched

    G = safe_read_gpickle(graph_path)
    llm = llm or (get_llm() if get_llm else None)

    extracted = llm_extract_intent_and_targets(llm, query)
    intent, targets = extracted.get("intent", "unknown"), extracted.get("targets", [])
    if llm: targets = llm_expand_semantic_targets(llm, query, targets, G)

    SELECT_THRESHOLD = 140
    matches, selected, per_target = {}, {}, {}
    for tok in targets:
        cands = fuzzy_find_nodes_with_scores(G, tok, 8)
        matches[tok] = cands
        selected[tok] = cands[0] if cands and cands[0][1] >= SELECT_THRESHOLD else None

    for tok, sel in selected.items():
        if not sel:
            per_target[tok] = {"selected": None}
            continue
        nid, score = sel
        ups = collect_semantic_upstream(G, nid)
        downs = collect_downstream(G, nid)
        per_target[tok] = {
            "selected": {"node": nid, "brief": node_brief(G, nid)},
            "upstream": [node_brief(G, n) for n in ups],
            "downstream": [node_brief(G, n) for n in downs],
        }

    reasoning = []
    for t, info in per_target.items():
        sel = info.get("selected")
        if not sel: continue
        u = [u["label"] for u in info.get("upstream", [])[:5]]
        reasoning.append(f"{t} derives from {', '.join(u) or 'unknown sources'}.")

    summary = ""
    if llm:
        ctx = "\n".join(reasoning) or "No clear lineage context."
        prompt = f"Summarize lineage context:\nQuestion: {query}\nContext:\n{ctx}\nRespond in 3 concise sentences."
        summary = _safe_call_llm(llm, prompt)

    if not summary:
        summary = "No clear summary generated. (Check /tmp/lineage_intent_debug.log for raw LLM output.)"

    return json.dumps({
        "intent": intent,
        "targets": targets,
        "matches": {t: [{"node": n, "score": s, "brief": node_brief(G, n)} for n, s in matches.get(t, [])] for t in targets},
        "per_target": per_target,
        "reasoning": "\n".join(reasoning),
        "summary": summary
    }, indent=2)
