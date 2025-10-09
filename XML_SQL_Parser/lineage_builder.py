"""
lineage_builder.py (CTE-aware, Enhanced)
SQL + XML ingestion -> fragment generation -> NetworkX DiGraph builder

Enhancements:
- Detects top-level WITH ... AS (CTE) clauses and creates one node per CTE.
- Builds edges between tables -> CTEs -> final query to represent multi-stage lineage.
- Creates expression/column nodes for SELECT expressions and heuristically links them to source tables/columns.
- Adds readable `label` for all nodes, especially expressions (Option A: Keep them but label meaningfully).
- Backwards-compatible with previous interface.

Functions:
- safe_read_gpickle(path)
- safe_write_gpickle(G, path)
- ingest_zip_or_dir(path, out_graph)
- parse_generic_xml_dataset(xml_text, source_path=None)
- parse_sql_file(sql_text, source_path=None)
- build_graph_from_fragments(fragments)
"""
import os
import re
import tempfile
import zipfile
import glob
import pickle
from typing import List, Dict, Any, Tuple

try:
    from lxml import etree
    LXML_AVAILABLE = True
except Exception:
    LXML_AVAILABLE = False

import networkx as nx

# ----------------- IO helpers -----------------
def safe_read_gpickle(path: str) -> nx.DiGraph:
    if not os.path.exists(path):
        return nx.DiGraph()
    with open(path, 'rb') as f:
        return pickle.load(f)

def safe_write_gpickle(G: nx.DiGraph, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(G, f)

# ----------------- XML parsing -----------------
def sanitize_xml_text(text: str) -> str:
    if not text:
        return ''
    return re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f]", '', text)

def parse_generic_xml_dataset(xml_text: str, source_path: str = None) -> Dict[str, Any]:
    fragment = {'source_type': 'xml', 'source_path': source_path, 'nodes': [], 'edges': []}
    text = sanitize_xml_text(xml_text)
    if LXML_AVAILABLE:
        try:
            root = etree.fromstring(text.encode('utf-8'))
            for el in root.iter():
                nid = el.get('id') or el.get('name') or f"xml:{id(el)}"
                n = {
                    'id': str(nid),
                    'type': el.tag,
                    'label': el.tag,
                    'raw': etree.tostring(el, encoding='utf-8', pretty_print=False).decode('utf-8')[:400],
                    'text': (el.text or '')[:400]
                }
                fragment['nodes'].append(n)
                for c in el:
                    cid = c.get('id') or c.get('name') or f"xml:{id(c)}"
                    fragment['edges'].append({'src': n['id'], 'dst': str(cid), 'type': 'contains'})
        except Exception:
            fragment['nodes'].append({'id': source_path or 'xml_unspecified', 'type': 'xml_document', 'raw': text[:400], 'label': 'xml_document'})
    else:
        fragment['nodes'].append({'id': source_path or 'xml_unspecified', 'type': 'xml_document', 'raw': text[:400], 'label': 'xml_document'})
    return fragment

# ----------------- SQL parsing (CTE-aware) -----------------
def _clean_sql_whitespace(sql: str) -> str:
    return re.sub(r"\s+", " ", sql).strip()

_TABLE_RE = re.compile(r"[A-Za-z0-9_\.]+")
_TABLE_COL_RE = re.compile(r"([A-Za-z0-9_]+)\.([A-Za-z0-9_]+)")

def extract_table_identifiers(sql: str) -> List[str]:
    if not sql:
        return []
    s = sql
    s = re.sub(r"'.*?'", "", s, flags=re.DOTALL)
    s = re.sub(r'".*?"', "", s, flags=re.DOTALL)
    patterns = [
        r'from\s+([A-Za-z0-9_\.]+)',
        r'join\s+([A-Za-z0-9_\.]+)',
        r'into\s+([A-Za-z0-9_\.]+)',
        r'update\s+([A-Za-z0-9_\.]+)',
        r'with\s+([A-Za-z0-9_\.]+)\s+as'
    ]
    found = []
    for p in patterns:
        for m in re.finditer(p, s, flags=re.IGNORECASE):
            tn = m.group(1).split()[0]
            found.append(tn)
    seen, out = set(), []
    for t in found:
        key = t.lower()
        if key not in seen:
            seen.add(key)
            out.append(t)
    return out

def extract_select_expressions(sql: str) -> List[str]:
    m = re.search(r'select\s+(.*?)\s+from\s', sql, flags=re.IGNORECASE | re.S)
    if not m:
        return []
    cols = m.group(1)
    parts, buf, depth = [], '', 0
    for ch in cols:
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth = max(0, depth-1)
        if ch == ',' and depth == 0:
            parts.append(buf.strip())
            buf = ''
        else:
            buf += ch
    if buf.strip():
        parts.append(buf.strip())
    cleaned = [re.sub(r'\s+as\s+\w+$', '', p, flags=re.IGNORECASE).strip() for p in parts]
    return cleaned

def split_ctes_and_main(sql: str) -> Tuple[Dict[str, str], str]:
    s = sql.strip()
    if not re.match(r'^\s*with\s', s, flags=re.IGNORECASE):
        return {}, s
    i, L = 0, len(s)
    m = re.match(r'^\s*with\s', s, flags=re.IGNORECASE)
    if not m:
        return {}, s
    i = m.end()
    ctes = {}
    while i < L:
        while i < L and s[i].isspace():
            i += 1
        nm_match = re.match(r'[A-Za-z0-9_]+', s[i:])
        if not nm_match:
            break
        name = nm_match.group(0)
        i += nm_match.end()
        while i < L and s[i].isspace():
            i += 1
        if i < L and s[i] == '(':
            depth = 0
            while i < L:
                if s[i] == '(':
                    depth += 1
                elif s[i] == ')':
                    depth -= 1
                    if depth == 0:
                        i += 1
                        break
                i += 1
            while i < L and s[i].isspace():
                i += 1
        as_match = re.match(r'as\s*\(', s[i:], flags=re.IGNORECASE)
        if not as_match:
            break
        open_paren_idx = i + as_match.end() - 1
        depth, j = 0, open_paren_idx
        while j < L:
            if s[j] == '(':
                depth += 1
            elif s[j] == ')':
                depth -= 1
                if depth == 0:
                    body = s[open_paren_idx+1:j]
                    ctes[name] = body.strip()
                    j += 1
                    break
            j += 1
        i = j
        while i < L and s[i].isspace():
            i += 1
        if i < L and s[i] == ',':
            i += 1
            continue
        break
    main_sql = s[i:].strip()
    return ctes, main_sql

# ---- new helper for clean expression nodes ----
def _make_expr_node(expr_text: str, owner_id: str) -> Dict[str, Any]:
    expr_text = expr_text.strip()
    label = re.sub(r"\s+", " ", expr_text)[:80]
    if len(expr_text) > 80:
        label += "..."
    expr_id = f"expr:{abs(hash((owner_id, expr_text))) % 10**8}"
    return {'id': expr_id, 'type': 'expression', 'expr': expr_text, 'label': label, 'belongs_to': owner_id}

# ----------------- main SQL parse -----------------
def parse_sql_file(sql_text: str, source_path: str = None) -> Dict[str, Any]:
    fragment = {'source_type': 'sql', 'source_path': source_path, 'nodes': [], 'edges': []}
    text = _clean_sql_whitespace(sql_text)
    if not text:
        return fragment
    qid = source_path or f"sql:{abs(hash(text))}"
    main_node = {'id': qid, 'type': 'sql_query', 'text': text[:500], 'raw': text, 'label': os.path.basename(source_path) if source_path else qid}
    fragment['nodes'].append(main_node)

    ctes, main_sql = split_ctes_and_main(text)
    discovered_tables = {}

    def add_table_node(tname):
        tid = f"table:{tname}"
        if not any(n['id'] == tid for n in fragment['nodes']):
            fragment['nodes'].append({'id': tid, 'type': 'table', 'table_name': tname, 'label': tname})
        discovered_tables[tname.lower()] = tname
        return tid

    cte_nodes = {}
    for cname, body in ctes.items():
        cid = f"cte:{cname}"
        cte_nodes[cname] = cid
        fragment['nodes'].append({'id': cid, 'type': 'cte', 'cte_name': cname, 'text': body[:500], 'raw': body, 'label': cname})
        for t in extract_table_identifiers(body):
            tid = add_table_node(t)
            fragment['edges'].append({'src': tid, 'dst': cid, 'type': 'reads'})
        for expr in extract_select_expressions(body):
            if not expr:
                continue
            expr_node = _make_expr_node(expr, cid)
            fragment['nodes'].append(expr_node)
            fragment['edges'].append({'src': expr_node['id'], 'dst': cid, 'type': 'selected_in'})
            for m in _TABLE_COL_RE.finditer(expr):
                t, col = m.groups()
                tmatch = discovered_tables.get(t.lower(), t)
                col_id = f"column:{tmatch}.{col}"
                fragment['nodes'].append({'id': col_id, 'type': 'column', 'column_name': col, 'table': tmatch, 'label': f"{tmatch}.{col}"})
                fragment['edges'].append({'src': col_id, 'dst': expr_node['id'], 'type': 'aggregates_from'})
            for t_low, t_orig in discovered_tables.items():
                if re.search(rf'\b{re.escape(t_orig)}\b', expr, flags=re.IGNORECASE):
                    fragment['edges'].append({'src': f"table:{t_orig}", 'dst': expr_node['id'], 'type': 'contributes_to'})

    for t in extract_table_identifiers(main_sql):
        if t in ctes:
            fragment['edges'].append({'src': cte_nodes[t], 'dst': qid, 'type': 'reads'})
        else:
            tid = add_table_node(t)
            fragment['edges'].append({'src': tid, 'dst': qid, 'type': 'reads'})
    for cname, cid in cte_nodes.items():
        if not any(e for e in fragment['edges'] if e['src'] == cid and e['dst'] == qid):
            fragment['edges'].append({'src': cid, 'dst': qid, 'type': 'reads'})

    for expr in extract_select_expressions(main_sql):
        if not expr:
            continue
        expr_node = _make_expr_node(expr, qid)
        fragment['nodes'].append(expr_node)
        fragment['edges'].append({'src': expr_node['id'], 'dst': qid, 'type': 'selected_in'})
        for m in _TABLE_COL_RE.finditer(expr):
            t, col = m.groups()
            tmatch = discovered_tables.get(t.lower(), t)
            col_id = f"column:{tmatch}.{col}"
            fragment['nodes'].append({'id': col_id, 'type': 'column', 'column_name': col, 'table': tmatch, 'label': f"{tmatch}.{col}"})
            fragment['edges'].append({'src': col_id, 'dst': expr_node['id'], 'type': 'aggregates_from'})
        for t_low, t_orig in discovered_tables.items():
            if re.search(rf'\b{re.escape(t_orig)}\b', expr, flags=re.IGNORECASE):
                fragment['edges'].append({'src': f"table:{t_orig}", 'dst': expr_node['id'], 'type': 'contributes_to'})

    for p in (r'insert\s+into\s+([A-Za-z0-9_\.]+)', r'update\s+([A-Za-z0-9_\.]+)'):
        for m in re.finditer(p, text, flags=re.IGNORECASE):
            t = m.group(1)
            tid = add_table_node(t)
            fragment['edges'].append({'src': qid, 'dst': tid, 'type': 'writes'})
    return fragment

# ----------------- Graph build + ingestion -----------------
def build_graph_from_fragments(fragments: List[Dict[str, Any]]) -> nx.DiGraph:
    G = nx.DiGraph()
    for frag in fragments:
        src_type = frag.get('source_type', 'unknown')
        for n in frag.get('nodes', []):
            nid = n.get('id')
            if not nid:
                continue
            attrs = dict(n)
            attrs['source_type'] = src_type
            G.add_node(nid, **attrs)
        for e in frag.get('edges', []):
            s, d = e.get('src'), e.get('dst')
            if s and d:
                G.add_edge(s, d, **{k: v for k, v in e.items() if k not in ('src', 'dst')})
    return G

def ingest_zip_or_dir(path: str, out_graph: str) -> Tuple[str, int, int]:
    tmpdir = None
    file_paths = []
    if zipfile.is_zipfile(path):
        tmpdir = tempfile.mkdtemp()
        with zipfile.ZipFile(path, 'r') as z:
            z.extractall(tmpdir)
        search_dir = tmpdir
    elif os.path.isdir(path):
        search_dir = path
    else:
        raise ValueError('path must be a zipfile or directory')
    for ext in ('*.xml', '*.XML'):
        file_paths.extend(glob.glob(os.path.join(search_dir, '**', ext), recursive=True))
    for ext in ('*.sql', '*.SQL', '*.txt'):
        file_paths.extend(glob.glob(os.path.join(search_dir, '**', ext), recursive=True))
    fragments = []
    for fp in file_paths:
        try:
            with open(fp, 'r', encoding='utf-8', errors='ignore') as f:
                txt = f.read()
        except Exception:
            continue
        if fp.lower().endswith('.xml'):
            fragments.append(parse_generic_xml_dataset(txt, source_path=fp))
        elif fp.lower().endswith('.sql') or fp.lower().endswith('.txt'):
            if re.search(r'\b(select|insert|update|delete|with)\b', txt, flags=re.IGNORECASE):
                fragments.append(parse_sql_file(txt, source_path=fp))
            else:
                fragments.append({'source_type': 'text', 'source_path': fp, 'nodes': [{'id': fp, 'type': 'text_blob', 'raw': txt[:400], 'label': 'text_blob'}], 'edges': []})
    G = build_graph_from_fragments(fragments)
    safe_write_gpickle(G, out_graph)
    return out_graph, G.number_of_nodes(), G.number_of_edges()

