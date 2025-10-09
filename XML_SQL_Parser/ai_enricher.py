"""
ai_enricher.py
Enriches a baseline NetworkX graph with labels, titles, summaries and layout.
SQL-aware: recognizes sql_query, table, column_expr nodes.
"""
import networkx as nx
from typing import Optional

def _safe_call_llm(llm, prompt: str) -> str:
    if not llm:
        return ''
    try:
        if hasattr(llm, 'complete'):
            return llm.complete(prompt)
        elif callable(llm):
            return llm(prompt)
    except Exception:
        return ''

def assign_positions(buckets, x_base=0, y_gap=60, x_gap=300):
    out = {}
    for i, bucket in enumerate(buckets):
        x = x_base + i * x_gap
        for j, n in enumerate(bucket):
            out[n] = {'x': x, 'y': j * y_gap}
    return out

def infer_node_kind(attrs: dict) -> str:
    t = str(attrs.get('type', '')).lower()
    src = str(attrs.get('source_type', '')).lower()
    if 'sql' in t or src == 'sql':
        if t == 'sql_query':
            return 'sql_query'
        if t == 'table':
            return 'table'
        if t == 'column_expr' or 'column' in t:
            return 'column'
        return 'sql_node'
    if src == 'xml' or t.startswith('xml'):
        return 'xml_node'
    return 'unknown'

def enrich_graph(G: nx.DiGraph, llm=None, autolayout=True, summarize=True) -> nx.DiGraph:
    buckets = {'sql_query': [], 'table': [], 'column': [], 'xml_node': [], 'unknown': []}
    for n, attrs in G.nodes(data=True):
        kind = infer_node_kind(attrs)
        attrs['node_kind'] = kind
        if kind == 'sql_query':
            attrs['label'] = attrs.get('id')
            attrs['title'] = attrs.get('text', attrs.get('raw', ''))
        elif kind == 'table':
            attrs['label'] = attrs.get('table_name', attrs.get('id'))
            attrs['title'] = f"Table: {attrs.get('table_name', attrs.get('id'))}"
        elif kind == 'column':
            attrs['label'] = attrs.get('expr', attrs.get('id'))[:40]
            attrs['title'] = attrs.get('expr', '')
        elif kind == 'xml_node':
            attrs['label'] = attrs.get('type', attrs.get('id'))
            attrs['title'] = attrs.get('text', attrs.get('raw', ''))
        else:
            attrs['label'] = attrs.get('id')
            attrs['title'] = attrs.get('raw', '')

        if summarize:
            prompt = f"Summarize the following node for a data lineage UI:\n{attrs.get('title','')[:1000]}"
            s = _safe_call_llm(llm, prompt)
            if s:
                attrs['summary'] = s

        if attrs['node_kind'] in buckets:
            buckets[attrs['node_kind']].append(n)
        else:
            buckets['unknown'].append(n)

    if autolayout:
        ordered = [buckets['sql_query'], buckets['table'], buckets['column'], buckets['xml_node'], buckets['unknown']]
        positions = assign_positions(ordered, x_base=0)
        for nid, pos in positions.items():
            nx.set_node_attributes(G, {nid: pos})
    return G

__all__ = ['enrich_graph']
