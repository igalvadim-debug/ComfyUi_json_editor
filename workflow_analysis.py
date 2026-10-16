# workflow_analysis.py - Node-Analyse, Prompt-Suche,
# Versions-Erkennung (2.3 / 2.5), Lese-/Schreibzugriff auf
# Node-Parameter in BEIDEN ComfyUI-Formaten.

import pandas as pd
from config import (
    LTX_NODE_TYPES, NODE_WIDGET_SCHEMA, LTX_VERSION_HINTS,
    LTX23_TO_LTX25_NODE_RENAME,
)

EMPTY_DF = pd.DataFrame(columns=["ID", "Titel", "Class Type", "Wichtige Inputs"])

# Vorbereitete Version hints: Kleinbuchstaben + vorgegeben
# Reihenfolge (dict-Reihenfolge von LTX_VERSION_HINTS bleibt
# erhalten - 'in'-Tests funktionieren identisch zum Original).
_VERSION_HINTS_LOWER = {
    version: tuple(h.lower() for h in hints)
    for version, hints in LTX_VERSION_HINTS.items()
}
# Node-Typen, die es nur in 2.5 gibt (fuer Versions-Erkennung)
_LTX25_NODE_TYPES = frozenset(LTX23_TO_LTX25_NODE_RENAME.values())
# Prompt-Node-Typen als Set statt Tuple (O(1)-Mitgliedschaftstest)
_PROMPT_NODE_TYPES = frozenset((
    "CLIPTextEncode", "CLIPTextEncodeSDXL", "smZ CLIPTextEncode",
    "PrimitiveStringMultiline",
))
# Vorab berechnete Index-Zugriffe aufs Widget-Schema
# (ersetzt das lineare schema.index(name) in heissen Schleifen).
_SCHEMA_INDEX = {
    ctype: {name: i for i, name in enumerate(schema)}
    for ctype, schema in NODE_WIDGET_SCHEMA.items()
}


# ═══════════════════════════════════════════════════════════════
#  Node-Parameter lesen/schreiben (Workflow-Format mit
#  widgets_values UND API-Format mit direkten inputs-Werten)
# ═══════════════════════════════════════════════════════════════

def _node_type(node):
    return str(node.get("type") or node.get("class_type") or "?")


def get_node_param(node, name, default=None):
    """Read a parameter from direct inputs or schema-defined widgets.

    A direct inputs value takes precedence unless it is a two-item list,
    which is treated as a connection. Widget lookup requires a known
    parameter in the node type's schema and supports positional lists,
    named dictionaries, and digit-keyed dictionaries using string indices.
    Return default when no value is found; a missing slot in a nonempty
    digit-keyed dictionary returns None.
    """
    inputs = node.get("inputs")

    # API-Format: inputs ist ein Dict mit direkten Werten
    # (keine Verbindungen wie [node_id, slot])
    if isinstance(inputs, dict) and name in inputs:
        val = inputs[name]
        if not (isinstance(val, list) and len(val) == 2):
            return val

    # Workflow-Format: Wert steckt in widgets_values
    if "widgets_values" in node:
        idx = _SCHEMA_INDEX.get(_node_type(node), {}).get(name)
        if idx is not None:
            wv = node.get("widgets_values") or []
            if isinstance(wv, dict):
                # Neuere ComfyUI-Saves: widgets_values ist ein benanntes
                # Dict statt einer positionsbasierten Liste
                # (z.B. VHS_VideoCombine) - direkt per Name lesen.
                if name in wv:
                    return wv[name]
                if all(str(k).isdigit() for k in wv):
                    wv = [wv.get(str(i)) for i in range(idx + 1)]
                else:
                    return default
            if idx < len(wv):
                return wv[idx]

    return default


def set_node_param(node, name, value):
    """Set a parameter in place and return the same node.

    Update an existing inputs entry first, even if it holds a connection.
    Otherwise, write schema-defined parameters when widgets_values exists:
    use a dictionary key or initialize/extend a list with None as needed.
    Fall back to creating an inputs entry, replacing non-dictionary inputs.
    """
    inputs = node.get("inputs")
    if isinstance(inputs, dict) and name in inputs:
        inputs[name] = value
        return node

    if "widgets_values" in node:
        idx = _SCHEMA_INDEX.get(_node_type(node), {}).get(name)
        if idx is not None:
            wv = node.get("widgets_values")
            if isinstance(wv, dict):
                # benanntes Dict (neuere Saves) - direkt per Name schreiben
                wv[name] = value
                return node
            if wv is None:
                wv = []
                node["widgets_values"] = wv
            while len(wv) <= idx:
                wv.append(None)
            wv[idx] = value
            return node

    # Fallback: als direkten inputs-Eintrag anlegen
    if not isinstance(inputs, dict):
        inputs = {}
    inputs[name] = value
    node["inputs"] = inputs
    return node


def find_max_id(nodes):
    ids = []
    for n in nodes:
        try:
            ids.append(int(n.get("id", 0)))
        except Exception:
            pass
    return max(ids) if ids else 0


# ═══════════════════════════════════════════════════════════════
#  Workflow-Analyse
# ═══════════════════════════════════════════════════════════════

def analyze_workflow(workflow):
    """Return a node summary, overview DataFrame, and prompt records.

    Expect a workflow with a nodes list. Summarize the node count and any
    LTX or prompt counts. Overview rows contain ID, Titel, Class Type, and
    Wichtige Inputs (up to five scalar parameters). Prompt records contain
    node_id, class_type, title, and nonblank text from recognized prompt
    node types. A falsey workflow or one without nodes returns
    "Kein Workflow geladen.", an empty overview, and an empty prompt list.
    """
    if not workflow or "nodes" not in workflow:
        return "Kein Workflow geladen.", EMPTY_DF.copy(), []

    nodes = workflow["nodes"]
    rows = []
    prompts = []
    ltx_count = 0

    for node in nodes:
        nid   = str(node.get("id", "?"))
        title = str(node.get("title") or node.get("_meta", {}).get("title", "") or "")
        ctype = _node_type(node)

        # "Wichtige Inputs" fuer die Uebersichtstabelle
        important = []
        schema = NODE_WIDGET_SCHEMA.get(ctype)
        if schema:
            for key in schema:
                val = get_node_param(node, key)
                if isinstance(val, (str, int, float, bool)) and val != "":
                    important.append(f"{key}: {val}")
        elif isinstance(node.get("inputs"), dict):
            for k, v in node["inputs"].items():
                if isinstance(v, (str, int, float, bool)):
                    important.append(f"{k}: {v}")
        important_str = " | ".join(important[:5])

        # Prompt-Nodes einsammeln
        if ctype in _PROMPT_NODE_TYPES:
            text = get_node_param(node, "text") or get_node_param(node, "value")
            if isinstance(text, str) and text.strip():
                prompts.append({"node_id": nid, "class_type": ctype, "title": title, "text": text})

        is_ltx = ctype in LTX_NODE_TYPES or "ltx" in ctype.lower() or "ltx" in title.lower()
        if is_ltx:
            ltx_count += 1
            title = "LTX " + title

        rows.append({"ID": nid, "Titel": title, "Class Type": ctype, "Wichtige Inputs": important_str})

    df = pd.DataFrame(rows, columns=["ID", "Titel", "Class Type", "Wichtige Inputs"])
    summary = f"{len(nodes)} Nodes"
    if ltx_count:
        summary += f" | LTX-Nodes: {ltx_count}"
    if prompts:
        summary += f" | Prompts: {len(prompts)}"
    return summary, df, prompts


def get_prompt_texts(workflow):
    _, _, prompts = analyze_workflow(workflow)
    if not prompts:
        return "Keine Prompt-Nodes gefunden."
    return "\n\n".join(
        f"[Node {p['node_id']} - {p.get('title') or p['class_type']}]\n{p['text']}"
        for p in prompts
    )


def get_first_positive_prompt(workflow):
    _, _, prompts = analyze_workflow(workflow)
    for p in prompts:
        if "neg" not in str(p.get("title", "")).lower():
            return p["text"]
    return prompts[0]["text"] if prompts else ""


# ═══════════════════════════════════════════════════════════════
#  Versions-Erkennung: ist das ein LTX-2.3- oder LTX-2.5-Workflow?
# ═══════════════════════════════════════════════════════════════

def detect_ltx_version(workflow):
    """Infer the LTX version from node types and string parameter values.

    Inspect string items in list widgets_values and dictionary inputs;
    match version hints case-insensitively in strings containing "ltx".
    Targets of configured node renames also count as 2.5 evidence.
    Return "2.3" or "2.5" for evidence of only that version, "gemischt"
    for multiple versions (including 19B), and "unbekannt" otherwise.
    A falsey workflow, missing nodes, or only 19B evidence is "unbekannt".
    """
    if not workflow or "nodes" not in workflow:
        return "unbekannt"

    found = set()
    for node in workflow["nodes"]:
        ctype = _node_type(node)
        # Node-Typ-Hinweise (z.B. VAEDecodeTiled existiert nur in 2.5)
        if ctype in _LTX25_NODE_TYPES:
            found.add("2.5")

        # Dateinamen in widgets_values/inputs durchsuchen
        candidates = []
        wv = node.get("widgets_values")
        if isinstance(wv, list):
            candidates += [v for v in wv if isinstance(v, str)]
        inputs = node.get("inputs")
        if isinstance(inputs, dict):
            candidates += [v for v in inputs.values() if isinstance(v, str)]

        for text in candidates:
            low = text.lower()
            # Kurzer Vorabtest: ohne 'ltx' im Text kann kein Hint matchen.
            if "ltx" not in low:
                continue
            for version, hints in _VERSION_HINTS_LOWER.items():
                if any(h in low for h in hints):
                    found.add(version)

    if found == {"2.3"}:
        return "2.3"
    if found == {"2.5"}:
        return "2.5"
    if len(found) > 1:
        return "gemischt"
    return "unbekannt"
