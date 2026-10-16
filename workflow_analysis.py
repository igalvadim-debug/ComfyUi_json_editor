# workflow_analysis.py - Node-Analyse, Prompt-Suche,
# Versions-Erkennung (2.3 / 2.5), Lese-/Schreibzugriff auf
# Node-Parameter in BEIDEN ComfyUI-Formaten.

import pandas as pd
from config import LTX_NODE_TYPES, NODE_WIDGET_SCHEMA, LTX_VERSION_HINTS

EMPTY_DF = pd.DataFrame(columns=["ID", "Titel", "Class Type", "Wichtige Inputs"])


# ═══════════════════════════════════════════════════════════════
#  Node-Parameter lesen/schreiben (Workflow-Format mit
#  widgets_values UND API-Format mit direkten inputs-Werten)
# ═══════════════════════════════════════════════════════════════

def _node_type(node):
    return str(node.get("type") or node.get("class_type") or "?")


def get_node_param(node, name, default=None):
    """Liest einen benannten Parameter aus einem Node, egal ob er
    im API-Format (inputs-Dict mit direktem Wert) oder im
    Workflow-Format (widgets_values, positionsbasiert ueber
    NODE_WIDGET_SCHEMA) gespeichert ist."""
    inputs = node.get("inputs")

    # API-Format: inputs ist ein Dict mit direkten Werten
    # (keine Verbindungen wie [node_id, slot])
    if isinstance(inputs, dict) and name in inputs:
        val = inputs[name]
        if not (isinstance(val, list) and len(val) == 2):
            return val

    # Workflow-Format: Wert steckt in widgets_values
    if "widgets_values" in node:
        schema = NODE_WIDGET_SCHEMA.get(_node_type(node))
        if schema and name in schema:
            idx = schema.index(name)
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
    """Schreibt einen benannten Parameter zurueck in den Node
    (in-place). Funktioniert fuer beide Formate."""
    inputs = node.get("inputs")
    if isinstance(inputs, dict) and name in inputs:
        inputs[name] = value
        return node

    if "widgets_values" in node:
        schema = NODE_WIDGET_SCHEMA.get(_node_type(node))
        if schema and name in schema:
            idx = schema.index(name)
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
    """Gibt (summary_str, DataFrame, prompts_list) zurueck."""
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
        if ctype in ("CLIPTextEncode", "CLIPTextEncodeSDXL", "smZ CLIPTextEncode",
                     "PrimitiveStringMultiline"):
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
    """Untersucht Node-Typen und referenzierte Modell-Dateinamen und
    liefert '2.3', '2.5', 'gemischt' oder 'unbekannt'."""
    if not workflow or "nodes" not in workflow:
        return "unbekannt"

    found = set()
    for node in workflow["nodes"]:
        ctype = _node_type(node)
        # Node-Typ-Hinweise (z.B. VAEDecodeTiled existiert nur in 2.5)
        from config import LTX23_TO_LTX25_NODE_RENAME
        if ctype in LTX23_TO_LTX25_NODE_RENAME.values():
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
            for version, hints in LTX_VERSION_HINTS.items():
                if any(h in low for h in hints):
                    found.add(version)

    if found == {"2.3"}:
        return "2.3"
    if found == {"2.5"}:
        return "2.5"
    if len(found) > 1:
        return "gemischt"
    return "unbekannt"
