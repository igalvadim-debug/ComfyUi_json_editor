# md_whitelist.py - MD-Whitelist: die App arbeitet ausschliesslich
# mit minimax H3 Workflows und nur mit den Modellen und Nodes, die in
# der MD-Sammlung (https://github.com/igalvadim-debug/ComfyUi_json_editor/tree/main/MD)
# aufgelistet sind.
#
# MD-Dateien (eine Liste pro Zeile, '#' = Kommentar, leer = Platzhalter):
#   MD/models.txt     -> erlaubte Modelldateien / Modellnamen
#   MD/nodes.txt      -> erlaubte Node-Typen            (optional)
#   MD/workflows.txt  -> erlaubte Workflow-/JSON-Dateien (optional)
#
# Wichtig: Die Liste wird NUR zur Pruefung/Freigabe benutzt - aus ihr
# werden keine Werte erzeugt oder geraten. Steht eine Datei noch leer
# (Platzhalter), meldet das Modul das deutlich, statt alles freizugeben.

import os
import re

MD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "MD")
MD_REPO_URL = "https://github.com/igalvadim-debug/ComfyUi_json_editor/tree/main/MD"

MODELS_FILE = os.path.join(MD_DIR, "models.txt")
NODES_FILE = os.path.join(MD_DIR, "nodes.txt")
WORKFLOWS_FILE = os.path.join(MD_DIR, "workflows.txt")


def _read_list_file(path):
    """Zeilen einer MD-Liste lesen (Kommentare/Leerzeilen ignorieren)."""
    items = []
    if not os.path.isfile(path):
        return items
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if line:
                items.append(line)
    return items


def load_md_lists():
    """Alle drei MD-Listen laden."""
    return {
        "models":    _read_list_file(MODELS_FILE),
        "nodes":     _read_list_file(NODES_FILE),
        "workflows": _read_list_file(WORKFLOWS_FILE),
    }


def models_allowed():
    """Gibt an, ob MD/models.txt mindestens einen Modelleintrag enthaelt."""
    return bool(load_md_lists()["models"])


def nodes_allowed():
    """Gibt an, ob MD/nodes.txt mindestens einen Node-Eintrag enthaelt."""
    return bool(load_md_lists()["nodes"])


def workflows_allowed():
    """Gibt an, ob MD/workflows.txt mindestens einen Workflow-Eintrag enthaelt."""
    return bool(load_md_lists()["workflows"])


# ═══════════════════════════════════════════════════════════════
#  minimax H3 Workflow-Erkennung
# ═══════════════════════════════════════════════════════════════

_MINIMAX_RE = re.compile(r"minimax[\s_-]*h3|minimax|hailuo", re.IGNORECASE)


def is_minimax_h3_workflow(workflow):
    """Erkennt ein Workflow-Dict als minimax-H3-Workflow.

    Belege: 'minimax'/'H3'/'hailuo' in Workflow-Name/-Titel, im
    Unterordner-Prefix einer Modellschwanz ('minimax/...'), in einem
    expliziten MiniMax-API-Node (z.B. MiniMaxH3VideoGeneration) oder
    in den MD-Feldern ('minimax_h3': ...) eines JSONs.
    """
    if not isinstance(workflow, dict):
        return False

    texts = []
    for key in ("_meta", "name", "title"):
        v = workflow.get(key)
        if isinstance(v, str):
            texts.append(v)
        elif isinstance(v, dict):
            texts += [str(x) for x in v.values() if isinstance(x, str)]

    def scan_strings(obj, depth=0):
        """Sammelt Strings und Dict-Schluessel bis Tiefe 6, je Liste maximal 200."""
        if depth > 6:
            return
        if isinstance(obj, str):
            texts.append(obj)
        elif isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(k, str):
                    texts.append(k)
                scan_strings(v, depth + 1)
        elif isinstance(obj, (list, tuple)):
            for v in obj[:200]:
                scan_strings(v, depth + 1)

    scan_strings(workflow.get("nodes", []))
    scan_strings(workflow.get("definitions", {}))

    return any(_MINIMAX_RE.search(t or "") for t in texts)


def describe_format(workflow):
    """Kurze Formatbeschreibung fuer Reports/API-Antworten."""
    if not isinstance(workflow, dict):
        return "unbekannt"
    if "nodes" in workflow:
        fmt = "Workflow-Format (nodes/links)"
    elif all(isinstance(v, dict) and "class_type" in v for v in workflow.values() if v):
        fmt = "API-Format"
    else:
        fmt = "unbekannt"
    h3 = "minimax H3" if is_minimax_h3_workflow(workflow) else "nicht minimax H3"
    return f"{fmt} | {h3}"


# ═══════════════════════════════════════════════════════════════
#  Pruefung: Models & Nodes gegen die MD-Whitelist
# ═══════════════════════════════════════════════════════════════

def _norm(s):
    """Normalisiert Text auf Kleinschreibung ohne Leerraum, _, - und Punkte."""
    return re.sub(r"[\s_\-.]+", "", str(s)).lower()


def _match_entry(value, entries):
    """True, wenn value (oder sein Dateiname) in den MD-Eintragen ist.

    Verglichen werden normalisierte Vollpfade UND reine Dateinamen,
    damit 'models/minimax/h3-diffusion.safetensors' gegen den Eintrag
    'h3-diffusion.safetensors' (und umgekehrt) matcht.
    """
    v = _norm(value)
    if not v:
        return False
    base = _norm(os.path.basename(str(value).replace("\\", "/")))
    for e in entries:
        en = _norm(e)
        if not en:
            continue
        eb = _norm(os.path.basename(e.replace("\\", "/")))
        if v == en or base == eb or (base and base in v) or (v and v in en):
            return True
    return False


_MODEL_VALUE_KEYS = (
    "ckpt_name", "unet_name", "vae_name", "clip_name", "lora_name",
    "model", "model_name", "text_encoder", "diffusion_model",
    "control_net_name", "upscale_name", "checkpoint",
)


def collect_models(workflow):
    """Alle Modelldatei-/namensreferenzen eines Workflows einsammeln."""
    found = []
    if not isinstance(workflow, dict):
        return found

    def add(val, where):
        """Merkt nichtleere Strings mit Modellendung oder Pfadtrenner samt Fundort."""
        if isinstance(val, str) and val.strip():
            low = val.lower()
            if any(ext in low for ext in (".safetensors", ".gguf", ".ckpt", ".pt",
                                          ".bin", ".pth", ".onnx")) \
               or "/" in val or "\\" in val:
                found.append((val, where))

    for node in workflow.get("nodes", []) or []:
        title = str(node.get("title") or "")
        ntype = str(node.get("type") or node.get("class_type") or "?")
        wv = node.get("widgets_values")
        if isinstance(wv, list):
            for i, val in enumerate(wv):
                add(val, f"node {node.get('id', '?')} ({ntype}) widget[{i}] {title}".strip())
        elif isinstance(wv, dict):
            for k, val in wv.items():
                add(val, f"node {node.get('id', '?')} ({ntype}) widget '{k}' {title}".strip())
        inputs = node.get("inputs")
        if isinstance(inputs, dict):
            for k, val in inputs.items():
                if str(k).lower() in _MODEL_VALUE_KEYS and not (isinstance(val, list) and len(val) == 2):
                    add(val, f"node {node.get('id', '?')} ({ntype}) input '{k}' {title}".strip())

    # dedup unter Beibehaltung der Reihenfolge
    seen, uniq = set(), []
    for val, where in found:
        if _norm(val) not in seen:
            seen.add(_norm(val))
            uniq.append((val, where))
    return uniq


def collect_nodes(workflow):
    """Alle Node-Typen (mit IDs) eines Workflows, auch in Subgraphs."""
    types = {}

    def walk(nodes):
        """Ergaenzt die Typ-zu-ID-Tabelle aus einer Node-Liste."""
        for node in nodes or []:
            t = str(node.get("type") or node.get("class_type") or "?")
            types.setdefault(t, []).append(str(node.get("id", "?")))
        # Subgraphs (definitions.subgraphs) ebenfalls pruefen
        defs = node.get("definitions") if isinstance(node.get("definitions"), dict) else None
        if defs:
            for sg in defs.get("subgraphs", []) or []:
                walk(sg.get("nodes", []))

    if isinstance(workflow, dict):
        walk(workflow.get("nodes", []))
        for sg in (workflow.get("definitions", {}) or {}).get("subgraphs", []) or []:
            walk(sg.get("nodes", []))
    return types


def check_workflow(workflow, source_name=""):
    """Prueft einen geladenen Workflow gegen alle MD-Vorgaben.

    Gibt (ok, text) zurueck. ok=False bei: Nicht-minimax-H3-Workflow,
    Modell- oder Node-Referenzen, die nicht in MD/models.txt bzw.
    MD/nodes.txt stehen. Leere/fehlende MD-Listen fuehren NICHT zu
    stillschweigender Freigabe, sondern zu einer klaren Meldung.
    """
    lists = load_md_lists()
    lines = ["MD-Pruefung" + (f" ({source_name})" if source_name else "") + ":"]
    problems = []

    # 1) minimax H3?
    if is_minimax_h3_workflow(workflow):
        lines.append("  [OK]   minimax-H3-Workflow erkannt.")
    else:
        problems.append("Nicht als minimax-H3-Workflow erkennbar "
                        "(kein 'minimax'/'H3'/'hailuo' in Name, Nodes oder Modellen). "
                        "Diese App arbeitet nur mit minimax H3 Workflows.")
        lines.append("  [FEHLT] minimax-H3-Zuordnung nicht belegt.")

    # 2) Models
    models = collect_models(workflow)
    if not lists["models"]:
        note = ("models.txt ist leer (nur Platzhalter) - keine Modelle freigegeben; "
                "Liste in MD/models.txt ergaenzen: " + MD_REPO_URL)
        problems.append(note)
        lines.append(f"  [LEER] {note}")
    else:
        bad = [(m, w) for m, w in models if not _match_entry(m, lists["models"])]
        good = [m for m, _ in models if _match_entry(m, lists["models"])]
        if not models:
            lines.append("  [OK]   Keine Modellreferenzen im Workflow gefunden.")
        else:
            lines.append(f"  [{'OK' if not bad else 'FEHLT'}]   Modelle: {len(good)} OK, "
                         f"{len(bad)} nicht in MD/models.txt ({len(lists['models'])} freigegeben).")
        for m, w in bad:
            problems.append(f"Modell nicht in MD/models.txt: '{m}' ({w})")
            lines.append(f"       - VERBOTEN: {m}  [{w}]")

    # 3) Nodes
    node_types = collect_nodes(workflow)
    if not lists["nodes"]:
        note = ("nodes.txt fehlt/ist leer - Node-Pruefung inaktiv; "
                "freigegebene Node-Typen nach MD/nodes.txt schreiben (eine pro Zeile)")
        lines.append(f"  [LEER] {note}")
    else:
        bad_t = [t for t in node_types if not _match_entry(t, lists["nodes"])]
        lines.append(f"  [{'OK' if not bad_t else 'FEHLT'}]   Nodes: "
                     f"{len(node_types) - len(bad_t)} OK, {len(bad_t)} nicht in MD/nodes.txt "
                     f"({len(lists['nodes'])} freigegeben).")
        for t in bad_t:
            problems.append(f"Node-Typ nicht in MD/nodes.txt: '{t}' (IDs: {', '.join(node_types[t])})")
            lines.append(f"       - VERBOTEN: {t}  [IDs: {', '.join(node_types[t])}]")

    lines.append("")
    if problems:
        lines.append(f"ERGEBNIS: NICHT FREIGEGEBEN ({len(problems)} Problem(en)):")
        lines += [f"  {i+1}. {p}" for i, p in enumerate(problems)]
    else:
        lines.append("ERGEBNIS: FREIGEGEBEN - Workflow entspricht den MD-Vorgaben.")
    return (len(problems) == 0), "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
#  minimax H3 API (OpenAI-kompatibel, nur MD-gelistete Modelle)
# ═══════════════════════════════════════════════════════════════

MINIMAX_BASE_URL = "https://api.minimax.io/v1"
MINIMAX_CHAT_ENDPOINT = MINIMAX_BASE_URL + "/chat/completions"
MINIMAX_API_KEY_ENV = "MINIMAX_API_KEY"


def allowed_api_models():
    """Modelle aus MD/models.txt, die wie API-Modelnamen aussehen.

    Nur MD-gelistete Namen kommen fuer API-Aufrufe in Frage; Datei-
    eintraege (.safetensors/.gguf/...) sind lokale ComfyUI-Modelle und
    werden gefiltert. Ohne MD-Eintrag: None (Aufruf muss abbrechen).
    """
    entries = load_md_lists()["models"]
    api_like = [e for e in entries
                if not re.search(r"\.(safetensors|gguf|ckpt|pt|bin|pth|onnx)$", e, re.I)]
    return api_like or None


def default_api_model():
    """Gibt das erste freigegebene API-Modell zurueck, sonst None."""
    allowed = allowed_api_models()
    return allowed[0] if allowed else None


def call_minimax_chat(messages, model=None, max_tokens=800, temperature=0.7):
    """Chat-Completion gegen die minimax H3 API.

    Das Modell MUSS in MD/models.txt stehen - sonst wird der Aufruf
    verweigert (auch wenn ein gueltiger API-Key vorhanden waere).
    Rueckgabe: (text oder None, statusmeldung).
    """
    api_key = os.getenv(MINIMAX_API_KEY_ENV, "").strip()
    allowed = allowed_api_models()
    if not allowed:
        return None, ("Keine Modelle in MD/models.txt freigegeben - "
                      "API-Aufruf verweigert. Bitte MD-Liste pflegen: " + MD_REPO_URL)
    if model is None:
        model = allowed[0]
    elif not _match_entry(model, allowed):
        return None, (f"Modell '{model}' ist nicht in MD/models.txt freigegeben. "
                      f"Erlaubt: {', '.join(allowed)}")
    if not api_key:
        return None, (f"Kein {MINIMAX_API_KEY_ENV} in .env gefunden - "
                      "minimax-H3-API nicht erreichbar.")
    try:
        import requests
        r = requests.post(
            MINIMAX_CHAT_ENDPOINT,
            headers={"Authorization": f"Bearer {api_key}",
                     "Content-Type": "application/json"},
            json={"model": model, "messages": messages,
                  "max_tokens": max_tokens, "temperature": temperature},
            timeout=90,
        )
        r.raise_for_status()
        data = r.json()
        content = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
        if not content.strip():
            return None, f"minimax H3 lieferte leere Antwort (status_code={data.get('base_resp', {}).get('status_code', '?')})"
        return content.strip(), f"OK (minimax H3, Modell: {model})"
    except Exception as e:
        return None, f"minimax-H3-Fehler: {e}"
