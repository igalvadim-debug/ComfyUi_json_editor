# workflow_io.py - Laden, Normalisieren und Speichern von
# Workflow-Dateien (JSON / PNG / WebP / MP4)

import json, io, re, tempfile
from pathlib import Path
from PIL import Image, PngImagePlugin


def resolve_file_path(file):
    """Gradio 6: 'file' kann dict, str oder Objekt mit .name sein."""
    if file is None:
        return None
    if isinstance(file, dict):
        return file.get("path") or file.get("name") or file.get("tmp_path")
    if isinstance(file, str):
        return file
    if hasattr(file, "name"):
        return file.name
    return str(file)


def normalize_workflow(workflow):
    """Bringt beide bekannten ComfyUI-Formate auf einen gemeinsamen
    Nenner mit Top-Level-Key 'nodes':

    1) 'Workflow'-/UI-Format (aus 'Save' in ComfyUI, PNG/MP4-Metadaten):
       hat bereits 'nodes' (Liste) + 'links' (Liste) - wird
       unveraendert durchgereicht. Werte stecken hier meist in
       node['widgets_values'] (siehe workflow_analysis.py).

    2) 'API'-/Prompt-Format (aus 'Save (API Format)'):
       ein Dict, dessen Keys Node-IDs sind und deren Werte
       {'class_type':..., 'inputs': {...direkte Werte...}} -
       wird in eine Liste von Nodes umgewandelt (inputs bleiben
       als Dict mit direkten Werten erhalten).
    """
    if not workflow:
        return None

    if isinstance(workflow, dict) and "nodes" in workflow and isinstance(workflow["nodes"], list):
        return workflow  # bereits Workflow-/UI-Format

    if isinstance(workflow, dict) and any(
        isinstance(v, dict) and "class_type" in v
        for v in workflow.values() if isinstance(v, dict)
    ):
        nodes = []
        for nid_str, node_data in workflow.items():
            if isinstance(node_data, dict) and "class_type" in node_data:
                node = node_data.copy()
                node["id"] = int(nid_str) if str(nid_str).isdigit() else nid_str
                node["type"] = node_data["class_type"]
                nodes.append(node)
        return {"nodes": nodes, "links": [], "_api_format": True}

    return workflow


def extract_workflow_from_png(image_bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes))
        for key in ["workflow", "comfyui_workflow", "prompt", "workflow_json"]:
            if key in img.info:
                try:
                    return normalize_workflow(json.loads(img.info[key]))
                except Exception:
                    continue
        if hasattr(img, "text"):
            for value in img.text.values():
                if isinstance(value, str) and value.strip().startswith("{"):
                    try:
                        return normalize_workflow(json.loads(value))
                    except Exception:
                        continue
    except Exception:
        pass
    return None


def extract_workflow_from_mp4(file_bytes):
    try:
        text = file_bytes.decode("latin-1")
        for m in re.finditer(r'{"nodes":', text):
            for end in range(min(len(text), m.start() + 500000), m.start() + 100, -1):
                try:
                    return normalize_workflow(json.loads(text[m.start():end]))
                except Exception:
                    continue
    except Exception:
        pass
    return None


def load_workflow_file(file):
    """Hauptfunktion zum Laden. Akzeptiert JSON / PNG / WebP / MP4."""
    if file is None:
        return None, "Keine Datei ausgewaehlt"
    try:
        raw_path = resolve_file_path(file)
        if not raw_path:
            return None, f"Pfad nicht ermittelbar: {type(file)} / {repr(file)}"
        path = Path(raw_path)
        suffix = path.suffix.lower()
        with open(path, "rb") as f:
            raw = f.read()

        if suffix == ".json":
            workflow = normalize_workflow(json.loads(raw.decode("utf-8")))
            n = len(workflow.get("nodes", [])) if workflow else 0
            return workflow, f"JSON geladen - {n} Nodes ({path.name})"
        elif suffix in (".png", ".webp"):
            workflow = extract_workflow_from_png(raw)
            if workflow:
                return workflow, f"Workflow aus {suffix.upper()} - {len(workflow.get('nodes', []))} Nodes"
            return None, f"Kein Workflow in {suffix.upper()} gefunden"
        elif suffix == ".mp4":
            workflow = extract_workflow_from_mp4(raw)
            if workflow:
                return workflow, f"Workflow aus MP4 - {len(workflow.get('nodes', []))} Nodes"
            return None, "Kein Workflow in MP4 gefunden"
        else:
            return None, f"Format nicht unterstuetzt: {suffix}"
    except Exception as e:
        return None, f"Fehler beim Laden: {e}"


def load_workflow_from_path(path_str):
    """Laedt einen Workflow direkt anhand eines Dateipfads (String) -
    praktisch fuer die Beispiel-Workflow-Dropdowns in der UI."""
    return load_workflow_file(path_str)


def save_as_json(workflow):
    if not workflow:
        return None
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8")
    json.dump(workflow, tmp, indent=2, ensure_ascii=False)
    tmp.close()
    return tmp.name


def save_as_png_with_workflow(workflow):
    """Exportiert ein PNG mit eingebettetem Workflow (ComfyUI-kompatibel:
    einfach in ComfyUI hineinziehen, Workflow laedt automatisch)."""
    if not workflow:
        return None
    try:
        img = Image.new("RGB", (512, 512), color=(30, 30, 40))
        meta = PngImagePlugin.PngInfo()
        wf_json = json.dumps(workflow, ensure_ascii=False)
        meta.add_text("workflow", wf_json)
        meta.add_text("prompt", wf_json)
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        img.save(tmp.name, "PNG", pnginfo=meta)
        tmp.close()
        return tmp.name
    except Exception:
        return None
