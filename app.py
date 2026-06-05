import gradio as gr
import json
import copy
from pathlib import Path
from PIL import Image
import io
import dotenv

dotenv.load_dotenv()

TITLE = "ComfyUI JSON Workflow Editor & GGUF Optimizer"

# ================== CONFIG ==================
GGUF_CONFIG = {
    "unet_class": "UnetLoaderGGUF",
    "clip_class": "DualCLIPLoaderGGUF",
    "vae_class": "VAELoader",
    "clip_type": "flux",          # anpassbar
}

# ================== HELPERS ==================

def find_max_id(nodes):
    """Sichere maximale Node-ID finden"""
    if not nodes:
        return 0
    ids = []
    for node in nodes:
        try:
            nid = int(node.get("id", 0))
            ids.append(nid)
        except (TypeError, ValueError):
            continue
    return max(ids) if ids else 0


def validate_workflow(workflow):
    """Einfache Validierung"""
    if not isinstance(workflow, dict):
        return False, "Kein gültiges Dictionary"
    if "nodes" not in workflow or not isinstance(workflow["nodes"], list):
        return False, "Fehlender oder ungültiger 'nodes'-Array"
    return True, "OK"


def normalize_workflow(workflow):
    """API-Format → Workflow-Format konvertieren"""
    if not workflow:
        return None
    if isinstance(workflow, dict) and "nodes" in workflow and isinstance(workflow["nodes"], list):
        return workflow
    
    # Prompt/API-Format (Keys sind Strings)
    if isinstance(workflow, dict) and any(isinstance(v, dict) and "class_type" in v for v in workflow.values()):
        nodes = []
        for nid_str, node_data in workflow.items():
            if isinstance(node_data, dict) and "class_type" in node_data:
                node = node_data.copy()
                node["id"] = int(nid_str) if nid_str.isdigit() else nid_str
                nodes.append(node)
        return {"nodes": nodes, "links": []}
    return workflow


def extract_workflow_from_png(image_bytes: bytes) -> dict | None:
    """Workflow aus PNG-Metadaten extrahieren"""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        info = img.info
        
        # Häufige Keys
        candidates = ["workflow", "comfyui_workflow", "prompt", "workflow_json"]
        for key in candidates:
            if key in info:
                try:
                    data = json.loads(info[key])
                    return normalize_workflow(data)
                except:
                    continue
        # Fallback: alle Text-Chunks durchsuchen
        if hasattr(img, "text"):
            for value in img.text.values():
                if isinstance(value, str) and value.strip().startswith("{"):
                    try:
                        return normalize_workflow(json.loads(value))
                    except:
                        continue
    except Exception:
        pass
    return None


def extract_workflow_from_file(file) -> tuple:
    """Haupt-Ladefunktion"""
    if not file:
        return None, "❌ Keine Datei ausgewählt", None

    try:
        path = Path(file.name if hasattr(file, "name") else file)
        suffix = path.suffix.lower()

        if suffix == ".json":
            with open(path, "r", encoding="utf-8") as f:
                workflow = json.load(f)
            status = f"✅ JSON geladen ({len(workflow.get('nodes', []))} Nodes)"
        elif suffix in [".png", ".webp"]:
            with open(path, "rb") as f:
                workflow = extract_workflow_from_png(f.read())
            status = f"✅ Workflow aus Bild extrahiert" if workflow else "❌ Kein Workflow in Bild gefunden"
        else:
            return None, "❌ Unterstützte Formate: .json, .png, .webp", None

        if workflow:
            workflow = normalize_workflow(workflow)
            valid, msg = validate_workflow(workflow)
            if not valid:
                return None, f"❌ {msg}", None

        return workflow, status, workflow

    except Exception as e:
        return None, f"❌ Fehler: {str(e)}", None


def replace_to_gguf(workflow):
    """Verbesserte GGUF-Ersetzung mit besserem Link-Handling"""
    if not workflow or "nodes" not in workflow:
        return workflow, "❌ Kein gültiger Workflow"

    workflow = copy.deepcopy(workflow)
    nodes = workflow["nodes"]
    links = workflow.get("links", [])

    new_nodes = []
    next_id = find_max_id(nodes) + 1000  # Sicherheitsabstand
    replaced = 0

    for node in nodes:
        class_type = node.get("class_type") or node.get("type")
        if class_type in ["CheckpointLoaderSimple", "CheckpointLoader", "UNETLoader"]:
            replaced += 1
            old_id = node.get("id")
            ckpt_name = node.get("inputs", {}).get("ckpt_name", "model.safetensors")
            base_name = Path(ckpt_name).stem

            # Neue Nodes erstellen
            unet_id = next_id
            clip_id = next_id + 1
            vae_id = next_id + 2
            next_id += 10

            unet_node = {
                "id": unet_id,
                "class_type": GGUF_CONFIG["unet_class"],
                "inputs": {"unet_name": f"{base_name}.gguf"},
                "outputs": [{"name": "MODEL"}]
            }

            clip_node = {
                "id": clip_id,
                "class_type": GGUF_CONFIG["clip_class"],
                "inputs": {
                    "clip_name1": f"{base_name}_clip_l.safetensors",
                    "clip_name2": f"{base_name}_clip_g.safetensors",
                    "type": GGUF_CONFIG["clip_type"]
                },
                "outputs": [{"name": "CLIP"}]
            }

            vae_node = {
                "id": vae_id,
                "class_type": GGUF_CONFIG["vae_class"],
                "inputs": {"vae_name": f"{base_name}_vae.safetensors"},
                "outputs": [{"name": "VAE"}]
            }

            new_nodes.extend([unet_node, clip_node, vae_node])

            # Links umleiten — alle Verbindungen berücksichtigen
            new_links = []
            for link in links:
                # Link-Format: [link_id, from_node_id, from_slot, to_node_id, to_slot, ...]
                if len(link) < 5:
                    new_links.append(link)
                    continue

                link_id, from_node, from_slot, to_node, to_slot, *extra = link

                if from_node == old_id:
                    # Von altem Checkpoint kommend
                    if from_slot == 0:   # MODEL
                        new_links.append([link_id, unet_id, 0, to_node, to_slot] + extra)
                    elif from_slot == 1: # CLIP
                        new_links.append([link_id, clip_id, 0, to_node, to_slot] + extra)
                    elif from_slot == 2: # VAE
                        new_links.append([link_id, vae_id, 0, to_node, to_slot] + extra)
                    else:
                        new_links.append(link)
                else:
                    new_links.append(link)

            links = new_links
        else:
            new_nodes.append(node)

    workflow["nodes"] = new_nodes
    workflow["links"] = links

    return workflow, f"✅ {replaced} Checkpoint(s) → GGUF Trio ersetzt"


def save_workflow_json(workflow):
    if not workflow:
        return None
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as tmp:
        json.dump(workflow, tmp, indent=2, ensure_ascii=False)
        return tmp.name


# ================== GRADIO UI ==================

with gr.Blocks(title=TITLE, theme=gr.themes.Soft()) as demo:
    gr.Markdown(f"# {TITLE}")

    workflow_state = gr.State(None)

    with gr.Tab("📥 Workflow laden"):
        file_input = gr.File(label="JSON / PNG / WebP", file_types=[".json", ".png", ".webp"])
        load_btn = gr.Button("Laden", variant="primary")
        status = gr.Textbox(label="Status", interactive=False)
        preview = gr.JSON(label="Vorschau", height=400)

        load_btn.click(
            extract_workflow_from_file,
            inputs=file_input,
            outputs=[preview, status, workflow_state]
        )

    with gr.Tab("🔄 GGUF Konvertierung"):
        gr.Markdown("Ersetzt CheckpointLoader durch GGUF-kompatible Nodes")
        convert_btn = gr.Button("Checkpoint → GGUF Trio ersetzen", variant="primary")
        convert_status = gr.Textbox(label="Ergebnis", interactive=False)

        convert_btn.click(
            lambda wf: replace_to_gguf(wf) if wf else (None, "Kein Workflow geladen"),
            inputs=workflow_state,
            outputs=[workflow_state, convert_status]
        )

    with gr.Tab("💾 Speichern"):
        save_btn = gr.Button("JSON herunterladen")
        download = gr.File(label="Download")

        save_btn.click(
            save_workflow_json,
            inputs=workflow_state,
            outputs=download
        )

    gr.Markdown("**Tipp:** Für andere Modelle (SD3, Aurora etc.) kannst du die Config oben im Code anpassen.")

demo.launch(
    server_name="127.0.0.1",
    server_port=7860,
    inbrowser=True
)
