import gradio as gr
import json
import yaml
import os
from pathlib import Path
from PIL import Image
import dotenv
import tempfile
import copy
import io

dotenv.load_dotenv()

TITLE = "ComfyUI JSON Workflow Manager & LTX/GGUF Optimizer"

# ================== FORMAT & HELPER ==================

def normalize_workflow(workflow):
    """Normalisiert Workflow in einheitliches Format mit 'nodes' und 'links'."""
    if not workflow:
        return None
    if isinstance(workflow, dict) and "nodes" in workflow:
        return workflow
    # API-Format (prompt) in Workflow-Format konvertieren
    if isinstance(workflow, dict) and any("class_type" in v for v in workflow.values() if isinstance(v, dict)):
        nodes = []
        for nid, node in workflow.items():
            if isinstance(node, dict) and "class_type" in node:
                node = node.copy()
                node["id"] = int(nid) if str(nid).isdigit() else nid
                nodes.append(node)
        return {"nodes": nodes, "links": []}
    return workflow


def extract_workflow_from_png(image_bytes: bytes) -> dict | None:
    """Extrahiert Workflow aus ComfyUI PNG-Metadaten (tEXt / zTXt Chunk)."""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        
        # PNG tEXt / zTXt / iTXt Chunks durchsuchen
        if hasattr(img, "text"):
            # Standard PIL text chunks
            for key, value in img.text.items():
                if key.lower() in ["workflow", "comfyui_workflow"]:
                    try:
                        return json.loads(value)
                    except json.JSONDecodeError:
                        # Manchmal ist es komprimiert oder escaped
                        pass
        
        # Fallback: ExifTool-ähnliche Suche oder raw metadata
        pnginfo = img.info
        for key, value in pnginfo.items():
            if isinstance(value, str) and ("workflow" in key.lower() or len(value) > 1000):
                try:
                    # Manchmal ist es direkt JSON
                    if value.strip().startswith("{"):
                        return json.loads(value)
                    # Oder base64 / komprimiert (selten)
                except:
                    continue
    except Exception:
        pass
    return None


def extract_workflow_from_file(file) -> tuple:
    """Hauptfunktion: Lädt Workflow aus .json oder .png/.webp."""
    if file is None:
        return None, "❌ Keine Datei ausgewählt", None

    try:
        file_path = file.name if hasattr(file, "name") else file
        suffix = Path(file_path).suffix.lower()

        if suffix == ".json":
            with open(file_path, "r", encoding="utf-8") as f:
                workflow = json.load(f)
            status = f"✅ JSON erfolgreich geladen ({len(workflow.get('nodes', []))} Nodes)"
            
        elif suffix in [".png", ".webp"]:
            with open(file_path, "rb") as f:
                image_bytes = f.read()
            workflow = extract_workflow_from_png(image_bytes)
            if workflow:
                status = f"✅ Workflow aus PNG extrahiert ({len(workflow.get('nodes', []))} Nodes)"
            else:
                status = "❌ Kein ComfyUI-Workflow in der PNG gefunden"
                workflow = None
        else:
            status = "❌ Nicht unterstütztes Dateiformat"
            workflow = None

        if workflow:
            workflow = normalize_workflow(workflow)

        return workflow, status, workflow

    except Exception as e:
        return None, f"❌ Fehler beim Laden: {str(e)}", None


def save_workflow_json(workflow_state):
    if not workflow_state:
        return None
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as tmp:
        json.dump(workflow_state, tmp, indent=2, ensure_ascii=False)
        return tmp.name


# ================== GGUF REPLACEMENT ==================

def replace_checkpoint_with_gguf(workflow):
    if not workflow or "nodes" not in workflow:
        return workflow, "❌ Kein gültiges Workflow geladen"

    workflow = copy.deepcopy(workflow)
    nodes = workflow["nodes"]
    links = workflow.get("links", [])

    new_nodes = []
    replaced_count = 0

    for node in nodes:
        if node.get("class_type") in ["CheckpointLoaderSimple", "CheckpointLoader"]:
            replaced_count += 1
            old_id = node["id"]
            old_inputs = node.get("inputs", {})
            ckpt_name = old_inputs.get("ckpt_name", "model.safetensors")
            base_name = Path(ckpt_name).stem

            # Unet GGUF
            unet_node = {
                "id": old_id,
                "class_type": "UnetLoaderGGUF",
                "inputs": {"unet_name": f"{base_name}.gguf"},
                "outputs": [{"name": "MODEL"}]
            }

            # CLIP GGUF (Flux-Beispiel)
            clip_id = old_id + 10000
            clip_node = {
                "id": clip_id,
                "class_type": "DualCLIPLoaderGGUF",
                "inputs": {
                    "clip_name1": f"{base_name}_clip_l.safetensors",
                    "clip_name2": f"{base_name}_clip_g.safetensors",
                    "type": "flux"
                },
                "outputs": [{"name": "CLIP"}]
            }

            # VAE
            vae_id = old_id + 20000
            vae_node = {
                "id": vae_id,
                "class_type": "VAELoader",
                "inputs": {"vae_name": f"{base_name}_vae.safetensors"},
                "outputs": [{"name": "VAE"}]
            }

            new_nodes.extend([unet_node, clip_node, vae_node])

            # Links umleiten (vereinfacht)
            new_links = []
            for link in links:
                link_id, from_node, from_slot, to_node, to_slot, *rest = link
                if from_node == old_id:
                    if from_slot == 0:   # MODEL
                        new_links.append([link_id, unet_id, 0, to_node, to_slot] + rest)
                    elif from_slot == 1: # CLIP
                        new_links.append([link_id, clip_id, 0, to_node, to_slot] + rest)
                    elif from_slot == 2: # VAE
                        new_links.append([link_id, vae_id, 0, to_node, to_slot] + rest)
                else:
                    new_links.append(link)
            links = new_links
        else:
            new_nodes.append(node)

    workflow["nodes"] = new_nodes
    workflow["links"] = links

    msg = f"✅ {replaced_count} Checkpoint(s) durch GGUF-Trio ersetzt"
    return workflow, msg


# ================== GRADIO APP ==================

with gr.Blocks(title=TITLE, theme=gr.themes.Soft()) as demo:
    gr.Markdown(f"# {TITLE}\n\nWorkflow Manager mit GGUF-Konvertierung")

    workflow_state = gr.State(value=None)

    with gr.Tab("1. Workflow Laden"):
        file_input = gr.File(
            label="JSON oder PNG hochladen",
            file_types=[".json", ".png", ".webp"]
        )
        load_btn = gr.Button("Workflow laden", variant="primary")
        status_text = gr.Textbox(label="Status", interactive=False)
        workflow_preview = gr.JSON(label="Workflow Vorschau", height=500)

        load_btn.click(
            fn=extract_workflow_from_file,
            inputs=file_input,
            outputs=[workflow_preview, status_text, workflow_state]
        )

    with gr.Tab("2. GGUF Ersatz"):
        gguf_btn = gr.Button("Checkpoint(s) → GGUF Trio ersetzen", variant="primary")
        replace_status = gr.Textbox(label="Ergebnis", interactive=False)

        gguf_btn.click(
            fn=lambda state: replace_checkpoint_with_gguf(state) if state else (None, "Kein Workflow geladen"),
            inputs=workflow_state,
            outputs=[workflow_state, replace_status]
        )

    with gr.Tab("3. Speichern"):
        save_btn = gr.Button("JSON herunterladen", variant="primary")
        download_file = gr.File(label="Herunterladen")

        save_btn.click(
            fn=save_workflow_json,
            inputs=workflow_state,
            outputs=download_file
        )

    gr.Markdown("### Hinweis: Für Flux/SD3 Modelle ggf. noch manuell anpassen (CLIP Loader Typ).")

demo.launch(
    server_name="127.0.0.1",
    server_port=7860,
    inbrowser=True,
    share=False
)
