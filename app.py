import gradio as gr
import json
import yaml
import os
from pathlib import Path
from PIL import Image
import dotenv
import tempfile
import copy

dotenv.load_dotenv()

TITLE = "ComfyUI JSON Workflow Manager & LTX Optimizer"

# ================== FORMAT & HELPER ==================

def normalize_workflow(workflow):
    if not workflow:
        return None
    if isinstance(workflow, dict) and "nodes" in workflow:
        return workflow
    # API-Format konvertieren
    if isinstance(workflow, dict):
        nodes = []
        for nid, node in workflow.items():
            if isinstance(node, dict) and "class_type" in node:
                node = node.copy()
                node["id"] = int(nid) if str(nid).isdigit() else nid
                nodes.append(node)
        return {"nodes": nodes, "links": []}
    return workflow


def extract_workflow_from_file(file):
    # ... (bleibt gleich wie vorher)
    # Ich lasse hier der Kürze halber die vorherige Funktion (kann ich bei Bedarf nochmal geben)
    pass  # Platzhalter - nutze die Version aus vorheriger Nachricht


def save_workflow_json(workflow_state):
    if not workflow_state:
        return None
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as tmp:
        json.dump(workflow_state, tmp, indent=2, ensure_ascii=False)
        return tmp.name


# ================== GGUF REPLACEMENT MIT RE-ROUTING ==================

def replace_checkpoint_with_gguf(workflow):
    if not workflow or "nodes" not in workflow or "links" not in workflow:
        return workflow, "❌ Kein gültiges Workflow mit Links gefunden"

    workflow = copy.deepcopy(workflow)  # Wichtig: Nicht das Original verändern
    nodes = workflow["nodes"]
    links = workflow["links"]
    
    new_nodes = []
    replaced_count = 0
    node_id_map = {}  # Alter Node ID → Neue Node IDs

    for node in nodes:
        if node.get("class_type") in ["CheckpointLoaderSimple", "CheckpointLoader"]:
            replaced_count += 1
            old_id = node["id"]
            old_inputs = node.get("inputs", {})
            ckpt_name = old_inputs.get("ckpt_name", "model.safetensors")
            
            base_name = Path(ckpt_name).stem

            # 1. UNET Loader GGUF
            unet_id = old_id
            unet_node = {
                "id": unet_id,
                "class_type": "UnetLoaderGGUF",
                "inputs": {"unet_name": f"{base_name}.gguf"},
                "outputs": [{"name": "MODEL"}]
            }
            node_id_map[old_id] = {"unet": unet_id}

            # 2. CLIP Loader (GGUF)
            clip_id = old_id + 10000
            clip_node = {
                "id": clip_id,
                "class_type": "DualCLIPLoaderGGUF",   # oder CLIPLoaderGGUF je nach Modell
                "inputs": {"clip_name1": f"{base_name}_clip_l.safetensors", "clip_name2": f"{base_name}_clip_g.safetensors", "type": "flux"},
                "outputs": [{"name": "CLIP"}]
            }
            node_id_map[old_id]["clip"] = clip_id

            # 3. VAE Loader
            vae_id = old_id + 20000
            vae_node = {
                "id": vae_id,
                "class_type": "VAELoader",
                "inputs": {"vae_name": f"{base_name}_vae.safetensors"},
                "outputs": [{"name": "VAE"}]
            }
            node_id_map[old_id]["vae"] = vae_id

            new_nodes.extend([unet_node, clip_node, vae_node])

            # Links umleiten
            new_links = []
            for link in links:
                link_id, from_node, from_slot, to_node, to_slot, *rest = link
                if from_node == old_id:
                    # Umleitung je nach Output-Slot
                    if from_slot == 0:   # MODEL
                        new_links.append([link_id, unet_id, 0, to_node, to_slot] + rest)
                    elif from_slot == 1: # CLIP
                        new_links.append([link_id, clip_id, 0, to_node, to_slot] + rest)
                    elif from_slot == 2: # VAE
                        new_links.append([link_id, vae_id, 0, to_node, to_slot] + rest)
                else:
                    new_links.append(link)
            links = new_links  # Aktualisierte Links

        else:
            new_nodes.append(node)

    workflow["nodes"] = new_nodes
    workflow["links"] = links

    msg = f"✅ {replaced_count} Checkpoint(s) durch GGUF-Trio ersetzt mit korrektem Re-Routing"
    return workflow, msg


# ================== GRADIO APP ==================

with gr.Blocks(title=TITLE, theme=gr.themes.Soft()) as demo:
    gr.Markdown(f"# {TITLE}")
    workflow_state = gr.State(value=None)

    with gr.Tab("1. Workflow Laden"):
        file_input = gr.File(label="Workflow hochladen", file_types=[".json",".png",".webp"])
        load_btn = gr.Button("Laden", variant="primary")
        status_text = gr.Textbox(label="Status")
        workflow_preview = gr.JSON(height=400)

        load_btn.click(
            fn=lambda f: (*extract_workflow_from_file(f), extract_workflow_from_file(f)[0]),
            inputs=file_input,
            outputs=[workflow_preview, status_text, workflow_state]
        )

    with gr.Tab("Modelle & GGUF Ersatz"):
        yaml_path = gr.Textbox(label="extra_model_paths.yaml", value=r"D:\ComfyUiVid\extra_model_paths.yaml")
        scan_btn = gr.Button("Modelle scannen")
        
        gguf_btn = gr.Button("Checkpoint(s) → GGUF Trio ersetzen (mit Re-Routing)", variant="primary")
        replace_status = gr.Textbox(label="Status")

        gguf_btn.click(
            fn=lambda state: replace_checkpoint_with_gguf(state) if state else (None, "Kein Workflow geladen"),
            inputs=workflow_state,
            outputs=[workflow_state, replace_status]
        )

    with gr.Tab("Speichern"):
        save_btn = gr.Button("JSON herunterladen", variant="primary")
        download_file = gr.File()

        save_btn.click(save_workflow_json, inputs=workflow_state, outputs=download_file)

demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)
