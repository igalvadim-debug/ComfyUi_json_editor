import gradio as gr
import json
import yaml
import os
import requests
from pathlib import Path
from PIL import Image
import io
import dotenv

# --- Config ---
dotenv.load_dotenv()  # wird später über Pfad gesteuert

TITLE = "ComfyUI LTX Workflow Manager & Optimizer"

# =============== HELPER FUNCTIONS ===============

def load_env_from_path(env_path):
    if env_path and os.path.exists(env_path):
        dotenv.load_dotenv(env_path)
    return "✅ .env geladen" if env_path else "Keine .env angegeben"

def extract_workflow_from_file(file):
    """Unterstützt .json, .png, .webp, .mp4"""
    if file is None:
        return None, "Keine Datei hochgeladen"
    
    suffix = Path(file.name).suffix.lower()
    
    try:
        if suffix == ".json":
            with open(file.name, "r", encoding="utf-8") as f:
                workflow = json.load(f)
            return workflow, "JSON direkt geladen"
        
        elif suffix in [".png", ".webp"]:
            img = Image.open(file.name)
            if "workflow" in img.info:
                workflow = json.loads(img.info["workflow"])
                return workflow, f"Workflow aus {suffix.upper()} Metadata extrahiert"
            elif "prompt" in img.info:
                prompt = json.loads(img.info["prompt"])
                return {"prompt": prompt}, "Nur Prompt extrahiert"
            else:
                return None, "Kein Workflow-Metadata gefunden"
        
        elif suffix in [".mp4"]:
            # Für MP4 später mit ffprobe oder custom node Metadata erweitern
            return None, "MP4 Metadata-Extraktion noch nicht implementiert (VideoHelperSuite Style)"
        
        return None, f"Nicht unterstütztes Format: {suffix}"
    
    except Exception as e:
        return None, f"Fehler: {str(e)}"

def get_models_from_yaml(yaml_path):
    """extra_model_paths.yaml parsen"""
    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        # Vereinfachte Version – du kannst das später erweitern
        return {"status": "✅ YAML geladen", "paths": data}
    except Exception as e:
        return {"status": f"Fehler: {e}", "paths": {}}

# =============== GRADIO APP ===============

with gr.Blocks(title=TITLE, theme=gr.themes.Soft()) as demo:
    gr.Markdown(f"# {TITLE}")
    gr.Markdown("Workflow Editor + LTX Optimizer + Prompt Verbesserung (llama.cpp)")

    with gr.Tab("Settings"):
        env_path = gr.Textbox(label="Pfad zur .env Datei", placeholder="D:\\ComfyUiVid\\.env", value="")
        env_status = gr.Textbox(label="Status", interactive=False)
        env_btn = gr.Button(" .env laden ")
        env_btn.click(load_env_from_path, inputs=env_path, outputs=env_status)

        llama_url = gr.Textbox(label="llama.cpp Server URL", value="http://127.0.0.1:8080", placeholder="http://127.0.0.1:8080")
        deepseek_key = gr.Textbox(label="Deepseek API Key", type="password", value=os.getenv("DEEPSEEK_KEY", ""))
        tavily_key = gr.Textbox(label="Tavily API Key", type="password", value=os.getenv("TAVILY_KEY", ""))

    with gr.Tab("Workflow Laden"):
        file_input = gr.File(label="Workflow hochladen (.json, .png, .webp, .mp4)", file_types=[".json",".png",".webp",".mp4"])
        load_btn = gr.Button("Workflow laden")
        status_text = gr.Textbox(label="Status", interactive=False)
        workflow_json = gr.JSON(label="Extrahierter Workflow", visible=True)

        load_btn.click(
            extract_workflow_from_file,
            inputs=file_input,
            outputs=[workflow_json, status_text]
        )

    with gr.Tab("LTX Video"):
        ltx_checkbox = gr.Checkbox(label="LTX Video Modus", value=True)
        with_ton = gr.Checkbox(label="Mit Ton", value=False)
        # Hier kommen später LTX-spezifische Controls

    with gr.Tab("Prompt Verbesserung"):
        improve_preset = gr.Dropdown(
            choices=["Cinematic / Filmisch", "Detailed / Hochdetailliert", "LTX-optimized", 
                     "Short & Strong", "More Style", "Technical / Precise", "German → English"],
            label="Verbesserungs-Modus",
            value="LTX-optimized"
        )
        improve_btn = gr.Button("Prompt global verbessern (llama.cpp)")
        improved_prompt = gr.Textbox(label="Verbesserter Prompt", lines=8)

    # Weitere Tabs: Model Manager, Node List, KI-Hilfe etc. folgen

    gr.Markdown("### Nächste Schritte nach diesem Starter:\n- Model Scanner aus extra_model_paths.yaml\n- Node-Erkennung & Dropdowns\n- GGUF Ersatz\n- Deepseek & Tavily Integration")

demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
