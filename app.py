import gradio as gr
import json
import yaml
import os
from pathlib import Path
from PIL import Image
import dotenv
import tempfile

dotenv.load_dotenv()

TITLE = "ComfyUI JSON Workflow Manager & LTX Optimizer"

# ================== HELPER FUNCTIONS ==================

def load_env_from_path(env_path):
    if env_path and os.path.exists(env_path):
        dotenv.load_dotenv(env_path, override=True)
        return "✅ .env erfolgreich geladen"
    return "❌ .env Pfad nicht gefunden oder leer"

def extract_workflow_from_file(file):
    if not file:
        return None, "Keine Datei ausgewählt"
    
    suffix = Path(file.name).suffix.lower()
    
    try:
        if suffix == ".json":
            with open(file.name, "r", encoding="utf-8") as f:
                workflow = json.load(f)
            return workflow, f"✅ JSON geladen ({len(workflow.get('nodes', []))} Nodes)"

        elif suffix in [".png", ".webp"]:
            img = Image.open(file.name)
            if "workflow" in img.info:
                workflow = json.loads(img.info["workflow"])
                return workflow, f"✅ Workflow aus {suffix.upper()} Metadata extrahiert"
            elif "prompt" in img.info:
                return {"prompt": json.loads(img.info["prompt"])}, "⚠️ Nur Prompt extrahiert"
            else:
                return None, "❌ Kein Workflow-Metadata im Bild gefunden"

        else:
            return None, f"❌ Format {suffix} wird noch nicht unterstützt"
            
    except Exception as e:
        return None, f"❌ Fehler beim Laden: {str(e)}"


def save_workflow_json(workflow_state):
    """Workflow als JSON-Datei zum Download bereitstellen"""
    if not workflow_state:
        return None
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as tmp:
        json.dump(workflow_state, tmp, indent=2, ensure_ascii=False)
        tmp_path = tmp.name
    return tmp_path


# ================== GRADIO INTERFACE ==================

with gr.Blocks(title=TITLE, theme=gr.themes.Soft()) as demo:
    gr.Markdown(f"# {TITLE}")
    gr.Markdown("Workflow Editor • LTX Video • GGUF • llama.cpp Prompt-Optimierung")

    # ---------- State ----------
    workflow_state = gr.State(value=None)   # ← WICHTIG: Hält den Workflow im Speicher

    with gr.Tab("Settings"):
        with gr.Row():
            env_path = gr.Textbox(
                label="Pfad zur .env Datei", 
                placeholder=r"D:\ComfyUiVid\.env",
                value=r"D:\ComfyUiVid\.env",
                scale=3
            )
            env_btn = gr.Button("Load .env", scale=1)
        env_status = gr.Textbox(label="Status", interactive=False)
        
        llama_url = gr.Textbox(label="llama.cpp Server URL (z.B. http://127.0.0.1:8080)", 
                              value="http://127.0.0.1:8080")
        
        deepseek_key = gr.Textbox(label="Deepseek API Key", type="password", value=os.getenv("DEEPSEEK_KEY", ""))
        tavily_key = gr.Textbox(label="Tavily API Key", type="password", value=os.getenv("TAVILY_KEY", ""))

        env_btn.click(load_env_from_path, inputs=env_path, outputs=env_status)

    with gr.Tab("1. Workflow Laden"):
        file_input = gr.File(
            label="Workflow hochladen (.json, .png, .webp, .mp4)",
            file_types=[".json", ".png", ".webp", ".mp4"]
        )
        load_btn = gr.Button("Workflow laden", variant="primary")
        
        status_text = gr.Textbox(label="Status", interactive=False)
        workflow_json_preview = gr.JSON(label="Workflow Vorschau", height=400)

        load_btn.click(
            fn=lambda f: (*extract_workflow_from_file(f), extract_workflow_from_file(f)[0]),
            inputs=file_input,
            outputs=[workflow_json_preview, status_text, workflow_state]
        )

    with gr.Tab("2. LTX Video"):
        ltx_mode = gr.Checkbox(label="LTX Video Modus aktivieren", value=True)
        with_audio = gr.Checkbox(label="Mit Ton (Audio Integration)", value=False)
        gr.Markdown("Weitere LTX-spezifische Einstellungen kommen hier rein...")

    with gr.Tab("3. Prompt Verbesserung"):
        improve_preset = gr.Dropdown(
            choices=["Cinematic / Filmisch", "Detailed / Hochdetailliert", "LTX-optimized", 
                     "Short & Strong", "More Style", "Technical / Precise", "German → English"],
            label="Verbesserungs-Modus",
            value="LTX-optimized"
        )
        improve_btn = gr.Button("Prompt global verbessern (llama.cpp)", variant="primary")
        improved_prompt = gr.Textbox(label="Verbesserter Prompt (global)", lines=6)

    with gr.Tab("Speichern & Export"):
        save_btn = gr.Button("Workflow als JSON herunterladen", variant="primary")
        download_file = gr.File(label="Download", interactive=False)

        save_btn.click(
            fn=save_workflow_json,
            inputs=workflow_state,
            outputs=download_file
        )

    gr.Markdown("**Tipp:** Der Workflow wird über `workflow_state` zwischen allen Tabs geteilt.")

demo.launch(
    server_name="127.0.0.1",
    server_port=7860,
    share=False,
    inbrowser=True
)
