# ComfyUi_json_editor

# ComfyUI JSON Workflow Manager & LTX Optimizer

Eine lokale **Gradio**-Oberfläche zum Bearbeiten von ComfyUI Workflows mit starkem Fokus auf **LTX Video**.

## Features

- Workflow laden aus **.json**, **.png**, **.webp** und **.mp4** (Metadata)
- Automatische Model-Auswahl aus `extra_model_paths.yaml`
- LTX Video spezifische Tools (mit/ohne Ton)
- Prompt-Verbesserung mit **llama.cpp** (lokal)
- GGUF-Ersetzung (Checkpoint → UNET + CLIP + VAE)
- Deepseek & Tavily Integration (optimale Settings + Workflow-Suche)
- Einfache Installation per `install.bat`

## Installation

1. Repository clonen oder herunterladen:
   ```bash
   git clone https://github.com/igalvadim-debug/ComfyUi_json_editor.git
   cd ComfyUi_json_editor
