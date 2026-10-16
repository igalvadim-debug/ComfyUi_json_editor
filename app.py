# app.py - Einstiegspunkt. Enthaelt keine Logik, nur Start.

from ui import build_app

demo = build_app()

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,
    )
