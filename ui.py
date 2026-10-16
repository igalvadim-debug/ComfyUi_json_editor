# ui.py - komplette Gradio-Oberflaeche.
# Enthaelt KEINE Geschaeftslogik - ruft nur Funktionen aus
# workflow_io / workflow_analysis / converter_ltx25 / transformers /
# llm_analysis / model_manager / workflow_modify / diagnostics /
# md_whitelist auf.
#
# MD-Pflicht: Die App arbeitet ausschliesslich mit minimax H3
# Workflows und nur mit den Modellen/Nodes aus der MD-Sammlung
# (https://github.com/igalvadim-debug/ComfyUi_json_editor/tree/main/MD).
# Jeder Load-Vorgang laeuft durch md_whitelist.check_workflow();
# nicht freigegebene Workflows werden zurueckgewiesen.
#
# Aufbau: zwei parallele, komplett unabhaengige "Spuren" A und B
# (zwei Workflows gleichzeitig geladen & bearbeitbar), in jedem
# Tab nebeneinander als zwei Spalten.

import os
import gradio as gr
import pandas as pd

from config import TITLE, YAML_PATH, PROMPT_PRESETS, GPU_VRAM_PRESETS

from model_manager import load_yaml_paths, get_all_model_lists, scan_ltx_models_by_version
from workflow_io import load_workflow_file, save_as_json, save_as_png_with_workflow
from workflow_analysis import analyze_workflow, get_prompt_texts, detect_ltx_version
from converter_ltx25 import convert_workflow_2_3_to_2_5, build_llm_context
from transformers import replace_checkpoint_with_gguf, apply_ltx_settings, apply_improved_prompt_to_workflow
from llm_analysis import improve_prompt, suggest_conversion_advice, search_workflows_tavily, md_models_status
from workflow_modify import detect_blocks, apply_mods, verify_modification, count_deactivated
from diagnostics import build_full_report, build_minimax_context
from md_whitelist import check_workflow, load_md_lists, describe_format, call_minimax_chat, MD_REPO_URL

_yaml_paths, _yaml_error = load_yaml_paths()
_model_catalog = get_all_model_lists(_yaml_paths)
_empty_df = pd.DataFrame(columns=["ID", "Titel", "Class Type", "Wichtige Inputs"])

EXAMPLE_DIR = os.path.join(os.path.dirname(__file__), "example_workflows")


_md_lists = load_md_lists()


def _md_filter_models(category, choices):
    """Model-Auswahl auf MD/models.txt beschraenken (Whitelist).

    Ohne Eintraege in models.txt gibt es KEINE Modell-Auswahl
    ('(keine Modelle)'-Platzhalter) - nichts wird aus anderen Quellen
    'geraten'. Lokale Pfade werden gegen den reinen Dateinamen
    gematcht.
    """
    entries = _md_lists["models"]
    if not entries:
        return ["(keine Modelle - MD/models.txt ist leer)"]
    import re as _re

    def _norm(s):
        return _re.sub(r"[\s_\-.]+", "", str(s)).lower()

    allowed_norm = {_norm(e) for e in entries}
    allowed_base = {_norm(str(e).replace("\\", "/").split("/")[-1]) for e in entries}
    out = []
    for c in choices:
        base = _norm(str(c).replace("\\", "/").split("/")[-1])
        if _norm(c) in allowed_norm or base in allowed_base:
            out.append(c)
    return out or ["(kein lokales Modell in MD/models.txt freigegeben)"]


def _make_choices(category):
    return _md_filter_models(category, _model_catalog.get(category, ["(keine Modelle)"]))


def _list_example_workflows():
    files = []
    for root, _dirs, names in os.walk(EXAMPLE_DIR):
        for n in names:
            if n.lower().endswith(".json"):
                files.append(os.path.join(root, n))
    return sorted(files)


_example_files = _list_example_workflows()


# ═══════════════════════════════════════════════════════════════
#  Tab 1: Workflow laden (pro Spur: Datei-Upload ODER Beispiel-
#  Workflow-Dropdown)
# ═══════════════════════════════════════════════════════════════

def _build_load_track(label, wf_state, wf_orig):
    gr.Markdown(f"### Workflow {label} (minimax H3 + MD-Freigabe noetig)")
    file_input = gr.File(label="JSON / PNG / WebP / MP4",
                          file_types=[".json", ".png", ".webp", ".mp4"])
    example_dd = gr.Dropdown(choices=_example_files, label="...oder Beispiel-Workflow waehlen")
    with gr.Row():
        load_btn = gr.Button(f"Laden {label}", variant="primary")
        reset_btn = gr.Button("Zuruecksetzen")
    status_lbl = gr.Textbox(label="Status (inkl. MD-Pruefung)", lines=8, interactive=False)
    version_lbl = gr.Textbox(label="Erkannte LTX-Version", interactive=False)
    node_summary = gr.Textbox(label="Uebersicht", interactive=False)
    node_table = gr.Dataframe(value=_empty_df, label="Node-Liste", interactive=False, wrap=True)
    prompt_preview = gr.Textbox(label="Gefundene Prompts", lines=8, interactive=False)

    def on_load(file, example_path):
        src = file if file is not None else example_path
        wf, msg = load_workflow_file(src)   # MD-Pruefung ist hier schon integriert
        if wf is None:
            empty = pd.DataFrame(columns=["ID", "Titel", "Class Type", "Wichtige Inputs"])
            return None, None, msg, "-", "Kein Workflow geladen (MD-Abweisung).", empty, ""
        summary, df, _ = analyze_workflow(wf)
        prompts = get_prompt_texts(wf)
        version = detect_ltx_version(wf)
        return wf, wf, msg, version, summary, df, prompts

    load_btn.click(
        on_load, inputs=[file_input, example_dd],
        outputs=[wf_state, wf_orig, status_lbl, version_lbl, node_summary, node_table, prompt_preview],
    )

    def on_reset(orig):
        summary, df, _ = analyze_workflow(orig)
        return orig, "Zurueckgesetzt", detect_ltx_version(orig), summary, df, get_prompt_texts(orig)

    reset_btn.click(
        on_reset, inputs=[wf_orig],
        outputs=[wf_state, status_lbl, version_lbl, node_summary, node_table, prompt_preview],
    )


# ═══════════════════════════════════════════════════════════════
#  Tab 2: 2.3 -> 2.5 Konvertierung
# ═══════════════════════════════════════════════════════════════

def _build_convert_track(label, wf_state):
    gr.Markdown(f"### Konvertierung {label}: LTX 2.3 -> 2.5")
    convert_btn = gr.Button(f"Konvertieren {label}", variant="primary")
    report_box = gr.Textbox(label="Konvertierungs-Report", lines=12, interactive=False)
    with gr.Row():
        mm_btn = gr.Button("minimax-H3-Zweitmeinung einholen")
    mm_box = gr.Textbox(label="minimax-H3-Empfehlung (nur Hinweise, aendert nichts automatisch)",
                            lines=10, interactive=False)

    def on_convert(wf):
        new_wf, report = convert_workflow_2_3_to_2_5(wf, _yaml_paths)
        return new_wf, "\n".join(report)

    convert_btn.click(on_convert, inputs=[wf_state], outputs=[wf_state, report_box])

    def on_minimax(wf, report_text):
        if not report_text.strip():
            return "Bitte zuerst 'Konvertieren' ausfuehren."
        context = build_llm_context(wf, report_text.split("\n"))
        return suggest_conversion_advice(context)

    mm_btn.click(on_minimax, inputs=[wf_state, report_box], outputs=[mm_box])


# ═══════════════════════════════════════════════════════════════
#  Tab 3: Modelle & GGUF
# ═══════════════════════════════════════════════════════════════

def _build_model_track(label, wf_state):
    gr.Markdown(f"### Modelle {label} "
                "(Dropdowns gefiltert auf MD/models.txt - nur freigegebene Modelle)")
    ckpt_dd = gr.Dropdown(
        choices=_make_choices("checkpoints") + _make_choices("unet") + _make_choices("diffusion_models"),
        label="Checkpoint / UNET", interactive=True)
    clip1_dd = gr.Dropdown(choices=_make_choices("text_encoders") + _make_choices("clip"),
                            label="CLIP 1", interactive=True)
    clip2_dd = gr.Dropdown(choices=_make_choices("text_encoders") + _make_choices("clip"),
                            label="CLIP 2", interactive=True)
    vae_dd = gr.Dropdown(choices=_make_choices("vae"), label="VAE", interactive=True)
    clip_type_dd = gr.Dropdown(choices=["flux", "sd3", "stable_diffusion", "ltxv", "wan"],
                                value="ltxv", label="CLIP Typ", interactive=True)
    gguf_btn = gr.Button(f"Checkpoint -> GGUF-Trio ersetzen {label}", variant="primary")
    gguf_status = gr.Textbox(label="Ergebnis", interactive=False)

    def on_gguf(wf, unet, c1, c2, vae, ctype):
        return replace_checkpoint_with_gguf(
            wf, unet or "model.gguf", c1 or "clip_l.safetensors",
            c2 or "clip_g.safetensors", vae or "vae.safetensors", ctype or "ltxv")

    gguf_btn.click(on_gguf, inputs=[wf_state, ckpt_dd, clip1_dd, clip2_dd, vae_dd, clip_type_dd],
                    outputs=[wf_state, gguf_status])


# ═══════════════════════════════════════════════════════════════
#  Tab 4: LTX Video Schnelleinstellungen
# ═══════════════════════════════════════════════════════════════

def _build_ltx_track(label, wf_state):
    gr.Markdown(f"### LTX-Einstellungen {label}")
    ltx_audio = gr.Checkbox(label="Mit Ton", value=False)
    ltx_width = gr.Slider(256, 1920, step=32, value=768, label="Width")
    ltx_height = gr.Slider(256, 1280, step=32, value=512, label="Height")
    ltx_frames = gr.Slider(8, 257, step=8, value=97, label="Frames")
    ltx_fps = gr.Slider(4, 60, step=1, value=24, label="FPS")
    ltx_steps = gr.Slider(1, 100, step=1, value=30, label="Steps")
    ltx_cfg = gr.Slider(1.0, 20.0, step=0.5, value=3.0, label="CFG")
    ltx_shift = gr.Slider(0.0, 10.0, step=0.1, value=2.0, label="Sigmas Shift")
    ltx_btn = gr.Button(f"Einstellungen anwenden {label}", variant="primary")
    ltx_status = gr.Textbox(label="Status", interactive=False)

    ltx_btn.click(
        apply_ltx_settings,
        inputs=[wf_state, ltx_width, ltx_height, ltx_frames, ltx_fps, ltx_steps, ltx_cfg, ltx_shift, ltx_audio],
        outputs=[wf_state, ltx_status],
    )


# ═══════════════════════════════════════════════════════════════
#  Tab 5: Prompt-Verbesserung (minimax H3, MD-models only)
# ═══════════════════════════════════════════════════════════════

def _build_prompt_track(label, wf_state):
    gr.Markdown(f"### Prompt-Verbesserung {label} "
                "(minimax H3 - Modellzwang aus MD/models.txt)")
    allowed, status = md_models_status()
    model_dd = gr.Dropdown(choices=allowed or ["(keine MD-Modelle freigegeben)"],
                            value=(allowed[0] if allowed else None),
                            label="minimax-H3-Modell (nur MD/models.txt)", interactive=True)
    preset_dd = gr.Dropdown(choices=list(PROMPT_PRESETS.keys()), value="LTX-optimiert", label="Preset")
    prompt_in = gr.Textbox(label="Original Prompt", lines=6)
    load_prompt_btn = gr.Button("Aus Workflow laden (Positive)")
    prompt_out = gr.Textbox(label="Verbesserter Prompt", lines=6)
    improve_status = gr.Textbox(label="Status", interactive=False)
    improve_btn = gr.Button(f"Prompt verbessern {label}", variant="primary")
    with gr.Row():
        apply_pos_btn = gr.Button("-> Positive Nodes")
        apply_neg_btn = gr.Button("-> Negative Nodes")

    def load_pos(wf):
        from workflow_analysis import get_first_positive_prompt
        return get_first_positive_prompt(wf)

    def on_improve(prompt_text, preset_name, model_choice):
        # Nur MD-freigegebene Modelle akzeptieren; sonst klare Absage.
        if not allowed:
            return prompt_text, status
        from md_whitelist import call_minimax_chat, default_api_model
        from config import PROMPT_PRESETS as _PP
        sys_prompt = _PP.get(preset_name, list(_PP.values())[0])
        model = model_choice if (allowed and model_choice in allowed) else default_api_model()
        result, call_status = call_minimax_chat([
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt_text},
        ], model=model)
        if result is None:
            return prompt_text, call_status
        return result, f"Verbessert mit minimax H3 ({model}) | {call_status}"

    load_prompt_btn.click(load_pos, inputs=[wf_state], outputs=[prompt_in])
    improve_btn.click(on_improve, inputs=[prompt_in, preset_dd, model_dd],
                       outputs=[prompt_out, improve_status])
    apply_pos_btn.click(lambda wf, t: apply_improved_prompt_to_workflow(wf, t, "positive"),
                         inputs=[wf_state, prompt_out], outputs=[wf_state, improve_status])
    apply_neg_btn.click(lambda wf, t: apply_improved_prompt_to_workflow(wf, t, "negative"),
                         inputs=[wf_state, prompt_out], outputs=[wf_state, improve_status])


# ═══════════════════════════════════════════════════════════════
#  Tab 6: Speichern & Export
# ═══════════════════════════════════════════════════════════════

def _build_save_track(label, wf_state):
    gr.Markdown(f"### Speichern {label}")
    json_btn = gr.Button(f"Als JSON herunterladen {label}", variant="primary")
    json_download = gr.File(label="JSON")
    json_btn.click(save_as_json, inputs=[wf_state], outputs=[json_download])

    png_btn = gr.Button(f"Als PNG exportieren {label}", variant="primary")
    png_download = gr.File(label="PNG")
    png_btn.click(save_as_png_with_workflow, inputs=[wf_state], outputs=[png_download])

    show_btn = gr.Button(f"Aktuellen Workflow anzeigen {label}")
    wf_preview = gr.JSON(label="Workflow-Vorschau", height=300)
    show_btn.click(lambda wf: wf, inputs=[wf_state], outputs=[wf_preview])


# ═══════════════════════════════════════════════════════════════
#  Tab 7: Workflow modifizieren (Bloeke abschalten, nodeschonend)
# ═══════════════════════════════════════════════════════════════

def _build_modify_track(label, wf_state, wf_orig):
    gr.Markdown(f"### Workflow {label} modifizieren")
    info_box = gr.Textbox(label="Verfuegbare Bloeke", lines=8, interactive=False)
    block_box = gr.CheckboxGroup(
        label="Bloecke abschalten (Mute + Bypass - Nodes bleiben erhalten)",
        choices=[], value=[],
    )
    with gr.Row():
        detect_btn = gr.Button(f"Bloeke erkennen {label}")
        apply_btn = gr.Button(f"Anwenden {label}", variant="primary")
        reset_btn = gr.Button("Zuruecksetzen")
    report_box = gr.Textbox(label="Modifikations-Report", lines=12, interactive=False)
    status_lbl = gr.Textbox(label="Status", interactive=False)
    file_out = gr.File(label="Modifizierter Workflow (JSON) - direkt herunterladen")

    def on_detect(wf):
        if not wf or "nodes" not in wf:
            return gr.update(choices=[], value=[]), "Kein Workflow geladen."
        blocks = detect_blocks(wf)
        found = [f"{name}: {info['label']} ({info['nodes']} Nodes)"
                 for name, info in blocks.items() if info["found"]]
        if not found:
            return gr.update(choices=[], value=[]), "Keine abschaltbaren Bloeke gefunden."
        return gr.update(choices=found, value=[]), "Gefunden:\n" + "\n".join(found)

    def on_apply(wf, selected):
        if not wf or "nodes" not in wf:
            return wf, gr.update(value=[]), "", "Kein Workflow geladen."
        if not selected:
            return wf, gr.update(value=[]), "", "Keine Bloeke ausgewaehlt."
        # CheckboxGroup liefert die Choice-Strings ("<name>: <label> ...") -
        # der Blockname steht immer vor dem ersten Doppelpunkt.
        names = [s.split(":", 1)[0].strip() for s in selected]
        mods = {name: True for name in names}
        modified, report = apply_mods(wf, mods)
        muted_count = count_deactivated(modified)
        ok, vmsg = verify_modification(wf, modified)
        summary = (f"Verify: {'OK' if ok else 'FEHLER'} | {muted_count} Nodes deaktiviert | "
                   f"Links: {len(wf.get('links', []))} -> {len(modified.get('links', []))}")
        # Direkt speichern: die modifizierte Datei liegt sofort als Download vor.
        save_path = save_as_json(modified)
        return (modified, gr.update(choices=[], value=[]),
                "\n".join(report + [vmsg]), summary, save_path)

    def on_reset(orig):
        return orig, gr.update(choices=[], value=[]), "", "Zurueckgesetzt (Original wiederhergestellt)", None

    detect_btn.click(on_detect, inputs=[wf_state], outputs=[block_box, info_box])
    apply_btn.click(on_apply, inputs=[wf_state, block_box],
                    outputs=[wf_state, block_box, report_box, status_lbl, file_out])
    reset_btn.click(on_reset, inputs=[wf_orig],
                    outputs=[wf_state, block_box, report_box, status_lbl, file_out])


# ═══════════════════════════════════════════════════════════════
#  Tab 8: Fehleranalyse & GGUF-Optimierung
# ═══════════════════════════════════════════════════════════════

def _build_diagnostics_track(label, wf_state):
    gr.Markdown(f"### Fehleranalyse {label}")
    gpu_dd = gr.Dropdown(choices=list(GPU_VRAM_PRESETS.keys()),
                          value="RTX 5060 Laptop (8 GB)", label="GPU-Preset")
    vram_num = gr.Number(value=8, label="VRAM (GB) - wird bei Preset automatisch gesetzt, "
                                        "oder 'Eigener Wert' waehlen und hier eintragen")
    ram_num = gr.Number(value=32, label="System-RAM (GB) - fuer Offloading-Einschaetzung")
    ltx_version_dd = gr.Dropdown(choices=["2.3", "2.5", "19B"], value="2.3",
                                  label="LTX-Version (fuer GGUF-Groessenschaetzung)")
    error_text = gr.Textbox(
        label="ComfyUI-Log/Fehlermeldung einfuegen ODER Problem in eigenen Worten beschreiben",
        lines=10, placeholder="z.B. kompletten Log-Auszug aus der Konsole hier reinkopieren...")
    analyze_btn = gr.Button(f"Analysieren {label}", variant="primary")
    report_out = gr.Textbox(label="Diagnose & Empfehlungen", lines=22, interactive=False)
    with gr.Row():
        mm_btn2 = gr.Button("minimax-H3-Zweitmeinung (fuer unklare Faelle)")
    mm_out2 = gr.Textbox(label="minimax-H3-Einschaetzung", lines=10, interactive=False)

    def on_gpu_change(gpu_name):
        return GPU_VRAM_PRESETS.get(gpu_name, 8)

    gpu_dd.change(on_gpu_change, inputs=[gpu_dd], outputs=[vram_num])

    def on_analyze(wf, text, vram_gb, ram_gb, ltx_version):
        report = build_full_report(wf, text or "", int(vram_gb or 8), ltx_version, int(ram_gb) if ram_gb else None)
        return report

    analyze_btn.click(on_analyze, inputs=[wf_state, error_text, vram_num, ram_num, ltx_version_dd],
                       outputs=[report_out])

    def on_minimax(wf, text, vram_gb, ltx_version, rule_report):
        if not rule_report.strip():
            return "Bitte zuerst 'Analysieren' ausfuehren."
        context = build_minimax_context(wf, text or "", int(vram_gb or 8), ltx_version, rule_report)
        messages = [
            {"role": "system", "content": (
                "You are an expert ComfyUI/minimax-H3-Video troubleshooter. Given a workflow summary, "
                "an error description, and an already-generated rule-based report, add anything "
                "the rules likely missed - especially subtle version-mismatch or VRAM issues. "
                "Only suggest models and nodes that appear in the user's MD whitelist lists. "
                "Be concise and concrete. Answer in German."
            )},
            {"role": "user", "content": context},
        ]
        result, status = call_minimax_chat(messages, max_tokens=700)
        return result if result is not None else status

    mm_btn2.click(on_minimax, inputs=[wf_state, error_text, vram_num, ltx_version_dd, report_out],
                    outputs=[mm_out2])


# ═══════════════════════════════════════════════════════════════
#  Gesamte App
# ═══════════════════════════════════════════════════════════════

def build_app():
    with gr.Blocks(title=TITLE) as demo:
        gr.Markdown(f"# {TITLE}")

        wf_state_a, wf_orig_a = gr.State(None), gr.State(None)
        wf_state_b, wf_orig_b = gr.State(None), gr.State(None)

        if _yaml_error:
            gr.Markdown(f"YAML-Fehler: {_yaml_error}")
        else:
            cats = {k: len(v) for k, v in _model_catalog.items() if v and v[0] != "(keine Modelle gefunden)"}
            gr.Markdown(f"YAML geladen - {len(_yaml_paths)} Kategorien | Modelle: {cats}")

        ltx_by_version = scan_ltx_models_by_version(_yaml_paths)
        gr.Markdown(
            f"Lokal gefundene LTX-Modelle - 2.3: {len(ltx_by_version['2.3'])} Dateien, "
            f"2.5: {len(ltx_by_version['2.5'])} Dateien."
        )

        with gr.Tab("Workflow laden"):
            with gr.Row():
                with gr.Column():
                    _build_load_track("A", wf_state_a, wf_orig_a)
                with gr.Column():
                    _build_load_track("B", wf_state_b, wf_orig_b)

        with gr.Tab("2.3 -> 2.5 Konvertierung"):
            with gr.Row():
                with gr.Column():
                    _build_convert_track("A", wf_state_a)
                with gr.Column():
                    _build_convert_track("B", wf_state_b)

        with gr.Tab("Fehleranalyse & GGUF-Optimierung"):
            gr.Markdown("Workflow (im Tab 'Workflow laden') laden, dann Log/Fehlermeldung einfuegen "
                        "oder Problem beschreiben, VRAM/Hardware waehlen, 'Analysieren'.")
            with gr.Row():
                with gr.Column():
                    _build_diagnostics_track("A", wf_state_a)
                with gr.Column():
                    _build_diagnostics_track("B", wf_state_b)

        with gr.Tab("Workflow modifizieren"):
            gr.Markdown("Bloeke abschalten - nodeschonend (Mute + Bypass, nichts wird geloescht). "
                        "Erst 'Bloeke erkennen', dann auswaehlen und 'Anwenden'.")
            with gr.Row():
                with gr.Column():
                    _build_modify_track("A", wf_state_a, wf_orig_a)
                with gr.Column():
                    _build_modify_track("B", wf_state_b, wf_orig_b)

        with gr.Tab("Modelle & GGUF"):
            with gr.Row():
                with gr.Column():
                    _build_model_track("A", wf_state_a)
                with gr.Column():
                    _build_model_track("B", wf_state_b)

        with gr.Tab("LTX Video"):
            with gr.Row():
                with gr.Column():
                    _build_ltx_track("A", wf_state_a)
                with gr.Column():
                    _build_ltx_track("B", wf_state_b)

        with gr.Tab("Prompt Verbesserung"):
            with gr.Row():
                with gr.Column():
                    _build_prompt_track("A", wf_state_a)
                with gr.Column():
                    _build_prompt_track("B", wf_state_b)

        with gr.Tab("Web-Suche (Tavily)"):
            gr.Markdown("Unabhaengig von Workflow A/B - fuer beide gemeinsam nutzbar.")
            tv_key = gr.Textbox(label="Tavily API Key", type="password")
            tv_query = gr.Textbox(label="Suchanfrage", value="ComfyUI LTX Video 2.5 workflow")
            tv_btn = gr.Button("Suchen", variant="primary")
            tv_result = gr.Textbox(label="Ergebnisse", lines=14, interactive=False)
            tv_btn.click(search_workflows_tavily, inputs=[tv_key, tv_query], outputs=[tv_result])

        with gr.Tab("Speichern & Export"):
            with gr.Row():
                with gr.Column():
                    _build_save_track("A", wf_state_a)
                with gr.Column():
                    _build_save_track("B", wf_state_b)

    return demo
