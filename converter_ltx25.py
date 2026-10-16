# converter_ltx25.py - Konvertiert einen LTX-2.3-Workflow in
# einen LTX-2.5-Workflow, basierend auf dem Vergleich der echten
# Beispiel-Workflows in example_workflows/2.3 und /2.5.
#
# Prinzip: Regelbasiert (schnell, kostenlos, deterministisch).
# Nur wenn etwas nicht eindeutig zuordenbar ist, wird das im
# Report vermerkt - eine LLM-Zweitmeinung (minimax H3) kann optional
# in llm_analysis.py dazu befragt werden, AUTOMATISCH aendert die
# KI dabei nichts am Workflow (Sicherheit / Nachvollziehbarkeit).

import copy

from config import (
    LTX23_TO_LTX25_NODE_RENAME,
    CHECKPOINT_LOADER_TYPES,
    LTX_VERSION_HINTS,
)
from workflow_analysis import get_node_param, set_node_param, find_max_id, _node_type
from model_manager import scan_ltx_models_by_version, find_25_equivalent


def _is_2_3_filename(text):
    if not isinstance(text, str):
        return False
    low = text.lower()
    return any(h in low for h in LTX_VERSION_HINTS["2.3"])


def _new_loader_node(node_id, node_type, title, widgets_values, output_name, output_type):
    return {
        "id": node_id,
        "type": node_type,
        "pos": [0, 0],
        "size": [340, 106],
        "flags": {},
        "order": 0,
        "mode": 0,
        "inputs": [],
        "outputs": [{"name": output_name, "type": output_type, "links": []}],
        "properties": {"Node name for S&R": node_type},
        "widgets_values": widgets_values,
        "title": title,
    }


# ═══════════════════════════════════════════════════════════════
#  Schritt 1: veraltete/umbenannte Node-Typen ersetzen
# ═══════════════════════════════════════════════════════════════

def rename_deprecated_nodes(workflow):
    report = []
    for node in workflow.get("nodes", []):
        old_type = _node_type(node)
        new_type = LTX23_TO_LTX25_NODE_RENAME.get(old_type)
        if new_type:
            node["type"] = new_type
            if "class_type" in node:
                node["class_type"] = new_type
            report.append(f"Node {node.get('id')}: '{old_type}' -> '{new_type}' umbenannt")
    return workflow, report


# ═══════════════════════════════════════════════════════════════
#  Schritt 2: CheckpointLoaderSimple -> UNET + CLIP + 2x VAE
# ═══════════════════════════════════════════════════════════════

def replace_checkpoint_with_split_loaders(workflow, yaml_paths=None):
    report = []
    nodes = workflow.get("nodes", [])
    links = workflow.get("links", [])

    ckpt_nodes = [n for n in nodes if _node_type(n) in CHECKPOINT_LOADER_TYPES]
    if not ckpt_nodes:
        return workflow, report

    next_id = find_max_id(nodes) + 1000
    ltx25_catalog = scan_ltx_models_by_version(yaml_paths)["2.5"] if yaml_paths else []

    new_nodes = [n for n in nodes if n not in ckpt_nodes]
    new_links = list(links)

    for ckpt in ckpt_nodes:
        old_id = ckpt.get("id")
        ckpt_name = get_node_param(ckpt, "ckpt_name") or ""

        if yaml_paths:
            # Kategorie-bewusst suchen: ein UNETLoader darf niemals ein
            # CLIP/Gemma-Modell zugewiesen bekommen (CATEGORY_KEYWORDS
            # in model_manager.py filtert auf transformer/unet/distilled).
            guess_25, _ = find_25_equivalent(ckpt_name, yaml_paths, category="transformer")
        else:
            guess_25 = None

        unet_name = guess_25 or (ltx25_catalog[0] if ltx25_catalog else "BITTE_2.5_UNET_WAEHLEN.safetensors")
        clip_name = next((c for c in ltx25_catalog if "clip" in c.lower() or "gemma" in c.lower()),
                          "BITTE_2.5_CLIP_WAEHLEN.safetensors")
        vae_video_name = next(
            (c for c in ltx25_catalog if "vae" in c.lower() and "video" in c.lower()),
            next((c for c in ltx25_catalog if "vae" in c.lower() and "audio" not in c.lower()),
                 "BITTE_2.5_VAE_WAEHLEN.safetensors"))
        vae_audio_name = next(
            (c for c in ltx25_catalog if "vae" in c.lower() and "audio" in c.lower()),
            "BITTE_2.5_VAE_WAEHLEN.safetensors")

        unet_id, clip_id, vae_v_id, vae_a_id = next_id, next_id + 1, next_id + 2, next_id + 3
        next_id += 10

        new_nodes.append(_new_loader_node(
            unet_id, "UNETLoader", "Load Diffusion Model",
            [unet_name, "default"], "MODEL", "MODEL"))
        new_nodes.append(_new_loader_node(
            clip_id, "CLIPLoader", "Load CLIP - Text Encoder",
            [clip_name, "ltxv", "default"], "CLIP", "CLIP"))
        new_nodes.append(_new_loader_node(
            vae_v_id, "VAELoader", "Load VAE - Video",
            [vae_video_name], "VAE", "VAE"))
        new_nodes.append(_new_loader_node(
            vae_a_id, "VAELoader", "Load VAE - Audio",
            [vae_audio_name], "VAE", "VAE"))

        # Links umbiegen: alte Checkpoint-Outputs waren
        # slot 0 = MODEL, slot 1 = CLIP, slot 2 = VAE
        rewired = []
        for lnk in new_links:
            if len(lnk) < 5 or lnk[1] != old_id:
                rewired.append(lnk)
                continue
            lid, fn, fs, tn, ts, *extra = lnk
            target_node = next((n for n in new_nodes if n.get("id") == tn), None)
            target_type = _node_type(target_node) if target_node else ""

            if fs == 0:
                new_fn = unet_id
            elif fs == 1:
                new_fn = clip_id
            elif fs == 2:
                new_fn = vae_a_id if "audio" in target_type.lower() else vae_v_id
            else:
                new_fn = unet_id
            rewired.append([lid, new_fn, 0, tn, ts] + extra)
        new_links = rewired

        report.append(
            f"Checkpoint-Node {old_id} ('{ckpt_name}') ersetzt durch "
            f"'Load Diffusion Model' (UNETLoader, id {unet_id}), "
            f"'Load CLIP - Text Encoder' (CLIPLoader, id {clip_id}), "
            f"2x 'Load VAE' (VAELoader, id {vae_v_id}/{vae_a_id})."
        )
        if guess_25 is None:
            report.append(
                f"  ACHTUNG: Fuer '{ckpt_name}' konnte keine 2.5-Datei automatisch "
                f"gefunden werden - bitte UNET/CLIP/VAE-Dateinamen manuell im "
                f"Modell-Tab pruefen und setzen."
            )

    workflow["nodes"] = new_nodes
    workflow["links"] = new_links
    return workflow, report


# ═══════════════════════════════════════════════════════════════
#  Schritt 3: uebrige 2.3-Dateinamen (Audio-VAE, Text-Encoder,
#  Gemma-Modellreferenzen) auf 2.5 umbiegen, wo moeglich
# ═══════════════════════════════════════════════════════════════

def rename_leftover_model_filenames(workflow, yaml_paths=None):
    report = []
    for node in workflow.get("nodes", []):
        wv = node.get("widgets_values")
        if not isinstance(wv, list):
            continue
        for i, val in enumerate(wv):
            if _is_2_3_filename(val):
                guess_25, _ = find_25_equivalent(val, yaml_paths) if yaml_paths else (None, [])
                if guess_25:
                    wv[i] = guess_25
                    report.append(f"Node {node.get('id')} ({_node_type(node)}): '{val}' -> '{guess_25}'")
                else:
                    report.append(
                        f"Node {node.get('id')} ({_node_type(node)}): '{val}' referenziert "
                        f"LTX 2.3 - bitte manuell auf ein 2.5-Modell umstellen."
                    )
    return workflow, report


# ═══════════════════════════════════════════════════════════════
#  Schritt 4: 2.3-LoRAs sind mit 2.5-Modellen inkompatibel ->
#  stummschalten (mode=4) statt loeschen, damit nichts verloren geht
# ═══════════════════════════════════════════════════════════════

def flag_incompatible_loras(workflow):
    report = []
    for node in workflow.get("nodes", []):
        ctype = _node_type(node)
        if "Lora" not in ctype:
            continue
        lora_name = get_node_param(node, "lora_name") or ""
        if _is_2_3_filename(lora_name):
            node["mode"] = 4  # in ComfyUI: Node stummgeschaltet/bypassed
            report.append(
                f"Node {node.get('id')} ({ctype}): LoRA '{lora_name}' ist auf LTX 2.3 "
                f"trainiert und mit 2.5 NICHT kompatibel - Node wurde stummgeschaltet "
                f"(mode=4). Neu trainierte 2.5-LoRA manuell eintragen, falls vorhanden."
            )
    return workflow, report


# ═══════════════════════════════════════════════════════════════
#  Hauptfunktion
# ═══════════════════════════════════════════════════════════════

def convert_workflow_2_3_to_2_5(workflow, yaml_paths=None):
    """Fuehrt alle Konvertierungsschritte aus und gibt
    (neuer_workflow, report_liste) zurueck. Aendert NIE das
    Original-Objekt (deepcopy)."""
    if not workflow or "nodes" not in workflow:
        return workflow, ["Kein Workflow geladen."]

    wf = copy.deepcopy(workflow)
    full_report = []

    wf, r = rename_deprecated_nodes(wf)
    full_report += r
    wf, r = replace_checkpoint_with_split_loaders(wf, yaml_paths)
    full_report += r
    wf, r = rename_leftover_model_filenames(wf, yaml_paths)
    full_report += r
    wf, r = flag_incompatible_loras(wf)
    full_report += r

    if not full_report:
        full_report.append(
            "Keine LTX-2.3-spezifischen Elemente gefunden - Workflow scheint "
            "bereits 2.5-kompatibel zu sein (oder ist kein LTX-Workflow)."
        )

    return wf, full_report


def build_llm_context(workflow, report):
    """Baut eine kompakte Textzusammenfassung fuer eine optionale
    LLM-Zweitmeinung (minimax H3) - z.B. bei unklaren/manuellen Punkten."""
    nodes = workflow.get("nodes", []) if workflow else []
    types = sorted(set(_node_type(n) for n in nodes))
    lines = [
        f"ComfyUI-Workflow mit {len(nodes)} Nodes.",
        "Node-Typen: " + ", ".join(types),
        "",
        "Bisherige automatische Konvertierungsschritte (2.3 -> 2.5):",
    ]
    lines += [f"- {line}" for line in report]
    return "\n".join(lines)
