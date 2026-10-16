# transformers.py - GGUF-Konvertierung, LTX-Schnelleinstellungen,
# Prompt-Einsetzen. Nutzt get_node_param/set_node_param aus
# workflow_analysis.py, damit es sowohl mit widgets_values
# (Workflow-Format) als auch mit direkten inputs-Werten
# (API-Format) funktioniert.

import copy

from workflow_analysis import get_node_param, set_node_param, find_max_id, _node_type


def replace_checkpoint_with_gguf(workflow, unet_name, clip1_name, clip2_name, vae_name, clip_type="flux"):
    """Ersetzt CheckpointLoader/UNETLoader durch UNET-GGUF + DualCLIPLoaderGGUF + VAELoader."""
    if not workflow or "nodes" not in workflow:
        return workflow, "Kein Workflow geladen"

    wf = copy.deepcopy(workflow)
    nodes = wf["nodes"]
    links = wf.get("links", [])
    next_id = find_max_id(nodes) + 1000
    replaced = 0
    new_nodes = []

    for node in nodes:
        ctype = _node_type(node)
        if ctype in ("CheckpointLoaderSimple", "CheckpointLoader", "UNETLoader"):
            old_id  = node.get("id")
            unet_id = next_id
            clip_id = next_id + 1
            vae_id  = next_id + 2
            next_id += 10
            new_nodes += [
                {
                    "id": unet_id, "type": "UnetLoaderGGUF", "pos": [0, 0], "size": [340, 60],
                    "flags": {}, "order": 0, "mode": 0, "inputs": [],
                    "outputs": [{"name": "MODEL", "type": "MODEL", "links": []}],
                    "properties": {"Node name for S&R": "UnetLoaderGGUF"},
                    "widgets_values": [unet_name], "title": "UNET GGUF",
                },
                {
                    "id": clip_id, "type": "DualCLIPLoaderGGUF", "pos": [0, 0], "size": [340, 100],
                    "flags": {}, "order": 0, "mode": 0, "inputs": [],
                    "outputs": [{"name": "CLIP", "type": "CLIP", "links": []}],
                    "properties": {"Node name for S&R": "DualCLIPLoaderGGUF"},
                    "widgets_values": [clip1_name, clip2_name, clip_type], "title": "Dual CLIP GGUF",
                },
                {
                    "id": vae_id, "type": "VAELoader", "pos": [0, 0], "size": [340, 60],
                    "flags": {}, "order": 0, "mode": 0, "inputs": [],
                    "outputs": [{"name": "VAE", "type": "VAE", "links": []}],
                    "properties": {"Node name for S&R": "VAELoader"},
                    "widgets_values": [vae_name], "title": "VAE",
                },
            ]

            new_links = []
            for lnk in links:
                if len(lnk) < 5 or lnk[1] != old_id:
                    new_links.append(lnk)
                    continue
                lid, fn, fs, tn, ts, *extra = lnk
                new_fn = {0: unet_id, 1: clip_id, 2: vae_id}.get(fs, unet_id)
                new_links.append([lid, new_fn, 0, tn, ts] + extra)
            links = new_links
            replaced += 1
        else:
            new_nodes.append(node)

    wf["nodes"] = new_nodes
    wf["links"] = links
    return wf, f"{replaced} Checkpoint(s) -> GGUF-Trio ersetzt"


def apply_ltx_settings(workflow, width, height, frames, fps, steps, cfg, shift, with_audio):
    """Schreibt LTX-Parameter in die passenden Nodes (width/height/
    frames, steps/cfg, shift, fps)."""
    if not workflow or "nodes" not in workflow:
        return workflow, "Kein Workflow geladen"

    wf = copy.deepcopy(workflow)
    changed = []

    for node in wf["nodes"]:
        ctype = _node_type(node)

        if ctype in ("LTXVideoLatentImage", "EmptyLTXVLatentVideo", "EmptyLatentImage"):
            if get_node_param(node, "width") is not None:
                set_node_param(node, "width", int(width)); changed.append(f"width={width}")
            if get_node_param(node, "height") is not None:
                set_node_param(node, "height", int(height)); changed.append(f"height={height}")
            if get_node_param(node, "length") is not None:
                set_node_param(node, "length", int(frames)); changed.append(f"frames={frames}")

        if ctype == "LTXVEmptyLatentAudio":
            if get_node_param(node, "frames_number") is not None:
                set_node_param(node, "frames_number", int(frames))
            if get_node_param(node, "frame_rate") is not None:
                set_node_param(node, "frame_rate", int(fps))

        if ctype in ("KSampler", "KSamplerAdvanced", "LTXVSampler"):
            if get_node_param(node, "steps") is not None:
                set_node_param(node, "steps", int(steps)); changed.append(f"steps={steps}")
            if get_node_param(node, "cfg") is not None:
                set_node_param(node, "cfg", float(cfg)); changed.append(f"cfg={cfg}")

        if "sigmas" in ctype.lower() or ctype in ("LTX23Sigmas", "LTXVScheduler"):
            if get_node_param(node, "shift") is not None:
                set_node_param(node, "shift", float(shift)); changed.append(f"shift={shift}")

        if ctype in ("VHS_VideoCombine", "SaveAnimatedWEBP", "CreateVideo"):
            if get_node_param(node, "frame_rate") is not None:
                set_node_param(node, "frame_rate", int(fps)); changed.append(f"fps={fps}")
            if get_node_param(node, "fps") is not None:
                set_node_param(node, "fps", int(fps)); changed.append(f"fps={fps}")

    msg = f"Gesetzt: {', '.join(sorted(set(changed)))}" if changed else "Keine passenden LTX-Nodes gefunden"
    return wf, msg


def apply_improved_prompt_to_workflow(workflow, improved_text, target="positive"):
    """Schreibt einen verbesserten Prompt in Positive- oder
    Negative-CLIPTextEncode-Nodes."""
    if not workflow or not improved_text.strip():
        return workflow, "Kein Workflow oder Prompt leer"

    wf = copy.deepcopy(workflow)
    changed = 0
    for node in wf["nodes"]:
        ctype = _node_type(node)
        title = str(node.get("title", "") or "").lower()
        if ctype in ("CLIPTextEncode", "CLIPTextEncodeSDXL", "smZ CLIPTextEncode"):
            is_neg = "neg" in title or "negativ" in title
            if (target == "positive" and not is_neg) or (target == "negative" and is_neg):
                set_node_param(node, "text", improved_text)
                changed += 1
    return wf, f"{changed} {target.capitalize()}-Node(s) aktualisiert"
