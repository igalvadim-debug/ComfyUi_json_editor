# workflow_modify.py - Modulare Workflow-Modifikation (Etappe 2, v3)
# Bloeke erkennen und abschalten OHNE den Graphen zu beschädigen.
#
# v3: Subgraph-Unterstuetzung. ComfyUI-Template-Workflows lagern Teile des
# Graphen in definitions.subgraphs aus (die UUID-Nodes im Top-Level sind
# nur Instanzen). Diese Subgraph-Definitionen haben EIGENE nodes/links-
# Listen und EIGENE Kopien der Set/Get-Bruecken (z.B. Get_upscale_model).
# Alle Block-Operationen arbeiten deshalb auf "Graph-Containern":
#   * Typ-basierte Block-Suche (audio/upscaler/patches) durchsucht ALLE
#     Container (Top-Level + jeder Subgraph).
#   * Die Bruecken-Kaskade wirkt UEBER Container-Grenzen: ist Set_X
#     deaktiviert, werden alle Get_X in allen Containern mitdeaktiviert.
#   * verify_modification prueft die Link-Integritaet und verlorene
#     Verbindungen in ALLEN Containern.
#
# v2-Basis (beibehalten): Bypass (mode 4) mit INTACTEN Links - ComfyUI
# reicht bei Bypass die Inputs positionsbasiert an die Outputs durch.
# Keine "Missing connection"-Warnungen, Reaktivieren ist schadlos.
# Ausnahme stage2 (Sampler-Kette): typ-validiertes Rewire auf den Decoder.

import copy

from workflow_io import normalize_workflow
from workflow_analysis import _node_type
from workflow_compare import compare_workflows

# ----------------------------------------------------------------------
#  Graph-Container: Top-Level + alle Subgraph-Definitionen
# ----------------------------------------------------------------------

def _graph_containers(wf):
    """Liefert [(nodes, links, label)] - label ist 0 fuer den Top-Level-
    Graphen, sonst die Subgraph-id. Die Listen sind Referenzen in wf."""
    containers = [(wf.get("nodes", []), wf.get("links", []), 0)]
    for sub in (wf.get("definitions") or {}).get("subgraphs") or []:
        if isinstance(sub, dict):
            containers.append((sub.get("nodes", []), sub.get("links", []), sub.get("id")))
    return containers


def count_deactivated(workflow):
    """Anzahl deaktivierter Nodes ueber ALLE Container."""
    wf = normalize_workflow(workflow)
    if not wf:
        return 0
    return sum(1 for nodes, _links, _lbl in _graph_containers(wf)
               for n in nodes if n.get("mode") in (2, 4))


def _type_ids(wf, types):
    """Alle (container_idx, node_id) mit Typ in types, ueber ALLE Container."""
    out = []
    for cidx, (nodes, _l, _label) in enumerate(_graph_containers(wf)):
        out += [(cidx, n.get("id")) for n in nodes if _node_type(n) in types]
    return out

# ----------------------------------------------------------------------
#  Link-Hilfsfunktionen (UI-Format: links + inputs/outputs synchron)
# ----------------------------------------------------------------------

def _by_id(nodes):
    return {n.get("id"): n for n in nodes}


def _remove_link(links, by_id, link_id):
    links[:] = [l for l in links if l[0] != link_id]
    for node in by_id.values():
        for out in node.get("outputs", []):
            out_links = out.get("links") or []
            if link_id in out_links:
                out_links.remove(link_id)
                out["links"] = out_links
        for inp in node.get("inputs", []):
            if inp.get("link") == link_id:
                inp["link"] = None


def _add_link(links, by_id, max_link, src_id, src_slot, tgt_id, tgt_slot, link_type):
    new_id = max_link + 1
    links.append([new_id, src_id, src_slot, tgt_id, tgt_slot, link_type])
    src_out = by_id[src_id]["outputs"][src_slot]
    src_out["links"] = (src_out.get("links") or []) + [new_id]
    by_id[tgt_id]["inputs"][tgt_slot]["link"] = new_id
    return new_id


def _input_slot(node, name):
    for i, inp in enumerate(node.get("inputs", [])):
        if inp.get("name") == name:
            return i, inp
    return None, None


def _link_tuple(l):
    """Normalisiert einen Link auf (id, from, from_slot, to, to_slot, type).
    Top-Level nutzt Listen, Subgraph-Links sind Dicts (id, origin_id,
    origin_slot, target_id, target_slot, type)."""
    if isinstance(l, dict):
        return (l.get("id"), l.get("origin_id"), l.get("origin_slot"),
                l.get("target_id"), l.get("target_slot"), l.get("type"))
    return l
# ----------------------------------------------------------------------
#  Kern: Bypass auf einer deepcopy anwenden
# ----------------------------------------------------------------------

def _apply(wf, mute_ids, rewires=None, bridges=None):
    """Setzt mute_ids (container_idx, node_id) auf BYPASS (mode 4) -
    Links bleiben intakt - wendet optionale typ-validierte rewires
    (Top-Level) an und deaktiviert Set/Get-Bruecken ueber alle
    Container. Rueckgabe (ok, report)."""
    report = []
    containers = _graph_containers(wf)
    top_nodes, top_links, _ = containers[0]
    by_id = _by_id(top_nodes)
    max_link = max((l[0] for l in top_links), default=0)

    # --- 1. Explizite Rewires (Top-Level: stage2, enhance_prompt) -----
    for src_id, src_slot, tgt_id, tgt_slot in (rewires or []):
        src, tgt = by_id.get(src_id), by_id.get(tgt_id)
        if src is None or tgt is None:
            report.append(f"FEHLER: Node {src_id} oder {tgt_id} nicht gefunden - Abbruch")
            return False, report
        if src_slot >= len(src.get("outputs", [])) or tgt_slot >= len(tgt.get("inputs", [])):
            report.append(f"FEHLER: Slot ausserhalb ({src_id}.out{src_slot} -> {tgt_id}.in{tgt_slot}) - Abbruch")
            return False, report
        out_type = src["outputs"][src_slot].get("type", "*")
        in_desc = tgt["inputs"][tgt_slot]
        in_type = in_desc.get("type", "*")
        if out_type not in ("*", "any") and in_type not in ("*", "any") and out_type != in_type:
            report.append(f"FEHLER: Typ-Mismatch {src.get('type')}({out_type}) -> "
                          f"{tgt.get('type')}.{in_desc.get('name')}({in_type}) - Abbruch")
            return False, report
        if in_desc.get("link") is not None:
            _remove_link(top_links, by_id, in_desc["link"])
        max_link = _add_link(top_links, by_id, max_link, src_id, src_slot, tgt_id, tgt_slot, out_type)
        report.append(f"Link {tgt.get('type')}.{in_desc.get('name')} umgeleitet: von {src.get('type')}")

    # --- 2. Bypass setzen, Links bleiben intakt ------------------------
    muted = {cidx: set() for cidx in range(len(containers))}
    for cidx, nid in mute_ids:
        if 0 <= cidx < len(containers):
            muted[cidx].add(nid)
    core_muted = {c: set(s) for c, s in muted.items()}
    for cidx, nids in muted.items():
        nodes_c = containers[cidx][0]
        for n in nodes_c:
            if n.get("id") in nids:
                n["mode"] = 4

    # --- 3. Bruecken-Kaskade ueber ALLE Container ----------------------
    # Set_X wird deaktiviert, wenn sein Eingang von einem KERN-deaktivierten
    # Node desselben Containers kommt (Set_upscale_model <- Loader).
    # Ist Set_X deaktiviert, folgen ALLE Get_X in ALLEN Containern
    # (die Subgraph-Kopien!) - sonst liefern sie weiter Werte und die
    # Szenen melden 'Missing connection'.
    while True:
        progressed = False
        for name in (bridges or []):
            for cidx, (nodes_c, links_c, _lbl) in enumerate(containers):
                for s in nodes_c:
                    if (_node_type(s) != "SetNode" or s.get("title") != f"Set_{name}"
                            or s.get("id") in muted[cidx] or not s.get("inputs")):
                        continue
                    sl = next((l for l in links_c if _link_tuple(l)[0] == s["inputs"][0].get("link")), None)
                    if sl is not None and _link_tuple(sl)[1] in core_muted[cidx]:
                        s["mode"] = 4
                        muted[cidx].add(s.get("id"))
                        progressed = True
            set_muted = any(
                _node_type(s) == "SetNode" and s.get("title") == f"Set_{name}"
                and s.get("id") in muted[cidx]
                for cidx, (nodes_c, _l, _lbl) in enumerate(containers)
                for s in nodes_c
            )
            if not set_muted:
                continue
            for cidx, (nodes_c, _l, _lbl) in enumerate(containers):
                for g in nodes_c:
                    if _node_type(g) == "GetNode" and g.get("title") == f"Get_{name}" \
                            and g.get("id") not in muted[cidx]:
                        g["mode"] = 4
                        muted[cidx].add(g.get("id"))
                        progressed = True
        if not progressed:
            break

    total = sum(len(s) for s in muted.values())
    report.append(f"{total} Node(s) auf BYPASS (mode 4) gesetzt - Links bleiben intakt, "
                  f"reaktivierbar ohne Schaden.")
    if isinstance(wf.get("extra"), dict) and "prompt" in wf["extra"]:
        del wf["extra"]["prompt"]
        report.append("Eingebettetes extra.prompt entfernt (wird beim Queue frisch generiert).")
    return True, report

# ----------------------------------------------------------------------
#  Block-Registry (erweiterbar: neuer Block = neue find()-Funktion)
#  find() liefert: (mute_ids [(cidx, nid)], rewires, bridges, hints)
# ----------------------------------------------------------------------

_AUDIO_TYPES = (
    "LTXVConcatAVLatent", "LTXVAudioVAEDecode", "MelBandRoFormerModelLoader",
    "MelBandRoFormerSampler", "ComfySwitchNode", "TrimAudioDuration",
    "LTXVAudioVAEEncode", "SetLatentNoiseMask", "LTXVEmptyLatentAudio",
    "LTXVAudioVAELoader", "VAEEncodeAudio",
    # bewusst NICHT LoadAudio: es speist Set_audio_original und damit das
    # Original-Audio an die VHS weiter (Musikvideo!)
)
_PATCH_TYPES = (
    "PathchSageAttentionKJ", "LTXVChunkFeedForward", "LTX2AttentionTunerPatch",
    "LTX2_NAG", "LTX2SamplingPreviewOverride",
)


def _find_audio(wf):
    mute = _type_ids(wf, _AUDIO_TYPES)
    # KEINE Bruecken muten: Get_audio_*-Bruecken speisen die VHS mit dem
    # Original-Audio weiter - nur die Erzeugung wird aus.
    bridges = []
    hints = ["Hinweis: Original-Audio (Get_Bruecke an VHS) wird weiterhin gemuxt - "
             "nur die Audio-Erzeugung ist deaktiviert."]
    return mute, [], bridges, hints


def _find_upscaler(wf):
    mute = _type_ids(wf, ("LTXVLatentUpsampler", "LatentUpscaleModelLoader"))
    return mute, [], ["upscale_model"], []


def _find_patches(wf):
    mute = _type_ids(wf, _PATCH_TYPES)
    # Hauptfluss-Bruecken (Set/Get_model_with_lora) bleiben AKTIV: das Modell
    # fliesst per Bypass-Durchreichung durch die Patches-Kette zum Guider.
    return mute, [], [], []


def _find_stage2(wf):
    """Zweite Sampling-Stufe (Top-Level): Sampler + Separator deaktivieren,
    das Video-Latent VOR der Stufe direkt auf den Decoder umleiten."""
    nodes, links = wf.get("nodes", []), wf.get("links", [])
    decoders = ("VAEDecode", "VAEDecodeTiled", "LTXVTiledVAEDecode")
    for sep in [n for n in nodes if _node_type(n) == "LTXVSeparateAVLatent"]:
        in_idx, _ = _input_slot(sep, "av_latent")
        if in_idx is None:
            continue
        src_link = next((l for l in links if l[0] == sep["inputs"][in_idx].get("link")), None)
        if src_link is None:
            continue
        sampler = next((x for x in nodes if x.get("id") == src_link[1]), None)
        if sampler is None or _node_type(sampler) != "SamplerCustomAdvanced":
            continue
        goes_to_decoder = any(
            _node_type(next((x for x in nodes if x.get("id") == l[3]), {}) or {}) in decoders
            for l in links if l[1] == sep.get("id")
        )
        if not goes_to_decoder:
            continue
        lat_idx, _ = _input_slot(sampler, "latent_image")
        src = None
        if lat_idx is not None:
            sl = next((l for l in links if l[0] == sampler["inputs"][lat_idx].get("link")), None)
            if sl is not None:
                src = (sl[1], sl[2])
        if src is None:
            continue
        targets = [(l[3], l[4]) for l in links if l[1] == sep.get("id")]
        rewires = [(*src, t[0], t[1]) for t in targets]
        hints = ["Hinweis: Guider/Noise/Sigmas-Bruecken der Stufe bleiben aktiv (ungenutzt)."]
        return [(0, sep.get("id")), (0, sampler.get("id"))], rewires, [], hints
    return [], [], [], []


def _find_outputs(wf):
    """Alle VHS_VideoCombine (Top-Level) ausser dem 'finalen' deaktivieren."""
    nodes = wf.get("nodes", [])
    vhs = [n for n in nodes if _node_type(n) == "VHS_VideoCombine"]
    if not vhs:
        return [], [], [], []
    final = next((n for n in vhs if "final" in str(n.get("title") or "").lower()), vhs[-1])
    mute = [(0, n.get("id")) for n in vhs if n.get("id") != final.get("id")]
    return mute, [], [], [f"{len(vhs)} Video-Outputs - '{final.get('title') or final.get('id')}' bleibt aktiv."]


def _find_enhance_prompt(wf):
    """Prompt-Enhancer-Bruecke (Top-Level) umgehen: Set-Quelle direkt an
    Get-Konsumenten."""
    nodes, links = wf.get("nodes", []), wf.get("links", [])
    sets = [n for n in nodes if _node_type(n) == "SetNode" and n.get("title") == "Set_enhance_prompt"]
    gets = [n for n in nodes if _node_type(n) == "GetNode" and n.get("title") == "Get_enhance_prompt"]
    if not sets or not gets:
        return [], [], [], []
    src_link = next((l for l in links if l[0] == sets[0]["inputs"][0].get("link")), None)
    if src_link is None:
        return [], [], [], []
    rewires = [(src_link[1], src_link[2], l[3], l[4])
               for g in gets
               for l in links if l[1] == g.get("id")]
    return [], rewires, [], ["enhance_prompt: Set-Quelle direkt an Get-Konsumenten verdrahtet."]


BLOCK_DEFS = {
    "audio":          {"label": "Audio (Vocals-Trennung + generiertes Audio)", "find": _find_audio},
    "upscaler":       {"label": "Upscaler (LTXVLatentUpsampler)", "find": _find_upscaler},
    "patches":        {"label": "Modell-Patches (SageAttention/Chunk/NAG/Tuner)", "find": _find_patches},
    "stage2":         {"label": "2. Sampling-Stufe (Refiner-Pass)", "find": _find_stage2},
    "outputs":        {"label": "Video-Outputs (nur finalen behalten)", "find": _find_outputs},
    "enhance_prompt": {"label": "Prompt-Enhancer-Bruecke", "find": _find_enhance_prompt},
}

# ----------------------------------------------------------------------
#  Oeffentliche API
# ----------------------------------------------------------------------

def detect_blocks(workflow):
    wf = normalize_workflow(workflow)
    if not wf or "nodes" not in wf:
        return {}
    result = {}
    for name, spec in BLOCK_DEFS.items():
        mute, rewires, bridges, _ = spec["find"](wf)
        result[name] = {
            "label": spec["label"],
            "found": bool(mute or rewires),
            "nodes": len(set(nid for _cidx, nid in mute)),
        }
    return result


def _run_block(wf, name):
    spec = BLOCK_DEFS.get(name)
    if not spec:
        return False, [f"Unbekannter Block: '{name}'"]
    mute, rewires, bridges, hints = spec["find"](wf)
    if not mute and not rewires:
        return True, [f"Block '{name}': nicht im Workflow gefunden (uebersprungen)"]
    ok, report = _apply(wf, mute, rewires, bridges)
    if ok:
        report = [f"--- {name} ---"] + report + hints
    else:
        report = [f"--- {name} ---"] + report + ["ABBRUCH an dieser Stelle."]
    return ok, report


def disable_block(workflow, name):
    wf = copy.deepcopy(workflow)
    ok, report = _run_block(wf, name)
    if not ok:
        return wf, report + ["Workflow unveraendert gelassen (nur deepcopy beruehrt)."]
    return wf, [f"Block '{name}' deaktiviert."] + report


def apply_mods(workflow, mods):
    """Wendet mehrere Modifikationen auf EINER deepcopy an."""
    wf = copy.deepcopy(workflow)
    full_report = []
    for name, enabled in mods.items():
        if not enabled:
            continue
        ok, report = _run_block(wf, name)
        full_report += report
        if not ok:
            break
    return wf, full_report


def verify_modification(original, modified):
    """Verifiziert die Modifikation ueber ALLE Graph-Container:
      (a) workflow_compare (Top-Level): keine Nodes geaendert/entfernt/neu,
      (b) Link-Integritaet in jedem Container (Top-Level + Subgraphs),
      (c) kein AKTIVER Node (egal welcher Ebene) hat einen Input verloren,
          der im Original verbunden war ('Missing connection'-Fruehwarnung).
    Rueckgabe (ok, message)."""
    problems = []

    comp = compare_workflows(original, modified, "Original", "Modifiziert")
    if comp.get("error"):
        problems.append(f"Vergleich fehlgeschlagen: {comp['error']}")
    if comp.get("added") or comp.get("removed"):
        problems.append(f"Nodes geaendert: +{len(comp['added'])} / -{len(comp['removed'])}")
    if comp.get("changed"):
        problems.append(f"Node-Parameter geaendert: {len(comp['changed'])}")

    wf = normalize_workflow(modified)
    orig = normalize_workflow(original)
    if not wf or "nodes" not in wf or not orig or "nodes" not in orig:
        return False, "Workflow unlesbar."

    new_containers = _graph_containers(wf)
    old_containers = _graph_containers(orig)

    # (b) + (c) pro Container; Subgraph-Container ueber die id matchen
    for new_nodes, new_links, label in new_containers:
        by_id = {n.get("id"): n for n in new_nodes}
        link_ids = set()
        for l in new_links:
            lid, l_from, l_fslot, l_to, l_tslot, _ltype = _link_tuple(l)
            link_ids.add(lid)
            # Negative IDs sind Subgraph-Grenz-Ports (origin_id=-10 =
            # Subgraph-Eingang, target_id=-20 = Subgraph-Ausgang) - valide.
            if l_from >= 0 and l_from not in by_id:
                problems.append(f"[{label}] Link {lid}: Quelle {l_from} existiert nicht.")
            if l_to >= 0 and l_to not in by_id:
                problems.append(f"[{label}] Link {lid}: Ziel {l_to} existiert nicht.")
            if l_to >= 0 and l_to in by_id and l_tslot >= len(by_id[l_to].get("inputs", [])):
                problems.append(f"[{label}] Link {lid}: Ziel-Input {l_tslot} ausserhalb.")
        for n in new_nodes:
            for i, inp in enumerate(n.get("inputs", [])):
                if inp.get("link") is not None and inp["link"] not in link_ids:
                    problems.append(f"[{label}] Node {n.get('id')}: Input {i} verweist auf toten Link {inp['link']}.")

    # (c) aktive Nodes duerfen keine vorher verbundenen Inputs verlieren
    for new_nodes, _l, label in new_containers:
        old_nodes = next((on for on, _ol, o_label in old_containers
                          if o_label == label), None)
        if old_nodes is None:
            problems.append(f"[{label}] Subgraph fehlt im Original - Struktur veraendert?")
            continue
        old_by_id = {n.get("id"): n for n in old_nodes}
        for n in new_nodes:
            if n.get("mode") == 4:
                continue
            old_node = old_by_id.get(n.get("id"))
            if not old_node:
                continue
            for i, inp in enumerate(n.get("inputs", [])):
                o_inp = old_node.get("inputs", [])[i] if i < len(old_node.get("inputs", [])) else {}
                if o_inp.get("link") is not None and inp.get("link") is None:
                    problems.append(
                        f"[{label}] Node {n.get('id')} ({n.get('type')}): aktiver Input "
                        f"'{inp.get('name')}' hat seine Verbindung verloren."
                    )

    if problems:
        return False, "\n".join(problems[:20])
    return True, ("Verify OK: keine Nodes geaendert, alle Links konsistent "
                  "(inkl. Subgraphs), keine verlorenen Verbindungen an aktiven Nodes.")
