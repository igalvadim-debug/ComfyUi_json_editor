# diagnostics.py - Fehleranalyse (ComfyUI-Logs/Fehlermeldungen) +
# GGUF-Optimierungsempfehlungen fuer gegebenes VRAM.
#
# Zwei Ebenen:
#   1) Regelbasiert (kostenlos, sofort): bekannte Fehlermuster per
#      Regex + strukturelle Workflow-Pruefung (Checkpoint vs. VRAM,
#      fehlendes Tiled-VAE-Decode, Aufloesung/Batch zu hoch usw.)
#   2) Optional Agnes AI fuer Faelle, die die Regeln nicht abdecken.
#
# GGUF-Groessenschaetzung basiert auf echten Dateigroessen aus
# LTX-2-19B-GGUF-Community-Repos (HuggingFace) und wird linear auf
# 22B (LTX 2.3/2.5) hochskaliert - SCHAETZUNG, keine Garantie.

import re

from config import (
    LTX_VERSION_HINTS, LTX_PARAMS_B, GGUF_BITS_PER_PARAM,
    GGUF_QUANT_ORDER, GGUF_QUANT_QUALITY_NOTE, GPU_VRAM_PRESETS,
)
from workflow_analysis import _node_type, get_node_param


# ═══════════════════════════════════════════════════════════════
#  1) Bekannte Fehlermuster (Regex -> Erklaerung + Empfehlung)
# ═══════════════════════════════════════════════════════════════

KNOWN_ERROR_PATTERNS = [
    (
        r"CUDA out of memory|OutOfMemoryError|torch\.cuda\.OutOfMemoryError",
        "VRAM-Ueberlauf (Out of Memory)",
        "Die GPU hatte nicht genug freien Speicher fuer diesen Schritt. "
        "Loesungen (in Reihenfolge): (1) kleineres GGUF-Quant fuer das "
        "Modell verwenden, (2) Aufloesung/Frames/Batch-Size reduzieren, "
        "(3) VAEDecodeTiled statt normalem VAEDecode nutzen, (4) andere "
        "GPU-Programme schliessen, (5) --lowvram / --novram Startparameter "
        "in ComfyUI pruefen.",
    ),
    (
        r"Error\(s\) in loading state_dict|size mismatch for",
        "Checkpoint/Version-Mismatch beim Laden",
        "Die Gewichte in der Datei passen nicht zur erwarteten Node-Struktur "
        "- meist weil eine LTX-2.3-Datei mit einem 2.5-Loader (oder "
        "umgekehrt) kombiniert wurde, oder eine LoRA auf das falsche "
        "Basismodell trifft. Pruefe, ob Checkpoint/UNET/CLIP/VAE/LoRA alle "
        "zur selben LTX-Version passen (Dateiname enthaelt meist die Version).",
    ),
    (
        r"not registered|Node type not found|NODE_CLASS_MAPPINGS",
        "Fehlender Custom Node",
        "Ein Node-Typ aus dem Workflow ist in deiner ComfyUI-Installation "
        "nicht registriert - die zugehoerige Custom-Node-Extension fehlt "
        "oder ist veraltet. Pruefe im ComfyUI-Manager, ob alle benoetigten "
        "Node-Packs (z.B. ComfyUI-GGUF, ComfyUI-LTXVideo, rgthree) "
        "installiert und aktuell sind.",
    ),
    (
        r"No module named",
        "Fehlendes Python-Paket",
        "Eine Custom-Node-Extension braucht ein Python-Paket, das in "
        "deiner venv fehlt. Meist hilft: die requirements.txt der "
        "jeweiligen Extension im ComfyUI-venv mit "
        "'pip install -r requirements.txt' nachinstallieren.",
    ),
    (
        r"expected all tensors to be on the same device|Expected all tensors on the same device",
        "Geraete-Mismatch (CPU/GPU)",
        "Es wird versucht, Tensoren von CPU und GPU (oder unterschiedlichen "
        "GPUs) zu verrechnen. Kommt oft vor, wenn GGUF- und Nicht-GGUF-Teile "
        "gemischt werden, oder bei Offloading-Problemen. Pruefe, ob "
        "UNET/CLIP/VAE konsistent ueber die GGUF- bzw. Standard-Loader "
        "geladen werden (nicht gemischt).",
    ),
    (
        r"invalid load key|PytorchStreamReader failed|unexpected EOF|failed finding central directory",
        "Beschaedigte/unvollstaendige Modelldatei",
        "Die Modell-Datei ist wahrscheinlich unvollstaendig heruntergeladen "
        "oder beschaedigt. Datei erneut herunterladen und Dateigroesse mit "
        "der Angabe auf HuggingFace vergleichen.",
    ),
    (
        r"mat1 and mat2 shapes cannot be multiplied|shape.*invalid for input of size|Sizes of tensors must match",
        "Shape-/Dimensions-Fehler",
        "Zwei Tensoren passen nicht zusammen - typische Ursachen: Breite/"
        "Hoehe nicht durch 32 teilbar (LTX-Standard), ein Node aus einer "
        "anderen LTX-Version im Graph, oder ein falsch verbundener Node "
        "nach manuellem Umbau des Workflows.",
    ),
    (
        r"Prompt outputs failed validation|Return type mismatch|Required input is missing",
        "Workflow-Validierungsfehler",
        "ComfyUI hat den Graphen vor dem Start abgelehnt - ein Pflicht-Input "
        "fehlt oder ein Node bekommt den falschen Typ geliefert. Haeufig "
        "nach manuellem Bearbeiten der JSON oder nach einem Block-Toggle, "
        "der eine Verbindung verloren hat (siehe 'Missing connection' in "
        "ComfyUI selbst).",
    ),
    (
        r"split_with_sizes",
        "AV-Transformer-Architektur-Fehler (LTX 2.3/2.5)",
        "Dieser Fehler tritt typischerweise auf, wenn ein LTX-2.3/2.5-"
        "Checkpoint (Audio+Video-Transformer) faelschlich mit einer "
        "Pipeline/einem Workflow fuer das aeltere LTX-2 19B (nur Video) "
        "kombiniert wird. Pruefe, ob Workflow-Version und Modell-Version "
        "wirklich zusammenpassen.",
    ),
]

_VRAM_LINE = re.compile(r"Total VRAM (\d+) MB")
_GPU_LINE = re.compile(r"Device: cuda:\d+\s+(.+?)\s*:")
_LOWVRAM_LINE = re.compile(r"Set vram state to:\s*(\w+)")


def parse_comfyui_log(log_text):
    """Durchsucht einen eingefuegten ComfyUI-Log/Fehlertext nach
    bekannten Mustern UND extrahiert nebenbei VRAM/GPU-Infos, falls
    vorhanden (z.B. aus den Start-Logzeilen). Gibt
    (gefundene_probleme[(titel, erklaerung)], erkannt_vram_mb, erkannter_gpu_name,
    erkannter_vram_modus) zurueck."""
    if not log_text or not log_text.strip():
        return [], None, None, None

    found = []
    for pattern, title, advice in KNOWN_ERROR_PATTERNS:
        if re.search(pattern, log_text, re.IGNORECASE):
            found.append((title, advice))

    vram_mb = None
    m = _VRAM_LINE.search(log_text)
    if m:
        vram_mb = int(m.group(1))

    gpu_name = None
    m = _GPU_LINE.search(log_text)
    if m:
        gpu_name = m.group(1).strip()

    vram_mode = None
    m = _LOWVRAM_LINE.search(log_text)
    if m:
        vram_mode = m.group(1)

    return found, vram_mb, gpu_name, vram_mode


# ═══════════════════════════════════════════════════════════════
#  2) Strukturelle Workflow-Pruefung (Aufloesung, Tiled-Decode,
#     GGUF vs. Nicht-GGUF, Batch-Size)
# ═══════════════════════════════════════════════════════════════

def _all_nodes(workflow):
    """Sammelt Top-Level-Nodes und Nodes der direkten Workflow-Subgraphs."""
    nodes = list(workflow.get("nodes", []))
    for sg in (workflow.get("definitions") or {}).get("subgraphs") or []:
        nodes += sg.get("nodes", [])
    return nodes


def analyze_workflow_structure(workflow, vram_gb):
    """Rein strukturelle Pruefungen am Workflow selbst - unabhaengig
    von einer konkreten Fehlermeldung. Gibt eine Liste von
    (schweregrad, text) zurueck. schweregrad: 'warn' oder 'info'."""
    if not workflow or "nodes" not in workflow:
        return []

    nodes = _all_nodes(workflow)
    issues = []

    heavy_loaders = [n for n in nodes if _node_type(n) in
                     ("CheckpointLoaderSimple", "UNETLoader")]
    gguf_loaders = [n for n in nodes if _node_type(n) in
                    ("UnetLoaderGGUF", "DualCLIPLoaderGGUF")]
    if heavy_loaders and not gguf_loaders and vram_gb <= 12:
        names = ", ".join(
            str(get_node_param(n, "ckpt_name") or get_node_param(n, "unet_name") or "?")
            for n in heavy_loaders
        )
        issues.append(("warn",
            f"Es wird ein NICHT-quantisiertes Modell geladen ({names}) bei "
            f"nur {vram_gb} GB VRAM. Fuer LTX-2/2.3/2.5 auf {vram_gb} GB "
            f"empfiehlt sich ein GGUF-Quant (siehe Empfehlung unten)."
        ))

    has_tiled = any(_node_type(n) in ("VAEDecodeTiled", "LTXVTiledVAEDecode") for n in nodes)
    has_plain_decode = any(_node_type(n) == "VAEDecode" for n in nodes)
    if has_plain_decode and not has_tiled and vram_gb <= 12:
        issues.append(("warn",
            "Kein Tiled-VAE-Decode (VAEDecodeTiled) im Workflow gefunden, aber "
            f"nur {vram_gb} GB VRAM - das Decodieren grosser/lange Videos kann "
            "hier zum OOM-Kandidaten Nr. 1 werden. VAEDecode gegen "
            "VAEDecodeTiled tauschen."
        ))

    for n in nodes:
        if _node_type(n) in ("EmptyLTXVLatentVideo",):
            w = get_node_param(n, "width")
            h = get_node_param(n, "height")
            length = get_node_param(n, "length")
            batch = get_node_param(n, "batch_size")
            if isinstance(w, int) and isinstance(h, int) and vram_gb <= 8 and w * h > 768 * 512:
                issues.append(("warn",
                    f"Node {n.get('id')}: Aufloesung {w}x{h} ist bei {vram_gb} GB VRAM "
                    "grosszuegig. Fuer stabile Laeufe eher <= 768x512 anpeilen "
                    "(oder Tiled-Decode + kleines GGUF-Quant einsetzen)."
                ))
            if isinstance(length, int) and vram_gb <= 8 and length > 97:
                issues.append(("info",
                    f"Node {n.get('id')}: {length} Frames bei {vram_gb} GB VRAM - "
                    "laengere Clips erhoehen den Speicherbedarf beim Sampling "
                    "und v.a. beim finalen Decode deutlich."
                ))
            if isinstance(batch, int) and batch > 1 and vram_gb <= 12:
                issues.append(("warn",
                    f"Node {n.get('id')}: batch_size={batch} bei {vram_gb} GB VRAM - "
                    "auf 1 reduzieren, wenn OOM-Probleme auftreten."
                ))
            if isinstance(w, int) and (w % 32 != 0):
                issues.append(("info", f"Node {n.get('id')}: width={w} ist nicht durch 32 teilbar (LTX-Konvention)."))
            if isinstance(h, int) and (h % 32 != 0):
                issues.append(("info", f"Node {n.get('id')}: height={h} ist nicht durch 32 teilbar (LTX-Konvention)."))

    return issues


# ═══════════════════════════════════════════════════════════════
#  3) GGUF-Groessenschaetzung + Quant-Empfehlung
# ═══════════════════════════════════════════════════════════════

def estimate_gguf_size_gb(params_b, quant):
    """Schaetzt GB aus Milliarden Parametern und Quant mit 5% Aufschlag.

    Gibt None zurueck, wenn die Quant-Stufe unbekannt ist.
    """
    bits = GGUF_BITS_PER_PARAM.get(quant)
    if bits is None:
        return None
    return round(params_b * bits / 8 * 1.05, 1)  # +5% Metadaten-Aufschlag


def recommend_gguf(vram_gb, ltx_version, ram_gb=None, overhead_gb=2.5):
    """Baut eine Empfehlungstabelle fuer alle Quant-Stufen der
    gegebenen LTX-Version und markiert, welche komplett ins VRAM
    passen bzw. RAM-Offloading brauchen. 'overhead_gb' reserviert
    Platz fuer VAE/CLIP/Aktivierungen neben dem Hauptmodell."""
    params_b = LTX_PARAMS_B.get(ltx_version)
    if not params_b:
        return f"Unbekannte LTX-Version '{ltx_version}' - keine Schaetzung moeglich.", None

    usable_vram = max(vram_gb - overhead_gb, 0.5)
    lines = [
        f"LTX {ltx_version} (~{params_b}B Parameter) - Schaetzung fuer {vram_gb} GB VRAM "
        f"(ca. {overhead_gb} GB Reserve fuer VAE/CLIP/Aktivierungen -> "
        f"{usable_vram:.1f} GB fuer das Hauptmodell nutzbar):",
        "",
    ]

    best_fit = None
    light_offload = None    # passt mit wenig Offloading (schnell genug)
    heavy_offload = None    # passt nur mit starkem Offloading (langsam)
    sizes = {}
    for quant in GGUF_QUANT_ORDER:
        size = estimate_gguf_size_gb(params_b, quant)
        sizes[quant] = size
        note = GGUF_QUANT_QUALITY_NOTE.get(quant, "")
        if size <= usable_vram:
            tag = "passt komplett ins VRAM"
            if best_fit is None:
                best_fit = quant
        elif size <= usable_vram * 1.6:
            tag = "leichtes RAM-Offloading (noch brauchbares Tempo)"
            if light_offload is None:
                light_offload = quant
        elif ram_gb and size <= usable_vram + ram_gb * 0.6:
            tag = "starkes RAM-Offloading (deutlich langsamer)"
            if heavy_offload is None:
                heavy_offload = quant
        else:
            tag = "zu gross, auch mit Offloading unrealistisch"
        lines.append(f"  {quant:8s} ~{size:5.1f} GB - {tag} - {note}")

    lines.append("")
    # Praktischer Sweet-Spot bevorzugen (Q4_K_M/Q5_K_M/Q3_K_M), statt
    # mechanisch das rein rechnerisch groesste noch "passende" Quant zu
    # waehlen - sonst wuerde bei viel System-RAM z.B. BF16 vorgeschlagen,
    # was mit staendigem Nachladen ueber PCIe in der Praxis sehr langsam ist.
    sweet_spot_order = ["Q4_K_M", "Q5_K_M", "Q3_K_M", "Q3_K_S", "Q6_K", "Q2_K"]

    if best_fit:
        lines.append(f"EMPFEHLUNG: {best_fit} passt komplett ins VRAM - beste Wahl ohne Offloading-Verlangsamung.")
    elif light_offload:
        lines.append(f"EMPFEHLUNG: {light_offload} - passt mit leichtem RAM-Offloading, sollte noch "
                     f"brauchbares Tempo liefern.")
    elif heavy_offload:
        sweet = next((q for q in sweet_spot_order
                     if sizes.get(q) and sizes[q] <= usable_vram + (ram_gb or 0) * 0.6), heavy_offload)
        lines.append(f"EMPFEHLUNG: {sweet} als praktischer Kompromiss (dein System hat "
                     f"{ram_gb} GB RAM fuers Offloading verfuegbar). Groessere Quants (bis "
                     f"{heavy_offload}) laufen technisch auch, aber mit staerkerem Nachladen "
                     f"ueber PCIe spuerbar langsamer - fuer den ersten Test eher unten anfangen "
                     f"und bei Bedarf hochgehen.")
    else:
        lines.append("EMPFEHLUNG: Selbst Q2_K ist bei diesem VRAM/RAM knapp - kleinere Aufloesung/"
                     "kuerzere Clips einplanen oder mehr RAM/VRAM in Betracht ziehen.")

    lines.append("")
    lines.append("Hinweis: Groessen sind Schaetzungen (linear von echten LTX-2-19B-GGUF-"
                 "Dateigroessen auf 22B hochgerechnet) - reale Dateigroesse im jeweiligen "
                 "HuggingFace-Repo vor dem Download pruefen.")
    return "\n".join(lines), (best_fit or light_offload or heavy_offload or "Q2_K")


# ═══════════════════════════════════════════════════════════════
#  4) Gesamtreport zusammenbauen
# ═══════════════════════════════════════════════════════════════

def build_full_report(workflow, user_text, vram_gb, ltx_version, ram_gb=None):
    """Buendelt Log-Analyse, Workflow-Pruefung und GGUF-Empfehlung als Text."""
    sections = []

    log_issues, det_vram_mb, det_gpu, det_mode = parse_comfyui_log(user_text)
    if det_gpu or det_vram_mb:
        info = []
        if det_gpu:
            info.append(f"GPU erkannt: {det_gpu}")
        if det_vram_mb:
            info.append(f"VRAM laut Log: {det_vram_mb} MB (~{det_vram_mb/1024:.1f} GB)")
        if det_mode:
            info.append(f"VRAM-Modus: {det_mode}")
        sections.append("=== Aus dem Log erkannt ===\n" + "\n".join(info))

    if log_issues:
        block = ["=== Erkannte Fehlermuster ==="]
        for title, advice in log_issues:
            block.append(f"\n[{title}]\n{advice}")
        sections.append("\n".join(block))
    elif user_text.strip():
        sections.append(
            "=== Erkannte Fehlermuster ===\nKein bekanntes Muster in deinem Text gefunden - "
            "entweder ein selteneres Problem, oder es ist gar keine Fehlermeldung (z.B. nur "
            "eine Beschreibung in eigenen Worten). Fuer eine Einordnung ggf. minimax-H3-Zweitmeinung nutzen."
        )

    struct_issues = analyze_workflow_structure(workflow, vram_gb)
    if struct_issues:
        block = ["=== Workflow-Struktur-Check ==="]
        for sev, text in struct_issues:
            marker = "WARNUNG" if sev == "warn" else "Info"
            block.append(f"[{marker}] {text}")
        sections.append("\n".join(block))

    gguf_text, _best = recommend_gguf(vram_gb, ltx_version, ram_gb)
    sections.append("=== GGUF-Empfehlung ===\n" + gguf_text)

    return "\n\n".join(sections)


def build_minimax_context(workflow, user_text, vram_gb, ltx_version, rule_based_report):
    """Kontext fuer eine minimax-H3-Zweitmeinung - der bereits erstellte
    regelbasierte Report wird mitgegeben, damit das minimax-Modell nicht
    bei Null anfaengt."""
    nodes = _all_nodes(workflow) if workflow else []
    types = sorted(set(_node_type(n) for n in nodes))
    lines = [
        f"ComfyUI-Workflow (minimax H3) mit {len(nodes)} Nodes (Top-Level + Subgraphs), "
        f"erkannte LTX-Version: {ltx_version}.",
        f"Ziel-Hardware: {vram_gb} GB VRAM.",
        "Node-Typen: " + ", ".join(types[:60]),
        "",
        "Nutzer-Fehlermeldung/-Beschreibung:",
        user_text.strip() or "(keine angegeben)",
        "",
        "Bereits erstellter regelbasierter Report:",
        rule_based_report,
    ]
    return "\n".join(lines)


# Abwaerts-Kompatibilitaet (alter Name)
build_agnes_context = build_minimax_context
