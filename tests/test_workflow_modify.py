# tests/test_workflow_modify.py - Tests fuer workflow_modify.py (Etappe 2, v2)
# Laufbar OHNE pytest:  python tests/test_workflow_modify.py
#
# v2-Prinzip: Bypass (mode 4) mit INTACTEN Links - keine Link-Entfernung.
# Die Tests pruefen deshalb: mode-Werte, Link-Anzahl/-Inhalt (unveraendert!),
# Kaskaden-Verhalten, Original-Schutz und Verify.

import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from workflow_modify import (
    BLOCK_DEFS, detect_blocks, disable_block, apply_mods, verify_modification,
)
from workflow_compare import compare_workflows

# ----------------------------------------------------------------------
#  Fixture-Helfer (ComfyUI-UI-Format)
# ----------------------------------------------------------------------

def node(nid, typ, title="", inputs=(), outputs=(), mode=0):
    return {
        "id": nid, "type": typ, "title": title, "mode": mode,
        "inputs": [{"name": nm, "type": ty, "link": None} for nm, ty in inputs],
        "outputs": [{"name": nm, "type": ty, "links": []} for nm, ty in outputs],
    }


def link(nodes, links, src, src_slot, tgt, tgt_slot, typ):
    lid = (links[-1][0] + 1) if links else 1
    links.append([lid, src, src_slot, tgt, tgt_slot, typ])
    by_id = {n["id"]: n for n in nodes}
    by_id[src]["outputs"][src_slot]["links"].append(lid)
    by_id[tgt]["inputs"][tgt_slot]["link"] = lid


def wf(nodes, links):
    return {"nodes": nodes, "links": links, "version": 0.4}


def _modes(mod):
    return {n["id"]: n.get("mode") for n in mod["nodes"]}

# ----------------------------------------------------------------------
#  Fixtures
# ----------------------------------------------------------------------

def mk_upscaler_fixture():
    nodes = [
        node(10, "LTXVSeparateAVLatent", inputs=(("av_latent", "LATENT"),),
             outputs=(("video_latent", "LATENT"), ("audio_latent", "LATENT"))),
        node(12, "LTXVLatentUpsampler", inputs=(("samples", "LATENT"),
                                                ("upscale_model", "LATENT_UPSCALE_MODEL"),
                                                ("vae", "VAE")),
             outputs=(("LATENT", "LATENT"),)),
        node(13, "LatentUpscaleModelLoader", outputs=(("LATENT_UPSCALE_MODEL", "LATENT_UPSCALE_MODEL"),)),
        node(14, "VAEDecode", inputs=(("samples", "LATENT"), ("vae", "VAE")),
             outputs=(("IMAGE", "IMAGE"),)),
    ]
    links = []
    link(nodes, links, 13, 0, 12, 1, "LATENT_UPSCALE_MODEL")
    link(nodes, links, 10, 0, 12, 0, "LATENT")
    link(nodes, links, 12, 0, 14, 0, "LATENT")
    return wf(nodes, links)


def mk_patches_fixture():
    nodes = [
        node(1, "LoraLoaderModelOnly", outputs=(("MODEL", "MODEL"),)),
        node(2, "PathchSageAttentionKJ", inputs=(("model", "MODEL"),), outputs=(("model", "MODEL"),)),
        node(3, "LTXVChunkFeedForward", inputs=(("model", "MODEL"),), outputs=(("model", "MODEL"),)),
        node(4, "SetNode", title="Set_model", inputs=(("in", "*"),)),
        node(5, "GetNode", title="Get_model", outputs=(("out", "*"),)),
        node(6, "CFGGuider", inputs=(("model", "MODEL"), ("positive", "CONDITIONING"),
                                     ("negative", "CONDITIONING"), ("cfg", "FLOAT")),
             outputs=(("GUIDER", "GUIDER"),)),
    ]
    links = []
    link(nodes, links, 1, 0, 2, 0, "MODEL")
    link(nodes, links, 2, 0, 3, 0, "MODEL")
    link(nodes, links, 3, 0, 4, 0, "MODEL")
    link(nodes, links, 5, 0, 6, 0, "MODEL")
    return wf(nodes, links)


def mk_audio_fixture():
    nodes = [
        node(7, "LTXVImgToVideoInplace", inputs=(("vae", "VAE"), ("image", "IMAGE"),
                                                 ("latent", "LATENT"), ("strength", "FLOAT")),
             outputs=(("latent", "LATENT"),)),
        node(8, "LTXVConcatAVLatent", inputs=(("video_latent", "LATENT"), ("audio_latent", "LATENT")),
             outputs=(("latent", "LATENT"),)),
        node(9, "LTXVAudioVAEEncode", inputs=(("audio", "AUDIO"), ("audio_vae", "VAE")),
             outputs=(("Audio Latent", "LATENT"),)),
        node(10, "SamplerCustomAdvanced", inputs=(("noise", "NOISE"), ("guider", "GUIDER"),
                                                  ("sampler", "SAMPLER"), ("sigmas", "SIGMAS"),
                                                  ("latent_image", "LATENT")),
             outputs=(("output", "LATENT"),)),
    ]
    links = []
    link(nodes, links, 7, 0, 8, 0, "LATENT")
    link(nodes, links, 9, 0, 8, 1, "LATENT")
    link(nodes, links, 8, 0, 10, 4, "LATENT")
    return wf(nodes, links)


def mk_stage2_fixture():
    nodes = [
        node(19, "LTXVImgToVideoInplace", inputs=(("vae", "VAE"), ("image", "IMAGE"),
                                                  ("latent", "LATENT"), ("strength", "FLOAT")),
             outputs=(("latent", "LATENT"),)),
        node(20, "SamplerCustomAdvanced", inputs=(("noise", "NOISE"), ("guider", "GUIDER"),
                                                  ("sampler", "SAMPLER"), ("sigmas", "SIGMAS"),
                                                  ("latent_image", "LATENT")),
             outputs=(("output", "LATENT"),)),
        node(21, "LTXVSeparateAVLatent", inputs=(("av_latent", "LATENT"),),
             outputs=(("video_latent", "LATENT"), ("audio_latent", "LATENT"))),
        node(22, "VAEDecode", inputs=(("samples", "LATENT"), ("vae", "VAE")),
             outputs=(("IMAGE", "IMAGE"),)),
    ]
    links = []
    link(nodes, links, 19, 0, 20, 4, "LATENT")
    link(nodes, links, 20, 0, 21, 0, "LATENT")
    link(nodes, links, 21, 0, 22, 0, "LATENT")
    return wf(nodes, links)


def mk_bridge_fixture():
    nodes = [
        node(30, "PrimitiveString", outputs=(("STRING", "STRING"),)),
        node(31, "SetNode", title="Set_enhance_prompt", inputs=(("in", "*"),)),
        node(32, "GetNode", title="Get_enhance_prompt", outputs=(("out", "*"),)),
        node(33, "StringConcatenate", inputs=(("string1", "STRING"), ("string2", "STRING")),
             outputs=(("STRING", "STRING"),)),
    ]
    links = []
    link(nodes, links, 30, 0, 31, 0, "STRING")
    link(nodes, links, 32, 0, 33, 0, "STRING")
    return wf(nodes, links)


def mk_bridge_mismatch_fixture():
    """enhance_prompt-Rewire mit Typ-Mismatch (BOOLEAN -> MODEL)."""
    nodes = [
        node(30, "PrimitiveBoolean", outputs=(("BOOLEAN", "BOOLEAN"),)),
        node(31, "SetNode", title="Set_enhance_prompt", inputs=(("in", "*"),)),
        node(32, "GetNode", title="Get_enhance_prompt", outputs=(("out", "*"),)),
        node(6, "CFGGuider", inputs=(("model", "MODEL"),), outputs=(("GUIDER", "GUIDER"),)),
    ]
    links = []
    link(nodes, links, 30, 0, 31, 0, "BOOLEAN")
    link(nodes, links, 32, 0, 6, 0, "MODEL")
    return wf(nodes, links)

# ----------------------------------------------------------------------
#  Tests
# ----------------------------------------------------------------------

def test_detect_blocks():
    d = detect_blocks(mk_upscaler_fixture())
    assert d["upscaler"]["found"] is True
    assert d["audio"]["found"] is False
    d2 = detect_blocks(mk_stage2_fixture())
    assert d2["stage2"]["found"] is True


def test_disable_upscaler_keeps_links():
    """Bypass statt Link-Entfernung: alle 3 Links bleiben intact."""
    orig = mk_upscaler_fixture()
    snapshot = json.dumps(orig)
    mod, report = disable_block(orig, "upscaler")
    assert any("deaktiviert" in r for r in report)
    assert json.dumps(orig) == snapshot, "Original wurde veraendert!"

    modes = _modes(mod)
    assert modes[12] == 4 and modes[13] == 4   # Upsampler + Loader bypassed
    assert modes[10] == 0 and modes[14] == 0   # SeparateAV + Decoder aktiv
    assert len(mod["links"]) == 3, "Links muessen intakt bleiben"
    assert mod["links"] == orig["links"], "Kein Link darf veraendert werden"

    ok, msg = verify_modification(orig, mod)
    assert ok, msg
    assert compare_workflows(orig, mod)["identical"] is True


def test_disable_patches_keeps_flow():
    """Patches bypassed, Modell fliesst durch; Bruecken bleiben aktiv."""
    orig = mk_patches_fixture()
    mod, report = disable_block(orig, "patches")
    modes = _modes(mod)
    assert modes[2] == 4 and modes[3] == 4
    assert modes[4] == 0 and modes[5] == 0 and modes[6] == 0  # Bruecken + Guider aktiv
    assert len(mod["links"]) == 4 and mod["links"] == orig["links"]
    ok, msg = verify_modification(orig, mod)
    assert ok, msg


def test_disable_audio_keeps_links():
    orig = mk_audio_fixture()
    mod, report = disable_block(orig, "audio")
    modes = _modes(mod)
    assert modes[8] == 4 and modes[9] == 4      # Concat + Encode bypassed
    assert modes[10] == 0                        # Sampler bleibt aktiv
    assert len(mod["links"]) == 3 and mod["links"] == orig["links"]
    ok, msg = verify_modification(orig, mod)
    assert ok, msg


def test_disable_stage2_rewires_decoder():
    """stage2: Sampler+Separator bypassed, Decoder bekommt das Latent
    VOR der Stufe; Sampler-Eingaenge bleiben intakt."""
    orig = mk_stage2_fixture()
    mod, report = disable_block(orig, "stage2")
    modes = _modes(mod)
    assert modes[20] == 4 and modes[21] == 4
    assert modes[22] == 0  # Decoder aktiv
    # Decoder.samples jetzt von 19 (ImgToVideo), nicht mehr von 21
    dec = next(n for n in mod["nodes"] if n["id"] == 22)
    src = next(l for l in mod["links"] if l[0] == dec["inputs"][0]["link"])
    assert src[1] == 19 and src[2] == 0
    # Sampler-Input bleibt intakt (keine Missing connection)
    sam = next(n for n in mod["nodes"] if n["id"] == 20)
    assert sam["inputs"][4]["link"] is not None
    ok, msg = verify_modification(orig, mod)
    assert ok, msg


def test_bridge_enhance_prompt():
    orig = mk_bridge_fixture()
    mod, report = disable_block(orig, "enhance_prompt")
    # PrimitiveString direkt an StringConcatenate; Set-Bruecke bleibt
    assert len(mod["links"]) == 2
    assert any(l[1:5] == [30, 0, 33, 0] for l in mod["links"])
    assert any(l[1:5] == [30, 0, 31, 0] for l in mod["links"])
    ok, msg = verify_modification(orig, mod)
    assert ok, msg


def test_type_mismatch_aborts():
    orig = mk_bridge_mismatch_fixture()
    snapshot = json.dumps(orig)
    mod, report = disable_block(orig, "enhance_prompt")
    assert any("Typ-Mismatch" in r for r in report)
    assert json.dumps(orig) == snapshot
    assert json.dumps(mod) == snapshot, "Bei Mismatch darf nichts veraendert werden"


def test_unknown_block():
    orig = mk_upscaler_fixture()
    snapshot = json.dumps(orig)
    mod, report = disable_block(orig, "gibt_es_nicht")
    assert any("Unbekannter Block" in r for r in report)
    assert json.dumps(mod) == snapshot


def test_apply_mods_multiple():
    orig = mk_audio_fixture()
    mod, report = apply_mods(orig, {"audio": True, "upscaler": False})
    assert any("--- audio ---" in r for r in report)
    assert _modes(mod)[8] == 4
    assert json.dumps(orig) != json.dumps(mod)


def test_stage2_and_audio_combined():
    """stage2 + audio zusammen: keine toten Links, keine verlorenen
    Verbindungen an aktiven Nodes."""
    nodes = [
        node(19, "LTXVImgToVideoInplace", inputs=(("vae", "VAE"), ("image", "IMAGE"),
                                                  ("latent", "LATENT"), ("strength", "FLOAT")),
             outputs=(("latent", "LATENT"),)),
        node(20, "SamplerCustomAdvanced", inputs=(("noise", "NOISE"), ("guider", "GUIDER"),
                                                  ("sampler", "SAMPLER"), ("sigmas", "SIGMAS"),
                                                  ("latent_image", "LATENT")),
             outputs=(("output", "LATENT"),)),
        node(21, "LTXVSeparateAVLatent", inputs=(("av_latent", "LATENT"),),
             outputs=(("video_latent", "LATENT"), ("audio_latent", "LATENT"))),
        node(22, "VAEDecode", inputs=(("samples", "LATENT"), ("vae", "VAE")),
             outputs=(("IMAGE", "IMAGE"),)),
        node(23, "LTXVAudioVAEDecode", inputs=(("samples", "LATENT"), ("audio_vae", "VAE")),
             outputs=(("Audio", "AUDIO"),)),
    ]
    links = []
    link(nodes, links, 19, 0, 20, 4, "LATENT")
    link(nodes, links, 20, 0, 21, 0, "LATENT")
    link(nodes, links, 21, 0, 22, 0, "LATENT")
    link(nodes, links, 21, 1, 23, 0, "LATENT")
    orig = wf(nodes, links)

    mod, report = apply_mods(orig, {"stage2": True, "audio": True})
    modes = _modes(mod)
    assert modes[20] == 4 and modes[21] == 4 and modes[23] == 4
    # Decoder (22) aktiv und korrekt verdrahtet
    dec = next(n for n in mod["nodes"] if n["id"] == 22)
    src = next(l for l in mod["links"] if l[0] == dec["inputs"][0]["link"])
    assert src[1] == 19
    ok, msg = verify_modification(orig, mod)
    assert ok, msg


def test_verify_detects_broken_link():
    orig = mk_upscaler_fixture()
    broken = copy.deepcopy(orig)
    broken["links"][0][3] = 999  # Ziel existiert nicht
    ok, msg = verify_modification(orig, broken)
    assert not ok


def test_verify_detects_lost_connection():
    """Aktiver Node verliert einen verbundenen Input -> Verify schlaegt fehl."""
    orig = mk_upscaler_fixture()
    bad = copy.deepcopy(orig)
    bad["links"] = [l for l in bad["links"] if not (l[1] == 12 and l[2] == 0)]
    bad = json.loads(json.dumps(bad))  # tiefe Kopie nach Link-Entfernung
    dec = next(n for n in bad["nodes"] if n["id"] == 14)
    # Decoder.samples ist jetzt leer (Link 12->14 existiert noch? nein: nur 10->12 und 13->12 entfernt?)
    ok, msg = verify_modification(orig, bad)
    assert not ok, "Verlorene Verbindung an aktivem Node muss erkannt werden"


def test_registry_is_extensible():
    assert "audio" in BLOCK_DEFS and "upscaler" in BLOCK_DEFS and "patches" in BLOCK_DEFS
    assert "stage2" in BLOCK_DEFS and "outputs" in BLOCK_DEFS and "enhance_prompt" in BLOCK_DEFS


# ----------------------------------------------------------------------
#  Subgraph-Tests (definitions.subgraphs - die "Scene"-Kopien der Bruecken)
# ----------------------------------------------------------------------

def _subgraph_wf(top, top_links, sub_nodes, sub_links, sub_id="scene-1"):
    return {
        "nodes": top, "links": top_links, "version": 0.4,
        "definitions": {"subgraphs": [{
            "id": sub_id, "name": "Generate Video", "version": 1, "revision": 0,
            "state": {}, "config": {}, "inputNode": 1, "outputNode": 2,
            "nodes": sub_nodes, "links": sub_links,
        }]},
    }


def mk_subgraph_fixture():
    """Top-Level-Upscaler + Subgraph-Kopie der Get_upscale_model-Bruecke
    und eines LTXVLatentUpsampler - die 'Missing connection'-Situation."""
    top = [
        node(100, "LatentUpscaleModelLoader", outputs=(("LATENT_UPSCALE_MODEL", "LATENT_UPSCALE_MODEL"),)),
        node(101, "SetNode", title="Set_upscale_model", inputs=(("in", "*"),)),
    ]
    top_links = []
    link(top, top_links, 100, 0, 101, 0, "LATENT_UPSCALE_MODEL")
    sub_nodes = [
        node(10, "GetNode", title="Get_upscale_model", outputs=(("out", "*"),)),
        node(11, "LTXVLatentUpsampler", inputs=(("samples", "LATENT"),
                                                ("upscale_model", "LATENT_UPSCALE_MODEL"),
                                                ("vae", "VAE")),
             outputs=(("LATENT", "LATENT"),)),
    ]
    sub_links = []
    link(sub_nodes, sub_links, 10, 0, 11, 1, "LATENT_UPSCALE_MODEL")
    return _subgraph_wf(top, top_links, sub_nodes, sub_links)


def mk_audio_subgraph_fixture():
    top = [node(200, "LTXVConcatAVLatent", inputs=(("video_latent", "LATENT"), ("audio_latent", "LATENT")),
                outputs=(("latent", "LATENT"),))]
    top_links = []
    sub_nodes = [
        node(20, "TrimAudioDuration", inputs=(("audio", "AUDIO"), ("duration", "FLOAT")),
             outputs=(("audio", "AUDIO"),)),
        node(21, "LTXVConcatAVLatent", inputs=(("video_latent", "LATENT"), ("audio_latent", "LATENT")),
             outputs=(("latent", "LATENT"),)),
    ]
    sub_links = []
    link(sub_nodes, sub_links, 20, 0, 21, 1, "AUDIO")
    return _subgraph_wf(top, top_links, sub_nodes, sub_links)


def test_disable_upscaler_mutes_subgraph_bridges():
    """Der Kern-Fall des Users: Top-Level-Set_upscale_model deaktiviert ->
    die Subgraph-Kopien von Get_upscale_model UND die subgraph-internen
    Block-Nodes muessen mit deaktiviert werden."""
    orig = mk_subgraph_fixture()
    mod, report = disable_block(orig, "upscaler")

    top_modes = {n["id"]: n.get("mode") for n in mod["nodes"]}
    sub_modes = {n["id"]: n.get("mode")
                 for n in mod["definitions"]["subgraphs"][0]["nodes"]}
    assert top_modes[100] == 4   # Loader
    assert top_modes[101] == 4   # Set_upscale_model
    assert sub_modes[10] == 4, "Subgraph-Get_upscale_model muss mit deaktiviert werden"
    assert sub_modes[11] == 4, "Subgraph-LTXVLatentUpsampler muss mit deaktiviert werden"

    ok, msg = verify_modification(orig, mod)
    assert ok, msg
    # Links in beiden Containern bleiben intakt
    assert len(mod["links"]) == 1
    assert len(mod["definitions"]["subgraphs"][0]["links"]) == 1


def test_disable_audio_mutes_subgraph_nodes():
    orig = mk_audio_subgraph_fixture()
    mod, report = disable_block(orig, "audio")
    top_modes = {n["id"]: n.get("mode") for n in mod["nodes"]}
    sub_modes = {n["id"]: n.get("mode")
                 for n in mod["definitions"]["subgraphs"][0]["nodes"]}
    assert top_modes[200] == 4
    assert sub_modes[20] == 4, "Subgraph-TrimAudioDuration muss deaktiviert werden"
    assert sub_modes[21] == 4, "Subgraph-Concat muss deaktiviert werden"
    ok, msg = verify_modification(orig, mod)
    assert ok, msg


def test_verify_ok_with_subgraphs():
    """Korrekte Subgraph-Modifikation darf verify nicht faelschlich
    scheitern lassen."""
    orig = mk_subgraph_fixture()
    mod, _ = disable_block(orig, "upscaler")
    ok, msg = verify_modification(orig, mod)
    assert ok, msg


def test_verify_detects_lost_connection_in_subgraph():
    """Aktiver Subgraph-Node verliert eine Verbindung -> Verify schlaegt
    fehl (Fruehwarnung vor dem Export)."""
    orig = mk_subgraph_fixture()
    bad = copy.deepcopy(orig)
    sub = bad["definitions"]["subgraphs"][0]
    sub["links"] = [l for l in sub["links"] if not (l[1] == 10 and l[2] == 0)]
    for n in sub["nodes"]:
        for inp in n.get("inputs", []):
            if inp.get("name") == "upscale_model":
                inp["link"] = None
    ok, msg = verify_modification(orig, bad)
    assert not ok
    assert "scene-1" in msg


def test_count_deactivated_includes_subgraphs():
    from workflow_modify import count_deactivated
    orig = mk_subgraph_fixture()
    mod, _ = disable_block(orig, "upscaler")
    assert count_deactivated(mod) == 4  # 100, 101 (Top) + 10, 11 (Subgraph)


def test_subgraph_dict_links():
    """ComfyUI speichert Subgraph-Links als Dicts (id/origin_id/...),
    Top-Level-Links als Tupel - beides muss funktionieren."""
    from workflow_modify import _link_tuple
    top = [
        node(100, "LatentUpscaleModelLoader", outputs=(("LATENT_UPSCALE_MODEL", "LATENT_UPSCALE_MODEL"),)),
        node(101, "SetNode", title="Set_upscale_model", inputs=(("in", "*"),)),
    ]
    top_links = []
    link(top, top_links, 100, 0, 101, 0, "LATENT_UPSCALE_MODEL")
    sub_nodes = [
        node(10, "GetNode", title="Get_upscale_model", outputs=(("out", "*"),)),
        node(11, "LTXVLatentUpsampler", inputs=(("samples", "LATENT"),
                                                ("upscale_model", "LATENT_UPSCALE_MODEL"),
                                                ("vae", "VAE")),
             outputs=(("LATENT", "LATENT"),)),
    ]
    sub_links = [{"id": 5001, "origin_id": 10, "origin_slot": 0,
                  "target_id": 11, "target_slot": 1, "type": "LATENT_UPSCALE_MODEL"}]
    sub_nodes[1]["inputs"][1]["link"] = 5001
    sub_nodes[0]["outputs"][0]["links"] = [5001]
    orig = _subgraph_wf(top, top_links, sub_nodes, sub_links)

    assert _link_tuple(sub_links[0]) == (5001, 10, 0, 11, 1, "LATENT_UPSCALE_MODEL")
    mod, report = disable_block(orig, "upscaler")
    ok, msg = verify_modification(orig, mod)
    assert ok, msg
    sub_modes = {n["id"]: n.get("mode") for n in mod["definitions"]["subgraphs"][0]["nodes"]}
    assert sub_modes[10] == 4 and sub_modes[11] == 4


def test_subgraph_boundary_ports():
    """Subgraph-Links auf negative IDs (Grenz-Ports -10/-20) sind valide
    und duerfen verify nicht scheitern lassen."""
    top = [node(100, "LatentUpscaleModelLoader", outputs=(("LATENT_UPSCALE_MODEL", "LATENT_UPSCALE_MODEL"),))]
    top_links = []
    sub_nodes = [
        node(10, "GetNode", title="Get_upscale_model", outputs=(("out", "*"),)),
        node(11, "LTXVLatentUpsampler", inputs=(("samples", "LATENT"),
                                                ("upscale_model", "LATENT_UPSCALE_MODEL"),
                                                ("vae", "VAE")),
             outputs=(("LATENT", "LATENT"),)),
    ]
    # Input-Port (-10) -> Upsampler.samples, Get_upscale_model ->
    # Upsampler.upscale_model, Upsampler -> Output-Port (-20)
    sub_links = [
        {"id": 1, "origin_id": -10, "origin_slot": 0, "target_id": 11, "target_slot": 0, "type": "LATENT"},
        {"id": 2, "origin_id": 10, "origin_slot": 0, "target_id": 11, "target_slot": 1, "type": "LATENT_UPSCALE_MODEL"},
        {"id": 3, "origin_id": 11, "origin_slot": 0, "target_id": -20, "target_slot": 0, "type": "LATENT"},
    ]
    for l in sub_links:
        if l["target_id"] >= 0:
            tgt = next(n for n in sub_nodes if n["id"] == l["target_id"])
            tgt["inputs"][l["target_slot"]]["link"] = l["id"]
        if l["origin_id"] >= 0:
            src = next(n for n in sub_nodes if n["id"] == l["origin_id"])
            src["outputs"][l["origin_slot"]]["links"] = src["outputs"][l["origin_slot"]].get("links", []) + [l["id"]]
    orig = _subgraph_wf(top, top_links, sub_nodes, sub_links)

    ok, msg = verify_modification(orig, orig)
    assert ok, msg
    mod, _ = disable_block(orig, "upscaler")
    ok2, msg2 = verify_modification(orig, mod)
    assert ok2, msg2

# ----------------------------------------------------------------------
#  Runner
# ----------------------------------------------------------------------

if __name__ == "__main__":
    import traceback

    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception:
            failed += 1
            print(f"FAIL  {name}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} Tests bestanden")
    sys.exit(1 if failed else 0)
