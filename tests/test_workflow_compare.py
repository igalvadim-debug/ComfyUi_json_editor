# tests/test_workflow_compare.py - Tests fuer workflow_compare.py (Phase 1)
#
# Laufbar OHNE pytest:  python tests/test_workflow_compare.py
# (fuegt das Projektverzeichnis zu sys.path hinzu, importiert also
#  config / workflow_analysis / workflow_io direkt).
# Unter pytest laufen dieselben Funktionen als test_* weiter.

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from workflow_compare import (
    compare_workflows,
    match_nodes,
    diff_params,
    node_key,
    effective_type,
    format_diff_report,
    _MISSING,
)

# ----------------------------------------------------------------------
#  Fixtures
# ----------------------------------------------------------------------

def wf_node(nid, ctype, title="", widgets=None, inputs=None, outputs=None, **extra):
    """Workflow-Format-Node (wie ComfyUI 'Save' ihn schreibt)."""
    node = {
        "id": nid, "type": ctype, "title": title,
        "pos": [nid * 10, 0], "size": [340, 60], "flags": {},
        "order": 0, "mode": 0,
    }
    if widgets is not None:
        node["widgets_values"] = widgets
    if inputs is not None:
        node["inputs"] = inputs
    if outputs is not None:
        node["outputs"] = outputs
    node.update(extra)
    return node


def wf(nodes, links=None):
    return {"nodes": nodes, "links": links or [], "version": 0.4}


def base_wf():
    """Zwei-Nodes-Workflow (Workflow-Format) fuer die meisten Tests."""
    return wf([
        wf_node(10, "CLIPTextEncode", "CLIP Text Encode (Prompt)",
                widgets=["a cat"],
                inputs=[{"name": "clip", "type": "CLIP", "link": 1}],
                outputs=[{"name": "CONDITIONING", "type": "CONDITIONING", "links": [2]}]),
        wf_node(20, "KSampler", "KSampler",
                widgets=[1, "fixed", 20, 8.0, "euler", "normal", 1.0],
                inputs=[{"name": "model", "type": "MODEL", "link": 3},
                        {"name": "positive", "type": "CONDITIONING", "link": 2},
                        {"name": "negative", "type": "CONDITIONING", "link": 4},
                        {"name": "latent_image", "type": "LATENT", "link": 5}],
                outputs=[{"name": "LATENT", "type": "LATENT", "links": [6]}]),
    ])


def api_wf():
    """Derselbe Inhalt im rohen API-/Prompt-Format (node_id-keyed Dict)."""
    return {
        "3": {"class_type": "CLIPTextEncode",
              "inputs": {"text": "a cat", "clip": ["4", 0]}},
        "4": {"class_type": "KSampler",
              "inputs": {"seed": 1, "steps": 20, "cfg": 8.0,
                         "sampler_name": "euler", "scheduler": "normal",
                         "denoise": 1.0,
                         "model": ["6", 0], "positive": ["3", 0],
                         "negative": ["5", 0], "latent_image": ["7", 0]}},
    }

# ----------------------------------------------------------------------
#  Tests
# ----------------------------------------------------------------------

def test_identical_workflow_format():
    r = compare_workflows(base_wf(), base_wf(), "A", "B")
    assert r["error"] is None
    assert r["identical"] is True
    assert r["similarity"] == 100
    assert not r["added"] and not r["removed"] and not r["renamed"] and not r["changed"]


def test_api_format_identical():
    r = compare_workflows(api_wf(), api_wf(), "A", "B")
    assert r["identical"] is True
    assert not r["added"] and not r["removed"] and not r["changed"]


def test_mixed_formats_equal():
    """Gleicher Inhalt in Workflow- UND API-Format -> keine Schein-Diffs
    (Dual-Format-Lesen via NODE_WIDGET_SCHEMA)."""
    r = compare_workflows(base_wf(), api_wf(), "Workflow", "API")
    assert r["identical"] is True, format_diff_report(r)
    assert not r["changed"]


def test_added_node_detected():
    b = base_wf()
    b["nodes"].append(wf_node(30, "VAELoader", "Load VAE", widgets=["vae.safetensors"],
                              outputs=[{"name": "VAE", "type": "VAE", "links": []}]))
    r = compare_workflows(base_wf(), b, "A", "B")
    assert r["identical"] is False
    assert len(r["added"]) == 1
    assert r["added"][0]["type"] == "VAELoader"
    assert r["stats"]["added"] == 1


def test_removed_node_detected():
    b = base_wf()
    b["nodes"].pop(0)
    r = compare_workflows(base_wf(), b, "A", "B")
    assert r["identical"] is False
    assert len(r["removed"]) == 1
    assert r["removed"][0]["type"] == "CLIPTextEncode"


def test_param_change_detected():
    a = base_wf()
    b = base_wf()
    b["nodes"][1]["widgets_values"][2] = 30  # KSampler.steps: 20 -> 30
    r = compare_workflows(a, b, "A", "B")
    assert r["identical"] is False
    assert len(r["changed"]) == 1
    diffs = r["changed"][0]["diffs"]
    assert any(d["param"] == "steps" and d["old"] == 20 and d["new"] == 30 for d in diffs)


def test_numeric_coercion():
    """'20' vs 20.0 ist KEIN Unterschied."""
    a = base_wf()
    b = base_wf()
    b["nodes"][1]["widgets_values"][2] = "20"   # str statt int
    r = compare_workflows(a, b, "A", "B")
    assert r["identical"] is True, format_diff_report(r)


def test_known_rename():
    """LTXVImgToVideoConditionOnly -> LTXVImgToVideoInplace kommt aus der
    config-Tabelle und wird als Rename gemeldet, nicht als add+remove."""
    a = wf([wf_node(1, "LTXVImgToVideoConditionOnly", "Img2Vid", widgets=[8])])
    b = wf([wf_node(2, "LTXVImgToVideoInplace", "Img2Vid", widgets=[8])])
    r = compare_workflows(a, b, "2.3", "2.5")
    assert r["identical"] is False
    assert len(r["renamed"]) == 1
    assert r["renamed"][0]["known"] is True
    assert r["renamed"][0]["old_type"] == "LTXVImgToVideoConditionOnly"
    assert r["renamed"][0]["new_type"] == "LTXVImgToVideoInplace"
    assert not r["added"] and not r["removed"]


def test_unknown_type_change_is_remove_add():
    """Rename ausserhalb der config-Tabelle -> konservativ entfernt+neu."""
    a = wf([wf_node(1, "SomeOldNode", "X", widgets=[1])])
    b = wf([wf_node(2, "SomeNewNode", "X", widgets=[1])])
    r = compare_workflows(a, b, "A", "B")
    assert len(r["removed"]) == 1
    assert len(r["added"]) == 1
    assert not r["renamed"]


def test_layout_metadata_ignored():
    a = base_wf()
    b = base_wf()
    b["nodes"][0]["pos"] = [9999, 9999]
    b["nodes"][0]["size"] = [10, 10]
    b["nodes"][0]["flags"] = {"collapsed": True}
    b["nodes"][0]["order"] = 42
    b["nodes"][0]["mode"] = 2
    b["nodes"][0]["color"] = "#ff0000"
    b["nodes"][0]["id"] = 777  # andere ID ist kein Unterschied
    r = compare_workflows(a, b, "A", "B")
    assert r["identical"] is True, format_diff_report(r)


def test_link_change_in_api_format():
    a = api_wf()
    b = api_wf()
    b["4"]["inputs"]["model"] = ["9", 0]  # anderer Vorgaenger-Node
    r = compare_workflows(a, b, "A", "B")
    assert r["identical"] is False
    diffs = r["changed"][0]["diffs"]
    assert any(d["param"] == "model (Verbindung)" for d in diffs)


def test_widget_diff_unknown_type():
    """Unbekannter Node-Typ: Vergleich positionsbasiert (widget[i])."""
    a = wf([wf_node(1, "MyCustomNode", "Custom", widgets=[1, 2])])
    b = wf([wf_node(1, "MyCustomNode", "Custom", widgets=[1, 3])])
    r = compare_workflows(a, b, "A", "B")
    assert r["identical"] is False
    diffs = r["changed"][0]["diffs"]
    assert any(d["param"] == "widget[1]" and d["old"] == 2 and d["new"] == 3 for d in diffs)


def test_output_signature_change():
    a = wf([wf_node(1, "UNETLoader", "UNET",
                   outputs=[{"name": "MODEL", "type": "MODEL", "links": []}])])
    b = wf([wf_node(1, "UNETLoader", "UNET",
                   outputs=[{"name": "MODEL", "type": "MODEL", "links": []},
                            {"name": "LATENT", "type": "LATENT", "links": []}])])
    r = compare_workflows(a, b, "A", "B")
    assert r["identical"] is False
    diffs = r["changed"][0]["diffs"]
    assert any(d["param"] == "outputs" for d in diffs)


def test_empty_vs_nonempty():
    r = compare_workflows(wf([]), base_wf(), "leer", "B")
    assert len(r["added"]) == 2
    assert r["similarity"] == 0
    r2 = compare_workflows(wf([]), wf([]), "leer", "leer")
    assert r2["identical"] is True
    assert r2["similarity"] == 100


def test_duplicate_type_matching():
    """Zwei CLIPTextEncode mit vertauschten Texten: Greedy-Matching ueber
    den Fingerprint paart ueber Kreuz -> Workflows sind inhaltlich
    identisch (Reihenfolge ist kein Unterschied)."""
    def two(texts):
        return wf([
            wf_node(1, "CLIPTextEncode", "Pos", widgets=[texts[0]],
                    outputs=[{"name": "CONDITIONING", "type": "CONDITIONING", "links": []}]),
            wf_node(2, "CLIPTextEncode", "Pos", widgets=[texts[1]],
                    outputs=[{"name": "CONDITIONING", "type": "CONDITIONING", "links": []}]),
        ])
    r = compare_workflows(two(["hello", "world"]), two(["world", "hello"]), "A", "B")
    assert r["stats"]["matched"] == 2
    assert not r["added"] and not r["removed"]
    assert r["stats"]["changed_nodes"] == 0
    assert r["identical"] is True


def test_duplicate_type_real_change():
    """Duplikate mit echter Textaenderung: 1 Node geaendert, keiner
    entfernt/neu."""
    def two(texts):
        return wf([
            wf_node(1, "CLIPTextEncode", "Pos", widgets=[texts[0]],
                    outputs=[{"name": "CONDITIONING", "type": "CONDITIONING", "links": []}]),
            wf_node(2, "CLIPTextEncode", "Pos", widgets=[texts[1]],
                    outputs=[{"name": "CONDITIONING", "type": "CONDITIONING", "links": []}]),
        ])
    r = compare_workflows(two(["hello", "world"]), two(["hello", "hi"]), "A", "B")
    assert r["stats"]["matched"] == 2
    assert r["stats"]["changed_nodes"] == 1
    assert not r["added"] and not r["removed"]


def test_ui_only_widget_ignored():
    """control_after_generate ist UI-only: Workflow mit 'fixed' vs 'random'
    ist inhaltlich identisch (und im API-Format existiert das Feld nicht)."""
    a = base_wf()
    b = base_wf()
    b["nodes"][1]["widgets_values"][1] = "random"
    r = compare_workflows(a, b, "A", "B")
    assert r["identical"] is True, format_diff_report(r)


def test_node_key_uses_title_not_id():
    a = wf_node(1, "KSampler", "Mein Sampler", widgets=[1])
    b = wf_node(99, "KSampler", "Mein Sampler", widgets=[1])
    assert node_key(a) == node_key(b)


def test_dict_shaped_widgets_values():
    """Neuere ComfyUI-Saves (z.B. VHS_VideoCombine) speichern
    widgets_values als benanntes Dict statt Liste - get_node_param
    darf da nicht crashen und muss den Wert per Name finden
    (Regressionstest fuer den KeyError-0-Fix in workflow_analysis)."""
    a = wf([wf_node(1, "VHS_VideoCombine", "Video",
                    widgets={"frame_rate": 24, "filename_prefix": "LTX-2",
                             "format": "video/h264-mp4"})])
    b = wf([wf_node(2, "VHS_VideoCombine", "Video",
                    widgets={"frame_rate": 30, "filename_prefix": "LTX-2",
                             "format": "video/h264-mp4"})])
    r = compare_workflows(a, b, "A", "B")
    assert r["identical"] is False
    assert len(r["changed"]) == 1
    diffs = r["changed"][0]["diffs"]
    assert any(d["param"] == "frame_rate" and d["old"] == 24 and d["new"] == 30 for d in diffs)


def test_effective_type_maps_rename_table():
    old = wf_node(1, "LTXVTiledVAEDecode")
    new = wf_node(2, "VAEDecodeTiled")
    assert effective_type(old) == effective_type(new) == "VAEDecodeTiled"


def test_missing_workflow_returns_error():
    r = compare_workflows(None, base_wf(), "A", "B")
    assert r["error"] is not None


def test_match_nodes_surface():
    """match_nodes ist die Phase-2-Schnittstelle und liefert Paare direkt."""
    a = [wf_node(1, "KSampler", "S", widgets=[1, "fixed", 20, 8.0, "euler", "normal", 1.0])]
    b = [wf_node(2, "KSampler", "S", widgets=[1, "fixed", 20, 8.0, "euler", "normal", 1.0])]
    matches, added, removed, renamed = match_nodes(a, b)
    assert len(matches) == 1
    assert not added and not removed and not renamed
    assert diff_params(matches[0][0], matches[0][1]) == []


# ----------------------------------------------------------------------
#  Kleiner Runner (funktioniert auch ohne pytest)
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
