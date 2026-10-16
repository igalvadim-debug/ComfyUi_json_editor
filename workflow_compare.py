# workflow_compare.py - Vergleich zweier ComfyUI-Workflows (Phase 1)
#
# Warum nicht einfach dict == dict? ComfyUI-Workflows enthalten viel
# Nicht-Semantisches: Node-IDs, Positionen (pos), Groessen (size),
# flags, order/mode usw. koennen sich aendern, ohne dass sich der
# Workflow inhaltlich aendert. Deshalb arbeitet der Vergleich mit
# einem stabilen Node-Schluessel (Typ + Titel) statt mit IDs:
#
#   * Rename-Awareness: Node-Typen, die zwischen LTX 2.3 und 2.5
#     umbenannt wurden (config.LTX23_TO_LTX25_NODE_RENAME), werden
#     vor dem Matching auf den 2.5-Typ abgebildet und als "Rename"
#     gemeldet statt als entfernt + hinzugefuegt.
#   * Param-Diff: liest Werte ueber workflow_analysis.get_node_param,
#     funktioniert damit fuer Workflow-Format (widgets_values,
#     positionsbasiert via NODE_WIDGET_SCHEMA) UND API-Format
#     (inputs-Dict mit direkten Werten).
#   * Layout-Metadaten (pos, size, flags, order, mode, color, ...)
#     werden ignoriert.
#   * Numerische Koerzierung: "20" == 20 == 20.0.
#
# Phase-2-Hook: match_nodes() ist die Schnittstelle, auf der
# migration_plan.py die Topologie-Analyse (Verbindungen zwischen
# gematchten Nodes) aufbauen kann.

from config import LTX23_TO_LTX25_NODE_RENAME, NODE_WIDGET_SCHEMA
from workflow_io import normalize_workflow
from workflow_analysis import get_node_param, _node_type

# ----------------------------------------------------------------------
#  Konstanten
# ----------------------------------------------------------------------

_MISSING = object()  # Sentinel: "Wert nicht vorhanden" (unterscheidet sich von None)
_MISSING_LABEL = "<fehlt>"

# Widgets, die nur in der ComfyUI-Oberflaeche existieren und keinen
# Einfluss auf die API-Ausfuehrung haben (im API-Format fehlen sie
# komplett). Unterschiede hier waeren reines Rauschen.
UI_ONLY_WIDGET_KEYS = {"control_after_generate", "choose_file_to_upload"}

# ----------------------------------------------------------------------
#  Node-Identitaet
# ----------------------------------------------------------------------

def _node_title(node):
    meta = node.get("_meta") if isinstance(node.get("_meta"), dict) else {}
    return str(node.get("title") or meta.get("title") or "").strip()


def node_key(node):
    """Stabiler Node-Schluessel (Typ, Titel) - bewusst NICHT die ID."""
    return (_node_type(node), _node_title(node) or None)


def effective_type(node):
    """Node-Typ mit Rename-Tabelle abgebildet (2.3-Name -> 2.5-Name).
    Typen ausserhalb der Tabelle bleiben unveraendert."""
    t = _node_type(node)
    return LTX23_TO_LTX25_NODE_RENAME.get(t, t)


def _scalar_fingerprint(node, schema=None):
    """Geordnete Liste skalarer Parameterwerte - Tiebreak beim Matching
    gleichartiger Nodes (z.B. mehrere CLIPTextEncode). 'schema' kann
    vom Aufrufer vorgegeben werden (gleicher effektiver Typ -> gleiche
    Schema-Liste), damit die Lookup-Tabelle nicht pro Node neu
    aufgebaut wird."""
    if schema is None:
        schema = NODE_WIDGET_SCHEMA.get(_node_type(node))
    out = []
    if schema:
        for key in schema:
            v = get_node_param(node, key, _MISSING)
            if v is not _MISSING and not isinstance(v, (list, dict)):
                num = _to_num(v)
                out.append((key, num if num is not None else v))
    return out

# ----------------------------------------------------------------------
#  Wertevergleich
# ----------------------------------------------------------------------

def _to_num(v):
    """Konvertiert Zahlen und numerische Strings nach float (bool nie)."""
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v.strip())
        except ValueError:
            return None
    return None


def _values_equal(a, b):
    """Toleranter Wertevergleich: "20" == 20 == 20.0; Listen elementweise;
    bool bleibt bool (True != 1)."""
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return float(a) == float(b)
    if isinstance(a, str) or isinstance(b, str):
        na, nb = _to_num(a), _to_num(b)
        if na is not None and nb is not None:
            return na == nb
        return a == b
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_values_equal(x, y) for x, y in zip(a, b))
    return a == b


def _fmt(v):
    """Kompakte, UI-taugliche Darstellung eines Vergleichswerts."""
    if v is _MISSING:
        return _MISSING_LABEL
    if isinstance(v, list):
        return ", ".join(_fmt(x) for x in v)
    s = str(v)
    return s if len(s) <= 80 else s[:77] + "..."

# ----------------------------------------------------------------------
#  Matching
# ----------------------------------------------------------------------

def _score(a, b, fa=None, fb=None):
    """Matching-Punktzahl: gleicher Roh-Typ +2, gleicher Titel +2,
    plus Anzahl uebereinstimmender skalarer Parameter.
    Die Fingerprints koennen vom Aufrufer vorgegeben werden
    (werden in match_nodes ohnehin pro Node benoetigt)."""
    s = 0
    if _node_type(a) == _node_type(b):
        s += 2
    ta, tb = _node_title(a), _node_title(b)
    if ta and ta == tb:
        s += 2
    if fa is None:
        fa = _scalar_fingerprint(a)
    if fb is None:
        fb = _scalar_fingerprint(b)
    s += sum(1 for x in fa if x in fb)
    return s


def match_nodes(nodes_a, nodes_b):
    """Greedy-Matching: fuer jeden Node aus A den besten noch freien
    Kandidaten aus B mit gleichem *effektivem* Typ waehlen (beste
    Punktzahl via _score, deterministisch in A-Reihenfolge).

    Effizienz: Nodes beider Seiten werden vorab in Buckets gleichen
    effektiven Typs einsortiert - die Suche laeuft dann nur noch über
    Kandidaten desselben Buckets (statt über alle Nodes von B). Inner-
    halb eines Buckets sind Typ und Widget-Schema identisch, daher
    wird jeder Skalar-Fingerprint genau EINMAL berechnet. Komplexitaet
    damit O(n * k + F) statt O(n^2) mit n * k Fingerprint-Aufbauten.

    Typ-Renames aus der config-Tabelle werden dadurch automatisch
    zugeordnet und als 'renamed' gemeldet. Renames AUSSERHALB der
    Tabelle werden konservativ als entfernt + neu gemeldet.

    Rueckgabe (matches, added, removed, renamed):
      matches: list[(node_a, node_b)]  - als gleich erkannte Paare
      added:   list[node]              - nur in B
      removed: list[node]              - nur in A
      renamed: list[dict]              - Paare mit Typ-Rename
    """
    # Vorab: Bucket-Zuordnung + je ein Fingerprint pro Node.
    # 'seq' merkt die urspruengliche Reihenfolge in nodes_b - fuer die
    # 'added'-Liste am Ende (bleibt damit exakt in B-Reihenfolge).
    buckets_b = {}          # eff_typ -> [Listeneintrag, ...]
    for j, b in enumerate(nodes_b):
        schema_b = NODE_WIDGET_SCHEMA.get(_node_type(b))
        entry = [b, _scalar_fingerprint(b, schema_b), False, j]
        buckets_b.setdefault(effective_type(b), []).append(entry)

    # Fingerprints der A-Nodes in Original-Reihenfolge vorraeps berechnen
    prepared_a = []
    for a in nodes_a:
        schema_a = NODE_WIDGET_SCHEMA.get(_node_type(a))
        prepared_a.append((a, effective_type(a), _scalar_fingerprint(a, schema_a)))

    matches, removed, renamed = [], [], []
    for a, etype, fa in prepared_a:
        pool = buckets_b.get(etype)
        if not pool:
            removed.append(a)
            continue
        best_i, best_s = None, -1
        for i, entry in enumerate(pool):
            if entry[2]:
                continue
            s = _score(a, entry[0], fa, entry[1])
            if s > best_s:
                best_s, best_i = s, i
        if best_i is None:
            removed.append(a)
            continue

        pool[best_i][2] = True
        b = pool[best_i][0]
        if _node_type(a) != _node_type(b):
            renamed.append({
                "old_node": a,
                "new_node": b,
                "old_type": _node_type(a),
                "new_type": _node_type(b),
                "known": LTX23_TO_LTX25_NODE_RENAME.get(_node_type(a)) == _node_type(b),
            })
        matches.append((a, b))

    unused = sorted((e for pool in buckets_b.values() for e in pool
                     if not e[2]), key=lambda e: e[3])
    added = [e[0] for e in unused]
    return matches, added, removed, renamed

# ----------------------------------------------------------------------
#  Parameter-Diff
# ----------------------------------------------------------------------

def _diff_widgets(wva, wvb):
    """Positionsbasierter Vergleich unbekannter Node-Typen."""
    diffs = []
    for i in range(max(len(wva), len(wvb))):
        va = wva[i] if i < len(wva) else _MISSING
        vb = wvb[i] if i < len(wvb) else _MISSING
        if not _values_equal(va, vb):
            diffs.append({"param": f"widget[{i}]", "old": va, "new": vb})
    return diffs


def _diff_inputs(ina, inb):
    """Benannter Vergleich von API-inputs-Dicts (inkl. Verbindungen)."""
    diffs = []
    for k in sorted(set(ina) | set(inb)):
        va, vb = ina.get(k, _MISSING), inb.get(k, _MISSING)
        if not _values_equal(va, vb):
            diffs.append({"param": k, "old": va, "new": vb})
    return diffs


def _output_signature(node):
    return [(o.get("name"), o.get("type")) for o in node.get("outputs", []) if isinstance(o, dict)]


def diff_params(node_a, node_b):
    """Parameter-Unterschiede zwischen zwei als gleich erkannten Nodes.

    Rueckgabe: list[dict] mit {"param", "old", "new"} - old/new kann
    der Sentinel _MISSING sein (Bedeutung: "nicht vorhanden").
    """
    diffs = []
    ctype = _node_type(node_a)
    if ctype != _node_type(node_b):
        diffs.append({"param": "node_type", "old": ctype, "new": _node_type(node_b)})

    schema = NODE_WIDGET_SCHEMA.get(ctype)
    if schema:
        for key in schema:
            if key in UI_ONLY_WIDGET_KEYS:
                continue
            va = get_node_param(node_a, key, _MISSING)
            vb = get_node_param(node_b, key, _MISSING)
            if va is _MISSING and vb is _MISSING:
                continue
            if not _values_equal(va, vb):
                diffs.append({"param": key, "old": va, "new": vb})

        # Verbindungen (API-Format: inputs-Werte als [node_id, slot]).
        # Nur wenn BEIDE Nodes inputs-Dicts haben - im Workflow-Format
        # liegen Verbindungen in der separaten "links"-Liste und werden
        # hier nicht verglichen (Topologie ist Phase-2-Thema).
        ina = node_a.get("inputs") if isinstance(node_a.get("inputs"), dict) else None
        inb = node_b.get("inputs") if isinstance(node_b.get("inputs"), dict) else None
        if ina is not None and inb is not None:
            for k in sorted(set(ina) | set(inb)):
                if k in schema:
                    continue  # Widget-Werte sind oben schon verglichen
                va, vb = ina.get(k, _MISSING), inb.get(k, _MISSING)
                if not _values_equal(va, vb):
                    diffs.append({"param": f"{k} (Verbindung)", "old": va, "new": vb})
    else:
        # Unbekannter Typ: kein Schema -> positionsbasiert bzw. nach
        # inputs-Name vergleichen.
        wva, wvb = node_a.get("widgets_values"), node_b.get("widgets_values")
        ina = node_a.get("inputs") if isinstance(node_a.get("inputs"), dict) else None
        inb = node_b.get("inputs") if isinstance(node_b.get("inputs"), dict) else None
        if isinstance(wva, list) and isinstance(wvb, list):
            diffs += _diff_widgets(wva, wvb)
        elif ina is not None and inb is not None:
            diffs += _diff_inputs(ina, inb)
        elif isinstance(wva, list) != isinstance(wvb, list):
            diffs.append({"param": "format", "old": wva, "new": wvb})

    # Output-Signatur nur vergleichen, wenn BEIDE Nodes Outputs
    # deklarieren (API-Format-Nodes haben keine -> waere Rauschen).
    out_a, out_b = _output_signature(node_a), _output_signature(node_b)
    if out_a and out_b and out_a != out_b:
        diffs.append({"param": "outputs", "old": out_a, "new": out_b})

    return diffs

# ----------------------------------------------------------------------
#  Top-Level: Vergleich zweier Workflows
# ----------------------------------------------------------------------

def _node_summary(node):
    return {"id": node.get("id"), "type": _node_type(node), "title": _node_title(node)}


def compare_workflows(workflow_a, workflow_b, title_a="Workflow A", title_b="Workflow B"):
    """Vergleicht zwei Workflows (jeweils Workflow- ODER API-Format,
    wird intern ueber workflow_io.normalize_workflow vereinheitlicht).

    Erwartet geladene Workflow-Dicts (das Laden ist Sache von
    workflow_io.load_workflow_file / load_workflow_from_path).

    Rueckgabe: Report-Dict mit
      - error        (str | None)
      - identical    (bool)
      - summary      (einzeilige Zusammenfassung fuer die UI)
      - similarity   (int 0-100; Renames zaehlen als Match)
      - stats        (nodes_a, nodes_b, matched, added, removed,
                      renamed, changed_nodes)
      - added/removed/renamed/changed (Details, siehe format_diff_report)
    """
    empty = {"identical": False, "stats": {}, "added": [], "removed": [],
             "renamed": [], "changed": [], "similarity": 0}

    if not workflow_a or not workflow_b:
        return {**empty, "error": "Beide Workflows muessen geladen sein.",
                "summary": f"{title_a} vs {title_b}: Abbruch - unvollstaendig"}

    wf_a = normalize_workflow(workflow_a)
    wf_b = normalize_workflow(workflow_b)
    if not isinstance(wf_a, dict) or "nodes" not in wf_a or not isinstance(wf_a["nodes"], list):
        return {**empty, "error": f"{title_a} ist kein erkennbarer ComfyUI-Workflow.",
                "summary": f"{title_a} vs {title_b}: Abbruch - unlesbar"}
    if not isinstance(wf_b, dict) or "nodes" not in wf_b or not isinstance(wf_b["nodes"], list):
        return {**empty, "error": f"{title_b} ist kein erkennbarer ComfyUI-Workflow.",
                "summary": f"{title_a} vs {title_b}: Abbruch - unlesbar"}

    nodes_a, nodes_b = wf_a["nodes"], wf_b["nodes"]
    matches, added, removed, renamed = match_nodes(nodes_a, nodes_b)

    changed = []
    for a, b in matches:
        diffs = [d for d in diff_params(a, b) if d["param"] != "node_type"]
        if diffs:
            same_type = _node_type(a) == _node_type(b)
            changed.append({
                "id_a": a.get("id"),
                "id_b": b.get("id"),
                "type": _node_type(b) if same_type else f"{_node_type(a)} -> {_node_type(b)}",
                "title": _node_title(b) or _node_title(a),
                "diffs": diffs,
            })

    stats = {
        "nodes_a": len(nodes_a), "nodes_b": len(nodes_b),
        "matched": len(matches), "added": len(added),
        "removed": len(removed), "renamed": len(renamed),
        "changed_nodes": len(changed),
    }
    total = stats["matched"] + stats["added"] + stats["removed"]
    similarity = round(100 * stats["matched"] / total) if total else 100
    identical = not (added or removed or renamed or changed)

    summary = (
        f"{title_a} -> {title_b}: {len(nodes_a)} vs {len(nodes_b)} Nodes | "
        f"{len(matches)} Match | +{len(added)} / -{len(removed)} | "
        f"{len(renamed)} Rename(s) | {len(changed)} geaendert"
    )
    return {
        "title_a": title_a, "title_b": title_b, "error": None,
        "identical": identical, "summary": summary, "stats": stats,
        "similarity": similarity,
        "added": [_node_summary(n) for n in added],
        "removed": [_node_summary(n) for n in removed],
        "renamed": [{
            "old_id": r["old_node"].get("id"), "new_id": r["new_node"].get("id"),
            "old_type": r["old_type"], "new_type": r["new_type"], "known": r["known"],
            "title": _node_title(r["new_node"]) or _node_title(r["old_node"]),
        } for r in renamed],
        "changed": changed,
    }


def format_diff_report(report):
    """Macht aus dem Report-Dict einen lesbaren deutschen Text (UI)."""
    if not report:
        return "Kein Vergleich durchgefuehrt."

    lines = [f"=== {report.get('title_a', 'A')} vs {report.get('title_b', 'B')} ==="]
    if report.get("error"):
        lines.append(f"Fehler: {report['error']}")
        return "\n".join(lines)

    lines.append(report.get("summary", ""))
    lines.append(
        f"Aehnlichkeit: {report.get('similarity', '?')}% | "
        f"Identisch: {'ja' if report.get('identical') else 'nein'}"
    )
    lines.append("")

    added, removed, renamed, changed = (
        report.get("added"), report.get("removed"),
        report.get("renamed"), report.get("changed"),
    )

    if added:
        lines.append(f"[+ neu] {len(added)}:")
        for n in added:
            lines.append(f"   Node '{n['title'] or n['type']}' ({n['type']}) [ID {n['id']}]")
    if removed:
        lines.append(f"[- entfernt] {len(removed)}:")
        for n in removed:
            lines.append(f"   Node '{n['title'] or n['type']}' ({n['type']}) [ID {n['id']}]")
    if renamed:
        lines.append(f"[* Rename] {len(renamed)}:")
        for r in renamed:
            known = "bekannt" if r["known"] else "UNBEKANNT"
            lines.append(f"   {r['old_type']} -> {r['new_type']} ('{r['title']}') [{known}]")
    if changed:
        lines.append(f"[~ geaendert] {len(changed)}:")
        for c in changed:
            lines.append(f"   Node '{c['title']}' ({c['type']}) [ID {c['id_a']} -> {c['id_b']}]")
            for d in c["diffs"]:
                lines.append(f"      {d['param']}: {_fmt(d['old'])} -> {_fmt(d['new'])}")

    if not (added or removed or renamed or changed):
        lines.append("Keine Unterschiede gefunden.")
    return "\n".join(lines)
