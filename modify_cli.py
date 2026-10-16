# modify_cli.py - Kommandozeilen-Zugriff auf workflow_modify (Etappe 2)
#
# Nutzung:
#   jsonvenv\Scripts\python.exe modify_cli.py <workflow.json>
#       -> zeigt verfuegbare Bloeke
#   jsonvenv\Scripts\python.exe modify_cli.py <workflow.json> ^
#       --disable audio --disable upscaler --disable patches ^
#       --disable stage2 --out MusicVideo_mod.json
#
# Der Eingabe-Workflow wird NIE veraendert (deepcopy); alle Bloeke
# werden vor dem Speichern per verify_modification geprueft.

import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from workflow_io import load_workflow_from_path, normalize_workflow
from workflow_modify import BLOCK_DEFS, detect_blocks, apply_mods, verify_modification, count_deactivated


def main():
    ap = argparse.ArgumentParser(description="Modulare Workflow-Modifikation")
    ap.add_argument("workflow", help="Pfad zur Workflow-Datei (JSON)")
    ap.add_argument("--disable", action="append", choices=list(BLOCK_DEFS),
                    help="Block abschalten (mehrfach angeben)")
    ap.add_argument("--out", help="Ausgabedatei (Default: <name>__mod.json neben der Eingabe)")
    args = ap.parse_args()

    wf, msg = load_workflow_from_path(args.workflow)
    if not wf:
        print(f"FEHLER: {msg}")
        sys.exit(1)

    blocks = detect_blocks(wf)
    print(f"{msg} | {len(wf.get('nodes', []))} Nodes, {len(wf.get('links', []))} Links")
    print("--- Verfuegbare Bloeke ---")
    for name, info in blocks.items():
        status = "VORHANDEN" if info["found"] else "nicht gefunden"
        print(f"  {name:16s} {status:15s} ({info['nodes']} Nodes) - {info['label']}")

    if not args.disable:
        print("\nKeine --disable angegeben - nur Analyse. Beispiel:")
        print("  --disable audio --disable upscaler --disable patches --disable stage2")
        return

    mods = {name: True for name in args.disable}
    modified, report = apply_mods(wf, mods)
    muted_count = count_deactivated(modified)

    ok, vmsg = verify_modification(wf, modified)
    print("\n--- Report ---")
    for line in report:
        print(" ", line)
    print(f"\nVerify: {'OK' if ok else 'FEHLER'}\n{vmsg}")
    print(f"Deaktiviert: {muted_count} Nodes (inkl. Subgraphs) | Links: "
          f"{len(wf.get('links', []))} -> {len(modified.get('links', []))}")

    if ok:
        out = args.out or (str(Path(args.workflow).with_name(Path(args.workflow).stem + "__mod.json")))
        with open(out, "w", encoding="utf-8") as f:
            json.dump(modified, f, indent=2, ensure_ascii=False)
        print(f"Gespeichert: {out}")


if __name__ == "__main__":
    main()
