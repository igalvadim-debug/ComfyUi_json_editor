# fetch_object_info.py - Holt die Node-Definitionen der LTX-2.3-relevanten
# Node-Typen von einer laufenden ComfyUI-Instanz und speichert sie als
# object_info_ltx.json (im Skript-Ordner).
#
# Beantwortet fuer die Architecture Review die Frage "core oder custom":
#   * Typ vorhanden  -> Definition inkl. input/output-Spezifikation
#   * Typ fehlt      -> in dieser ComfyUI-Instanz nicht installiert
#
# Bewusst NICHT das komplette /object_info (1-2 Mio Zeichen), sondern nur
# die ~31 relevanten Typen (einige hundert KB).
#
# Nutzung:  jsonvenv\Scripts\python.exe fetch_object_info.py [port]
# Default-Port: 8197  (deine Instanz)

import json
import sys
from pathlib import Path

import requests

BASE_URL = "http://127.0.0.1:{port}/object_info"

NODE_NAMES = [
    # Model / LoRA
    "CheckpointLoaderSimple", "LoraLoaderModelOnly",
    # Text-Encoder / Conditioning
    "LTXAVTextEncoderLoader", "CLIPTextEncode", "GemmaAPITextEncode",
    "LTXVConditioning", "LTXVImgToVideoConditionOnly",
    # Latent (Video + Audio)
    "EmptyLTXVLatentVideo", "LTXVEmptyLatentAudio",
    "LTXVAudioVAELoader", "LTXVAudioVAEEncode", "LTXVAudioVAEDecode",
    "LTXVConcatAVLatent", "LTXVSeparateAVLatent",
    # Custom-Sampling-Kette
    "RandomNoise", "ManualSigmas", "BasicScheduler", "LTXVScheduler",
    "CFGGuider", "KSamplerSelect", "SamplerCustomAdvanced",
    # Upscaler / Refiner / Decode
    "LTXVLatentUpsampler", "LatentUpscaleModelLoader",
    "LTXVTiledVAEDecode", "VAEDecodeTiled", "VAEEncodeTiled",
    # Output / Helfer
    "CreateVideo", "SaveVideo", "VHS_VideoCombine",
    "PrimitiveString", "LTXFloatToInt",
]


def main():
    port = sys.argv[1] if len(sys.argv) > 1 else "8197"
    base = BASE_URL.format(port=port)
    out = {}
    missing = []

    for name in NODE_NAMES:
        try:
            r = requests.get(f"{base}/{name}", timeout=10)
            r.raise_for_status()
            data = r.json()
            if isinstance(data, dict) and data.get(name):
                out[name] = data[name]
            else:
                missing.append(name)
        except Exception as e:
            missing.append(f"{name} ({type(e).__name__}: {e})")

    target = Path(__file__).resolve().parent / "object_info_ltx.json"
    with open(target, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"OK: {len(out)}/{len(NODE_NAMES)} Node-Typen geladen -> {target.name}")
    print(f"Dateigroesse: {target.stat().st_size / 1024:.0f} KB")
    if missing:
        print("\nFEHLT in dieser ComfyUI-Instanz (custom nicht installiert):")
        for m in missing:
            print("  -", m)
    else:
        print("\nAlle angefragten Typen sind in dieser ComfyUI-Instanz vorhanden.")


if __name__ == "__main__":
    main()
