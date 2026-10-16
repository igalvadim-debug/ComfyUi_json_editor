# model_manager.py - YAML lesen, Modell-Ordner scannen,
# LTX-Versionen (2.3 / 2.5 / 19B) erkennen und zueinander mappen

from pathlib import Path
import yaml
from config import YAML_PATH, MODEL_EXTENSIONS, LTX_VERSION_HINTS


def load_yaml_paths(yaml_path=YAML_PATH):
    """Liest die ComfyUI extra_model_paths.yaml und liefert
    {kategorie: [ordnerpfad, ...]}."""
    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        paths = {}
        for block_name, block in data.items():
            if not isinstance(block, dict):
                continue
            base = block.get("base_path", "")
            for key, rel in block.items():
                if key == "base_path":
                    continue
                full = str(Path(base) / str(rel).lstrip("./\\"))
                category = key.lower()
                paths.setdefault(category, []).append(full)
        return paths, None
    except Exception as e:
        return {}, str(e)


def scan_models(category, yaml_paths):
    """Listet alle Modell-Dateien einer Kategorie (nur Dateinamen)."""
    exts = MODEL_EXTENSIONS.get(category, [".safetensors", ".gguf", ".bin", ".pt"])
    found = []
    for folder in yaml_paths.get(category, []):
        p = Path(folder)
        if p.exists():
            for ext in exts:
                found += [f.name for f in p.glob(f"*{ext}")]
    return sorted(set(found)) if found else ["(keine Modelle gefunden)"]


def get_all_model_lists(yaml_paths):
    """Liefert {kategorie: [dateiname, ...]} fuer alle Kategorien."""
    return {cat: scan_models(cat, yaml_paths) for cat in MODEL_EXTENSIONS}


def _version_of_filename(filename):
    """Erkennt anhand des Dateinamens, ob es sich um eine
    LTX-2.3-, LTX-2.5- oder LTX-2-19B-Datei handelt. Gibt den
    Versions-Schluessel aus LTX_VERSION_HINTS oder None zurueck."""
    low = filename.lower()
    for version, hints in LTX_VERSION_HINTS.items():
        if any(h in low for h in hints):
            return version
    return None


def scan_ltx_models_by_version(yaml_paths):
    """Durchsucht ALLE Modell-Kategorien nach LTX-Dateien und
    gruppiert sie nach Version: {'2.3': [...], '2.5': [...], '19B': [...]}.
    Die Versions-Schluessel werden dynamisch aus LTX_VERSION_HINTS
    uebernommen, damit neue Versionen nicht zu einem KeyError fuehren."""
    result = {version: [] for version in LTX_VERSION_HINTS}
    seen = set()
    for category in MODEL_EXTENSIONS:
        names = scan_models(category, yaml_paths)
        for name in names:
            if name == "(keine Modelle gefunden)" or name in seen:
                continue
            seen.add(name)
            v = _version_of_filename(name)
            if v:
                result[v].append(name)
    for version in result:
        result[version] = sorted(result[version])
    return result


# Schluesselwoerter, die eine Datei-Kategorie erkennen lassen -
# damit ein VAE-Node niemals faelschlich mit einem Transformer/
# UNET-Dateinamen befuellt wird und umgekehrt.
CATEGORY_KEYWORDS = {
    "vae":         ["vae"],
    "clip":        ["clip", "gemma", "text_encoder", "textencoder"],
    "transformer": ["transformer", "unet", "dev", "distilled"],
}


def find_25_equivalent(filename_23, yaml_paths, category=None):
    """Versucht zu einer LTX-2.3-Datei die passende 2.5-Datei im
    lokalen Modell-Katalog zu finden.

    'category' schraenkt die Suche auf eine Datei-Kategorie ein
    ('vae', 'clip' oder 'transformer'), damit z.B. ein VAE-Loader
    niemals einen Transformer-Dateinamen zugewiesen bekommt.
    Wird nichts Passendes gefunden, wird bewusst None
    zurueckgegeben statt zu raten.

    Gibt (dateiname_oder_None, kandidatenliste) zurueck."""
    ltx_files = scan_ltx_models_by_version(yaml_paths)
    candidates_25 = ltx_files.get("2.5", [])

    if category and category in CATEGORY_KEYWORDS:
        keywords = CATEGORY_KEYWORDS[category]
        candidates_25 = [c for c in candidates_25 if any(k in c.lower() for k in keywords)]

    if not candidates_25:
        return None, ltx_files.get("2.5", [])

    # 1) Exakter Namensabgleich: 'ltx-2.3' -> 'ltx-2.5'
    guess = None
    low = filename_23.lower()
    for hint in LTX_VERSION_HINTS["2.3"]:
        if hint in low:
            guess = low.replace(hint, "ltx-2.5")
            break
    if guess:
        for c in candidates_25:
            if c.lower() == guess:
                return c, candidates_25

    # 2) Fallback: Token-Ueberlappung, aber nur innerhalb der
    #    bereits nach Kategorie gefilterten Kandidaten, und nur
    #    wenn es eine echte Uebereinstimmung gibt (score > 1)
    stem_tokens = set(low.replace("-", "_").split("_"))
    best, best_score = None, 1
    for c in candidates_25:
        c_tokens = set(c.lower().replace("-", "_").split("_"))
        score = len(stem_tokens & c_tokens)
        if score > best_score:
            best, best_score = c, score
    return best, candidates_25
