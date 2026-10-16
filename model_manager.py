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


# ----------------------------------------------------------------------
#  Modell-Katalog-Cache
# ----------------------------------------------------------------------
# scan_models() liest pro Aufruf den kompletten Ordner-Bestand von der
# Platte. find_25_equivalent() wird vom Konverter fuer JEDEN betroffenen
# Dateinamen aufgerufen und war ohne Cache ein Full-Disk-Rescan pro
# Datei. Gecacht wird pro (Kategorie, Ordnerliste); Gueltigkeit ueber
# eine leichte "Bestands-Signatur" (Anzahl + groester/mittelster
# File-Mtime - kein globbing, ~ein stat()-Aufruf pro Datei). Neue oder
# geloeschte Dateien aendern die Signatur und fuehren beim naechsten
# Aufruf automatisch zu einem frischen Einlesen.
_MODEL_SCAN_CACHE = {}      # (category, folders) -> (stamp, [names])
_LTX_VERSION_CACHE = {}     # (all_folders, stamp) -> result


def _folder_stamp(folder):
    """Return (entry_count, max_mtime_ns) for a folder's immediate entries.

    Count directories and symlinks too. Skip timestamps that raise OSError,
    using zero if none are readable. Empty folders or directory-scan
    OSError produce (0, 0.0). Changes preserving both values can be missed.

    Currently raises NameError because this module does not import os.
    """
    try:
        with os.scandir(folder) as it:
            entries = list(it)
        if not entries:
            return (0, 0.0)
        mtimes = []
        for e in entries:
            try:
                mt = e.stat(follow_symlinks=False).st_mtime_ns
            except OSError:
                continue
            if mt is None:
                try:
                    mt = os.stat(e.path).st_mtime_ns
                except OSError:
                    continue
            mtimes.append(mt)
        return (len(entries), max(mtimes) if mtimes else 0)
    except OSError:
        return (0, 0.0)


def invalidate_model_cache():
    """Setzt den Modell-Katalog-Cache zurueck (z.B. nach dem Anlegen
    neuer Modell-Ordner oder in Tests)."""
    _MODEL_SCAN_CACHE.clear()
    _LTX_VERSION_CACHE.clear()


def scan_models(category, yaml_paths):
    """Return sorted, unique matching basenames from a category's folders.

    yaml_paths maps categories to folder lists. Match immediate children
    using MODEL_EXTENSIONS, or .safetensors/.gguf/.bin/.pt for unknown
    categories. Return ["(keine Modelle gefunden)"] when nothing matches.
    Cache names until folder entry counts or maximum entry mtimes change;
    changes preserving both can leave stale results.

    NameError from _folder_stamp propagates when any folder is configured,
    because this module does not import os.
    """
    exts = MODEL_EXTENSIONS.get(category, [".safetensors", ".gguf", ".bin", ".pt"])
    folders = tuple(yaml_paths.get(category, []))
    key = (category, folders)
    stamp = tuple(_folder_stamp(f) for f in folders)
    cached = _MODEL_SCAN_CACHE.get(key)
    if cached is not None and cached[0] == stamp:
        found = list(cached[1])
    else:
        found = []
        for folder in folders:
            p = Path(folder)
            if p.exists():
                for ext in exts:
                    found += [f.name for f in p.glob(f"*{ext}")]
        _MODEL_SCAN_CACHE[key] = (stamp, sorted(set(found)))
        found = list(_MODEL_SCAN_CACHE[key][1])
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
    """Return sorted, deduplicated LTX basenames grouped by version.

    Scan MODEL_EXTENSIONS categories using yaml_paths' folder lists.
    Include every LTX_VERSION_HINTS key, even when its list is empty.
    Reuse cached groups for the same combined folder set and stamps;
    return independent copies of the lists. Changes to category assignments
    alone do not invalidate the cached grouping.

    NameError from _folder_stamp propagates when any relevant folder is
    configured, because this module does not import os.
    """
    # Cache-Key: alle relevanten Ordner + deren Stamps (ueber den
    # Kategorie-Cache von scan_models cheap abgeleitet)
    all_folders = tuple(sorted({f for cat in MODEL_EXTENSIONS
                                for f in yaml_paths.get(cat, [])}))
    stamp = tuple(_folder_stamp(f) for f in all_folders)
    ckey = (all_folders, stamp)
    cached = _LTX_VERSION_CACHE.get(ckey)
    if cached is not None:
        return {v: list(names) for v, names in cached.items()}

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
    # Nur einen Eintrag pro Ordner-Kombination halten (Stamp-Aenderungen
    # erzeugen neue Keys - altes Limit: die letzten 8 Katalog-Staende).
    if len(_LTX_VERSION_CACHE) >= 8:
        _LTX_VERSION_CACHE.clear()
    _LTX_VERSION_CACHE[ckey] = result
    return {v: list(names) for v, names in result.items()}


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
