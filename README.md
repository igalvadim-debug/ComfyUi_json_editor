# ComfyUI JSON Workflow Manager & LTX Optimizer

Gradio-Tool zum Laden, Analysieren, Reparieren und Optimieren von
ComfyUI-Workflows - mit Schwerpunkt auf LTX Video (2.3, 2.5, 19B).
Unterstuetzt sowohl klassische Workflows als auch solche mit
ComfyUI-**Subgraphs** (verschachtelte Node-Gruppen, z.B. mehrere
"Scene"-Instanzen in einem Musikvideo-Workflow).

## Start

```
start.bat
```
bzw. manuell:
```
jsonvenv\Scripts\python.exe app.py
```
Oeffnet die Gradio-Oberflaeche im Browser (Standard: http://127.0.0.1:7860).

Einmalige Einrichtung: `install.bat` (legt `jsonvenv` an, installiert
`requirements.txt`).

---

## Aktive Module (werden von der App importiert)

### Einstieg

| Datei | Zustaendigkeit |
|---|---|
| `app.py` | Einstiegspunkt. Keine Logik - importiert `ui.py`, startet `demo.launch()`. |
| `ui.py` | Komplette Gradio-Oberflaeche. Enthaelt **keine** Geschaeftslogik, nur UI-Aufbau + Verdrahtung der Buttons mit den Funktionen aus den anderen Modulen. Zwei parallele Spuren **A/B** (zwei Workflows gleichzeitig geladen), pro Tab als zwei Spalten. |
| `config.py` | Alle Konstanten an einer Stelle: Node-Mapping-Tabellen (2.3->2.5), Widget-Schema (siehe unten), Prompt-Presets, LTX-Versions-Erkennungsmuster, GGUF-Groessen/Quant-Tabellen, GPU/VRAM-Presets, Agnes-AI-Endpunkt. |

### Workflow lesen/schreiben/verstehen

| Datei | Zustaendigkeit |
|---|---|
| `workflow_io.py` | Laden (JSON / PNG / WebP / MP4 mit eingebettetem Workflow) und Speichern (JSON, PNG mit Workflow-Metadaten). Normalisiert das aeltere "API-Format" (dict-basiert) auf das gaengige "Workflow-Format" (`nodes`/`links`-Listen). |
| `workflow_analysis.py` | Node-Uebersicht (Tabelle), Prompt-Extraktion, LTX-Versions-Erkennung (2.3/2.5/19B). Kernstueck: `get_node_param()`/`set_node_param()` - liest/schreibt Werte egal ob sie in `widgets_values` (positionsbasiert, echtes ComfyUI-Workflow-Format) oder in `inputs` (direkte Werte, API-Format) stehen. Das **Widget-Schema** dazu liegt in `config.NODE_WIDGET_SCHEMA`. |
| `workflow_compare.py` | Vergleicht zwei Workflows inhaltlich (nicht `dict==dict`, da IDs/Position/Groesse sich aendern koennen, ohne dass sich der Workflow aendert). Matched Nodes ueber Typ+Titel+Parameter-Fingerprint, erkennt bekannte 2.3->2.5-Umbenennungen automatisch als "Rename" statt "entfernt+neu". Wird intern von `workflow_modify.verify_modification()` genutzt, um vor/nach einer Modifikation zu pruefen, dass keine Nodes veraendert wurden. *(Aktuell kein eigener UI-Tab, aber `format_diff_report()` liefert fertigen Text fuer die Anzeige.)* |

### LTX 2.3 -> 2.5 Konvertierung

| Datei | Zustaendigkeit |
|---|---|
| `converter_ltx25.py` | Regelbasierte Konvertierung eines LTX-2.3-Workflows nach 2.5: Node-Umbenennungen, Checkpoint-Loader -> UNET+CLIP+2xVAE-Split-Loader, uebrige 2.3-Dateireferenzen umbiegen (wo lokal ein 2.5-Pendant gefunden wird), inkompatible 2.3-LoRAs stummschalten statt loeschen. Liefert einen Text-Report; nichts wird "erraten" ohne das im Report zu vermerken. |
| `model_manager.py` | Liest `extra_model_paths.yaml`, scannt Modell-Ordner nach Kategorie, erkennt LTX-Dateien nach Version (2.3/2.5/19B) anhand des Dateinamens, sucht zu einer 2.3-Datei die passende 2.5-Datei im lokalen Katalog (kategorie-bewusst: VAE nur gegen VAE-Dateien usw., rein LTX-19B). |

### Workflow modifizieren (Bloecke abschalten)

| Datei | Zustaendigkeit |
|---|---|
| `workflow_modify.py` | Schaltet definierte Bloecke (audio, upscaler, patches, stage2, outputs, enhance_prompt) per **Bypass** (`mode=4`) ab statt Links zu loeschen - jederzeit gefahrlos reaktivierbar. **Subgraph-bewusst**: durchsucht Top-Level UND alle `definitions.subgraphs` (z.B. mehrere "Scene"-Instanzen in einem Musikvideo-Workflow), inkl. globaler rgthree-Set/Get-Bruecken, die ueber Subgraph-Grenzen hinweg wirken. `verify_modification()` prueft danach automatisiert: keine Nodes veraendert, alle Links konsistent, keine aktiven Nodes mit verlorener Verbindung. |
| `modify_cli.py` | Kommandozeilen-Version von `workflow_modify.py` fuer Batch-Verarbeitung ohne UI: `python modify_cli.py workflow.json --disable audio --disable upscaler --out result.json`. |

### Sonstige Workflow-Transformationen

| Datei | Zustaendigkeit |
|---|---|
| `transformers.py` | GGUF-Ersetzung (Checkpoint/UNET -> UnetLoaderGGUF + DualCLIPLoaderGGUF + VAELoader), LTX-Schnelleinstellungen (Aufloesung/Frames/Steps/CFG/Shift in einem Rutsch setzen), verbesserten Prompt in Positive/Negative-Nodes einsetzen. Nutzt durchgehend `get_node_param`/`set_node_param` (siehe `workflow_analysis.py`). |

### KI-Anbindung

| Datei | Zustaendigkeit |
|---|---|
| `llm_analysis.py` | **Aktive** LLM-Anbindung: Agnes AI (bezahlt, OpenAI-kompatibel, `AGNES_API_KEY` aus `.env`) fuer Prompt-Verbesserung und Konvertierungs-/Fehleranalyse-Zweitmeinungen; lokaler llama.cpp/LM-Studio/Ollama-Server als kostenlose Alternative; Tavily-Websuche. |
| `diagnostics.py` | Fehleranalyse & GGUF-Optimierung: parst eingefuegte ComfyUI-Logs/Fehlermeldungen nach bekannten Mustern (OOM, Checkpoint-Mismatch, fehlender Custom-Node, Geraete-Mismatch, korrupte Datei, Shape-Fehler, ...), erkennt GPU/VRAM automatisch aus dem Log, prueft den Workflow strukturell (GGUF vs. Vollpraezision bei wenig VRAM, fehlendes Tiled-VAE-Decode, Aufloesung/Batch), und schaetzt GGUF-Dateigroessen je Quant-Stufe (kalibriert an echten LTX-2-19B-Dateigroessen, linear auf 22B skaliert) mit konkreter Quant-Empfehlung. |

---

## Legacy / Standalone-Tools (nicht Teil der Haupt-App)

| Datei | Status |
|---|---|
| `ai_services.py` | **Veraltet, wird von `ui.py` nicht mehr importiert.** Fruehere LLM-Anbindung inkl. DeepSeek - abgeloest durch `llm_analysis.py` (Agnes AI). Kann geloescht werden, wurde aber vorerst nicht angefasst. |
| `fetch_object_info.py` | Eigenstaendiges CLI-Skript: fragt eine laufende ComfyUI-Instanz nach den echten Input/Output-Definitionen bestimmter LTX-Node-Typen ab (`.../object_info/<NodeType>`) und speichert sie als `object_info_ltx.json`. Nuetzlich, um zu pruefen, ob ein Node-Typ "core" oder "custom" ist bzw. welche Custom-Node-Pakete in der eigenen Installation fehlen. Aufruf: `python fetch_object_info.py [port]` (Default-Port 8197). |
| `object_info_ltx.json` | Datenausgabe von `fetch_object_info.py` (kein Code). |

## Tests

| Datei | Zustaendigkeit |
|---|---|
| `tests/test_workflow_compare.py` | Unit-Tests fuer `workflow_compare.py` (Node-Matching, Rename-Erkennung, Parameter-Diff). Laufbar mit oder ohne pytest: `python tests/test_workflow_compare.py`. |
| `tests/test_workflow_modify.py` | Unit-Tests fuer `workflow_modify.py` (Bypass-Verhalten, Link-Unversehrtheit, Subgraph-Kaskaden, Verify). Laufbar mit oder ohne pytest: `python tests/test_workflow_modify.py`. |

Alle Tests ausfuehren:
```
jsonvenv\Scripts\python.exe -m pytest tests/
```

---

## Modul-Abhaengigkeiten (grob)

```
app.py
 └─ ui.py
     ├─ config.py
     ├─ model_manager.py       ──> config.py
     ├─ workflow_io.py
     ├─ workflow_analysis.py   ──> config.py
     ├─ converter_ltx25.py     ──> config.py, workflow_analysis.py, model_manager.py
     ├─ transformers.py        ──> workflow_analysis.py
     ├─ llm_analysis.py        ──> config.py
     ├─ workflow_modify.py     ──> workflow_io.py, workflow_analysis.py, workflow_compare.py
     └─ diagnostics.py         ──> config.py, workflow_analysis.py

modify_cli.py  ──> workflow_io.py, workflow_modify.py   (unabhaengig von ui.py)
fetch_object_info.py                                     (komplett unabhaengig)
```

## Datei-Formate, die verstanden werden

- **ComfyUI-Workflow-Format** (Standard, aus "Save" in der UI oder PNG/MP4-Metadaten): Top-Level `nodes`/`links`-Listen, Node-Parameter in `widgets_values` (positionsbasiert - siehe `config.NODE_WIDGET_SCHEMA`), optional `definitions.subgraphs` fuer verschachtelte Node-Gruppen.
- **ComfyUI-API-Format** (aus "Save (API Format)"): dict, Keys = Node-IDs, Werte enthalten `class_type` + `inputs` (direkte Werte). Wird beim Laden automatisch auf das Workflow-Format normalisiert.

## .env (nicht eingecheckt)

Erwartete Variable: `agnes_API_KEY` (Agnes AI, siehe `llm_analysis.py`).
