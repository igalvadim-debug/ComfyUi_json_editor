# llm_analysis.py - LLM-Anbindung: ausschliesslich minimax H3 API.
# Die Modelle fuer alle KI-Aufrufe kommen NUR aus der MD-Sammlung
# (MD/models.txt, siehe md_whitelist.py) - Agnes AI und lokale
# LLM-Server sind entfernt. Ohne freigegebene MD-Modelle werden
# keine KI-Funktionen ausgefuehrt (klare Meldung statt Raten).

import os
import requests
from dotenv import load_dotenv

from config import PROMPT_PRESETS
from md_whitelist import (
    allowed_api_models, default_api_model, call_minimax_chat,
    MINIMAX_API_KEY_ENV, MD_REPO_URL,
)

load_dotenv()


def md_models_status():
    """(erlaubte Modelle oder None, menschenlesbare Statuszeile)."""
    allowed = allowed_api_models()
    if not allowed:
        return None, ("Keine Modelle in MD/models.txt freigegeben "
                      f"(Datei ist leer/Platzhalter). Pflegequelle: {MD_REPO_URL}")
    return allowed, f"Erlaubte MD-Modelle: {', '.join(allowed)}"


# ═══════════════════════════════════════════════════════════════
#  Einheitliche Prompt-Verbesserung (minimax H3, MD-Modellzwang)
# ═══════════════════════════════════════════════════════════════

def improve_prompt(prompt_text, preset_name, mode="minimax", server_url=""):
    """mode: 'minimax' (einzige erlaubte Quelle; Parameter nur
    Ruewaarts-Kompatibilitaet fuer die UI)."""
    if not prompt_text.strip():
        return "", "Kein Prompt eingegeben"

    allowed, status = md_models_status()
    if not allowed:
        return prompt_text, status

    system_prompt = PROMPT_PRESETS.get(preset_name, list(PROMPT_PRESETS.values())[0])
    result, call_status = call_minimax_chat([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt_text},
    ], model=default_api_model())
    if result is None:
        return prompt_text, call_status
    return result, f"Verbessert mit minimax H3 ({preset_name}) | {call_status}"


# ═══════════════════════════════════════════════════════════════
#  Konvertierungs-Zweitmeinung: minimax H3 bekommt den Report und
#  gibt (nur textuellen!) Rat - aendert NIE automatisch etwas.
# ═══════════════════════════════════════════════════════════════

def suggest_conversion_advice(context_text):
    """Holt minimax-H3-Konvertierungsrat als Text oder gibt den Fehlerstatus aus."""
    allowed, status = md_models_status()
    if not allowed:
        return status
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert on ComfyUI and minimax H3 video workflows. "
                "Given a summary of an automatic rule-based conversion, point out "
                "anything the automatic conversion likely missed or got wrong, and "
                "give concrete, actionable advice. Only reference models and nodes "
                "that appear in the user's MD whitelist lists. Be concise. "
                "Answer in German."
            ),
        },
        {"role": "user", "content": context_text},
    ]
    result, call_status = call_minimax_chat(messages, max_tokens=700)
    return result if result is not None else call_status


# ═══════════════════════════════════════════════════════════════
#  Tavily Web-Suche (unabhaengig vom LLM-Anbieter, unveraendert)
# ═══════════════════════════════════════════════════════════════

def search_workflows_tavily(api_key, query):
    """Sucht mit Tavily und gibt Ergebnisse oder eine Fehlermeldung als Text aus."""
    if not api_key.strip():
        return "Kein Tavily API-Key eingetragen"
    try:
        r = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": api_key, "query": query,
                "max_results": 5, "search_depth": "advanced", "include_answer": True,
            }, timeout=30,
        )
        r.raise_for_status()
        data = r.json()
        lines = []
        if data.get("answer"):
            lines.append(f"Zusammenfassung:\n{data['answer']}\n")
        for res in data.get("results", []):
            lines.append(f"{res['title']}\n{res['url']}\n{res.get('content','')[:200]}...\n")
        return "\n".join(lines) or "Keine Ergebnisse gefunden."
    except Exception as e:
        return f"Tavily-Fehler: {e}"
