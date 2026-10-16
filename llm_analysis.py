# llm_analysis.py - LLM-Anbindung: Agnes AI (bezahlt, ueber .env
# Guthaben) und lokales llama.cpp/LM-Studio/Ollama (kostenlos).
# DeepSeek wurde komplett entfernt.

import os
import requests
from dotenv import load_dotenv

from config import PROMPT_PRESETS, AGNES_CHAT_ENDPOINT, AGNES_DEFAULT_MODEL

load_dotenv()

AGNES_API_KEY = os.getenv("agnes_API_KEY", "")


# ═══════════════════════════════════════════════════════════════
#  Agnes AI (OpenAI-kompatibel)
# ═══════════════════════════════════════════════════════════════

def call_agnes_chat(messages, model=AGNES_DEFAULT_MODEL, max_tokens=800, temperature=0.7):
    """Ruft Agnes AI (OpenAI-kompatible Chat-Completion) auf.
    'messages' im ueblichen OpenAI-Format: [{'role':.., 'content':..}]."""
    if not AGNES_API_KEY:
        return None, "Kein AGNES_API_KEY in .env gefunden (Schluessel-Name: agnes_API_KEY)"
    try:
        r = requests.post(
            AGNES_CHAT_ENDPOINT,
            headers={
                "Authorization": f"Bearer {AGNES_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "messages": messages,
            },
            timeout=60,
        )
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"].strip()
        return content, "OK"
    except Exception as e:
        return None, f"Agnes-Fehler: {e}"


# ═══════════════════════════════════════════════════════════════
#  Lokaler LLM-Server (llama.cpp / LM Studio / Ollama / Kobold,
#  alle OpenAI-kompatibel oder /completion wie llama.cpp)
# ═══════════════════════════════════════════════════════════════

def call_local_llm(system_prompt, user_text, server_url, max_tokens=512, temperature=0.7):
    if not server_url.strip():
        return None, "Keine Server-URL angegeben"
    url = server_url.rstrip("/")
    try:
        if url.endswith("/completion"):
            r = requests.post(url, json={
                "prompt": f"<|system|>{system_prompt}<|user|>{user_text}<|assistant|>",
                "max_tokens": max_tokens, "temperature": temperature,
                "stop": ["<|user|>", "<|system|>"],
            }, timeout=60)
            r.raise_for_status()
            return r.json().get("content", "").strip(), "OK"
        else:
            endpoint = url if url.endswith("/v1/chat/completions") else url + "/v1/chat/completions"
            r = requests.post(endpoint, json={
                "model": "local", "temperature": temperature, "max_tokens": max_tokens,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_text},
                ],
            }, timeout=60)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip(), "OK"
    except Exception as e:
        return None, f"Lokaler LLM-Fehler: {e}"


# ═══════════════════════════════════════════════════════════════
#  Einheitliche Prompt-Verbesserung (waehlt Agnes ODER lokal)
# ═══════════════════════════════════════════════════════════════

def improve_prompt(prompt_text, preset_name, mode="local", server_url=""):
    """mode: 'local' oder 'agnes'."""
    if not prompt_text.strip():
        return "", "Kein Prompt eingegeben"

    system_prompt = PROMPT_PRESETS.get(preset_name, list(PROMPT_PRESETS.values())[0])

    if mode == "agnes":
        result, status = call_agnes_chat([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_text},
        ])
        if result is None:
            return prompt_text, status
        return result, f"Verbessert mit Agnes ({preset_name})"

    result, status = call_local_llm(system_prompt, prompt_text, server_url)
    if result is None:
        return prompt_text, status
    return result, f"Verbessert lokal ({preset_name})"


# ═══════════════════════════════════════════════════════════════
#  Konvertierungs-Zweitmeinung: Agnes bekommt den Konvertierungs-
#  Report und gibt (nur textuellen!) Rat zu offenen Punkten -
#  aendert NIE automatisch etwas am Workflow.
# ═══════════════════════════════════════════════════════════════

def suggest_conversion_advice(context_text):
    if not AGNES_API_KEY:
        return "Kein AGNES_API_KEY in .env gefunden (Schluessel-Name: agnes_API_KEY)"
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert on ComfyUI and Lightricks LTX-Video workflows. "
                "You help migrate LTX 2.3 workflows to LTX 2.5. Given a summary of "
                "an automatic rule-based conversion, point out anything the automatic "
                "conversion likely missed or got wrong, and give concrete, actionable "
                "advice. Be concise. Answer in German."
            ),
        },
        {"role": "user", "content": context_text},
    ]
    result, status = call_agnes_chat(messages, max_tokens=700)
    return result if result is not None else status


# ═══════════════════════════════════════════════════════════════
#  Tavily Web-Suche (unveraendert, unabhaengig vom LLM-Anbieter)
# ═══════════════════════════════════════════════════════════════

def search_workflows_tavily(api_key, query):
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
