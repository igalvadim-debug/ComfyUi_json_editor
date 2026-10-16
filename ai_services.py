# ai_services.py – llama.cpp, Deepseek, Tavily

import requests
from config import PROMPT_PRESETS
from workflow_analysis import analyze_workflow


def improve_prompt_llamacpp(prompt_text, server_url, preset_name):
    """Улучшает промпт через локальный llama.cpp сервер."""
    if not prompt_text.strip():
        return "", "Kein Prompt eingegeben"
    if not server_url.strip():
        return prompt_text, "Keine Server-URL angegeben"

    system_prompt = PROMPT_PRESETS.get(preset_name, list(PROMPT_PRESETS.values())[0])
    url = server_url.rstrip("/")

    try:
        if url.endswith("/completion"):
            r = requests.post(url, json={
                "prompt": f"<|system|>{system_prompt}<|user|>{prompt_text}<|assistant|>",
                "max_tokens": 512, "temperature": 0.7,
                "stop": ["<|user|>", "<|system|>"]
            }, timeout=60)
            r.raise_for_status()
            result = r.json().get("content", "").strip()
        else:
            endpoint = url if url.endswith("/v1/chat/completions") else url + "/v1/chat/completions"
            r = requests.post(endpoint, json={
                "model": "local", "temperature": 0.7, "max_tokens": 512,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": prompt_text}
                ]
            }, timeout=60)
            r.raise_for_status()
            result = r.json()["choices"][0]["message"]["content"].strip()

        return result, f"Verbessert ({preset_name})"
    except Exception as e:
        return prompt_text, f"Fehler: {e}"


def suggest_settings_deepseek(workflow, api_key, with_ltx):
    """Запрашивает у Deepseek оптимальные настройки для workflow."""
    if not api_key.strip():
        return "Kein Deepseek API-Key eingetragen"
    if not workflow:
        return "Kein Workflow geladen"

    _, df, _ = analyze_workflow(workflow)
    node_types = list(set(df["Class Type"].tolist())) if not df.empty else []
    context = f"ComfyUI workflow with nodes: {', '.join(node_types)}"
    if with_ltx:
        context += ". This is an LTX Video workflow."

    try:
        r = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": "deepseek-chat", "max_tokens": 600,
                "messages": [{
                    "role": "user",
                    "content": (
                        f"Based on this {context}, suggest optimal settings:\n"
                        "- Steps, CFG, Sampler, Scheduler\n"
                        "- Resolution (width x height)\n"
                        "- For LTX Video: Frames, FPS, Sigmas Shift\n"
                        "Be specific with numbers. Format as a clear list."
                    )
                }]
            }, timeout=30
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Deepseek-Fehler: {e}"


def search_workflows_tavily(api_key, query):
    """Ищет похожие workflow через Tavily."""
    if not api_key.strip():
        return "Kein Tavily API-Key eingetragen"
    try:
        r = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": api_key, "query": query,
                "max_results": 5, "search_depth": "advanced", "include_answer": True
            }, timeout=30
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
