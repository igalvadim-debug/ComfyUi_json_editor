# config.py - Konstanten, Presets, Node-Mapping-Tabellen

TITLE = "ComfyUI Workflow Manager & Optimizer"
YAML_PATH = r"D:\ComfyUiVid\extra_model_paths.yaml"

# ---------------------------------------------------------------
#  Modell-Kategorien (fuer den YAML/Ordner-Scanner)
# ---------------------------------------------------------------
MODEL_EXTENSIONS = {
    "checkpoints":       [".safetensors", ".ckpt"],
    "unet":              [".safetensors", ".gguf", ".bin"],
    "diffusion_models":  [".safetensors", ".gguf", ".bin"],
    "vae":               [".safetensors", ".ckpt", ".pt"],
    "clip":              [".safetensors", ".bin"],
    "text_encoders":     [".safetensors", ".bin", ".gguf"],
    "loras":             [".safetensors", ".ckpt", ".pt"],
    "controlnet":        [".safetensors", ".pth"],
    "upscale_models":    [".pth", ".safetensors"],
}

# ---------------------------------------------------------------
#  LTX Node-Typen (2.3 UND 2.5 zusammen, fuer die Erkennung
#  "ist das ueberhaupt ein LTX-Workflow")
# ---------------------------------------------------------------
LTX_NODE_TYPES = {
    "LTXVLoader", "LTX23Sigmas", "LTXVSampler", "LTXVScheduler",
    "LTXVConditioning", "LTXVAddNoise", "LTXVImgToVideo",
    "LTXVImgToVideoConditionOnly", "LTXVImgToVideoInplace",
    "LTXVideoLatentImage", "LTXVModel", "LTXVPreprocess",
    "EmptyLTXVLatentVideo", "LTXVEmptyLatentAudio",
    "LTXVConcatAVLatent", "LTXVSeparateAVLatent",
    "LTXVAudioVAELoader", "LTXVAudioVAEDecode", "LTXVTiledVAEDecode",
    "VAEDecodeTiled", "LTXAVTextEncoderLoader", "GemmaAPITextEncode",
    "MultimodalGuider", "GuiderParameters", "TextGenerateLTX2Prompt",
    "LTXFloatToInt",
}

# ---------------------------------------------------------------
#  Node-Typen, die zwischen 2.3 und 2.5 umbenannt wurden.
#  (aus dem Vergleich der echten Beispiel-Workflows ermittelt)
# ---------------------------------------------------------------
LTX23_TO_LTX25_NODE_RENAME = {
    "LTXVTiledVAEDecode":            "VAEDecodeTiled",
    "LTXVImgToVideoConditionOnly":   "LTXVImgToVideoInplace",
}

# ---------------------------------------------------------------
#  Node-Typen, die den Checkpoint-Loader ersetzen (2.5 laedt
#  UNET / CLIP / VAE getrennt statt eines einzelnen Checkpoints)
# ---------------------------------------------------------------
CHECKPOINT_LOADER_TYPES = {"CheckpointLoaderSimple", "CheckpointLoader"}

# Vorlage fuer die neuen Split-Loader-Nodes (Werte werden zur
# Laufzeit befuellt, siehe converter_ltx25.py)
SPLIT_LOADER_TEMPLATE = {
    "unet":        {"type": "UNETLoader",  "title": "UNET (LTX 2.5)"},
    "clip_encode": {"type": "CLIPLoader",  "title": "Load CLIP - Text Encoder"},
    "vae_video":   {"type": "VAELoader",   "title": "Load VAE - Video"},
    "vae_audio":   {"type": "VAELoader",   "title": "Load VAE - Audio"},
}

# ---------------------------------------------------------------
#  Widget-Schema: welche Werte stecken (in welcher Reihenfolge)
#  in "widgets_values" der jeweiligen Node-Typen.
#  Nur fuer die Nodes hinterlegt, die fuer Analyse/Konvertierung
#  relevant sind - alles andere wird generisch behandelt.
# ---------------------------------------------------------------
NODE_WIDGET_SCHEMA = {
    "CheckpointLoaderSimple":     ["ckpt_name"],
    "CheckpointLoader":           ["ckpt_name"],
    "UNETLoader":                 ["unet_name", "weight_dtype"],
    "CLIPLoader":                 ["clip_name", "type", "device"],
    "DualCLIPLoaderGGUF":         ["clip_name1", "clip_name2", "type"],
    "VAELoader":                  ["vae_name"],
    "UnetLoaderGGUF":             ["unet_name"],
    "LoraLoaderModelOnly":        ["lora_name", "strength_model"],
    "LoraLoader":                 ["lora_name", "strength_model", "strength_clip"],
    "CLIPTextEncode":             ["text"],
    "CLIPTextEncodeSDXL":         ["text"],
    "smZ CLIPTextEncode":         ["text"],
    "PrimitiveStringMultiline":   ["value"],
    "PrimitiveString":            ["value"],
    "PrimitiveInt":               ["value", "control_after_generate"],
    "PrimitiveFloat":             ["value", "control_after_generate"],
    "PrimitiveBoolean":           ["value"],
    "EmptyLTXVLatentVideo":       ["width", "height", "length", "batch_size"],
    "LTXVEmptyLatentAudio":       ["frames_number", "frame_rate", "batch_size"],
    "KSampler":                   ["seed", "control_after_generate", "steps", "cfg",
                                    "sampler_name", "scheduler", "denoise"],
    "KSamplerAdvanced":           ["add_noise", "noise_seed", "control_after_generate",
                                    "steps", "cfg", "sampler_name", "scheduler",
                                    "start_at_step", "end_at_step", "return_with_leftover_noise"],
    "LTXVScheduler":              ["steps", "max_shift", "base_shift",
                                    "stretch", "terminal"],
    "LTXVConditioning":           ["frame_rate"],
    "VHS_VideoCombine":           ["frame_rate"],
    "CreateVideo":                ["fps"],
    "GemmaAPITextEncode":         ["prompt", "system_prompt", "model"],
    "LTXVAudioVAELoader":         ["vae_name"],
    "LTXAVTextEncoderLoader":     ["clip_name", "model_type", "device"],
    "LoadImage":                  ["image", "choose_file_to_upload"],
}

# ---------------------------------------------------------------
#  Muster, mit denen wir erkennen, ob eine Datei/ein Node zu
#  LTX 2.3, LTX 2.5 oder LTX-2 19B gehoert (fuer Versions-
#  Erkennung, automatisches Datei-Mapping und Fehleranalyse).
# ---------------------------------------------------------------
LTX_VERSION_HINTS = {
    "2.3": ["ltx-2.3", "ltx2.3", "ltx_2_3"],
    "2.5": ["ltx-2.5", "ltx2.5", "ltx_2_5"],
    "19B": ["ltx-2-19b", "ltx2-19b", "ltx_2_19b", "ltx-19b"],
}

# ---------------------------------------------------------------
#  LTX-Modellgroessen (Parameter in Milliarden) - fuer die
#  GGUF-Groessenschaetzung in diagnostics.py
# ---------------------------------------------------------------
LTX_PARAMS_B = {
    "19B": 19,
    "2.3": 22,
    "2.5": 22,
}

# Effektive Bits/Parameter je Quant-Stufe - kalibriert an echten
# Dateigroessen von LTX-2-19B-GGUF-Repos (HuggingFace, Stand
# aktuelle Community-Uploads). Fuer 2.3/2.5 (22B) wird linear
# nach Parameterzahl hochskaliert -> Werte sind SCHAETZUNGEN,
# reale Dateigroesse im jeweiligen HF-Repo pruefen.
GGUF_BITS_PER_PARAM = {
    "BF16":   16.0,
    "Q8_0":    8.6,
    "Q6_K":    7.0,
    "Q5_K_M":  6.3,
    "Q4_K_M":  5.6,
    "Q3_K_M":  4.3,
    "Q3_K_S":  4.0,
    "Q2_K":    3.4,
}
GGUF_QUANT_ORDER = ["BF16", "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q3_K_S", "Q2_K"]
GGUF_QUANT_QUALITY_NOTE = {
    "BF16":   "volle Praezision, Referenzqualitaet",
    "Q8_0":   "praktisch verlustfrei",
    "Q6_K":   "minimaler Qualitaetsverlust",
    "Q5_K_M": "guter Kompromiss, kaum sichtbarer Verlust",
    "Q4_K_M": "spuerbarer, meist akzeptabler Verlust - gaengiger Sweet-Spot fuer 8-12GB",
    "Q3_K_M": "deutlicherer Verlust, fuer sehr knappes VRAM",
    "Q3_K_S": "deutlicher Verlust, kleinste praktikable Stufe",
    "Q2_K":   "starker Qualitaetsverlust, nur als letzter Ausweg",
}

# ---------------------------------------------------------------
#  GPU-VRAM-Presets fuer die Fehleranalyse/Optimierung
# ---------------------------------------------------------------
GPU_VRAM_PRESETS = {
    "RTX 5060 Laptop (8 GB)":  8,
    "RTX 4060 (8 GB)":         8,
    "RTX 3060 (12 GB)":        12,
    "RTX 4070 (12 GB)":        12,
    "RTX 5070 (12 GB)":        12,
    "RTX 4070 Ti / Super (16 GB)": 16,
    "RTX 4080 (16 GB)":        16,
    "RTX 5080 (16 GB)":        16,
    "RTX 3090 (24 GB)":        24,
    "RTX 4090 (24 GB)":        24,
    "RTX 5090 (32 GB)":        32,
    "Eigener Wert":            8,
}

# ---------------------------------------------------------------
#  Prompt-Presets fuer die Prompt-Verbesserung
# ---------------------------------------------------------------
PROMPT_PRESETS = {
    "LTX-optimiert": (
        "You are an expert prompt engineer for LTX Video generation. "
        "Rewrite the following prompt to be highly detailed, temporally coherent, "
        "with rich motion descriptions, camera movement, lighting. "
        "Keep it under 200 words. Output ONLY the improved prompt, nothing else."
    ),
    "Cinematic / Filmisch": (
        "You are a cinematographer writing prompts for AI video. "
        "Rewrite the following prompt with cinematic language: camera angles, lens type, "
        "color grading, lighting mood, film grain. Output ONLY the improved prompt."
    ),
    "Detailliert / Hochdetailliert": (
        "Rewrite the following prompt with maximum visual detail: textures, materials, "
        "colors, shapes, spatial relationships, lighting. Output ONLY the improved prompt."
    ),
    "Kurz & Stark": (
        "Rewrite the following prompt as a short, powerful, punchy version (max 50 words). "
        "Keep the core idea but make every word count. Output ONLY the improved prompt."
    ),
    "Mehr Stil / Artistisch": (
        "Rewrite the following prompt with strong artistic style: painterly, illustrative, "
        "inspired by great artists and movements. Output ONLY the improved prompt."
    ),
    "Technisch / Praezise": (
        "Rewrite the following prompt with precise technical language for AI image/video generation. "
        "Include technical parameters, render style, quality tokens. Output ONLY the improved prompt."
    ),
    "Deutsch -> Englisch": (
        "Translate the following German prompt to English. "
        "Improve vocabulary for AI image/video generation. Output ONLY the translated prompt."
    ),
}

# ---------------------------------------------------------------
#  Agnes AI (OpenAI-kompatibel) - ersetzt DeepSeek
# ---------------------------------------------------------------
AGNES_BASE_URL = "https://apihub.agnes-ai.com/v1"
AGNES_CHAT_ENDPOINT = AGNES_BASE_URL + "/chat/completions"
AGNES_DEFAULT_MODEL = "agnes-2.0-flash"

# ---------------------------------------------------------------
#  Lokaler LLM-Server (llama.cpp / LM Studio / Ollama / Kobold)
# ---------------------------------------------------------------
LOCAL_LLM_DEFAULT_URL = "http://127.0.0.1:8080"
