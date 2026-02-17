import os
import json
import time
from hashlib import sha256
from pathlib import Path

from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv(dotenv_path=Path(__file__).with_name(".env"))



app = Flask(__name__)

# Reads OPENAI_API_KEY from .env
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
CACHE = {}         # {cache_key: (timestamp, parsed_dict)}
CACHE_TTL = 300    # seconds (5 minutes)

SYSTEM = """
You are PromptSwitcher.

Follow ALL rules in this engine strictly.

SECTION 1 — PURPOSE
Create one input field (ANY language).
Output:
- Clean English version
- 5 optimized prompts (one for each AI model)

SECTION 2 — TRANSLATION RULES
Translate meaning, not words.
Keep only visual details.
Remove idioms, slang, cultural phrases.
One short English sentence only.

SECTION 8 — OUTPUT FORMAT (CRITICAL)
Return ONLY valid JSON.
No backticks.
No explanations.
No extra text before or after the JSON.

All values must be SINGLE-LINE strings.
Do NOT include literal line breaks inside values.
If you need line breaks, write "\\n" literally.

If you include a double quote inside a value, escape it as \\".

JSON keys (exactly):
english, midjourney, leonardo, dalle, ideogram, firefly


JSON keys:
english, midjourney, leonardo, dalle, ideogram, firefly
""".strip()


@app.route("/")
def home():
    return render_template("index.html")


def _extract_text(resp) -> str:
    """
    Safer than resp.output_text (which can throw ValueError if empty).
    Tries to extract any text blocks from the response output.
    """
    # First try the convenience property
    try:
        t = (resp.output_text or "").strip()
        if t:
            return t
    except Exception:
        pass

    # Fallback: walk the raw output structure
    parts = []
    for item in (resp.output or []):
        if getattr(item, "type", None) == "message":
            for c in getattr(item, "content", []) or []:
                # Common text fields in SDK objects
                if hasattr(c, "text") and c.text:
                    parts.append(c.text)

    return ("\n".join(parts)).strip()
def _safe_json_loads(text: str):
    """
    Tries hard to parse JSON even if the model accidentally adds extra text.
    """
    if not text:
        raise ValueError("Empty response text")

    s = text.strip()

    # Extract first JSON object if extra text exists
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        s = s[start:end + 1].strip()

    return json.loads(s)

def _parse_json_or_repair(text: str) -> dict:
    """
    1) Try to parse JSON normally.
    2) If the model produced broken JSON (common with quotes/newlines), ask OpenAI to repair it.
    """
    # First attempt
    try:
        return json.loads(text)
    except Exception as e:
        print("JSON PARSE FAILED (first attempt):", repr(e))

    # Repair attempt (cheap + strict)
    repair_instructions = """
You are a JSON repair tool.
Return ONLY valid JSON (no backticks, no commentary).
You must output EXACTLY these keys:
english, midjourney, leonardo, dalle, ideogram, firefly

Rules:
- Keep the meaning as close as possible.
- Escape quotes properly.
- Use \\n inside strings if needed.
- Do not add extra keys.
""".strip()

    repair_resp = client.responses.create(
        model="gpt-5-mini",
        reasoning={"effort": "low"},
        instructions=repair_instructions,
        input=f"Broken JSON to repair:\n{text}",
        max_output_tokens=900,
    )

    repaired_text = _extract_text(repair_resp)
    print("REPAIRED TEXT (raw):", repaired_text[:300], "..." if len(repaired_text) > 300 else "")

    # Second attempt (this should succeed)
    return json.loads(repaired_text)

@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json(silent=True) or {}
    idea = (data.get("idea") or "").strip()

    if not idea:
        return jsonify({"error": "No idea provided"}), 400

    # ---- CACHE (5 min) ----
    cache_key = sha256(idea.encode("utf-8")).hexdigest()
    now = time.time()

    if cache_key in CACHE:
        ts, cached = CACHE[cache_key]
        if now - ts < CACHE_TTL:
            return jsonify(cached)
        else:
            del CACHE[cache_key]
    # -----------------------

    try:
        resp = client.responses.create(
            model="gpt-5-mini",
            reasoning={"effort": "low"},
            instructions=SYSTEM,
            input=f"User idea: {idea}",
            max_output_tokens=800,
            
        )

        text = _extract_text(resp)
        if not text:
            print("OPENAI ERROR: No text returned. Raw output:", resp.output)
            return jsonify({"error": "OpenAI returned no text output"}), 500

        parsed = _parse_json_or_repair(text)



        # Store successful result in cache
        CACHE[cache_key] = (time.time(), parsed)

        return jsonify(parsed)

    except Exception as e:
        print("OPENAI/JSON ERROR:", repr(e))
        return jsonify({"error": f"{type(e).__name__}: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)


