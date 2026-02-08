import os
import json
import time
from hashlib import sha256

from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

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
           response_format={"type": "json_object"},
 
      )

        text = _extract_text(resp)
        if not text:
            print("OPENAI ERROR: No text returned. Raw output:", resp.output)
            return jsonify({"error": "OpenAI returned no text output"}), 500

        parsed = json.loads(text)

        # Store successful result in cache
        CACHE[cache_key] = (time.time(), parsed)

        return jsonify(parsed)

    except Exception as e:
        print("OPENAI/JSON ERROR:", repr(e))
        return jsonify({"error": "OpenAI failed"}), 500



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
