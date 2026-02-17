
PromptSwitcher MVP

PromptSwitcher is a lightweight backend tool that converts a single creative idea into optimized prompts for multiple AI image generation models.

The core problem it solves:

Different AI image models respond best to different prompt structures and syntax. Writing separate optimized prompts manually for each model is repetitive and inconsistent.

PromptSwitcher takes one idea (in any language) and outputs:

Clean English interpretation

A Midjourney-optimized prompt

A Leonardo-optimized prompt

A DALL·E-optimized prompt

An Ideogram-optimized prompt

A Firefly-optimized prompt

Live Demo

Deployed on Render:
https://promptswitcher-mvp.onrender.com

What This MVP Does

Accepts a single text idea (any language)

Translates it into clean, visual-focused English

Generates 5 model-specific prompts

Enforces structured JSON output

Repairs malformed JSON automatically

Caches repeated inputs (5-minute TTL)

Runs without accounts, database, or authentication

Architecture Overview
Backend

Python

Flask

OpenAI Responses API

In-memory cache (SHA256 key, 5-minute TTL)

JSON repair fallback for stability

Frontend

Single HTML page

Vanilla JS fetch call

Loading state

Copy-to-clipboard functionality

Dark UI theme

Hosting

GitHub repository (public)

Render (free tier auto-deploy)

Environment variable for OpenAI API key

Environment Setup

Create a .env file locally:

OPENAI_API_KEY=your_key_here


Install dependencies:

pip install -r requirements.txt


Run locally:

python app.py


The app runs on port 8000.

Known Limitations

No authentication

No usage tracking

No rate limiting

Free-tier hosting may sleep when idle

JSON repair step adds extra token usage in edge cases

Project Status

MVP complete.
Stable deployment.
Ready for backend hardening or public beta testing.

Built as an exploration into multi-model prompt optimization.
