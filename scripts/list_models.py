"""Print the models this Groq API key can actually reach.

Groq account tiers differ in which models they expose, so a model id from the
docs is not guaranteed to work for you:

    python -m scripts.list_models
"""

from __future__ import annotations

import sys

from groq import Groq

from src.agent.config import CONFIG

# Models that are useless for this agent even when the key can see them.
NON_CHAT_HINTS = ("whisper", "orpheus", "prompt-guard", "safeguard")


def main() -> int:
    if not CONFIG.groq_api_key or CONFIG.groq_api_key == "gsk_your_key_here":
        print("GROQ_API_KEY is not set in .env", file=sys.stderr)
        return 1

    model_ids = sorted(m.id for m in Groq(api_key=CONFIG.groq_api_key).models.list().data)
    chat = [m for m in model_ids if not any(h in m for h in NON_CHAT_HINTS)]

    print(f"{len(model_ids)} models visible to this key. Usable for the agent:\n")
    for model_id in chat:
        marker = "  <- current AGENT_MODEL" if model_id == CONFIG.model else ""
        print(f"  {model_id}{marker}")

    if CONFIG.model not in model_ids:
        print(f"\nWARNING: AGENT_MODEL={CONFIG.model!r} is NOT available to this key.")
        print("Set AGENT_MODEL in .env to one of the ids above.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
