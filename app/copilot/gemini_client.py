"""
Thin Gemini API wrapper. Deliberately dumb: no prompt construction, no
grounding, no citation logic lives here -- that's app.copilot.service /
app.copilot.grounding. This module's only job is to call the API and
degrade gracefully on any failure, so a production Gemini outage never
becomes an unhandled exception in the dashboard.
"""
import os

from google import genai

GEMINI_NOT_CONFIGURED_MESSAGE = "Copilot belum dikonfigurasi karena GEMINI_API_KEY belum tersedia."
GEMINI_CALL_FAILED_MESSAGE = "Copilot gagal menghubungi Gemini API. Coba lagi sebentar lagi."


def generate(prompt: str) -> str:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        return GEMINI_NOT_CONFIGURED_MESSAGE

    model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    try:
        client = genai.Client(api_key=key)
        result = client.models.generate_content(model=model, contents=prompt)
        text = getattr(result, "text", None)
        if not text:
            return GEMINI_CALL_FAILED_MESSAGE
        return text
    except Exception:
        # Network errors, auth failures, quota errors, malformed SDK
        # responses, etc. all degrade to the same plain message here --
        # app.copilot.service also wraps its call to this function, so
        # this is a belt-and-suspenders fail-safe, not the only guard.
        return GEMINI_CALL_FAILED_MESSAGE
