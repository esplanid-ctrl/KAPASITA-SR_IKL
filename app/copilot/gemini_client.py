import os
from google import genai

def generate(prompt):
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        return "Copilot belum dikonfigurasi karena GEMINI_API_KEY belum tersedia."
    client = genai.Client(api_key=key)
    model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    return client.models.generate_content(model=model, contents=prompt).text
