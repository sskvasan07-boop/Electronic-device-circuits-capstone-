"""Quick diagnostic with new SDK -- run: python test_api.py"""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env", override=True)
    print("dotenv loaded")
except ImportError:
    print("python-dotenv not installed")

key = os.environ.get("GEMINI_API_KEY", "").strip()
print(f"Key present: {bool(key)}")
if key:
    print(f"Key preview: {key[:12]}...{key[-4:]}")
else:
    print("ERROR: GEMINI_API_KEY is empty! Edit .env and add your key.")
    exit(1)

from google import genai

client = genai.Client(api_key=key)

print("\n-- Listing available models --")
try:
    models = [m.name for m in client.models.list()]
    flash_models = [m for m in models if "flash" in m or "pro" in m]
    for m in flash_models[:8]:
        print(f"  {m}")
except Exception as e:
    print(f"list() failed: {e}")

print("\n-- Testing generate_content --")
for model_name in ["gemini-2.0-flash-lite", "gemini-2.0-flash", "gemini-1.5-flash"]:
    print(f"  Trying {model_name}...", end=" ")
    try:
        resp = client.models.generate_content(model=model_name, contents="Say OK")
        print(f"SUCCESS: {resp.text.strip()}")
        break
    except Exception as e:
        err = str(e)
        print(f"FAILED")
        print(f"    Error: {err[:400]}")
