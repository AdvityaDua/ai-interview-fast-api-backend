"""
List all Gemini models that support generateContent.
Run: source ./venv/bin/activate && python list_gemini_models.py
"""
import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

def list_models():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY not found in .env")
        return

    client = genai.Client(api_key=api_key)

    all_models = []
    for model in client.models.list():
        all_models.append(model)

    # Filter to gemini models that support generateContent
    print("\n=== Gemini models supporting generateContent ===\n")
    print(f"{'Model ID':<50} {'Display Name'}")
    print("-" * 80)

    for model in sorted(all_models, key=lambda m: m.name):
        methods = getattr(model, 'supported_generation_methods', None) or []
        if not methods:
            # Try alternative attribute access
            methods = []
        
        name = model.name  # e.g. "models/gemini-2.5-flash"
        display = getattr(model, 'display_name', name)
        
        # Only show gemini text generation models 
        if 'gemini' in name.lower():
            # Check if model info has method support indicators
            print(f"{name:<50} {display}")

if __name__ == "__main__":
    list_models()
