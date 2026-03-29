import os
import google.generativeai as genai
genai.configure(api_key=os.environ.get('GOOGLE_API_KEY') or '')
try:
    models = genai.list_models()
    print("SUPPORTED EMBEDDING MODELS:")
    for m in models:
        if 'embedContent' in m.supported_generation_methods:
            print(f"- {m.name}")
except Exception as e:
    print(e)
