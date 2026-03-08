import asyncio
from google import genai
import os
from dotenv import load_dotenv
load_dotenv(".env")
api_key = os.getenv("GOOGLE_API_KEY")

async def test():
    client = genai.Client(api_key=api_key)
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents="test"
        )
        print(response.text)
    except Exception as e:
        print("Failed!")
        print(repr(e))
        
    try:
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents="test"
        )
        print("Success 2.0-flash!")
    except Exception as e:
        print(repr(e))

asyncio.run(test())
