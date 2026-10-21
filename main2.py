import os
import time
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=api_key)

context = """
Raghu knows Java and Python.
He also has knowledge of HTML, CSS, and JavaScript.
He has worked with OpenCV and MediaPipe.
"""

question = "What programming languages does Raghu know?"

prompt = f"""
Answer the question using only the information provided in the context.

Context:
{context}

Question:
{question}

Answer:
"""

for attempt in range(3):
    try:
        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=prompt
        )

        print("\nAnswer:")
        print(response.text)
        break

    except Exception as e:
        print(f"\nAttempt {attempt + 1} failed: {e}")

        if attempt < 2:
            wait_time = 5 * (2 ** attempt)
            print(f"Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
        else:
            print("Gemini is currently unavailable. Try again later.")