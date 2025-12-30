from dotenv import load_dotenv
import os

load_dotenv(override=True)
print(os.getenv("OPENAI_API_KEY"))
