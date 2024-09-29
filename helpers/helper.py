import os
from dotenv import load_dotenv, find_dotenv


# Load env variables
def load_env():
    load_dotenv(find_dotenv())


# Load env variable once
load_env()


# Get OpenAI Key
def get_openai_api_key():
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key is None:
        raise ValueError("OPENAI_API_KEY not found in environment variables.")
    return openai_api_key
