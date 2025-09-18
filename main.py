from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

load_dotenv()

# instatiate a chat model:
# choice 1: OpenAI "gpt-5" for chat or "o4-mini-deep-research" for the specific deep research
llm = ChatOpenAI(model="gpt-4o-mini")

#choice 2: Anthropic
#llm2 = ChatAnthropic(model= "claude-3-7-sonnet-20250219")

#choice 3: gemini

from langchain.chat_models import init_chat_model
llm3 = init_chat_model("gemini-2.5-flash", model_provider="google_genai")

# testing the llm and API_KEY:
#response = llm3.invoke("What is a graph theory?")
#print(response) #OK

# Create prompt template:

