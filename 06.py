from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
import requests
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import initialize_agent
from dotenv import load_dotenv
import os

load_dotenv()
HF_KEY = os.getenv('HF_KEY')
WEATHER_API = os.getenv('WEATHER_API')

# Tools
search_tool = DuckDuckGoSearchRun()

@tool
def get_weather_data(city: str) -> str:
    """
    Returns current weather for a given city
    """
    url = f'https://api.weatherstack.com/current?access_key={WEATHER_API}&query={city}'
    response = requests.get(url)
    return str(response.json())

# LLM
llm = ChatOpenAI(
    model="deepseek-ai/DeepSeek-V3-0324",
    api_key=HF_KEY,
    base_url="https://router.huggingface.co/v1",
    temperature=0.7,
    max_tokens=500
)

# Initialize ReAct agent (no need to pull hub manually)
agent = initialize_agent(
    tools=[search_tool, get_weather_data],
    llm=llm,
    agent="zero-shot-react-description",
    verbose=True
)

# Run agent
response = agent.run("What is the current temp of Islamabad?")
print(response)
