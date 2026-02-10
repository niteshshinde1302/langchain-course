from dotenv import load_dotenv
load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from tavily import TavilyClient
'''
client = TavilyClient()

@tool
def search(query: str) -> str:
    """
    Search the web for information
    Args:
        query: The query to search for
    Returns:
        The search results
    """
    print(f"Searching for: {query}")
    return client.search(query)
    #return "Tokyo weather is sunny"
'''

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools)

def main():
    print("Hello from langchain-course!")
    response = agent.invoke({"messages": HumanMessage(content="Search for 3 jobs for an AI Engineer using Langchain in New York City area on LinkedIn and list their details")})
    print(response)

if __name__ == "__main__":
    main()
