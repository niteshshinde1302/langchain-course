from unittest import result
from dotenv import load_dotenv
load_dotenv()

from langchain import hub
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain.agents import AgentExecutor
from langchain.agents.react.agent import create_react_agent
from langchain_tavily import TavilySearch
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from schemas import AgentResponse
from prompt import REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS

tools = [TavilySearch()]
#llm = ChatOllama(model="gemma3:270m")
#llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
llm = ChatOpenAI(model="gpt-4")
react_prompt = hub.pull("hwchase17/react")
output_parser = PydanticOutputParser(pydantic_object=AgentResponse)
react_prompt_with_format_instructions = PromptTemplate(
    template=REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS,
    input_variables=["input", "agent_scratchpad", "tools"],
    ).partial(
        format_instructions=output_parser.get_format_instructions()
    )

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=react_prompt_with_format_instructions,
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
extract_output = RunnableLambda(lambda x: x["output"])
parse_output = RunnableLambda(lambda x: output_parser.parse(x))
chain = agent_executor | extract_output | parse_output

def main():
    print("Hello from langchain-course!")
    result = chain.invoke(
        input={
            "input": "search for 3 job postings for an AI engineer in the bay area on LinkedIn and list their details",
        }
    )
    print(result)


if __name__ == "__main__":
    main()
