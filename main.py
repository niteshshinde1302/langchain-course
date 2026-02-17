from functools import partial
from dotenv import load_dotenv
from langchain_core import output_parsers
from langchain_core.language_models.chat_models import agenerate_from_stream
from langchain_core.tools import structured

load_dotenv()

from langchain_classic import hub
from langchain_classic.agents import AgentExecutor
from langchain_classic.agents.react.agent import create_react_agent
#from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

from schemas import AgentResponse
from prompt import REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
structured_llm = llm.with_structured_output(AgentResponse)
tools = [TavilySearch()]
react_prompt = hub.pull("hwchase17/react")
#output_parser = PydanticOutputParser(pydantic_object=AgentResponse)
react_prompt_with_format_instructions = PromptTemplate(
    template=REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS,
    input_variables=["input","agent_scratchpad", "tool_names", "tools"]
).partial(format_instructions="")

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=react_prompt_with_format_instructions,
)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
extract_output = RunnableLambda(lambda x: x["output"])
#parse_output = RunnableLambda(lambda x: output_parser.parse(x))
chain = agent_executor | extract_output | structured_llm


def main():
    print("Hello from langchain-course!")
    result = chain.invoke(
        input={
            "input": "Search for 3 job postings for an AI Engineer using Langchain in the new york city area on linkedIn and list their details"
        }
    )
    print(result)


if __name__ == "__main__":
    main()
