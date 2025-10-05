from typing import List, Union
from dotenv import load_dotenv
from langchain.agents import Tool, tool
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.prompts import PromptTemplate, prompt
from langchain_core.tools import render_text_description
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain_openai import ChatOpenAI

# from langchain_google_genai import ChatGoogleGenerativeAI
from callbacks import AgentCallbackHandler

load_dotenv()

@tool
def get_text_length(text: str) -> int:
    """Returns the length of the text by characters"""
    print(f"get_text_length enter with {text=}")
    text = text.strip("'\n").strip(
        '"'
    )  # stripping away non alphabetic characters just in case
    return len(text)


def find_tool_by_name(tools: List[Tool], tool_name: str) -> Tool:
    for tool in tools:
        if tool.name == tool_name:
            return tool
    return ValueError(f"Tool {tool_name} not found")


if __name__ == "__main__":
    print("Hello React Langchain")
    tools = [get_text_length]
    template = """
    Answer the following questions as best you can. You have access to the following tools:

    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin!

    Question: {input}
    Thought: {agent_scratchpad}
    """
    prompt = PromptTemplate.from_template(template=template).partial(
        tools=render_text_description(tools),
        tool_names=", ".join([t.name for t in tools]),
    )
    llm = ChatOpenAI(
        temperature=0,
        stop=["\nObservation", "Observation", "Observation:"],
        callbacks=[AgentCallbackHandler()],
    )
    # llm = ChatGoogleGenerativeAI(model="gemini-2.0.flash", temperature=0, model_kwargs={"stop":["\nObservation"]})
    intermediate_steps = []
    from langchain.agents.format_scratchpad import format_log_to_str

    """
    In LCEL, the output of first action acts as input to next action.
    prompt expect a dictionary as input and return output of type PromptValue; 
    llm expect a string input or PromptValue and ouputs a string
    """
    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_log_to_str(x["agent_scratchpad"]),
        }
        | prompt
        | llm
        | ReActSingleInputOutputParser()
    )

    agent_step = ""
    while not isinstance(agent_step, AgentFinish):
        agent_step: Union[AgentAction, AgentFinish] = agent.invoke(
            {
                "input": "What is the text length of DOG in characters",
                "agent_scratchpad": intermediate_steps,
            }
        )
        print(agent_step)

        """
        ReAct agent Flow is like:
                    LLM         Parsing         Tool            OK
        Query--->Agent---->Thought--------->Tool----------->Output---->Answer
                ^     Call                      Execution    |
                |                                            |Not
                |--------------------------------------------|OK
        """

        if isinstance(agent_step, AgentAction):
            tool_name = agent_step.tool
            tool_to_use = find_tool_by_name(tools=tools, tool_name=tool_name)
            tool_input = agent_step.tool_input

            observation = tool_to_use(str(tool_input))
            print(observation)
            intermediate_steps.append((agent_step, str(observation)))

    if isinstance(agent_step, AgentFinish):
        print(agent_step.return_values)
