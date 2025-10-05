from typing import List, Union
from dotenv import load_dotenv
from langchain.agents import Tool, tool
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
    llm = ChatOpenAI(
        temperature=0,
        callbacks=[AgentCallbackHandler()],
    )
    from langchain_core.messages import (
        SystemMessage,
        HumanMessage,
        AIMessage,
        ToolMessage,
    )

    bound_llm = llm.bind_tools(tools)

    messages = [
        SystemMessage(
            content=(
                "You are a helpful assistant. Use tools when helpful to answer."
            )
        ),
        HumanMessage(content="What is the text length of DOG in characters"),
    ]

    response = bound_llm.invoke(messages)
    while isinstance(response, AIMessage) and getattr(response, "tool_calls", None):
        tool_messages = []
        for tc in response.tool_calls:
            tool_name = tc.get("name")
            tool_to_use = find_tool_by_name(tools=tools, tool_name=tool_name)
            tool_args = tc.get("args", {})
            observation = tool_to_use.invoke(tool_args)
            tool_messages.append(
                ToolMessage(tool_call_id=tc.get("id"), content=str(observation))
            )

        messages = messages + [response] + tool_messages
        response = bound_llm.invoke(messages)

    # Final response without further tool calls
    print(getattr(response, "content", response))
