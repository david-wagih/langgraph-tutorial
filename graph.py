from typing import Annotated
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.checkpoint.memory import MemorySaver

# An in-memory checkpoint saver., in production, we would use a database so we will need to change this to PostgresSaver or SqliteSaver
memory = MemorySaver()

# to get the state of the graph at any time
# snapshot = graph.get_state(config)


class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)


tool = DuckDuckGoSearchRun()
tools = [tool]
llm = ChatOllama(model="mistral")
llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)

# when chatbot is called, we check if it needs tools then call tools, if not, the graph will stop
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
# Any time a tool is called, we return to the chatbot to decide the next step
graph_builder.add_edge("tools", "chatbot")
graph_builder.set_entry_point("chatbot")
graph = graph_builder.compile(
    checkpointer=memory,
    # this one is to interrupt the tools node before it is called (human in the loop),
    # # `None` will append nothing new to the current state, letting it resume as if it had never been interrupted,
    # after interrupting, we can manually update the state using the update_state function
    interrupt_before=["tools"],
    # we can also interrupt the chatbot node after it is called (human in the loop)
    # interrupt_after=["chatbot"],s
)


# we can get back to a certain point using get_state_history() method
