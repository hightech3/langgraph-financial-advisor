from langgraph.graph import MessagesState, END
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from state import AgentState
from nodes import agent_node, get_user_allocation, initialize_llm
from typing import Literal
from typing_extensions import TypedDict

api_key = "AIzaSyD_V5dkAj-HG0_fLH_WLDYEd9WdQPodCy0"
members = ["financial_advisor", "portfolio_manager"]
options = members + ["FINISH"]
system_prompt = (
    "You are a supervisor tasked with managing a conversation between the"
    f" following workers: {members}. Given the following user request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status. When finished,"
    " respond with FINISH."
)

class Router(TypedDict):
    next: Literal["financial_advisor", "portfolio_manager", "FINISH"]

llm = initialize_llm(api_key=api_key)

class State(MessagesState):
    next: str

def supervisor_node(state: State) -> Command[Literal["financial_advisor", "portfolio_manager", "__end__"]]:
    messages = [
        {"role": "system", "content": system_prompt},
    ] + state["messages"]
    response = llm.with_structured_output(Router).invoke(messages)
    goto = response["next"]
    if goto == "FINISH":
        goto = END

    return Command(goto=goto, update={"next": goto})