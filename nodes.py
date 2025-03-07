from langchain_core.messages import HumanMessage, AIMessage
from state import State
from typing import Literal
from langgraph.types import Command
from agent import financial_advisor_agent, portfolio_management_agent

def financial_advisor_node(state: State) -> Command[Literal["supervisor"]]:
    print("🚀financial_advisor_node")
    result = financial_advisor_agent.invoke(state)
    print("✨result:", result)
    return Command(
        update={
            "messages": [
                AIMessage(content=result["messages"][-1].content, name="financial_advisor")
            ]
        },
        goto="supervisor",
    )

def portfolio_management_node(state: State) -> Command[Literal["supervisor"]]:
    result = portfolio_management_agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="portfolio_manager")
            ]
        },
        goto="supervisor",
    )