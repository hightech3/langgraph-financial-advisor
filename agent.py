from langgraph.graph import END, StateGraph, START, add_messages, Graph
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.types import Command
from langgraph.prebuilt import create_react_agent
# from langgraph.prebuilt.chat_agent_executor import AgentState
from langgraph_supervisor import create_supervisor
from state import State, AgentState
from tools import initialize_llm, financial_advisor_tool, portfolio_manager_tool
from typing import Literal
from typing_extensions import TypedDict
from langgraph.func import entrypoint, task
import json
from langchain.schema.runnable import RunnableLambda

from dotenv import load_dotenv
import os
load_dotenv()

api_key = os.getenv("API_KEY")
# members = ["financial_advisor", "portfolio_manager"]
# options = members + ["FINISH"]
system_prompt = (
        "You are a team supervisor managing a financial advisor and a portfolio manager. "
        "For finanaical advice, use financial_advisor. "
        "Don't direct to portfolio manager about the datas for financial advice."
        "For portfolio management, use portfolio_manager. "
    )


llm = initialize_llm(api_key=api_key)

def financial_advice_workflow():
    """Create the workflow with the provided API key."""

    workflow = StateGraph(AgentState)
    # Define edges
    workflow.add_node("financial_advisor", financial_advisor_tool)
    workflow.set_entry_point("financial_advisor")

    # Compile workflow
    return workflow.compile(name="financial_advisor_agent")
    
def portfoliio_management_workflow():
    """Create the workflow with the provided API key."""
    workflow = StateGraph(AgentState)

    # Define edges
    workflow.add_node("portfolio_manager", portfolio_manager_tool)
    workflow.set_entry_point("portfolio_manager")

    # Compile workflow
    return workflow.compile(name="portfolio_manager_agent")


financial_advisor_agent = financial_advice_workflow()
portfolio_manager_agent = portfoliio_management_workflow()

# ✅ Pass agents correctly to the supervisor
workflow = create_supervisor(
    agents=[financial_advisor_agent, portfolio_manager_agent],
    model=llm,  # Ensure LLM is initialized
    prompt=system_prompt,
    output_mode="last_message",
    state_schema=AgentState
)


def run_financial_advisor(query: str):
    """
    Run the financial advisor with a user query.
    
    Args:
        query: The user's question or request
        api_key: Optional Google API key to use for this request
    """
    financial_advisor_app = workflow.compile()
    messages = [HumanMessage(content=query)]
    result = financial_advisor_app.invoke({
        "messages": messages,
    })
    messages = result.get("messages", [])
    for msg in messages:
        if isinstance(msg, AIMessage) and isinstance(msg.content, list):
            for item in msg.content:
                if isinstance(item, dict) and item.get("type") == "text":
                    yield item["text"]

        elif isinstance(msg, AIMessage):
            yield msg.content
