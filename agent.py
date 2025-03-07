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
        "For portfolio management, use portfolio_manager. "
    )

# system_prompt = (
#     "You are a supervisor tasked with managing a conversation between the"
#     f" following workers: {members}. Given the following user request,"
#     " respond with the worker to act next. Each worker will perform a"
#     " task and respond with their results and status. When finished,"
#     " respond with FINISH."
# )

# class Router(TypedDict):
#     next: Literal["financial_advisor", "portfolio_manager", "FINISH"]

llm = initialize_llm(api_key=api_key)



# def supervisor_node(state: State) -> Command[Literal["financial_advisor", "portfolio_manager", "__end__"]]:
#     messages = [
#         {"role": "system", "content": system_prompt},
#     ] + state["messages"]
#     response = llm.with_structured_output(Router).invoke(messages)
#     print("✨response.next:",  response["next"])
#     goto = response["next"]
#     if goto == "FINISH":
#         goto = END

#     return Command(goto=goto, update={"next": goto})

# financial_advisor_agent = create_react_agent(
#     llm,
#     tools=[build_portfolio],
#     name="financial_advisor",
#     state_schema=AgentState,
# )

# @entrypoint()
# def financial_advisor_agent(state: State):
#     messages = state["messages"]
# #     # Initialize the LLM with the provided API key
#     llm = initialize_llm(api_key)
#     system_content = ("You are a helpful financial advisor."
#                       "You help clients build investment portfolios based on their financial goals, risk tolerance, and investment horizon."
#                       "Use the build_portfolio function to generate recommendations."
#                       "Do not show python code in your reply."
#                       "Don't make disclaimer!" )
#     filtered_messages = [msg for msg in messages if hasattr(msg, "content") and msg.content.strip()]
#     filtered_messages = [{
#         "role": "user",
#         "content": system_content
#     }]
#     # if len(filtered_messages) == 1:
#     #     filtered_messages.append(HumanMessage(content="Hello, I need financial advice."))
#     # if sum(1 for msg in filtered_messages if isinstance(msg, AIMessage)) >= 5:
#     #     return {"messages": filtered_messages, "next": END}
#     response = llm.invoke(filtered_messages)

#     print("🚀financial_advisor_tool", response)
#     filtered_messages.append({
#         "role": "system",
#         "content": response.content
#     })
#     message = add_messages(state["messages"], filtered_messages)
#     if hasattr(response, "tool_calls") and response.tool_calls:
#         for tool_call in response.tool_calls:
#             tool_name = tool_call.function.name
#             tool_arguments = json.loads(tool_call.function.arguments)

#             # Call the original build_portfolio tool
#             tool_result = build_portfolio(**tool_arguments)
#             filtered_messages.append({"role": "user", "content": f"Tool Result: {tool_result}"})
#             message = add_messages(state["messages"], filtered_messages)
#             return {"messages": message}

#     return {"messages": message}

# def financial_advisor_node(state: State) -> Command[Literal["supervisor"]]:
#     print("🚀financial_advisor_node")
#     result = financial_advisor_agent.invoke(state)
#     print("✨result:", result)
#     return Command(
#         update={
#             "messages": [
#                 AIMessage(content=result["messages"][-1].content, name="financial_advisor")
#             ]
#         },
#         goto="supervisor",
#     )

# portfolio_management_agent = create_react_agent(
#     llm,
#     tools=[portfolio_optimization],
#     name="portfolio_manager",
#     state_schema=AgentState,
# )

# def portfolio_management_node(state: State) -> Command[Literal["supervisor"]]:
#     result = portfolio_management_agent.invoke(state)
#     return Command(
#         update={
#             "messages": [
#                 HumanMessage(content=result["messages"][-1].content, name="portfolio_manager")
#             ]
#         },
#         goto="supervisor",
#     )


def financial_advice_workflow():
    """Create the workflow with the provided API key."""

    workflow = StateGraph(AgentState)
    # Define edges
    workflow.add_node("financial_advisor", financial_advisor_tool)
    workflow.set_entry_point("financial_advisor")
    workflow.add_edge("financial_advisor", END)

    # Compile workflow
    return workflow.compile(name="financial_advisor_agent")
    
def portfoliio_management_workflow():
    """Create the workflow with the provided API key."""
    workflow = StateGraph(AgentState)

    # Define edges
    workflow.add_node("portfolio_manager", portfolio_manager_tool)
    workflow.set_entry_point("portfolio_manager")
    workflow.add_edge("portfolio_manager", END)

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
    result = financial_advisor_app.stream({
        "messages": messages,
    }, stream_mode="value")

    for s in result:
        message = s["messages"][-1]
        print("content ============> ", message.content)
        yield message.content
        # if isinstance(message, tuple):
        #     print(message)
        #     yield message
        # else:
        #     message.pretty_print()
        #     yield message.content
