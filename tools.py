import json
from typing import List
from langchain_core.tools import tool, Tool

import math
import numpy as np
import pandas as pd
import riskfolio as rf
import matplotlib.pyplot as plt

from state import State, AgentState
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage

from dotenv import load_dotenv
import os
load_dotenv()

api_key = os.getenv("API_KEY")

def initialize_llm(api_key=None):
    return ChatAnthropic(model="claude-3-5-sonnet-latest", temperature=1.0, api_key=api_key)


def financial_advisor_tool(state: AgentState) -> AgentState:
    """Process the agent state and determine the next step."""
    messages = state["messages"]
    # Initialize the LLM with the provided API key
    llm = initialize_llm(api_key)
    system_content = ("You are a helpful financial advisor."
                      "You help clients build investment portfolios based on their financial goals, risk tolerance, and investment horizon."
                      "Use the build_portfolio function to generate recommendations."
                      "Do not show python code in your reply."
                      "Don't make disclaimer!" )
    filtered_messages = [msg for msg in messages if hasattr(msg, "content")]
    if not filtered_messages or not (isinstance(filtered_messages[0], HumanMessage) and filtered_messages[0].content.startswith("You are a helpful financial advisor")):
        filtered_messages = [HumanMessage(content=system_content)] + filtered_messages
    if len(filtered_messages) == 1:
        filtered_messages.append(HumanMessage(content="Hello, I need financial advice."))
    # if sum(1 for msg in filtered_messages if isinstance(msg, AIMessage)) >= 5:
    #     return {"messages": filtered_messages, "next": END}
    response = llm.invoke(filtered_messages)

    filtered_messages.append(response)
    print("🚀financial_advisor_tool", response)
    if hasattr(response, "tool_calls") and response.tool_calls:
        for tool_call in response.tool_calls:
            tool_name = tool_call.function.name
            tool_arguments = json.loads(tool_call.function.arguments)

            # Call the original build_portfolio tool
            tool_result = build_portfolio(**tool_arguments)
            filtered_messages.append(HumanMessage(content=f"Tool Result: {tool_result}"))

            return {"messages": filtered_messages}

    return {"messages": filtered_messages}

# @tool
def build_portfolio(risk_tolerance: str, investment_amount: float, investment_horizon: int, goals: List[str]) -> str:
    
    """
    Build a portfolio based on the given risk tolerance, investment amount, investment horizon, and goals.

    Args:
        risk_tolerance (str): The risk tolerance level (e.g., 'low', 'medium', 'high').
        investment_amount (float): The amount of money to invest.
        investment_horizon (int): The investment horizon in years.
        goals (List[str]): The investment goals.

    Returns:
        str: A JSON string representing the portfolio allocation.
    """
    print("🚀financial_advisor_tool", risk_tolerance)
    portfolio = {
        "low": {"bonds": 0.60, "stocks": 0.30, "cash": 0.10},
        "medium": {"bonds": 0.40, "stocks": 0.50, "cash": 0.10},
        "high": {"bonds": 0.20, "stocks": 0.70, "alternatives": 0.10},
    }
    selected_portfolio = portfolio.get(risk_tolerance.lower(), portfolio["medium"])
    allocation = {asset: round(percentage * investment_amount, 2) for asset, percentage in selected_portfolio.items()}
    return json.dumps({
        "risk_profile": risk_tolerance,
        "investment_amount": investment_amount,
        "time_horizon": investment_horizon,
        "goals": goals,
        "allocation_percentages": selected_portfolio,
        "allocation_amounts": allocation,
        "expected_annual_return": {"low": "3-5%", "medium": "5-8%", "high": "8-12%"}.get(risk_tolerance.lower(), "5-8%"),
    }, indent=2)

financial_tool = Tool(
    name="build_portfolio",
    description="Build an investment portfolio based on client parameters",
    func=build_portfolio
)



def calc_risk_aversion(value):
    if not 1 <= value <= 10:
        raise ValueError("Input value must be between 1 and 10.")
    # Logarithmic mapping
    log_min = math.log10(0.01)  # Log base 10 of the minimum output value
    log_max = math.log10(100)   # Log base 10 of the maximum output value
    log_value = log_max + (value - 10) * (log_min - log_max) / (1 - 10)
    mapped_value = 10**log_value
    return mapped_value



def portfolio_manager_tool(state: AgentState):
    """Process user-provided allocation and generate a chart."""

    messages = state["messages"]
    # Initialize the LLM with the provided API key

    llm = initialize_llm(api_key)
    system_content = ("You are a helpful portfolio manager."
                      "You help clients can optimize their portfolios based on allocations."
                      "Use the portfolio_optimization function to optimize portfolio."
                      "Do not show python code in your reply."
                      "Don't make disclaimer!" )

    # Extract last user message (expected to contain allocation)
    filtered_messages = messages
    filtered_messages = [HumanMessage(content=system_content)] + filtered_messages
    if len(filtered_messages) == 1:
        filtered_messages.append(HumanMessage(content="Hello, I have to optimize my financial portfolio with given allocations."))
    # Parse user-provided allocation (Expected format: "Bonds: 50, Stocks: 30, Real Estate: 20")
    response = llm.invoke(filtered_messages)
    filtered_messages.append(response)

    if hasattr(response, "tool_calls") and response.tool_calls:
        for tool_call in response.tool_calls:
            tool_name = tool_call.function.name
            tool_arguments = json.loads(tool_call.function.arguments)

            # Call the original build_portfolio tool
            tool_result = portfolio_optimization(**tool_arguments)
            print("🚀portfolio_manager_tool", tool_result)
            filtered_messages.append(AIMessage(content=f" Tool Result: {tool_result}"))
            return {"messages": filtered_messages}
    return {"messages": filtered_messages}

# @tool
def portfolio_optimization(stocks: float, bonds: float, international_developed_markets: float, international_emerging_markets: float, cryptocurrency: float) -> str:
    """
    Generate portfolio constraints based on asset class, allocation strategy, liquidity levels, covariance matrix, and expected returns.
    And optimize the portfolio based on the constraints by using riskfolio.
    
    Args:
        stocks (float): The percentage of the portfolio allocated to stocks.
        bonds (float): The percentage of the portfolio allocated to bonds.
        international_developed_markets (float): The percentage of the portfolio allocated to international developed markets.
        international_emerging_markets (float): The percentage of the portfolio allocated to international emerging markets.
        cryptocurrency (float): The percentage of the portfolio allocated to cryptocurrency.

    Returns:
        str: A JSON string representing the optimized portfolio data.

    """
    constraints = []
    asset_data = pd.DataFrame({
        "Asset Class": ["Equity", "Fixed Income", "Cryptocurrency", "International Developed", "International Emerging Market"],
        "Liquidity": ["High", "Medium", "Low", "Medium", "High"]
    })


    # Determine liquidity strategy dynamically
    high_liquidity_assets = asset_data[asset_data['Liquidity'].isin(['High', 'Medium'])]
    low_liquidity_assets = asset_data[asset_data['Liquidity'] == 'Low']
    liquidity_condition = 'high' if len(high_liquidity_assets) > len(low_liquidity_assets) else 'low'
    
    # Define base constraints using allocation strategy
    base_constraints = {
        'Equity': stocks,
        'Fixed Income': bonds,
        'International Developed': international_developed_markets,
        'International Emerging Market': international_emerging_markets,
        'Cryptocurrency': cryptocurrency if liquidity_condition == 'low' else None,
    }
    asset_class_cov = pd.read_csv(r'CMA - Cov.csv', header=None)
    asset_class_mu = pd.read_csv(r'CMA - Mu.csv', header=None)
    
    # Adjust constraints based on covariance risk and expected returns
    risk_adjustment_factor = 0.1  # Can be tuned based on risk appetite
    for i, asset in enumerate(asset_data["Asset Class"]):
        asset_risk = asset_class_cov.iloc[i, i]  # Variance of asset
        asset_return = asset_class_mu.iloc[i, 0]  # Expected return
        
        # Modify weight limits based on risk-return characteristics
        if asset in base_constraints:
            adj_limit = base_constraints[asset][liquidity_condition] * (1 + risk_adjustment_factor * (asset_return / asset_risk))
            constraints.append(["FALSE", "Assets", "", asset, "<=", min(adj_limit, 1.0), "", "", "", ""])
    
    # Liquidity constraints
    liquidity_limits = {'low': 0.6, 'high': 0.08}
    constraints.append(["FALSE", "Classes", "Liquidity", "Low", ">=", liquidity_limits[liquidity_condition], "", "", "", ""])
    
    # Minimum allocation per asset
    min_allocation = {'low': 0.02, 'high': 0.01}
    constraints.append(["FALSE", "All Assets", "", "", ">=", min_allocation[liquidity_condition], "", "", "", ""])
    
    # Special case: Ensuring a minimum allocation to a specific asset (e.g., TLT for Fixed Income)
    constraints.append(["TRUE", "Each asset in a class", "Class 1", "Equity", ">=", "", "Assets", "", "TLT", 0.4])
    
    # Convert to DataFrame
    columns = ["Disabled", "Type", "Set", "Position", "Sign", "Weight", "Type Relative", "Relative Set", "Relative", "Factor"]
    asset_class_constraints = pd.DataFrame(constraints, columns=columns)

    asset_class_names = pd.read_csv(r'CMA - Asset Class.csv')
    asset_class_names['Asset Class'] = asset_class_names['Asset Class'].str.strip()
    
    asset_class_names = asset_class_names.drop(columns='ETF Ticker').sort_values('Asset Class')

    asset_class = sorted(asset_class_names['Asset Class'])
    returns = pd.DataFrame([[0]*len(asset_class)]*2, columns=asset_class)

    port = rf.Portfolio(returns=returns)
    port.mu = asset_class_mu[0].values
    port.cov = asset_class_cov.values
    model = 'Classic'
    rm = 'MV'
    obj = 'MaxRet' # 'Utility'
    riskfree_rate = 0.02
    risk_score = 1.0 # 1 to 10
    risk_aversion = calc_risk_aversion(risk_score)
    A, B = rf.assets_constraints(asset_class_constraints, asset_class_names)
    port.ainequality = A
    port.binequality = B
    port.solver = 'ECOS'
    w = port.optimization(model=model, rm=rm, obj=obj, rf=riskfree_rate, l=risk_aversion)
    frontier = port.efficient_frontier(model=model, rm=rm, points=50)
    ax = rf.plot_pie(w=w, title='Sharpe Mean Variance', others=0.05, nrow=25, cmap = "tab20", height=6, width=10, ax=None)
    plt.show()
    return json.dumps({
        "weights": w,
        "frontier": frontier,
    }, indent=2)

portfolio_tool = Tool(
    name="portfolio_optimization",
    description="Build an investment portfolio based on client parameters",
    func=portfolio_optimization
)