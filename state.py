from typing import Annotated, Optional, TypedDict
from langgraph.graph.message import add_messages
from langgraph.graph import MessagesState, END


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    remaining_steps: Optional[int]

class State(MessagesState):
    next: str