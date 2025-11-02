from langgraph.graph import StateGraph, MessageState, START, END

def mock_llm(state: MessagesState):
    return {"messages": [{"role": "assistant", "content": "Mock response"}]}

graph = StateGraph(MessageState)
graph.add_node(mock_llm)
graph.add_edge(START, "mock_llm")
graph.add_edge("mock_llm", END)
graph = graph.compile()

graph.invoke({"messages": [{"role": "user", "content": "Hello"}]})
