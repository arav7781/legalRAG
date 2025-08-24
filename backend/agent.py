from typing import Literal
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import tools_condition, ToolNode
from config import llm, embeddings_model, qdrant_client, logger

@tool
def retriever_tool(query: str, collection_name: str) -> str:
    """Retrieve relevant documents from Qdrant based on the query."""
    try:
        if embeddings_model is None:
            return "Embeddings model not available. Cannot perform retrieval."
        
        # Generate query embedding
        query_embedding = embeddings_model.embed_query(query)
        
        # Search in Qdrant
        search_result = qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=5
        )
        
        # Format results
        documents = []
        for result in search_result:
            documents.append(result.payload["text"])
        
        if not documents:
            return "No relevant documents found."
        
        return "\n\n".join(documents)
        
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        return "No relevant documents found."

def grade_documents(state) -> Literal["generate", "rewrite"]:
    """Determines whether the retrieved documents are relevant to the question."""
    logger.info("---CHECK RELEVANCE---")
    
    messages = state["messages"]
    last_message = messages[-1]
    question = messages[0].content
    docs = last_message.content
    
    # Create a simple relevance check prompt
    prompt = f"""
    You are assessing the relevance of retrieved documents to a user question.
    
    Question: {question}
    Documents: {docs[:500]}...
    
    Are these documents relevant to answer the question? Respond with only 'yes' or 'no'.
    """
    
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        decision = response.content.strip().lower()
        
        if "yes" in decision:
            logger.info("---DECISION: DOCS RELEVANT---")
            return "generate"
        else:
            logger.info("---DECISION: DOCS NOT RELEVANT---")
            return "rewrite"
    except Exception as e:
        logger.error(f"Document grading failed: {e}")
        # Default to generate if assessment fails
        return "generate"

def agent(state):
    """Agent that decides whether to retrieve documents or end."""
    logger.info("---CALL AGENT---")
    messages = state["messages"]
    collection_name = state["collection_name"]
    
    # Bind the retriever tool to the model
    tools = [retriever_tool]
    model_with_tools = llm.bind_tools(tools)
    
    # Add system message about using retrieval
    system_prompt = HumanMessage(
        content=f"""You are an AI assistant with access to a document retrieval tool. 
        Use the retriever_tool to find relevant information from the collection '{collection_name}' 
        to answer user questions. Always use the tool first before providing an answer."""
    )
    
    messages_with_system = [system_prompt] + messages
    response = model_with_tools.invoke(messages_with_system)
    
    return {"messages": [response]}

def rewrite(state):
    """Transform the query to produce a better question."""
    logger.info("---TRANSFORM QUERY---")
    messages = state["messages"]
    question = messages[0].content
    
    rewrite_prompt = f"""
    Look at the input and try to reason about the underlying semantic intent/meaning.
    
    Original question: {question}
    
    Formulate an improved, more specific question that would help retrieve better documents:
    """
    
    try:
        response = llm.invoke([HumanMessage(content=rewrite_prompt)])
        return {"messages": [response]}
    except Exception as e:
        logger.error(f"Rewrite failed: {e}")
        return {"messages": [HumanMessage(content=question)]}

def generate(state):
    """Generate final answer based on retrieved documents."""
    logger.info("---GENERATE---")
    messages = state["messages"]
    question = messages[0].content
    last_message = messages[-1]
    
    docs = last_message.content
    
    # RAG prompt
    rag_prompt = f"""
    Use the following pieces of context to answer the question at the end. 
    If you don't know the answer based on the context, just say that you don't know, 
    don't try to make up an answer.
    
    Context:
    {docs}
    
    Question: {question}
    
    Answer:
    """
    
    try:
        response = llm.invoke([HumanMessage(content=rag_prompt)])
        return {"messages": [response]}
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        return {"messages": [AIMessage(content="I apologize, but I encountered an error generating the response.")]}

def create_workflow():
    """Create the agent workflow graph."""
    workflow = StateGraph(AgentState)
    
    workflow.add_node("agent", agent)
    retrieve = ToolNode([retriever_tool])
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("rewrite", rewrite)
    workflow.add_node("generate", generate)
    
    # Add edges
    workflow.add_edge(START, "agent")
    
    workflow.add_conditional_edges(
        "agent",
        tools_condition,
        {
            "tools": "retrieve",
            END: END,
        },
    )
    
    workflow.add_conditional_edges(
        "retrieve",
        grade_documents,
        {
            "generate": "generate",
            "rewrite": "rewrite"
        }
    )
    
    workflow.add_edge("generate", END)
    workflow.add_edge("rewrite", "agent")
    
    return workflow.compile()