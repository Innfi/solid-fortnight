"""
A2A Server Agent Example
=========================

This creates a server-side agent that exposes tools via A2A protocol.
Other agents can connect to this agent using A2AClientToolProvider.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, List, Optional
from strands import Agent
from strands.models import OpenAIModel
import os
from dotenv import load_dotenv
import uvicorn

load_dotenv()

# Create FastAPI app
app = FastAPI(title="A2A Server Agent", version="1.0.0")

# Initialize agent
server_agent = Agent(
    name="ServerAgent",
    description="Server agent that provides tools via A2A protocol",
    model=OpenAIModel(
        api_key=os.getenv('OPENAI_API_KEY'),
        model="gpt-4"
    )
)


# Define tools
@server_agent.tool(
    name="get_system_status",
    description="Get current system status and health metrics"
)
def get_system_status() -> Dict[str, Any]:
    """Get system status"""
    import psutil
    import platform
    
    return {
        "status": "healthy",
        "cpu_percent": psutil.cpu_percent(interval=1),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_percent": psutil.disk_usage('/').percent,
        "platform": platform.system(),
        "python_version": platform.python_version()
    }


@server_agent.tool(
    name="search_database",
    description="Search database for records"
)
def search_database(
    query: str,
    limit: int = 10,
    filters: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """Search database (simulated)"""
    # Simulated database search
    results = [
        {
            "id": i,
            "query": query,
            "title": f"Result {i} for '{query}'",
            "score": 0.9 - (i * 0.1),
            "filters_applied": filters or {}
        }
        for i in range(1, min(limit + 1, 6))
    ]
    return results


@server_agent.tool(
    name="process_batch",
    description="Process a batch of items"
)
def process_batch(
    items: List[str],
    operation: str = "uppercase"
) -> List[str]:
    """Process batch of items"""
    if operation == "uppercase":
        return [item.upper() for item in items]
    elif operation == "lowercase":
        return [item.lower() for item in items]
    elif operation == "reverse":
        return [item[::-1] for item in items]
    else:
        return items


# Pydantic models for API
class ToolDiscoveryResponse(BaseModel):
    tools: List[Dict[str, Any]]


class ToolExecutionRequest(BaseModel):
    tool_name: str
    parameters: Dict[str, Any]


class ToolExecutionResponse(BaseModel):
    success: bool
    result: Any
    error: Optional[str] = None


# API endpoints
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": server_agent.name,
        "description": server_agent.description,
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "agent": server_agent.name}


@app.get("/tools", response_model=ToolDiscoveryResponse)
async def discover_tools():
    """
    Discover available tools.
    This is the endpoint that A2AClientToolProvider calls.
    """
    tools_info = []
    
    for tool_name, tool in server_agent.tools.items():
        tool_info = {
            "name": tool.name,
            "description": tool.description,
            "parameters": [
                {
                    "name": param.name,
                    "type": param.type,
                    "description": param.description,
                    "required": param.required,
                    "default": param.default
                }
                for param in tool.parameters
            ]
        }
        tools_info.append(tool_info)
    
    return ToolDiscoveryResponse(tools=tools_info)


@app.post("/tools/{tool_name}/execute", response_model=ToolExecutionResponse)
async def execute_tool(tool_name: str, request: ToolExecutionRequest):
    """
    Execute a specific tool.
    This is called by A2AClientToolProvider when invoking a tool.
    """
    try:
        # Check if tool exists
        if tool_name not in server_agent.tools:
            raise HTTPException(
                status_code=404,
                detail=f"Tool '{tool_name}' not found"
            )
        
        # Get the tool
        tool = server_agent.tools[tool_name]
        
        # Execute the tool with provided parameters
        result = tool.function(**request.parameters)
        
        return ToolExecutionResponse(
            success=True,
            result=result,
            error=None
        )
        
    except Exception as e:
        return ToolExecutionResponse(
            success=False,
            result=None,
            error=str(e)
        )


@app.get("/tools/{tool_name}")
async def get_tool_info(tool_name: str):
    """Get information about a specific tool"""
    if tool_name not in server_agent.tools:
        raise HTTPException(status_code=404, detail="Tool not found")
    
    tool = server_agent.tools[tool_name]
    
    return {
        "name": tool.name,
        "description": tool.description,
        "parameters": [
            {
                "name": param.name,
                "type": param.type,
                "description": param.description,
                "required": param.required,
                "default": param.default
            }
            for param in tool.parameters
        ]
    }


def main():
    """Run the A2A server agent"""
    print("Starting A2A Server Agent...")
    print(f"Agent: {server_agent.name}")
    print(f"Available tools: {list(server_agent.tools.keys())}")
    print("\nServer will be available at: http://localhost:8000")
    print("Tool discovery endpoint: http://localhost:8000/tools")
    print("API documentation: http://localhost:8000/docs")
    
    # Run the server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )


if __name__ == "__main__":
    main()
