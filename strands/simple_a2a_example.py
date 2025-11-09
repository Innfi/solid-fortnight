"""
Simple A2A Client Tool Provider Example
========================================

A minimal, easy-to-understand example of using A2AClientToolProvider
in Strands Python for agent-to-agent communication.
"""

import asyncio
from strands import Agent, Task
from strands.tools import A2AClientToolProvider
from strands.models import OpenAIModel
import os
from dotenv import load_dotenv

load_dotenv()


async def simple_example():
    """
    Simple example: Connect to a remote agent and use its tools.
    """
    print("Simple A2A Tool Provider Example")
    print("=" * 40)
    
    # Step 1: Create your local agent
    my_agent = Agent(
        name="MyAgent",
        description="My local agent that uses remote tools",
        model=OpenAIModel(
            api_key=os.getenv('OPENAI_API_KEY'),
            model="gpt-4"
        )
    )
    
    # Step 2: Create A2A client to connect to remote agent
    remote_tools = A2AClientToolProvider(
        name="calculator_tools",
        agent_url="http://localhost:8000/calculator",  # Remote agent URL
        description="Mathematical calculation tools from remote agent",
        timeout=30.0
    )
    
    # Step 3: Discover available tools from remote agent
    print("\nDiscovering tools from remote agent...")
    try:
        await remote_tools.discover_tools()
        
        discovered_tools = remote_tools.get_tools()
        print(f"Found {len(discovered_tools)} tools:")
        for tool in discovered_tools:
            print(f"  - {tool.name}: {tool.description}")
        
    except Exception as e:
        print(f"Error discovering tools: {e}")
        return
    
    # Step 4: Register remote tools with your local agent
    my_agent.add_tool_provider(remote_tools)
    print("\n✅ Remote tools registered with local agent")
    
    # Step 5: Create and execute a task that uses remote tools
    task = Task(
        description="Calculate the square root of 144 and then multiply it by 5",
        agent=my_agent
    )
    
    print("\nExecuting task...")
    result = await task.execute()
    
    print(f"\nTask Result: {result}")


async def local_simulation_example():
    """
    Simulated example that works without a remote server.
    This demonstrates the concept locally.
    """
    print("\nLocal Simulation Example")
    print("=" * 40)
    
    # Create a "remote" agent (simulated locally)
    remote_agent = Agent(
        name="MathAgent",
        description="Agent with mathematical tools",
        model=OpenAIModel(
            api_key=os.getenv('OPENAI_API_KEY'),
            model="gpt-4"
        )
    )
    
    # Add tools to the "remote" agent
    @remote_agent.tool(
        name="add",
        description="Add two numbers"
    )
    def add(a: float, b: float) -> float:
        """Add two numbers together"""
        return a + b
    
    @remote_agent.tool(
        name="multiply",
        description="Multiply two numbers"
    )
    def multiply(a: float, b: float) -> float:
        """Multiply two numbers"""
        return a * b
    
    @remote_agent.tool(
        name="calculate_average",
        description="Calculate average of a list of numbers"
    )
    def calculate_average(numbers: list[float]) -> float:
        """Calculate the average of a list of numbers"""
        if not numbers:
            return 0
        return sum(numbers) / len(numbers)
    
    # Create local agent
    local_agent = Agent(
        name="CoordinatorAgent",
        description="Coordinates tasks using remote math tools",
        model=OpenAIModel(
            api_key=os.getenv('OPENAI_API_KEY'),
            model="gpt-4"
        )
    )
    
    # In a real scenario, you would use A2AClientToolProvider here
    # For this simulation, we'll directly share tools
    print("\n✅ Simulated remote tools available:")
    for tool_name, tool in remote_agent.tools.items():
        print(f"  - {tool_name}: {tool.description}")
        local_agent.tools[tool_name] = tool
    
    # Execute task using the simulated remote tools
    task = Task(
        description="Calculate: (10 + 15) * 3, then find the average of [25, 30, 35, 40]",
        agent=local_agent
    )
    
    print("\nExecuting task with simulated remote tools...")
    result = await task.execute()
    print(f"Result: {result}")


async def main():
    """Run examples"""
    
    # Try the simple example
    # Note: This requires a remote agent server to be running
    print("Attempting to connect to remote agent...")
    try:
        await simple_example()
    except Exception as e:
        print(f"\nRemote agent not available: {e}")
        print("Running local simulation instead...\n")
    
    # Run local simulation (always works)
    await local_simulation_example()


if __name__ == "__main__":
    asyncio.run(main())
