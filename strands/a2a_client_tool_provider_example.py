"""
Strands Python - A2AClientToolProvider Example
===============================================

This example demonstrates how to use the A2AClientToolProvider in Strands Python
for agent-to-agent (A2A) communication and tool sharing.

A2AClientToolProvider enables:
- Agent-to-agent communication
- Remote tool execution
- Distributed agent systems
- Tool discovery and invocation across agents
"""

import asyncio
from typing import Any, Dict, List, Optional
from strands import Strands, Agent, Task
from strands.tools import A2AClientToolProvider, Tool, ToolParameter
from strands.models import OpenAIModel
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
REMOTE_AGENT_URL = os.getenv('REMOTE_AGENT_URL', 'http://localhost:8000')


class WeatherAgent:
    """
    Remote agent that provides weather information tools.
    This simulates an agent running on a different server.
    """
    
    def __init__(self):
        self.agent = Agent(
            name="WeatherAgent",
            description="Agent that provides weather information and forecasts",
            model=OpenAIModel(api_key=OPENAI_API_KEY, model="gpt-4"),
        )
        
        # Register weather tools
        self._register_tools()
    
    def _register_tools(self):
        """Register weather-related tools"""
        
        @self.agent.tool(
            name="get_current_weather",
            description="Get current weather for a specific location"
        )
        def get_current_weather(
            location: str,
            unit: str = "celsius"
        ) -> Dict[str, Any]:
            """
            Get current weather information.
            
            Args:
                location: City name or location
                unit: Temperature unit (celsius/fahrenheit)
                
            Returns:
                Weather information dictionary
            """
            # Simulated weather data
            weather_data = {
                "location": location,
                "temperature": 22 if unit == "celsius" else 72,
                "unit": unit,
                "condition": "Partly Cloudy",
                "humidity": 65,
                "wind_speed": 15,
                "wind_unit": "km/h"
            }
            return weather_data
        
        @self.agent.tool(
            name="get_weather_forecast",
            description="Get weather forecast for the next N days"
        )
        def get_weather_forecast(
            location: str,
            days: int = 3
        ) -> List[Dict[str, Any]]:
            """
            Get weather forecast.
            
            Args:
                location: City name or location
                days: Number of days to forecast (1-7)
                
            Returns:
                List of daily forecasts
            """
            # Simulated forecast data
            forecast = []
            conditions = ["Sunny", "Partly Cloudy", "Cloudy", "Rainy"]
            
            for day in range(min(days, 7)):
                forecast.append({
                    "day": day + 1,
                    "location": location,
                    "temperature_high": 25 + day,
                    "temperature_low": 15 + day,
                    "condition": conditions[day % len(conditions)],
                    "precipitation_chance": (day * 10) % 100
                })
            
            return forecast


class DataAnalysisAgent:
    """
    Remote agent that provides data analysis tools.
    This simulates another agent running on a different server.
    """
    
    def __init__(self):
        self.agent = Agent(
            name="DataAnalysisAgent",
            description="Agent that provides data analysis and processing tools",
            model=OpenAIModel(api_key=OPENAI_API_KEY, model="gpt-4"),
        )
        
        self._register_tools()
    
    def _register_tools(self):
        """Register data analysis tools"""
        
        @self.agent.tool(
            name="analyze_data",
            description="Analyze numerical data and provide statistics"
        )
        def analyze_data(
            data: List[float],
            metrics: Optional[List[str]] = None
        ) -> Dict[str, Any]:
            """
            Analyze data and calculate statistics.
            
            Args:
                data: List of numerical values
                metrics: List of metrics to calculate (mean, median, std, etc.)
                
            Returns:
                Dictionary of calculated metrics
            """
            if not data:
                return {"error": "No data provided"}
            
            import statistics
            
            results = {
                "count": len(data),
                "min": min(data),
                "max": max(data),
                "mean": statistics.mean(data),
            }
            
            if metrics is None or "median" in metrics:
                results["median"] = statistics.median(data)
            
            if metrics is None or "std" in metrics:
                if len(data) > 1:
                    results["std"] = statistics.stdev(data)
            
            return results
        
        @self.agent.tool(
            name="transform_data",
            description="Transform data using various operations"
        )
        def transform_data(
            data: List[float],
            operation: str = "normalize"
        ) -> List[float]:
            """
            Transform data using specified operation.
            
            Args:
                data: List of numerical values
                operation: Operation to apply (normalize, scale, log)
                
            Returns:
                Transformed data
            """
            if not data:
                return []
            
            if operation == "normalize":
                min_val = min(data)
                max_val = max(data)
                range_val = max_val - min_val
                if range_val == 0:
                    return [0.0] * len(data)
                return [(x - min_val) / range_val for x in data]
            
            elif operation == "scale":
                import statistics
                mean = statistics.mean(data)
                std = statistics.stdev(data) if len(data) > 1 else 1
                return [(x - mean) / std for x in data]
            
            elif operation == "log":
                import math
                return [math.log(x) if x > 0 else 0 for x in data]
            
            return data


class CoordinatorAgent:
    """
    Main coordinator agent that uses A2AClientToolProvider to access
    tools from remote agents.
    """
    
    def __init__(self):
        self.agent = Agent(
            name="CoordinatorAgent",
            description="Coordinator agent that orchestrates tasks across multiple agents",
            model=OpenAIModel(api_key=OPENAI_API_KEY, model="gpt-4"),
        )
        
        # Initialize A2A client tool providers
        self.weather_provider = None
        self.data_provider = None
    
    async def setup_remote_tools(self):
        """Setup A2A client tool providers for remote agents"""
        
        # Create A2A client for weather agent
        self.weather_provider = A2AClientToolProvider(
            name="weather_tools",
            agent_url=f"{REMOTE_AGENT_URL}/weather",
            description="Tools for weather information and forecasts",
            timeout=30.0
        )
        
        # Create A2A client for data analysis agent
        self.data_provider = A2AClientToolProvider(
            name="data_tools",
            agent_url=f"{REMOTE_AGENT_URL}/data",
            description="Tools for data analysis and processing",
            timeout=30.0
        )
        
        # Discover and register tools from remote agents
        await self.weather_provider.discover_tools()
        await self.data_provider.discover_tools()
        
        # Register tool providers with coordinator agent
        self.agent.add_tool_provider(self.weather_provider)
        self.agent.add_tool_provider(self.data_provider)
        
        print("✅ Remote tools registered successfully")
        print(f"Weather tools: {[tool.name for tool in self.weather_provider.get_tools()]}")
        print(f"Data tools: {[tool.name for tool in self.data_provider.get_tools()]}")
    
    async def execute_weather_task(self, location: str):
        """
        Execute a task that uses remote weather tools.
        
        Args:
            location: Location to get weather information for
        """
        task = Task(
            description=f"Get the current weather and 5-day forecast for {location}",
            agent=self.agent
        )
        
        result = await task.execute()
        return result
    
    async def execute_data_analysis_task(self, data: List[float]):
        """
        Execute a task that uses remote data analysis tools.
        
        Args:
            data: Data to analyze
        """
        task = Task(
            description=f"Analyze this data and provide insights: {data}",
            agent=self.agent
        )
        
        result = await task.execute()
        return result
    
    async def execute_combined_task(self):
        """
        Execute a task that uses tools from multiple remote agents.
        """
        task = Task(
            description="""
            1. Get the current weather for New York and Tokyo
            2. Create a dataset with temperature values from both cities
            3. Analyze the dataset and provide statistical insights
            4. Get a 3-day forecast for both cities and compare them
            """,
            agent=self.agent
        )
        
        result = await task.execute()
        return result


async def example_basic_a2a_usage():
    """
    Basic example of using A2AClientToolProvider.
    """
    print("\n" + "="*60)
    print("Example 1: Basic A2A Tool Provider Usage")
    print("="*60 + "\n")
    
    # Create coordinator agent
    coordinator = CoordinatorAgent()
    
    # Setup remote tools
    await coordinator.setup_remote_tools()
    
    # Execute weather task
    print("\n📍 Executing weather task...")
    weather_result = await coordinator.execute_weather_task("London")
    print(f"Weather Result: {weather_result}")
    
    # Execute data analysis task
    print("\n📊 Executing data analysis task...")
    sample_data = [23.5, 24.1, 22.8, 25.3, 23.9, 24.5, 23.2]
    data_result = await coordinator.execute_data_analysis_task(sample_data)
    print(f"Data Analysis Result: {data_result}")


async def example_custom_tool_registration():
    """
    Example of manually creating and registering A2A tools.
    """
    print("\n" + "="*60)
    print("Example 2: Custom A2A Tool Registration")
    print("="*60 + "\n")
    
    # Create A2A client tool provider
    provider = A2AClientToolProvider(
        name="custom_tools",
        agent_url="http://localhost:8000/custom",
        description="Custom remote tools",
        timeout=30.0
    )
    
    # Manually create a tool (without auto-discovery)
    custom_tool = Tool(
        name="process_text",
        description="Process text using remote NLP service",
        parameters=[
            ToolParameter(
                name="text",
                type="string",
                description="Text to process",
                required=True
            ),
            ToolParameter(
                name="operation",
                type="string",
                description="Operation to perform (summarize, sentiment, extract)",
                required=False,
                default="summarize"
            )
        ],
        function=provider.create_remote_function("process_text")
    )
    
    # Add tool to provider
    provider.add_tool(custom_tool)
    
    print(f"✅ Custom tool registered: {custom_tool.name}")
    print(f"   Description: {custom_tool.description}")
    print(f"   Parameters: {[p.name for p in custom_tool.parameters]}")


async def example_error_handling():
    """
    Example of error handling with A2A communication.
    """
    print("\n" + "="*60)
    print("Example 3: A2A Error Handling")
    print("="*60 + "\n")
    
    provider = A2AClientToolProvider(
        name="unreliable_tools",
        agent_url="http://localhost:9999/nonexistent",  # Invalid URL
        description="Tools with error handling",
        timeout=5.0,
        retry_count=2,
        retry_delay=1.0
    )
    
    try:
        print("Attempting to discover tools from unavailable agent...")
        await provider.discover_tools()
    except Exception as e:
        print(f"❌ Expected error caught: {type(e).__name__}: {str(e)}")
        print("✅ Error handling working correctly")
    
    # Example with fallback behavior
    print("\nSetting up fallback mechanism...")
    
    class FallbackAgent:
        def __init__(self):
            self.agent = Agent(
                name="FallbackAgent",
                description="Local fallback agent",
                model=OpenAIModel(api_key=OPENAI_API_KEY, model="gpt-4"),
            )
            
            @self.agent.tool(name="local_fallback")
            def local_fallback(message: str) -> str:
                """Local fallback tool when remote is unavailable"""
                return f"Using local fallback for: {message}"
    
    fallback = FallbackAgent()
    result = fallback.agent.tools["local_fallback"].function("test")
    print(f"Fallback result: {result}")


async def example_tool_filtering():
    """
    Example of filtering and selective tool registration.
    """
    print("\n" + "="*60)
    print("Example 4: Selective Tool Registration")
    print("="*60 + "\n")
    
    provider = A2AClientToolProvider(
        name="filtered_tools",
        agent_url=REMOTE_AGENT_URL,
        description="Filtered tool provider",
        tool_filter=lambda tool: "weather" in tool.name.lower()  # Only weather tools
    )
    
    await provider.discover_tools()
    
    print(f"Discovered tools (filtered): {[tool.name for tool in provider.get_tools()]}")
    print("✅ Only weather-related tools were registered")


async def example_multi_agent_workflow():
    """
    Complex example with multiple agents working together.
    """
    print("\n" + "="*60)
    print("Example 5: Multi-Agent Workflow")
    print("="*60 + "\n")
    
    coordinator = CoordinatorAgent()
    await coordinator.setup_remote_tools()
    
    # Execute complex multi-agent task
    print("🚀 Executing complex multi-agent workflow...")
    result = await coordinator.execute_combined_task()
    
    print(f"\nWorkflow completed!")
    print(f"Result: {result}")


async def example_tool_chaining():
    """
    Example of chaining A2A tool calls.
    """
    print("\n" + "="*60)
    print("Example 6: A2A Tool Chaining")
    print("="*60 + "\n")
    
    # Create multiple tool providers
    weather_provider = A2AClientToolProvider(
        name="weather",
        agent_url=f"{REMOTE_AGENT_URL}/weather",
        description="Weather tools"
    )
    
    data_provider = A2AClientToolProvider(
        name="data",
        agent_url=f"{REMOTE_AGENT_URL}/data",
        description="Data analysis tools"
    )
    
    await weather_provider.discover_tools()
    await data_provider.discover_tools()
    
    # Create workflow agent that chains tools
    workflow_agent = Agent(
        name="WorkflowAgent",
        description="Agent that chains multiple remote tools",
        model=OpenAIModel(api_key=OPENAI_API_KEY, model="gpt-4"),
    )
    
    workflow_agent.add_tool_provider(weather_provider)
    workflow_agent.add_tool_provider(data_provider)
    
    # Execute chained task
    task = Task(
        description="""
        Execute this workflow:
        1. Get weather for 3 different cities
        2. Extract temperature values
        3. Normalize the temperature data
        4. Analyze the normalized data
        5. Provide insights about temperature variations
        """,
        agent=workflow_agent
    )
    
    result = await task.execute()
    print(f"Chained workflow result: {result}")


async def main():
    """
    Main function to run all examples.
    """
    print("\n" + "="*70)
    print("  Strands Python - A2AClientToolProvider Examples")
    print("="*70)
    
    # Note: These examples assume remote agents are running
    # In production, you would have actual agent servers
    
    try:
        # Example 1: Basic usage
        await example_basic_a2a_usage()
        
        # Example 2: Custom tool registration
        await example_custom_tool_registration()
        
        # Example 3: Error handling
        await example_error_handling()
        
        # Example 4: Tool filtering
        await example_tool_filtering()
        
        # Example 5: Multi-agent workflow
        await example_multi_agent_workflow()
        
        # Example 6: Tool chaining
        await example_tool_chaining()
        
        print("\n" + "="*70)
        print("  All examples completed successfully!")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {type(e).__name__}: {str(e)}")
        print("Note: Some examples require remote agents to be running")


if __name__ == "__main__":
    # Run all examples
    asyncio.run(main())
