"""
Advanced A2A Use Cases
======================

Advanced examples demonstrating complex A2A scenarios:
- Distributed agent orchestration
- Failover and redundancy
- Load balancing
- Tool versioning
- Security and authentication
"""

import asyncio
from typing import List, Optional, Dict, Any
from strands import Agent, Task
from strands.tools import A2AClientToolProvider, Tool
from strands.models import OpenAIModel
import os
from dotenv import load_dotenv
import hashlib
import time

load_dotenv()


class RedundantA2AProvider:
    """
    A2A provider with multiple backend servers for failover.
    """
    
    def __init__(
        self,
        name: str,
        agent_urls: List[str],
        description: str = "",
        timeout: float = 30.0
    ):
        self.name = name
        self.agent_urls = agent_urls
        self.description = description
        self.timeout = timeout
        self.providers: List[A2AClientToolProvider] = []
        self.current_index = 0
    
    async def setup(self):
        """Initialize all backup providers"""
        for i, url in enumerate(self.agent_urls):
            provider = A2AClientToolProvider(
                name=f"{self.name}_backend_{i}",
                agent_url=url,
                description=self.description,
                timeout=self.timeout
            )
            self.providers.append(provider)
        
        # Try to discover tools from all providers
        for provider in self.providers:
            try:
                await provider.discover_tools()
                print(f"✅ Connected to backup server: {provider.agent_url}")
            except Exception as e:
                print(f"⚠️ Backup server unavailable: {provider.agent_url} - {e}")
    
    async def execute_with_failover(
        self,
        tool_name: str,
        parameters: Dict[str, Any]
    ) -> Any:
        """
        Execute tool with automatic failover to backup servers.
        """
        last_error = None
        
        # Try each provider in order
        for i in range(len(self.providers)):
            provider_index = (self.current_index + i) % len(self.providers)
            provider = self.providers[provider_index]
            
            try:
                result = await provider.execute_tool(tool_name, parameters)
                
                # Update current provider for next call (round-robin)
                self.current_index = provider_index
                
                return result
                
            except Exception as e:
                last_error = e
                print(f"❌ Failed on {provider.agent_url}: {e}")
                continue
        
        raise Exception(f"All backup servers failed. Last error: {last_error}")


class LoadBalancedA2AProvider:
    """
    A2A provider that distributes load across multiple servers.
    """
    
    def __init__(
        self,
        name: str,
        agent_urls: List[str],
        strategy: str = "round-robin"  # round-robin, random, least-loaded
    ):
        self.name = name
        self.agent_urls = agent_urls
        self.strategy = strategy
        self.providers: List[A2AClientToolProvider] = []
        self.current_index = 0
        self.load_counters: Dict[str, int] = {}
    
    async def setup(self):
        """Initialize all providers"""
        for i, url in enumerate(self.agent_urls):
            provider = A2AClientToolProvider(
                name=f"{self.name}_lb_{i}",
                agent_url=url
            )
            await provider.discover_tools()
            self.providers.append(provider)
            self.load_counters[url] = 0
    
    def _select_provider(self) -> A2AClientToolProvider:
        """Select provider based on load balancing strategy"""
        
        if self.strategy == "round-robin":
            provider = self.providers[self.current_index]
            self.current_index = (self.current_index + 1) % len(self.providers)
            return provider
        
        elif self.strategy == "random":
            import random
            return random.choice(self.providers)
        
        elif self.strategy == "least-loaded":
            # Find provider with lowest load
            min_load_url = min(self.load_counters, key=self.load_counters.get)
            for provider in self.providers:
                if provider.agent_url == min_load_url:
                    return provider
        
        return self.providers[0]
    
    async def execute_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any]
    ) -> Any:
        """Execute tool using load balancing"""
        
        provider = self._select_provider()
        
        # Track load
        self.load_counters[provider.agent_url] += 1
        
        try:
            result = await provider.execute_tool(tool_name, parameters)
            return result
        finally:
            # Decrease load counter after execution
            self.load_counters[provider.agent_url] -= 1


class SecureA2AProvider:
    """
    A2A provider with authentication and encryption.
    """
    
    def __init__(
        self,
        name: str,
        agent_url: str,
        api_key: str,
        secret_key: str
    ):
        self.name = name
        self.agent_url = agent_url
        self.api_key = api_key
        self.secret_key = secret_key
        self.provider: Optional[A2AClientToolProvider] = None
    
    def _generate_signature(self, payload: str) -> str:
        """Generate HMAC signature for request"""
        import hmac
        
        signature = hmac.new(
            self.secret_key.encode(),
            payload.encode(),
            hashlib.sha256
        ).hexdigest()
        
        return signature
    
    def _create_auth_headers(self, payload: str) -> Dict[str, str]:
        """Create authentication headers"""
        timestamp = str(int(time.time()))
        signature = self._generate_signature(f"{timestamp}:{payload}")
        
        return {
            "X-API-Key": self.api_key,
            "X-Timestamp": timestamp,
            "X-Signature": signature
        }
    
    async def setup(self):
        """Initialize secure provider"""
        
        # Create payload for discovery
        payload = "discover_tools"
        headers = self._create_auth_headers(payload)
        
        self.provider = A2AClientToolProvider(
            name=self.name,
            agent_url=self.agent_url,
            headers=headers
        )
        
        await self.provider.discover_tools()
        print(f"🔒 Secure connection established to {self.agent_url}")
    
    async def execute_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any]
    ) -> Any:
        """Execute tool with authentication"""
        
        import json
        payload = json.dumps({"tool": tool_name, "params": parameters})
        headers = self._create_auth_headers(payload)
        
        # Update provider headers for this request
        self.provider.headers.update(headers)
        
        return await self.provider.execute_tool(tool_name, parameters)


class VersionedA2AProvider:
    """
    A2A provider that handles tool versioning.
    """
    
    def __init__(
        self,
        name: str,
        agent_url: str,
        version: str = "v1"
    ):
        self.name = name
        self.agent_url = agent_url
        self.version = version
        self.providers: Dict[str, A2AClientToolProvider] = {}
    
    async def setup(self):
        """Initialize provider with versioned endpoints"""
        
        versioned_url = f"{self.agent_url}/{self.version}"
        
        provider = A2AClientToolProvider(
            name=f"{self.name}_{self.version}",
            agent_url=versioned_url
        )
        
        await provider.discover_tools()
        self.providers[self.version] = provider
        
        print(f"📦 Loaded tools version {self.version} from {versioned_url}")
    
    async def execute_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        version: Optional[str] = None
    ) -> Any:
        """Execute tool with specific version"""
        
        version = version or self.version
        
        if version not in self.providers:
            raise ValueError(f"Version {version} not available")
        
        provider = self.providers[version]
        return await provider.execute_tool(tool_name, parameters)
    
    async def migrate_version(self, new_version: str):
        """Migrate to a new version"""
        
        print(f"🔄 Migrating from {self.version} to {new_version}...")
        
        # Load new version
        new_url = f"{self.agent_url}/{new_version}"
        new_provider = A2AClientToolProvider(
            name=f"{self.name}_{new_version}",
            agent_url=new_url
        )
        
        await new_provider.discover_tools()
        self.providers[new_version] = new_provider
        
        # Update current version
        old_version = self.version
        self.version = new_version
        
        print(f"✅ Migration complete: {old_version} → {new_version}")


async def example_failover():
    """Example: Failover and redundancy"""
    print("\n" + "="*60)
    print("Example: Failover and Redundancy")
    print("="*60 + "\n")
    
    # Create redundant provider with multiple backends
    provider = RedundantA2AProvider(
        name="redundant_tools",
        agent_urls=[
            "http://primary-agent:8000",
            "http://backup-agent-1:8000",
            "http://backup-agent-2:8000"
        ],
        description="Tools with automatic failover"
    )
    
    await provider.setup()
    
    # Execute with automatic failover
    try:
        result = await provider.execute_with_failover(
            tool_name="process_data",
            parameters={"data": [1, 2, 3, 4, 5]}
        )
        print(f"Result: {result}")
    except Exception as e:
        print(f"All servers failed: {e}")


async def example_load_balancing():
    """Example: Load balancing across multiple servers"""
    print("\n" + "="*60)
    print("Example: Load Balancing")
    print("="*60 + "\n")
    
    # Create load-balanced provider
    lb_provider = LoadBalancedA2AProvider(
        name="balanced_tools",
        agent_urls=[
            "http://server1:8000",
            "http://server2:8000",
            "http://server3:8000"
        ],
        strategy="least-loaded"
    )
    
    await lb_provider.setup()
    
    # Execute multiple tasks - they'll be distributed
    tasks = []
    for i in range(10):
        task = lb_provider.execute_tool(
            tool_name="analyze",
            parameters={"id": i}
        )
        tasks.append(task)
    
    results = await asyncio.gather(*tasks)
    print(f"Completed {len(results)} tasks across load-balanced servers")


async def example_security():
    """Example: Secure A2A communication"""
    print("\n" + "="*60)
    print("Example: Secure Communication")
    print("="*60 + "\n")
    
    # Create secure provider
    secure_provider = SecureA2AProvider(
        name="secure_tools",
        agent_url="https://secure-agent.example.com",
        api_key=os.getenv("A2A_API_KEY", "your-api-key"),
        secret_key=os.getenv("A2A_SECRET_KEY", "your-secret-key")
    )
    
    await secure_provider.setup()
    
    # Execute with authentication
    result = await secure_provider.execute_tool(
        tool_name="sensitive_operation",
        parameters={"data": "confidential"}
    )
    
    print(f"Secure result: {result}")


async def example_versioning():
    """Example: Tool versioning"""
    print("\n" + "="*60)
    print("Example: Tool Versioning")
    print("="*60 + "\n")
    
    # Create versioned provider
    versioned_provider = VersionedA2AProvider(
        name="api_tools",
        agent_url="http://api-server:8000",
        version="v1"
    )
    
    await versioned_provider.setup()
    
    # Use v1 tools
    result_v1 = await versioned_provider.execute_tool(
        tool_name="process",
        parameters={"data": "test"}
    )
    print(f"V1 result: {result_v1}")
    
    # Migrate to v2
    await versioned_provider.migrate_version("v2")
    
    # Use v2 tools
    result_v2 = await versioned_provider.execute_tool(
        tool_name="process",
        parameters={"data": "test"}
    )
    print(f"V2 result: {result_v2}")


async def main():
    """Run advanced examples"""
    print("Advanced A2A Use Cases")
    print("="*60)
    
    try:
        await example_failover()
        await example_load_balancing()
        await example_security()
        await example_versioning()
        
        print("\n" + "="*60)
        print("Advanced examples completed!")
        print("="*60)
        
    except Exception as e:
        print(f"\nNote: Some examples require running servers")
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
