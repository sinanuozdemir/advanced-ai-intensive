#!/usr/bin/env python3
"""
Generic Agent Builder implementation with MCP integration
"""

import asyncio
import json
import os
from typing import Dict, List, Any
from datetime import datetime

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage

from prompts import get_sdr_system_prompt


class GenericAgent:
    """Generic Agent Builder with MCP integration"""
    
    def __init__(self, config_path: str = "agent_config.json"):
        self.config_path = config_path
        self.config = None
        self.mcp_client = None
        self.agent = None
        self.tools = []
        self.checkpointer = MemorySaver()  # For conversation persistence
        self.initialized = False
        
    def _get_system_prompt(self) -> str:
        """Get the appropriate system prompt based on agent configuration"""
        if not self.config:
            self.load_config()
            
        agent_name = self.config.get("name", "").lower()
        agent_settings = self.config.get("agent_settings", {})
        
        extra_instructions = agent_settings.get("extra_instructions", "")
        if extra_instructions:
            print(f"🔍 Using extra instructions: {extra_instructions[:20]}...")
        prompt = get_sdr_system_prompt(extra_instructions=extra_instructions)
        
        return prompt
    
    def load_config(self) -> Dict[str, Any]:
        """Load agent configuration from JSON file"""
        config_file = os.path.join(os.path.dirname(__file__), self.config_path)
        
        try:
            with open(config_file, 'r') as f:
                self.config = json.load(f)
            print(f"✅ Loaded configuration from {config_file}")
            return self.config
        except FileNotFoundError:
            print(f"❌ Configuration file not found: {config_file}")
            raise
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON in configuration file: {e}")
            raise
    
    def get_enabled_mcp_servers(self) -> Dict[str, Dict[str, Any]]:
        """Get only enabled MCP servers from configuration"""
        if not self.config:
            self.load_config()
            
        enabled_servers = {}
        for server_name, server_config in self.config.get("mcp_servers", {}).items():
            if server_config.get("enabled", False):
                enabled_servers[server_name] = {
                    "command": server_config["command"],
                    "args": server_config["args"],
                    "transport": server_config["transport"]
                }
                print(f"✅ Enabled MCP server: {server_name} - {server_config.get('description', '')}")
            else:
                print(f"⏸️  Disabled MCP server: {server_name}")
                
        return enabled_servers
    
    async def reload_config(self):
        """Reload configuration and reinitialize if needed"""
        print("🔄 Reloading agent configuration...")
        old_config = self.config.copy() if self.config else {}
        
        try:
            self.load_config()
            
            # Check if MCP server configuration changed
            old_servers = old_config.get("mcp_servers", {})
            new_servers = self.config.get("mcp_servers", {})
            
            if old_servers != new_servers:
                print("🔄 MCP server configuration changed, reinitializing...")
                self.initialized = False
                await self.initialize()
            else:
                print("✅ Configuration reloaded, no changes to MCP servers")
                
        except Exception as e:
            print(f"❌ Error reloading configuration: {e}")
            # Restore old config if reload failed
            self.config = old_config
            raise
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get a summary of current configuration"""
        if not self.config:
            self.load_config()
            
        enabled_count = sum(1 for server in self.config.get("mcp_servers", {}).values() 
                          if server.get("enabled", False))
        total_count = len(self.config.get("mcp_servers", {}))
        
        return {
            "agent_name": self.config.get("name", "Unknown"),
            "description": self.config.get("description", ""),
            "mcp_servers": {
                "enabled": enabled_count,
                "total": total_count,
                "servers": {name: {"enabled": config.get("enabled", False), 
                                 "description": config.get("description", "")}
                          for name, config in self.config.get("mcp_servers", {}).items()}
            },
            "agent_settings": self.config.get("agent_settings", {}),
            "initialized": self.initialized,
            "tools_loaded": len(self.tools) if self.tools else 0
        }

    async def _merge_tools(self, tools: List[Any]) -> List[Any]:
        """Hook: subclasses extend the MCP tool list (e.g. memory tools)."""
        return tools

    async def initialize(self):
        """Initialize the MCP client and LangGraph agent"""
        if self.initialized:
            return
            
        try:
            print("🔌 Initializing MCP client...")
            
            # Load configuration and get enabled servers
            enabled_servers = self.get_enabled_mcp_servers()
            
            if not enabled_servers:
                raise ValueError("No MCP servers enabled in configuration")
            
            # Configure MCP client with enabled servers
            self.mcp_client = MultiServerMCPClient(enabled_servers)
            
            # Get tools from MCP servers
            self.tools = await self.mcp_client.get_tools()
            print(f"✅ Loaded {len(self.tools)} tools from {len(enabled_servers)} MCP servers")
            self.tools = await self._merge_tools(self.tools)
            
            # Get agent settings from config
            agent_settings = self.config.get("agent_settings", {})
            model_name = agent_settings.get("model", "gpt-4.1-mini")
            temperature = agent_settings.get("temperature", 0.7)
            max_tokens = agent_settings.get("max_tokens", 4000)
            
            # Initialize the language model with config settings
            llm = ChatOpenAI(
                base_url="https://openrouter.ai/api/v1",
                model=model_name,
                api_key=os.getenv("OPENROUTER_API_KEY"),
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            print(f"🤖 Using model: {model_name} (temp: {temperature}, max_tokens: {max_tokens})")
            
            # Get the custom system prompt
            prompt = self._get_system_prompt()

            # Create the agent with checkpointer for memory
            self.agent = create_agent(
                llm, 
                self.tools, 
                system_prompt=prompt,
                checkpointer=self.checkpointer
            )
            self.initialized = True
            print("✅ Generic Agent initialized successfully with persistent memory")
            
        except Exception as e:
            print(f"❌ Error initializing Generic Agent: {e}")
            raise

    async def chat(self, message: str, conversation_id: str = None) -> Dict[str, Any]:
        """Process a chat message and return the response"""
        if not self.initialized:
            await self.initialize()
        
        try:
            # Create thread configuration for this conversation
            thread_config = {
                "configurable": {
                    "thread_id": conversation_id or "default"
                }
            }
            
            # Create input state with user message
            input_state = {
                "messages": [HumanMessage(content=message)]
            }
            
            # Invoke the agent with the checkpointer handling conversation history
            print(f"🤖 Processing message: {message[:100]}...")
            response = await self.agent.ainvoke(input_state, config=thread_config)
            print(f"🔍 Agent response: {response['messages'][-1].content[:100]}...")
            
            # Extract the AI response
            ai_response = response['messages'][-1].content
            
            # Extract tools used from the response
            tools_used = self._extract_tools_used(response)
            
            return {
                "success": True,
                "response": ai_response,
                "conversation_id": conversation_id,
                "timestamp": datetime.now().isoformat(),
                "tools_used": tools_used
            }
                
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            print(f"❌ Error in chat: {type(e).__name__}: {e}\n{tb}", flush=True)
            return {
                "success": False,
                "error": f"{type(e).__name__}: {e}" if str(e) else type(e).__name__,
                "timestamp": datetime.now().isoformat()
            }
    
    def _extract_tools_used(self, response: Dict) -> List[str]:
        """Extract which tools were used in the response"""
        tools_used = []
        for message in response.get("messages", []):
            if hasattr(message, 'tool_calls') and message.tool_calls:
                for tool_call in message.tool_calls:
                    print(f"🔍 Tool call: {tool_call}")
                    tools_used.append(tool_call.get('name', 'unknown'))
        return list(set(tools_used))
    
    async def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get list of available tools"""
        if not self.initialized:
            await self.initialize()
            
        tool_list = []
        for tool in self.tools:
            # Try to extract OpenAPI-style schema if available
            if hasattr(tool, "openapi_schema"):
                parameters = tool.openapi_schema
            elif hasattr(tool, "args_schema") and hasattr(tool.args_schema, "schema"):
                parameters = tool.args_schema.schema()
            elif hasattr(tool, "args") and isinstance(tool.args, dict):
                parameters = tool.args
            else:
                parameters = {}

            tool_list.append({
                "name": getattr(tool, "name", ""),
                "description": getattr(tool, "description", ""),
                "parameters": parameters
            })
        return tool_list
    
    async def get_conversation_history(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Get conversation history for a specific conversation ID using checkpointer"""
        if not self.initialized:
            await self.initialize()
            
        try:
            # Create thread configuration
            thread_config = {
                "configurable": {
                    "thread_id": conversation_id
                }
            }
            
            # Get the current state from checkpointer
            state = await self.agent.aget_state(config=thread_config)
            
            # Convert messages to serializable format
            serialized_history = []
            for msg in state.values.get("messages", []):
                if isinstance(msg, HumanMessage):
                    serialized_history.append({
                        "type": "human",
                        "content": msg.content,
                        "timestamp": getattr(msg, 'timestamp', datetime.now().isoformat())
                    })
                elif isinstance(msg, AIMessage):
                    serialized_history.append({
                        "type": "ai", 
                        "content": msg.content,
                        "timestamp": getattr(msg, 'timestamp', datetime.now().isoformat())
                    })
            
            return serialized_history
            
        except Exception as e:
            print(f"❌ Error getting conversation history: {e}")
            return []

    async def get_full_trace(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Return the full per-turn trace for a thread: system, human, ai
        (with tool_calls), tool, etc. Used by the subtask trace inspector.
        """
        if not self.initialized:
            await self.initialize()

        try:
            cfg = {"configurable": {"thread_id": conversation_id}}
            state = await self.agent.aget_state(config=cfg)
            out: List[Dict[str, Any]] = []
            for msg in state.values.get("messages", []):
                content = msg.content if hasattr(msg, "content") else ""
                if isinstance(content, list):
                    parts = []
                    for c in content:
                        if isinstance(c, dict):
                            parts.append(c.get("text") or json.dumps(c))
                        else:
                            parts.append(str(c))
                    content = "\n".join(p for p in parts if p)

                if isinstance(msg, SystemMessage):
                    out.append({"type": "system", "content": content})
                elif isinstance(msg, HumanMessage):
                    out.append({"type": "human", "content": content})
                elif isinstance(msg, AIMessage):
                    tool_calls = []
                    for tc in (getattr(msg, "tool_calls", None) or []):
                        tool_calls.append({
                            "id": tc.get("id"),
                            "name": tc.get("name"),
                            "args": tc.get("args") or {},
                        })
                    out.append({
                        "type": "ai",
                        "content": content,
                        "tool_calls": tool_calls,
                    })
                elif isinstance(msg, ToolMessage):
                    out.append({
                        "type": "tool",
                        "name": getattr(msg, "name", None),
                        "tool_call_id": getattr(msg, "tool_call_id", None),
                        "content": content,
                    })
                else:
                    out.append({"type": getattr(msg, "type", "unknown"), "content": content})
            return out
        except Exception as e:
            print(f"❌ Error getting full trace: {e}")
            return []
    
    async def clear_conversation(self, conversation_id: str) -> bool:
        """Clear conversation history for a specific conversation ID"""
        if not self.initialized:
            await self.initialize()
            
        try:
            # Create thread configuration
            thread_config = {
                "configurable": {
                    "thread_id": conversation_id
                }
            }
            
            # Clear the conversation by creating a fresh state
            # Note: MemorySaver doesn't have a direct delete method,
            # but we can overwrite with empty state
            empty_state = {"messages": []}
            await self.agent.aupdate_state(config=thread_config, values=empty_state)
            
            return True
            
        except Exception as e:
            print(f"❌ Error clearing conversation: {e}")
            return False 