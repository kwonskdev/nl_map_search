import asyncio
import os
import json
import logging
from typing import Optional, Dict, List, Any
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from anthropic import Anthropic
from dotenv import load_dotenv
from nicegui import ui, app, run

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mcp_client.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# Suppress noisy loggers
logging.getLogger('watchfiles.main').setLevel(logging.WARNING)
logging.getLogger('watchfiles').setLevel(logging.WARNING)
logging.getLogger('nicegui').setLevel(logging.WARNING)
logging.getLogger('uvicorn').setLevel(logging.WARNING)
logging.getLogger('fastapi').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

load_dotenv()

class MCPWebClient:
    def __init__(self):
        logger.info("Initializing MCPWebClient")
        self.sessions: Dict[str, ClientSession] = {}
        self.servers: Dict[str, dict] = {}
        self.exit_stack = AsyncExitStack()
        self.anthropic = Anthropic()
        self.connected_servers = []
        self.server_tools: Dict[str, List[dict]] = {}
        self.server_prompts: Dict[str, List[dict]] = {}  # 프롬프트 목록 저장
        self.enabled_servers: Dict[str, bool] = {}
        self.enabled_tools: Dict[str, bool] = {}
        self.conversation_history: List[Dict[str, Any]] = []  # 대화 히스토리 저장
        self.chat_sessions: Dict[str, List[Dict[str, Any]]] = {}  # 멀티 탭 채팅 세션
        self.current_chat_id: str = "default"  # 현재 활성 채팅 세션 ID
        logger.info("MCPWebClient initialized successfully")
        
    def load_mcp_config(self, config_path: str) -> Dict[str, Any]:
        """Load MCP configuration from JSON file"""
        logger.info(f"Loading MCP configuration from: {config_path}")
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            if 'mcpServers' not in config:
                logger.error("Invalid MCP configuration format. Expected 'mcpServers' field.")
                raise ValueError("Invalid MCP configuration format. Expected 'mcpServers' field.")
            
            mcp_servers = config['mcpServers']
            logger.info(f"Loaded {len(mcp_servers)} servers from configuration")
            return mcp_servers
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {config_path}")
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in configuration file: {e}")
            raise ValueError(f"Invalid JSON in configuration file: {e}")

    async def connect_to_server(self, server_name: str, server_config: Dict[str, Any]):
        """Connect to an MCP server based on its configuration"""
        logger.info(f"Attempting to connect to server: {server_name}")
        try:
            command = server_config['command']
            args = server_config.get('args', [])
            env = server_config.get('env', {})
            
            logger.debug(f"Server {server_name} command: {command}")
            logger.debug(f"Server {server_name} args: {args}")
            logger.debug(f"Server {server_name} env vars: {list(env.keys())}")
            
            # Check if command exists
            import shutil
            if not shutil.which(command):
                logger.error(f"Command not found: {command}")
                raise FileNotFoundError(f"Command not found: {command}")
            
            full_env = {**os.environ, **env}

            server_params = StdioServerParameters(
                command=command,
                args=args,
                env=full_env
            )
            
            logger.debug(f"Creating stdio transport for server: {server_name}")
            stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
            stdio, write = stdio_transport
            session = await self.exit_stack.enter_async_context(ClientSession(stdio, write))

            logger.debug(f"Initializing session for server: {server_name}")
            
            # Add timeout for initialization
            try:
                await asyncio.wait_for(session.initialize(), timeout=30.0)
            except asyncio.TimeoutError:
                logger.error(f"Timeout during session initialization for server: {server_name}")
                raise Exception(f"Session initialization timeout for {server_name}")
            except Exception as e:
                logger.error(f"Session initialization failed for {server_name}: {str(e)}")
                raise

            self.sessions[server_name] = session
            self.servers[server_name] = {
                'config': server_config,
                'stdio': stdio,
                'write': write
            }
            
            logger.debug(f"Listing tools for server: {server_name}")
            
            # Add timeout for listing tools
            try:
                response = await asyncio.wait_for(session.list_tools(), timeout=10.0)
                tools = response.tools
            except asyncio.TimeoutError:
                logger.error(f"Timeout while listing tools for server: {server_name}")
                raise Exception(f"Tool listing timeout for {server_name}")
            except Exception as e:
                logger.error(f"Failed to list tools for {server_name}: {str(e)}")
                raise
            
            logger.info(f"Server {server_name} has {len(tools)} tools available")
            
            # Store tools for this server
            self.server_tools[server_name] = []
            for tool in tools:
                tool_info = {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.inputSchema
                }
                self.server_tools[server_name].append(tool_info)
                tool_key = f"{server_name}::{tool.name}"
                self.enabled_tools[tool_key] = True  # Enable all tools by default
                logger.debug(f"Registered tool: {tool_key}")
            
            # List prompts for this server
            logger.debug(f"Listing prompts for server: {server_name}")
            try:
                prompts_response = await asyncio.wait_for(session.list_prompts(), timeout=10.0)
                prompts = prompts_response.prompts
            except asyncio.TimeoutError:
                logger.warning(f"Timeout while listing prompts for server: {server_name}")
                prompts = []
            except Exception as e:
                logger.warning(f"Failed to list prompts for {server_name}: {str(e)}")
                prompts = []
            
            logger.info(f"Server {server_name} has {len(prompts)} prompts available")
            
            # Store prompts for this server
            self.server_prompts[server_name] = []
            for prompt in prompts:
                prompt_info = {
                    "name": prompt.name,
                    "description": prompt.description,
                    "arguments": prompt.arguments
                }
                self.server_prompts[server_name].append(prompt_info)
                logger.debug(f"Registered prompt: {server_name}::{prompt.name}")
            
            self.enabled_servers[server_name] = True  # Enable server by default
            self.connected_servers.append(f"✓ {server_name} ({len(tools)} tools, {len(prompts)} prompts)")
            logger.info(f"Successfully connected to server: {server_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to server {server_name}: {str(e)}", exc_info=True)
            self.connected_servers.append(f"✗ {server_name}: {str(e)}")
            return False

    async def get_prompt(self, prompt_name: str, args: Dict[str, Any] = None) -> str:
        """Get a prompt from any available server"""
        logger.info(f"Getting prompt: {prompt_name} with args: {args}")
        
        for server_name, prompts in self.server_prompts.items():
            if not self.enabled_servers.get(server_name, False):
                continue
                
            for prompt in prompts:
                if prompt['name'] == prompt_name:
                    try:
                        logger.debug(f"Found prompt {prompt_name} on server {server_name}")
                        result = await self.sessions[server_name].get_prompt(prompt_name, args or {})
                        logger.info(f"Successfully retrieved prompt: {prompt_name}")
                        return result.content
                    except Exception as e:
                        logger.error(f"Error getting prompt {prompt_name} from server {server_name}: {str(e)}")
                        continue
        
        logger.warning(f"Prompt {prompt_name} not found on any enabled server")
        return f"프롬프트 '{prompt_name}'을 찾을 수 없습니다."

    async def list_available_prompts(self) -> List[Dict[str, Any]]:
        """List all available prompts from enabled servers"""
        available_prompts = []
        
        for server_name, prompts in self.server_prompts.items():
            if not self.enabled_servers.get(server_name, False):
                continue
                
            for prompt in prompts:
                available_prompts.append({
                    "server": server_name,
                    "name": prompt['name'],
                    "description": prompt['description'],
                    "arguments": prompt['arguments']
                })
        
        return available_prompts

    async def auto_connect_all_servers(self):
        """Automatically connect to all servers from mcp.json"""
        logger.info("Starting auto-connect to all servers")
        config_path = 'mcp.json'
        if not os.path.exists(config_path):
            logger.error(f"Configuration file not found: {config_path}")
            return False, f"설정 파일을 찾을 수 없습니다: {config_path}"
        
        try:
            mcp_servers = self.load_mcp_config(config_path)
            
            # Exclude example servers
            filtered_servers = {
                name: config for name, config in mcp_servers.items()
                if not name.startswith('example')
            }
            
            logger.info(f"Found {len(filtered_servers)} non-example servers to connect to")
            
            if not filtered_servers:
                logger.warning("No servers to connect to (example servers excluded)")
                return False, "연결할 서버가 없습니다 (example 서버 제외됨)."
            
            connected_count = 0
            for server_name, server_config in filtered_servers.items():
                logger.info(f"Connecting to server: {server_name}")
                if await self.connect_to_server(server_name, server_config):
                    connected_count += 1
            
            logger.info(f"Auto-connect completed: {connected_count}/{len(filtered_servers)} servers connected")
            return connected_count > 0, f"{connected_count}/{len(filtered_servers)}개 서버 연결 성공 (example 서버 제외)"
            
        except Exception as e:
            logger.error(f"Error during auto-connect: {str(e)}", exc_info=True)
            return False, f"연결 중 오류: {str(e)}"

    async def process_query(self, query: str) -> str:
        """Process a query using Claude and available tools from enabled servers only"""
        logger.info(f"Processing query: {query[:100]}{'...' if len(query) > 100 else ''}")
        
        # Check if this is a prompt request
        prompt_result = await self._handle_prompt_request(query)
        if prompt_result:
            return prompt_result
        
        # Add user message to conversation history
        self.conversation_history.append({
            "role": "user",
            "content": query
        })
        
        # Save current session
        self._save_current_session()
        
        # Truncate conversation history if it gets too long
        self._truncate_conversation_history()
        
        # Use conversation history for context
        messages = self.conversation_history.copy()

        available_tools = []
        
        # Only use enabled servers and tools
        for server_name, tools in self.server_tools.items():
            if not self.enabled_servers.get(server_name, False):
                logger.debug(f"Skipping disabled server: {server_name}")
                continue
                
            for tool in tools:
                tool_key = f"{server_name}::{tool['name']}"
                if self.enabled_tools.get(tool_key, False):
                    available_tools.append(tool)
                    logger.debug(f"Added available tool: {tool_key}")

        logger.info(f"Available tools for query: {len(available_tools)}")

        # Create system message
        system_message = """당신은 장소 추천 전문가입니다. 사용자가 특정 조건에 맞는 장소를 찾아달라고 요청할 때, 다음과 같은 방식으로 답변해주세요:

1. **장소 분석**: 사용자의 요구사항을 분석하여 적합한 장소 유형을 파악합니다.

2. **장소 추천**: 조건에 맞는 장소 3-5곳을 추천합니다. 각 장소에 대해 다음 정보를 포함합니다:
   - **이름**: 장소의 정확한 이름
   - **주소**: 구체적인 주소
   - **특징**: 사용자 요구사항에 맞는 주요 특징
   - **추천 이유**: 왜 이 장소를 추천하는지

3. **지도 생성**: 추천한 장소들을 지도에 표시하기 위해 places_to_map 도구를 사용합니다. 장소명과 주소를 정확히 입력하여 지도를 생성하고, 생성된 HTML을 답변 끝에 포함시킵니다.

4. **답변 형식**:
   ```
   ## 🗺️ [사용자 요청 요약]
   
   ### 추천 장소:
   
   **1. [장소명]**
   - 📍 주소: [주소]
   - ✨ 특징: [주요 특징]
   - 💡 추천 이유: [추천 이유]
   
   **2. [장소명]**
   - 📍 주소: [주소]
   - ✨ 특징: [주요 특징]
   - 💡 추천 이유: [추천 이유]
   
   [추천 장소들...]
   
   ### 📍 지도 보기
   [places_to_map 도구로 생성된 지도 HTML이 여기에 표시됩니다]
   ```

5. **주의사항**:
   - 장소명과 주소는 정확하고 검색 가능한 형태로 제공
   - 사용자의 구체적인 요구사항(가격, 거리, 특별한 조건 등)을 반영
   - 한국어로 자연스럽게 답변
   - 지도 생성 후 HTML을 답변 끝에 포함

예시 시나리오:
- 홍대입구역 근처에 있는 리터당 가격 1300원 이하의 주유소
- 서울 시내 너무 높지 않은, 등산 초심자가 도전할만한 산을 추천
- 잠실에서 송도까지 야외를 가장 적게 거치면서 지나 갈 수 있는 대중교통 출근길
- 마곡역 반경 1km 내 비밀번호 없이 이용할 수 있는 공용 화장실
- 정규 3점라인이 있는 시간당 1만원 이하 서울 시내 농구 코트
- 강남역 근처에서 1인용 샤워 가능한 공간

이러한 요청이 들어오면 위의 형식에 따라 답변하고, places_to_map 도구를 사용하여 지도를 생성한 후 생성된 HTML을 답변 끝에 그대로 포함시켜주세요. 지도 HTML은 답변의 마지막 부분에 있어야 합니다."""

        # Create request parameters
        request_params = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 1000,
            "messages": messages,
            "system": system_message
        }
        
        # Only add tools if we have any available
        if available_tools:
            request_params["tools"] = available_tools
            logger.debug(f"Added {len(available_tools)} tools to request")

        # Handle multiple rounds of tool usage
        max_rounds = 10  # Prevent infinite loops
        round_count = 0
        all_tool_usage_messages = []  # Track all tool usage across rounds
        
        while round_count < max_rounds:
            logger.debug(f"Processing round {round_count + 1}/{max_rounds}")
            response = self.anthropic.messages.create(**request_params)
            
            # Check if response contains tool calls
            has_tool_calls = any(content.type == 'tool_use' for content in response.content)
            
            if not has_tool_calls:
                logger.info("No tool calls in response, returning final result")
                # No more tool calls, return the final response
                final_response = []
                
                # Add tool usage summary if any tools were used
                if all_tool_usage_messages:
                    final_response.append("\n".join(all_tool_usage_messages))
                    final_response.append("")  # Empty line for spacing
                
                for content in response.content:
                    if content.type == 'text':
                        final_response.append(content.text)
                return "\n".join(final_response)
            
            # Process tool calls in this round
            assistant_content = []
            tool_results = []
            tool_usage_messages = []  # Track which tools were used
            
            logger.debug(f"Processing {len([c for c in response.content if c.type == 'tool_use'])} tool calls")
            
            for content in response.content:
                if content.type == 'text':
                    assistant_content.append({
                        "type": "text",
                        "text": content.text
                    })
                elif content.type == 'tool_use':
                    tool_name = content.name
                    tool_args = content.input
                    tool_use_id = content.id
                    
                    logger.info(f"Executing tool: {tool_name} with args: {tool_args}")
                    
                    assistant_content.append({
                        "type": "tool_use",
                        "id": tool_use_id,
                        "name": tool_name,
                        "input": tool_args
                    })
                    
                    # Find the correct server for this tool
                    server_name = None
                    for srv_name, srv_tools in self.server_tools.items():
                        if not self.enabled_servers.get(srv_name, False):
                            continue
                        for srv_tool in srv_tools:
                            tool_key = f"{srv_name}::{srv_tool['name']}"
                            if srv_tool['name'] == tool_name and self.enabled_tools.get(tool_key, False):
                                server_name = srv_name
                                break
                        if server_name:
                            break
                    
                    # Execute the tool
                    if server_name and server_name in self.sessions:
                        try:
                            logger.debug(f"Calling tool {tool_name} on server {server_name}")
                            result = await self.sessions[server_name].call_tool(tool_name, tool_args)
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": tool_use_id,
                                "content": result.content
                            })
                            tool_usage_messages.append(f"🔧 도구 실행: {tool_name} (서버: {server_name})")
                            logger.info(f"Tool {tool_name} executed successfully")
                        except Exception as e:
                            logger.error(f"Error executing tool {tool_name}: {str(e)}", exc_info=True)
                            tool_results.append({
                                "type": "tool_result", 
                                "tool_use_id": tool_use_id,
                                "content": f"도구 실행 오류: {str(e)}"
                            })
                            tool_usage_messages.append(f"❌ 도구 실행 실패: {tool_name} - {str(e)}")
                    else:
                        logger.warning(f"Tool {tool_name} not found or server not available")
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tool_use_id, 
                            "content": f"활성화된 도구 {tool_name}를 찾을 수 없음"
                        })
                        tool_usage_messages.append(f"❌ 도구를 찾을 수 없음: {tool_name}")
            
            # Add current round's tool usage messages to the global list
            all_tool_usage_messages.extend(tool_usage_messages)
            
            # Add assistant message and tool results to conversation history
            self.conversation_history.append({
                "role": "assistant",
                "content": assistant_content
            })
            
            self.conversation_history.append({
                "role": "user",
                "content": tool_results
            })
            
            # Save current session after each round
            self._save_current_session()
            
            # Update request parameters for next round
            request_params["messages"] = self.conversation_history.copy()
            round_count += 1
        
        # If we hit max rounds, return a message
        logger.warning(f"Reached maximum rounds ({max_rounds}), stopping conversation")
        return "대화가 너무 길어져 중단되었습니다. 다시 시도해 주세요."
    
    def clear_conversation_history(self):
        """Clear the conversation history to start a new conversation"""
        logger.info("Clearing conversation history")
        self.conversation_history.clear()
    
    def create_new_chat_session(self) -> str:
        """Create a new chat session and return its ID"""
        import uuid
        chat_id = str(uuid.uuid4())[:8]  # Short ID for display
        self.chat_sessions[chat_id] = []
        
        # Ensure default session exists if this is the first session
        if "default" not in self.chat_sessions:
            self.chat_sessions["default"] = []
            self.current_chat_id = "default"
        
        logger.info(f"Created new chat session: {chat_id}")
        return chat_id
    
    def switch_chat_session(self, chat_id: str):
        """Switch to a different chat session"""
        if chat_id in self.chat_sessions:
            self.current_chat_id = chat_id
            self.conversation_history = self.chat_sessions[chat_id].copy()
            logger.info(f"Switched to chat session: {chat_id}")
        else:
            logger.warning(f"Chat session {chat_id} not found")
    
    def delete_chat_session(self, chat_id: str):
        """Delete a chat session"""
        if chat_id in self.chat_sessions:
            del self.chat_sessions[chat_id]
            logger.info(f"Deleted chat session: {chat_id}")
            
            # If we deleted the current session, switch to default
            if chat_id == self.current_chat_id:
                if "default" in self.chat_sessions:
                    self.switch_chat_session("default")
                else:
                    # Create new default session
                    self.current_chat_id = "default"
                    self.chat_sessions["default"] = []
                    self.conversation_history = []
    
    def _save_current_session(self):
        """Save current conversation to the current chat session"""
        self.chat_sessions[self.current_chat_id] = self.conversation_history.copy()
    
    def _truncate_conversation_history(self, max_messages: int = 20):
        """Truncate conversation history to prevent it from getting too long"""
        if len(self.conversation_history) > max_messages:
            logger.info(f"Truncating conversation history from {len(self.conversation_history)} to {max_messages} messages")
            # Keep the most recent messages
            self.conversation_history = self.conversation_history[-max_messages:]

    async def _handle_prompt_request(self, query: str) -> Optional[str]:
        """Handle prompt requests in the query"""
        import re
        
        # Check for prompt call patterns
        # Pattern 1: "create_interactive_map(검색어, 제목, 중심점, 반경)"
        prompt_pattern = r'create_interactive_map\s*\(\s*["\']([^"\']+)["\']\s*(?:,\s*["\']([^"\']+)["\']\s*(?:,\s*["\']([^"\']+)["\']\s*(?:,\s*(\d+(?:\.\d+)?)\s*)?)?)?\s*\)'
        
        match = re.search(prompt_pattern, query, re.IGNORECASE)
        if match:
            search_query = match.group(1)
            map_title = match.group(2) or "Interactive Map"
            center_location = match.group(3) or ""
            radius_km = float(match.group(4)) if match.group(4) else None
            
            logger.info(f"Detected create_interactive_map prompt call: {search_query}, {map_title}, {center_location}, {radius_km}")
            
            try:
                prompt_result = await self.get_prompt("create_interactive_map", {
                    "search_query": search_query,
                    "map_title": map_title,
                    "center_location": center_location,
                    "radius_km": radius_km
                })
                
                return f"## 🗺️ 인터랙티브 지도 생성 가이드\n\n{prompt_result}\n\n**이제 위의 가이드를 따라 실제 지도를 생성해보세요!**"
                
            except Exception as e:
                logger.error(f"Error handling create_interactive_map prompt: {str(e)}")
                return f"프롬프트 처리 중 오류가 발생했습니다: {str(e)}"
        
        # Pattern 2: "프롬프트: create_interactive_map" 형태
        if "프롬프트:" in query or "prompt:" in query.lower():
            prompt_name_match = re.search(r'(?:프롬프트|prompt):\s*(\w+)', query, re.IGNORECASE)
            if prompt_name_match:
                prompt_name = prompt_name_match.group(1)
                logger.info(f"Detected prompt request: {prompt_name}")
                
                try:
                    prompt_result = await self.get_prompt(prompt_name)
                    return f"## 📝 프롬프트: {prompt_name}\n\n{prompt_result}"
                except Exception as e:
                    logger.error(f"Error getting prompt {prompt_name}: {str(e)}")
                    return f"프롬프트 '{prompt_name}'을 가져오는 중 오류가 발생했습니다: {str(e)}"
        
        return None
    
    async def cleanup(self):
        """Clean up resources"""
        logger.info("Cleaning up MCPWebClient resources")
        await self.exit_stack.aclose()
        logger.info("MCPWebClient cleanup completed")

# Global client instance
client = MCPWebClient()

# Global UI components that need to be updated
server_control_section = None
chat_section = None
status_area = None
server_controls_container = None
chat_tabs_container = None
chat_content_container = None

# Global notification function that's safe to use after UI updates
def safe_notify(message: str, type: str = 'info'):
    """Safely show notification without depending on specific UI context"""
    try:
        ui.notify(message, type=type)
    except Exception as e:
        logger.info(f"Notification failed: {message} (error: {e})")

async def update_server_controls():
    """Update server control switches and tool buttons"""
    logger.debug("Updating server controls UI")
    if not server_controls_container:
        logger.warning("Server controls container not available")
        return
        
    server_controls_container.clear()
    for server_name in client.enabled_servers.keys():
        with server_controls_container:
            # Server switch
            enabled = client.enabled_servers[server_name]
            tools_count = len(client.server_tools.get(server_name, []))
            prompts_count = len(client.server_prompts.get(server_name, []))
            switch = ui.switch(f'{server_name} ({tools_count} tools, {prompts_count} prompts)', 
                             value=enabled)
            
            def make_server_toggle(name):
                async def toggle_server(value):
                    logger.info(f"Toggling server {name} to {value.value}")
                    client.enabled_servers[name] = value.value
                    await update_server_controls()  # Refresh entire UI
                return toggle_server
            
            switch.on_value_change(make_server_toggle(server_name))
            
            # Tool and prompt buttons (only show when server is enabled)
            if enabled:
                with ui.column().classes('ml-8 mt-2 mb-4'):
                    # Tools section
                    tools = client.server_tools.get(server_name, [])
                    if tools:
                        ui.label('🔧 도구 (Tools)').classes('text-sm font-bold mt-2 mb-1')
                        # Group tools in rows of 4
                        for i in range(0, len(tools), 4):
                            with ui.row().classes('gap-2'):
                                for tool in tools[i:i+4]:
                                    tool_key = f"{server_name}::{tool['name']}"
                                    tool_enabled = client.enabled_tools.get(tool_key, True)
                                    
                                    def make_tool_button(key, tool_name, description):
                                        def toggle_tool():
                                            current_state = client.enabled_tools.get(key, True)
                                            new_state = not current_state
                                            logger.info(f"Toggling tool {key} from {current_state} to {new_state}")
                                            client.enabled_tools[key] = new_state
                                            # Update UI immediately
                                            asyncio.create_task(update_server_controls())
                                        return toggle_tool
                                    
                                    # Tool button with color based on enabled state
                                    color = 'primary' if tool_enabled else 'grey'
                                    btn = ui.button(
                                        tool['name'], 
                                        color=color,
                                        on_click=make_tool_button(tool_key, tool['name'], tool['description'])
                                    ).props('size=sm').tooltip(tool['description'])
                    
                    # Prompts section
                    prompts = client.server_prompts.get(server_name, [])
                    if prompts:
                        ui.label('📝 프롬프트 (Prompts)').classes('text-sm font-bold mt-4 mb-1')
                        # Group prompts in rows of 3
                        for i in range(0, len(prompts), 3):
                            with ui.row().classes('gap-2'):
                                for prompt in prompts[i:i+3]:
                                    def make_prompt_button(prompt_name, prompt_description):
                                        async def show_prompt():
                                            try:
                                                prompt_result = await client.get_prompt(prompt_name)
                                                # Show prompt in a dialog
                                                with ui.dialog() as dialog, ui.card():
                                                    ui.label(f'📝 프롬프트: {prompt_name}').classes('text-h6 mb-2')
                                                    ui.html(f'<pre style="white-space: pre-wrap; word-wrap: break-word;">{prompt_result}</pre>').classes('max-w-2xl max-h-96 overflow-y-auto')
                                                    ui.button('닫기', on_click=dialog.close).props('color=primary')
                                                dialog.open()
                                            except Exception as e:
                                                logger.error(f"Error showing prompt {prompt_name}: {str(e)}")
                                                ui.notify(f'프롬프트 표시 오류: {str(e)}', type='error')
                                        return show_prompt
                                    
                                    btn = ui.button(
                                        prompt['name'], 
                                        color='secondary',
                                        on_click=make_prompt_button(prompt['name'], prompt['description'])
                                    ).props('size=sm').tooltip(prompt['description'])
    logger.debug("Server controls UI updated")

async def update_tool_controls():
    """Deprecated - tool controls are now integrated with server controls"""
    logger.debug("update_tool_controls called (deprecated)")
    pass

async def update_chat_tabs():
    """Update chat tabs UI"""
    logger.debug("Updating chat tabs UI")
    if not chat_tabs_container:
        logger.warning("Chat tabs container not available")
        return
    
    chat_tabs_container.clear()
    
    # Ensure default session exists
    if "default" not in client.chat_sessions:
        client.chat_sessions["default"] = []
        client.current_chat_id = "default"
    
    # If no current chat ID is set, use default
    if not client.current_chat_id:
        client.current_chat_id = "default"
    
    for chat_id in client.chat_sessions.keys():
        with chat_tabs_container:
            # Determine if this is the active tab
            is_active = chat_id == client.current_chat_id
            active_class = "bg-blue-100 border-b-0" if is_active else "bg-gray-50 hover:bg-gray-100"
            
            # Create tab with separate close button
            with ui.row().classes(f'border border-gray-300 {active_class} rounded-t-lg cursor-pointer').style('min-width: 120px;'):
                # Tab label (clickable for switching)
                tab_label = ui.label(f'채팅 {chat_id}').classes('flex-grow px-3 py-1')
                
                # Close button (separate from tab label)
                close_btn = ui.button('✕', color='red').props('size=sm round flat').classes('opacity-0 hover:opacity-100 transition-opacity')
                
                # Add click handler for tab switching
                def make_tab_click_handler(chat_id):
                    async def switch_tab():
                        client.switch_chat_session(chat_id)
                        await update_chat_content()
                        await update_chat_tabs()  # Refresh tabs to show active state
                    return switch_tab
                
                # Add click handler for close button
                def make_close_handler(chat_id):
                    async def close_tab():
                        if len(client.chat_sessions) > 1:  # Don't close if it's the last tab
                            client.delete_chat_session(chat_id)
                            await update_chat_tabs()
                            await update_chat_content()
                            safe_notify(f'채팅 탭이 닫혔습니다.', 'info')
                        else:
                            safe_notify(f'마지막 탭은 닫을 수 없습니다.', 'warning')
                    return close_tab
                
                # Set up event handlers
                tab_label.on('click', make_tab_click_handler(chat_id))
                close_btn.on('click', make_close_handler(chat_id))
    
    logger.debug("Chat tabs UI updated")

async def update_chat_content():
    """Update chat content area"""
    logger.debug("Updating chat content area")
    if not chat_content_container:
        logger.warning("Chat content container not available")
        return
    
    chat_content_container.clear()
    
    with chat_content_container:
        # Conversation management buttons
        with ui.row().classes('mb-4 gap-2'):
            clear_btn = ui.button('🗑️ 대화 초기화', color='warning')
            
            async def clear_conversation():
                client.clear_conversation_history()
                await update_chat_content()
                safe_notify('대화가 초기화되었습니다.', 'info')
            
            clear_btn.on_click(clear_conversation)
        
        # Chat history display
        chat_history = ui.html().classes('w-full border p-4 mb-4 bg-gray-50 min-h-64 max-h-96 overflow-y-auto')
        
        # Display existing messages
        if client.conversation_history:
            history_html = ""
            for msg in client.conversation_history:
                if msg["role"] == "user":
                    if isinstance(msg["content"], str):
                        history_html += f'<div class="mb-2"><strong>사용자:</strong> {msg["content"]}</div>'
                elif msg["role"] == "assistant":
                    if isinstance(msg["content"], list):
                        for content in msg["content"]:
                            if content.get("type") == "text":
                                history_html += f'<div class="mb-2"><strong>AI:</strong> <pre style="white-space: pre-wrap; word-wrap: break-word;">{content["text"]}</pre></div>'
            chat_history.content = history_html
        else:
            chat_history.content = '<div class="text-gray-500">새로운 대화를 시작하세요.</div>'
        
        # Input area
        with ui.row().classes('w-full'):
            query_input = ui.input('질문을 입력하세요...').classes('flex-grow')
            send_btn = ui.button('전송', color='primary')
        
        async def send_query():
            if not query_input.value.strip():
                logger.debug("Empty query, ignoring")
                return
            
            query = query_input.value
            query_input.value = ''
            logger.info(f"User sent query: {query[:100]}{'...' if len(query) > 100 else ''}")
            
            # Add user message to chat
            chat_history.content += f'<div class="mb-2"><strong>사용자:</strong> {query}</div>'
            
            try:
                # Add loading indicator
                chat_history.content += '<div class="mb-2"><strong>AI:</strong> <em>응답 생성 중...</em></div>'
                
                response = await client.process_query(query)
                logger.info("Query processed successfully")
                
                # Replace loading indicator with actual response
                chat_content = chat_history.content
                last_loading_idx = chat_content.rfind('<div class="mb-2"><strong>AI:</strong> <em>응답 생성 중...</em></div>')
                if last_loading_idx != -1:
                    chat_history.content = (
                        chat_content[:last_loading_idx] + 
                        f'<div class="mb-2"><strong>AI:</strong> <pre style="white-space: pre-wrap; word-wrap: break-word;">{response}</pre></div>'
                    )
                
            except Exception as e:
                logger.error(f"Error processing query: {str(e)}", exc_info=True)
                # Replace loading indicator with error message
                chat_content = chat_history.content
                last_loading_idx = chat_content.rfind('<div class="mb-2"><strong>AI:</strong> <em>응답 생성 중...</em></div>')
                if last_loading_idx != -1:
                    chat_history.content = (
                        chat_content[:last_loading_idx] + 
                        f'<div class="mb-2"><strong>오류:</strong> {str(e)}</div>'
                    )
        
        send_btn.on_click(send_query)
        query_input.on('keydown.enter', send_query)
    
    logger.debug("Chat content area updated")

async def initialize_app():
    """Initialize the application by connecting to all servers"""
    global status_area, server_control_section, chat_section
    
    logger.info("Initializing application")
    try:
        if status_area:
            status_area.value = "자동 서버 연결 중..."
        
        success, message = await client.auto_connect_all_servers()
        
        if status_area:
            status_area.value = message + "\n\n연결된 서버:\n" + "\n".join(client.connected_servers)
        
        if success:
            logger.info("Application initialization successful")
            if server_control_section:
                server_control_section.visible = True
            if chat_section:
                chat_section.visible = True
            await update_server_controls()
            await update_chat_tabs()
            await update_chat_content()
        else:
            logger.warning(f"Application initialization failed: {message}")
    except Exception as e:
        logger.error(f"Error during application initialization: {str(e)}", exc_info=True)
        if status_area:
            status_area.value = f"자동 연결 중 오류: {str(e)}"

@ui.page('/')
def main_page():
    global status_area, server_control_section, chat_section
    global server_controls_container, chat_tabs_container, chat_content_container
    
    logger.info("Setting up main page")
    ui.page_title('MCP Web Client')
    
    with ui.header():
        ui.label('MCP Web Client').classes('text-h6')
    
    with ui.column().classes('w-full max-w-6xl mx-auto p-4'):
        # Connection status section
        with ui.card().classes('w-full mb-4'):
            ui.label('서버 연결 상태').classes('text-h6 mb-2')
            status_area = ui.textarea('초기화 중...').classes('w-full')
            status_area.props('readonly')
        
        # Server and tool control section (initially hidden)
        with ui.card().classes('w-full mb-4') as server_control_section:
            server_control_section.visible = False
            ui.label('서버 및 도구 제어').classes('text-h6 mb-2')
            ui.label('서버를 끄면 해당 서버의 모든 도구와 프롬프트가 비활성화됩니다. 개별 도구와 프롬프트는 버튼을 클릭해서 켜고 끌 수 있습니다.').classes('text-sm text-gray-600 mb-4')
            server_controls_container = ui.column()
            
        # Chat section (initially hidden)
        with ui.card().classes('w-full') as chat_section:
            chat_section.visible = False
            ui.label('채팅').classes('text-h6 mb-2')
            
            # Chat tabs and new chat button
            with ui.row().classes('mb-4 gap-2 items-center'):
                chat_tabs_container = ui.row().classes('flex-grow gap-1')
                new_chat_btn = ui.button('➕', color='primary').props('size=sm round')
                
                async def create_new_chat():
                    chat_id = client.create_new_chat_session()
                    client.switch_chat_session(chat_id)
                    await update_chat_tabs()
                    await update_chat_content()
                    safe_notify(f'새 채팅이 생성되었습니다.', 'info')
                
                new_chat_btn.on_click(create_new_chat)
            
            # Chat content area
            chat_content_container = ui.column().classes('w-full')
            
            # Add prompt usage examples
            with ui.expansion('📝 프롬프트 사용 예시', icon='help').classes('mb-4'):
                ui.markdown("""
                ### 프롬프트 사용 방법:
                
                **1. create_interactive_map 프롬프트:**
                ```
                create_interactive_map("홍대 카페", "홍대 카페 지도", "홍대입구역", 2.0)
                ```
                
                **2. 일반 프롬프트 호출:**
                ```
                프롬프트: create_interactive_map
                ```
                
                **3. 프롬프트 버튼 클릭:**
                - 위의 서버 제어 섹션에서 프롬프트 버튼을 클릭하면 프롬프트 내용을 확인할 수 있습니다.
                
                ### 멀티 탭 채팅:
                - ➕ 버튼을 클릭하여 새로운 채팅 탭을 생성할 수 있습니다.
                - 각 탭은 독립적인 대화 히스토리를 가집니다.
                - 탭에 마우스를 올리면 X 버튼이 나타나 탭을 닫을 수 있습니다.
                """)
        
        # Auto-initialize when page loads (after all UI components are created)
        ui.timer(0.1, initialize_app, once=True)
    
    logger.info("Main page setup completed")

# Handle app shutdown
async def cleanup():
    logger.info("Application shutdown initiated")
    await client.cleanup()

app.on_shutdown(cleanup)

if __name__ in {"__main__", "__mp_main__"}:
    logger.info("Starting MCP Web Client application")
    ui.run(title='MCP Web Client', port=8080, host='0.0.0.0')