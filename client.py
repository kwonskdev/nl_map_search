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

        # Create request parameters
        request_params = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 1000,
            "messages": messages
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
            
            # Update request parameters for next round
            request_params["messages"] = self.conversation_history.copy()
            round_count += 1
        
        # If we hit max rounds, return a message
        logger.warning(f"Reached maximum rounds ({max_rounds}), stopping conversation")
        return "대화가 너무 길어져 중단되었습니다. 다시 시도해 주세요."

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
        else:
            logger.warning(f"Application initialization failed: {message}")
    except Exception as e:
        logger.error(f"Error during application initialization: {str(e)}", exc_info=True)
        if status_area:
            status_area.value = f"자동 연결 중 오류: {str(e)}"

@ui.page('/')
def main_page():
    global status_area, server_control_section, chat_section
    global server_controls_container
    
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
            
        # Auto-initialize when page loads
        ui.timer(0.1, initialize_app, once=True)
        
        # Chat section (initially hidden)
        with ui.card().classes('w-full') as chat_section:
            chat_section.visible = False
            ui.label('채팅').classes('text-h6 mb-2')
            
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
                
                ### 멀티턴 대화:
                - 이제 이전 대화 내용을 기억하여 연속적인 대화가 가능합니다.
                - "대화 초기화" 버튼으로 새로운 대화를 시작할 수 있습니다.
                - "대화 요약" 버튼으로 현재 대화 상태를 확인할 수 있습니다.
                """)
            
            chat_history = ui.html().classes('w-full border p-4 mb-4 bg-gray-50 min-h-64 max-h-96 overflow-y-auto')
            
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
    
    logger.info("Main page setup completed")

# Handle app shutdown
async def cleanup():
    logger.info("Application shutdown initiated")
    await client.cleanup()

app.on_shutdown(cleanup)

if __name__ in {"__main__", "__mp_main__"}:
    logger.info("Starting MCP Web Client application")
    ui.run(title='MCP Web Client', port=8080, host='0.0.0.0')