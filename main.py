import asyncio
import json
import logging
import os
import re
import shutil
from contextlib import AsyncExitStack
from typing import Any

import httpx
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

TRIGGER_WORDS = {"reports", "report", "summary", "summarize", "document", "pdf"}

class Configuration:
    """Manages configuration and environment variables for the MCP client."""

    def __init__(self) -> None:
        """Initialize configuration with environment variables."""
        self.load_env()
        self.api_key = os.getenv("LLM_API_KEY")

    @staticmethod
    def load_env() -> None:
        """Load environment variables from .env file."""
        load_dotenv()

    @staticmethod
    def load_config(file_path: str) -> dict[str, Any]:
        """Load server configuration from JSON file.

        Args:
            file_path: Path to the JSON configuration file.

        Returns:
            Dict containing server configuration.

        Raises:
            FileNotFoundError: If configuration file doesn't exist.
            JSONDecodeError: If configuration file is invalid JSON.
        """
        with open(file_path, "r") as f:
            return json.load(f)

    @property
    def llm_api_key(self) -> str:
        """Get the LLM API key.

        Returns:
            The API key as a string.

        Raises:
            ValueError: If the API key is not found in environment variables.
        """
        if not self.api_key:
            raise ValueError("LLM_API_KEY not found in environment variables")
        return self.api_key


class Server:
    """Manages MCP server connections and tool execution."""

    def __init__(self, name: str, config: dict[str, Any]) -> None:
        self.name: str = name
        self.config: dict[str, Any] = config
        self.stdio_context: Any | None = None
        self.session: ClientSession | None = None
        self._cleanup_lock: asyncio.Lock = asyncio.Lock()
        self.exit_stack: AsyncExitStack = AsyncExitStack()

    async def initialize(self) -> None:
        """Initialize the server connection."""
        command = shutil.which("npx") if self.config["command"] == "npx" else self.config["command"]
        if command is None:
            raise ValueError("The command must be a valid string and cannot be None.")

        server_params = StdioServerParameters(
            command=command,
            args=self.config["args"],
            env={**os.environ, **self.config["env"]} if self.config.get("env") else None,
        )
        try:
            stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
            read, write = stdio_transport
            session = await self.exit_stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            self.session = session
        except Exception as e:
            logging.error(f"Error initializing server {self.name}: {e}")
            await self.cleanup()
            raise

    async def list_tools(self) -> list[Any]:
        """List available tools from the server.

        Returns:
            A list of available tools.

        Raises:
            RuntimeError: If the server is not initialized.
        """
        if not self.session:
            raise RuntimeError(f"Server {self.name} not initialized")

        tools_response = await self.session.list_tools()
        tools = []

        for item in tools_response:
            if isinstance(item, tuple) and item[0] == "tools":
                tools.extend(Tool(tool.name, tool.description, tool.inputSchema, tool.title) for tool in item[1])

        return tools

    async def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        retries: int = 2,
        delay: float = 1.0,
    ) -> Any:
        """Execute a tool with retry mechanism.

        Args:
            tool_name: Name of the tool to execute.
            arguments: Tool arguments.
            retries: Number of retry attempts.
            delay: Delay between retries in seconds.

        Returns:
            Tool execution result.

        Raises:
            RuntimeError: If server is not initialized.
            Exception: If tool execution fails after all retries.
        """
        if not self.session:
            raise RuntimeError(f"Server {self.name} not initialized")

        attempt = 0
        while attempt < retries:
            try:
                logging.info(f"Executing {tool_name}...")
                result = await self.session.call_tool(tool_name, arguments)

                return result

            except Exception as e:
                attempt += 1
                logging.warning(f"Error executing tool: {e}. Attempt {attempt} of {retries}.")
                if attempt < retries:
                    logging.info(f"Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                else:
                    logging.error("Max retries reached. Failing.")
                    raise

    async def cleanup(self) -> None:
        """Clean up server resources."""
        async with self._cleanup_lock:
            try:
                await self.exit_stack.aclose()
                self.session = None
                self.stdio_context = None
            except Exception as e:
                logging.error(f"Error during cleanup of server {self.name}: {e}")


class Tool:
    """Represents a tool with its properties and formatting."""

    def __init__(
        self,
        name: str,
        description: str,
        input_schema: dict[str, Any],
        title: str | None = None,
    ) -> None:
        self.name: str = name
        self.title: str | None = title
        self.description: str = description
        self.input_schema: dict[str, Any] = input_schema

    def format_for_llm(self) -> str:
        """Format tool information for LLM.

        Returns:
            A formatted string describing the tool.
        """
        args_desc = []
        if "properties" in self.input_schema:
            for param_name, param_info in self.input_schema["properties"].items():
                arg_desc = f"- {param_name}: {param_info.get('description', 'No description')}"
                if param_name in self.input_schema.get("required", []):
                    arg_desc += " (required)"
                args_desc.append(arg_desc)

        # Build the formatted output with title as a separate field
        output = f"Tool: {self.name}\n"

        # Add human-readable title if available
        if self.title:
            output += f"User-readable title: {self.title}\n"

        output += f"""Description: {self.description}
Arguments:
{chr(10).join(args_desc)}
"""

        return output


"""class LLMClient:
    #Manages communication with the LLM provider.

    def __init__(self, api_key: str) -> None:
        self.api_key: str = api_key

    def get_response(self, messages: list[dict[str, str]]) -> str:
        #Get a response from the LLM.

        #Args:
        #    messages: A list of message dictionaries.

        #Returns:
        #    The LLM's response as a string.

        #Raises:
            httpx.RequestError: If the request to the LLM fails.
        
        url = "https://api.groq.com/openai/v1/chat/completions"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "messages": messages,
            "model": "meta-llama/llama-4-scout-17b-16e-instruct",
            "temperature": 0.7,
            "max_tokens": 4096,
            "top_p": 1,
            "stream": False,
            "stop": None,
        }

        try:
            with httpx.Client() as client:
                response = client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"]

        except httpx.RequestError as e:
            error_message = f"Error getting LLM response: {str(e)}"
            logging.error(error_message)

            if isinstance(e, httpx.HTTPStatusError):
                status_code = e.response.status_code
                logging.error(f"Status code: {status_code}")
                logging.error(f"Response details: {e.response.text}")

            return f"I encountered an error: {error_message}. Please try again or rephrase your request."
"""


class QwenClient:
    """replacing LLMClient using my fine-tuned Qwen."""

    def __init__(self, model_path: str, base_model: str, use_adapter: bool = True) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        try:
            from peft import PeftModel
            PEFT = True
        except Exception:
            PEFT = False

        self.torch = torch
        tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "left"
        self.tokenizer = tok

        base_m = AutoModelForCausalLM.from_pretrained(
            base_model,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )

        if use_adapter and PEFT:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(base_m, model_path)
            logging.info(f"[Qwen] Loaded adapter from {model_path}")
        else:
            self.model = base_m
            logging.info(f"[Qwen] Loaded base model {base_model}")

        self.model.eval()
        self.known_mcp_tools: set = set()  # populated after MCP init
    
    def set_known_tools(self, tool_names: list[str]) -> None:
        self.known_mcp_tools = set(tool_names)
        logging.info(f"[Qwen] Known MCP tools: {self.known_mcp_tools}")
        
    def _extract_first_json(self, text: str) -> str:
        """Extract only the first complete JSON object from model output."""
        start = text.find("{")
        if start == -1:
            return text
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i+1]
        return text
    
    def get_response(self, messages: list[dict], mcp_tools: list = None) -> str:
        user_content = ""
        for m in reversed(messages):
            if m["role"] == "user":
                user_content = m["content"]
                break

        has_trigger = any(t in user_content.lower() for t in TRIGGER_WORDS)

        if has_trigger:
            # Tool-calling format — matches training format exactly
            prompt = f"{user_content}\n### Response:\n"
        else:
            # Conversational format — steers model away from tool JSON
            prompt = (
                "You are a helpful assistant. Answer the following question "
                "conversationally without using any tools.\n\n"
                f"User: {user_content}\nAssistant:"
            )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        ).to(next(self.model.parameters()).device)

        with self.torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.3,
            )

        decoded = self.tokenizer.decode(
            output[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True,
        ).strip()

        if "TOOL CALL INVOKED" in decoded:
            decoded = decoded.replace("TOOL CALL INVOKED", "").strip()

        if has_trigger:
            decoded = self._extract_first_json(decoded)

        return decoded
    
class ChatSession:
    """Orchestrates the interaction between user, LLM, and tools."""

    def __init__(self, servers: list[Server], llm_client: QwenClient) -> None:
        self.servers: list[Server] = servers
        self.llm_client: QwenClient = llm_client

    async def cleanup_servers(self) -> None:
        """Clean up all servers properly."""
        for server in reversed(self.servers):
            try:
                await server.cleanup()
            except Exception as e:
                logging.warning(f"Warning during final cleanup: {e}")

    async def process_llm_response(self, llm_response: str, messages: list = None) -> str:
        """Process the LLM response and execute tools if needed.
        Args:
            llm_response: The response from the LLM.
        Returns:
            The result of tool execution or the original response.
        """
        try:
            tool_call = json.loads(llm_response)
            if "tool" in tool_call and "arguments" in tool_call:
                
                # Block hallucinated tool names not in MCP registry
                if tool_call["tool"] not in self.llm_client.known_mcp_tools:
                    logging.info(f"[INFO] '{tool_call['tool']}' not a valid MCP tool — ignoring")
                    return llm_response
                
                # Inject required arguments if empty
                if tool_call["tool"] == "search_documents_with_ai":
                    if not tool_call["arguments"].get("document_id"):
                        user_msg = ""
                        if messages:
                            for m in reversed(messages):
                                if m["role"] == "user":
                                    user_msg = m["content"]
                                    break

                        # Try to match user query against loaded document keywords
                        resolved_id = ""
                        for keyword, doc_id in self.doc_registry.items():
                            if keyword in user_msg.lower():
                                resolved_id = doc_id
                                logging.info(f"[DocRegistry] Matched keyword '{keyword}' → {doc_id}")
                                break

                        # Fallback to default (first document)
                        if not resolved_id:
                            resolved_id = self.default_doc_id
                            logging.info(f"[DocRegistry] Using default doc_id: {resolved_id}")

                        tool_call["arguments"]["document_id"] = resolved_id

                    if not tool_call["arguments"].get("query"):
                        user_msg = ""
                        if messages:
                            for m in reversed(messages):
                                if m["role"] == "user":
                                    user_msg = m["content"]
                                    break
                        tool_call["arguments"]["query"] = user_msg

                logging.info(f"Executing tool: {tool_call['tool']}")
                logging.info(f"With arguments: {tool_call['arguments']}")

                for server in self.servers:
                    tools = await server.list_tools()
                    if any(tool.name == tool_call["tool"] for tool in tools):
                        try:
                            result = await server.execute_tool(
                                tool_call["tool"], tool_call["arguments"]
                            )
                            if isinstance(result, dict) and "progress" in result:
                                progress = result["progress"]
                                total = result["total"]
                                percentage = (progress / total) * 100
                                logging.info(f"Progress: {progress}/{total} ({percentage:.1f}%)")
                            return f"Tool execution result: {result}"
                        except Exception as e:
                            error_msg = f"Error executing tool: {str(e)}"
                            logging.error(error_msg)
                            return error_msg

                # Option C fallthrough
                logging.info(f"[INFO] Tool '{tool_call['tool']}' not in MCP, answering directly")
                return llm_response

            return llm_response
        except json.JSONDecodeError:
            return llm_response

    async def _load_document_registry(self) -> None:
        """Load available documents from MCP and cache id+title mapping."""
        self.doc_registry: dict[str, str] = {}  # keyword → document_id
        self.default_doc_id: str = ""

        try:
            for server in self.servers:
                tools = await server.list_tools()
                if any(t.name == "list_documents" for t in tools):
                    result = await server.execute_tool("list_documents", {})
                    # Safe text extraction
                    text = ""
                    if hasattr(result, "content") and result.content:
                        for c in result.content:
                            if hasattr(c, "text"):
                                text += c.text
                            else:
                                text = str(result)

                            if not text.strip():
                                logging.warning("[DocRegistry] Empty response from list_documents")
                                break
                            
                            docs = json.loads(text)
                            for doc in docs:
                                doc_id = doc.get("id", "")
                                title = doc.get("title", "").lower()
                                filename = doc.get("metadata", {}).get(
                                    "originalFilename", ""
                                ).lower().replace(".", " ").replace("_", " ")

                                for keyword in set([title] + title.split() + filename.split()):
                                    if len(keyword) > 2:
                                        self.doc_registry[keyword] = doc_id

                                if not self.default_doc_id and doc_id:
                                    self.default_doc_id = doc_id
                                    
                            logging.info(f"[DocRegistry] Loaded {len(docs)} documents")
                            logging.info(f"[DocRegistry] Keywords: {list(self.doc_registry.keys())}")
                            logging.info(f"[DocRegistry] Default doc_id: {self.default_doc_id}")
                            break
                    # Parse the JSON from the result text
                    """if hasattr(result, "content"):
                        for c in result.content:
                            if hasattr(c, "text"):
                                text += c.text
                    else:
                        text = str(result)

                    docs = json.loads(text)
                    for doc in docs:
                        doc_id = doc.get("id", "")
                        title = doc.get("title", "").lower()
                        filename = doc.get("metadata", {}).get(
                            "originalFilename", ""
                        ).lower().replace(".", " ").replace("_", " ")

                        # Register multiple keywords per document
                        for keyword in set([title] + title.split() + filename.split()):
                            if len(keyword) > 2:  # skip short words
                                self.doc_registry[keyword] = doc_id

                        # First document becomes default
                        if not self.default_doc_id and doc_id:
                            self.default_doc_id = doc_id

                    logging.info(f"[DocRegistry] Loaded {len(docs)} documents")
                    logging.info(f"[DocRegistry] Keywords: {list(self.doc_registry.keys())}")
                    logging.info(f"[DocRegistry] Default doc_id: {self.default_doc_id}")
                    break
                    """
        except Exception as e:
            logging.warning(f"[DocRegistry] Failed to load documents: {e}")
            self.doc_registry = {}
            self.default_doc_id = ""

    async def start(self) -> None:
        """Main chat session handler."""
        try:
            for server in self.servers:
                try:
                    await server.initialize()
                except Exception as e:
                    logging.error(f"Failed to initialize server: {e}")
                    await self.cleanup_servers()
                    return

                all_tools = []
                for server in self.servers:
                    tools = await server.list_tools()
                    all_tools.extend(tools)

                # Register known MCP tool names with Qwen so it can validate outputs
                self.llm_client.set_known_tools([t.name for t in all_tools])
                # Dynamically load document registry from MCP
                await self._load_document_registry()

                tools_description = "\n".join([tool.format_for_llm() for tool in all_tools])

                system_message = (
                    "You are a helpful assistant with access to these tools:\n\n"
                    f"{tools_description}\n"
                    "Choose the appropriate tool based on the user's question. "
                    "If no tool is needed, reply directly.\n\n"
                    "IMPORTANT: When you need to use a tool, you must ONLY respond with "
                    "the exact JSON object format below, nothing else:\n"
                    "{\n"
                    '    "tool": "tool-name",\n'
                    '    "arguments": {\n'
                    '        "argument-name": "value"\n'
                    "    }\n"
                    "}\n\n"
                    "After receiving a tool's response:\n"
                    "1. Transform the raw data into a natural, conversational response\n"
                    "2. Keep responses concise but informative\n"
                    "3. Focus on the most relevant information\n"
                    "4. Use appropriate context from the user's question\n"
                    "5. Avoid simply repeating the raw data\n\n"
                    "Please use only the tools that are explicitly defined above."
                )

                messages = [{"role": "system", "content": system_message}]

                # Define direct commands once, before the loop
                DIRECT_TOOL_COMMANDS = {
                    "process uploads": ("process_uploads", {}),
                    "list documents": ("list_documents", {}),
                    "list uploads": ("list_uploads_files", {}),
                    "get uploads path": ("get_uploads_path", {}),
                }

                while True:
                    try:
                        user_input = input("You: ").strip().lower()
                        if user_input in ["quit", "exit"]:
                            logging.info("\nExiting...")
                            break

                        # Check direct tool commands BEFORE passing to Qwen
                        direct = DIRECT_TOOL_COMMANDS.get(user_input.strip().lower())
                        if direct:
                            tool_name, tool_args = direct
                            logging.info(f"[Direct] Calling tool: {tool_name}")
                            for server in self.servers:
                                srv_tools = await server.list_tools()
                                if any(t.name == tool_name for t in srv_tools):
                                    try:
                                        result = await server.execute_tool(tool_name, tool_args)
                                        logging.info(f"[Direct] Result: {result}")
                                        messages.append({"role": "user", "content": user_input})
                                        messages.append({"role": "assistant", "content": str(result)})

                                        # Reload document registry after ingesting new files
                                        if tool_name == "process_uploads":
                                            await self._load_document_registry()
                                            logging.info("[DocRegistry] Reloaded after process_uploads")

                                    except Exception as e:
                                        logging.error(f"[Direct] Tool error: {e}")
                                    break
                            continue  # skip Qwen inference for this turn

                        # Normal Qwen flow
                        messages.append({"role": "user", "content": user_input})
                        llm_response = self.llm_client.get_response(messages)
                        logging.info("\nAssistant: %s", llm_response)

                        result = await self.process_llm_response(llm_response, messages)

                        if result != llm_response:
                            messages.append({"role": "assistant", "content": llm_response})
                            messages.append({"role": "system", "content": result})
                            final_response = self.llm_client.get_response(messages)
                            logging.info("\nFinal response: %s", final_response)
                            messages.append({"role": "assistant", "content": final_response})
                        else:
                            messages.append({"role": "assistant", "content": llm_response})

                    except KeyboardInterrupt:
                        logging.info("\nExiting...")
                        break

        finally:
            await self.cleanup_servers()

"""async def main() -> None:
    #Initialize and run the chat session.
    config = Configuration()
    server_config = config.load_config("servers_config.json")
    servers = [Server(name, srv_config) for name, srv_config in server_config["mcpServers"].items()]
    llm_client = LLMClient(config.llm_api_key)
    chat_session = ChatSession(servers, llm_client)
    await chat_session.start()"""

async def main() -> None:
    config = Configuration()
    server_config = config.load_config("servers_config.json")
    servers = [Server(name, srv_config) for name, srv_config in server_config["mcpServers"].items()]

    llm_client = QwenClient(
        model_path="/scratch/general/vast/u1457424/mcp_chatbot/results/evalsets_mcp_5k_11103275",
        base_model="Qwen/Qwen3-4B-Instruct-2507",
        use_adapter=True,  # False if saved a merged model
    )
    chat_session = ChatSession(servers, llm_client)
    await chat_session.start()


if __name__ == "__main__":
    asyncio.run(main())
