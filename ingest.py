import asyncio
import shutil
import os
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from contextlib import AsyncExitStack

async def main():
    with open("servers_config.json") as f:
        config = json.load(f)

    srv_cfg = config["mcpServers"]["documentation"]
    exit_stack = AsyncExitStack()
    command = shutil.which("npx") if srv_cfg["command"] == "npx" else srv_cfg["command"]
    params = StdioServerParameters(
        command=command,
        args=srv_cfg["args"],
        env={**os.environ, **srv_cfg.get("env", {})},
    )
    transport = await exit_stack.enter_async_context(stdio_client(params))
    read, write = transport
    session = await exit_stack.enter_async_context(ClientSession(read, write))
    await session.initialize()

    print("[1] Processing uploads...")
    result = await session.call_tool("process_uploads", {})
    print("process_uploads result:", result)

    print("[2] Listing documents...")
    result = await session.call_tool("list_documents", {})
    print("list_documents result:", result)

    await exit_stack.aclose()

asyncio.run(main())