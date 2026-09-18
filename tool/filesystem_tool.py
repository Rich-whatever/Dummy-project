
from langchain_mcp_adapters.client import MultiServerMCPClient

from config import SANDBOX_DIR
from tool.safe_shell import SafeShellTool


async def get_filesystem_tool():
    client = MultiServerMCPClient({
        "filesystem": {
        "command": "cmd",
        "args": [
            "/c",
            "npx",
            "-y",
            "@modelcontextprotocol/server-filesystem",
            SANDBOX_DIR,
        ],
        "transport": "stdio",
    }})

    FILESYSTEM_TOOLS = await client.get_tools() + [SafeShellTool()]
    return [client, FILESYSTEM_TOOLS]
