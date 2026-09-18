import os

from langchain_mcp_adapters.client import MultiServerMCPClient


async def get_github_tool():
    client = MultiServerMCPClient({
            "github": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-github"],
            "transport": "stdio",
            "env": {
                # This will pull from your computer's system variables
                "GITHUB_PERSONAL_ACCESS_TOKEN": os.environ.get("GIT_API_KEY")
            }
        }})
    GIT_TOOLS = await client.get_tools()
    return [client, GIT_TOOLS]
