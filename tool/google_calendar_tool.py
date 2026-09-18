
import os

from langchain_mcp_adapters.client import MultiServerMCPClient

ALLOWED_TOOLS = {
    "list-calendars",
    "list-events",
    "search-events",
    "get-event",
    "create-event",
    "update-event",
    "delete-event",
    "get-freebusy",
    "respond-to-event",
}
async def get_google_calendar_tools():
    client = MultiServerMCPClient({
            "google-calendar": {
            "command": "npx",
            "transport": "stdio",
            "args": ["@cocal/google-calendar-mcp"],
            "env": {
                # Path to the Google OAuth client-secret JSON, supplied through
                # the environment (see .env.example). When unset, the MCP
                # server reports its own authentication error.
                "GOOGLE_OAUTH_CREDENTIALS": os.environ.get("GOOGLE_OAUTH_CREDENTIALS", "")
            }
            }
        })
    Tools = await client.get_tools()

    GC_TOOLS = [
            tool for tool in Tools
            if tool.name in ALLOWED_TOOLS
        ]
    return [client, GC_TOOLS]

