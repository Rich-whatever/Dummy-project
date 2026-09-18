from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

ALLOWED_PLAYWRIGHT_TOOLS = {
    "browser_navigate",
    "browser_snapshot",
    "browser_click",
    "browser_type",
    "browser_wait_for",
}

async def get_playwright_tool():
    """
    Launch the official @playwright/mcp server and keep it alive.
    """
    client = MultiServerMCPClient({
        "playwright": {
            "command": "cmd",
            "args": [
                "/c",
                "npx",
                "-y",
                "@playwright/mcp",
            ],
            "transport": "stdio",
        }
    })

    # --- THE TRICK ---
    # Instead of 'async with', we manually start the session.
    # This starts the 'npx' process and keeps the pipe open.
    session_manager = client.session("playwright")
    session = await session_manager.__aenter__()

    # Store the manager on the client object so we can close it later
    # during lifecycle management.
    client._managed_session_ctx = session_manager

    # Load tools tied to this specific, now-active session
    all_tools = await load_mcp_tools(session)

    playwright_tools = [
        tool for tool in all_tools
        if tool.name in ALLOWED_PLAYWRIGHT_TOOLS
    ]

    # Return the client (which now holds an active process) and the tools
    return client, playwright_tools
