from config import SANDBOX_DIR

# Names (keys) of the tool categories exposed by the execution system.
TOOL_CATEGORY_NAMES = [
    "github",
    "filesystem_and_shell",
    "web_search",
    "browser_automation",
    "google_calendar",
]


async def build_toolkit():
    """
    Build the complete tool registry.

    Returns
    -------
    list
        ``[client_list, tool_category_dict]`` where:
        - client_list: list of MCP clients (for lifecycle management)
        - tool_category_dict: dict[str, list[tool]], e.g.
          ``{"github": [...], "filesystem": [...], "web_search": [...],
          "browser_automation": [...]}``
    """
    # Lazy imports to avoid forcing installation of optional dependencies
    from tool.browser_automation_tool import get_playwright_tool
    from tool.crawl4ai import fetch_relevant_webpage, fetch_webpage
    from tool.filesystem_tool import get_filesystem_tool
    from tool.github_tool import get_github_tool
    from tool.google_calendar_tool import get_google_calendar_tools
    from tool.searxng import web_search

    # Call MCP-based tool providers
    github_client, github_tools = await get_github_tool()
    fs_client, fs_tools = await get_filesystem_tool()
    playwright_client, playwright_tools = await get_playwright_tool()
    gc_client, gc_tools = await get_google_calendar_tools()

    # Collect all MCP clients so they can be kept alive
    client_list = [github_client, fs_client, playwright_client, gc_client]

    # Build category dict: name -> list of tool instances
    tool_category_dict = {
        "github": github_tools,
        "filesystem_and_shell": fs_tools,
        "web_search": [fetch_relevant_webpage, fetch_webpage, web_search],
        "browser_automation": playwright_tools,
        "google_calendar": gc_tools,
    }

    return [client_list, tool_category_dict]

TOOL_MAP_TEXT = f"""
Avaliable tool categories:
they follow "category_name - description" format

1. github
— manage Github repos, code, issues, PRs, workflows, and projects via natural language

2. filesystem_and_shell
— manage filesystem and able to use cmd for code / command execution
- Note: U're operating on window system. Filesystem tools access are limited to {SANDBOX_DIR}

3. web_search
— Search the internet and extract webpage content.
- Prefer for general information retrieval; lower cost and more reliable than browser automation.
- Use when information is available from search results or static webpages.
- If nothing returned, the tool is likely broken

4. browser_automation
— Control a browser (navigate, click, type, inspect pages, scroll, close).
- Higher cost; use only when interaction is required or content cannot be obtained through search/fetch.
- Use for JavaScript-rendered pages, dynamic content, forms, login flows, or user-requested browser actions.

5. google_calendar
— Manage Google Calendar events, schedules, calendars, and availability.
"""
