from urllib.parse import urlparse

import requests
from langchain_core.tools import tool

SEARXNG_URL = "http://localhost:8080/search"


@tool
def web_search(query: str, num_results: int = 3) -> list[dict]:
    """
    Search the internet using SearXNG.
    num_result (default as 3) can be modified to change the number of results returned
    Returns a list of useful web results containing:
    - title
    - url
    - source domain
    - snippet/content
    - published date if available
    """
    response = requests.get(
        SEARXNG_URL,
        params={
            "q": query,
            "format": "json"
        },
        timeout=10
    )

    response.raise_for_status()

    data = response.json()

    results = []

    for item in data.get("results", [])[:num_results]:
        url = item.get("url", "")

        results.append(
            {
                "title": item.get("title"),
                "url": url,
                "source": urlparse(url).netloc,
                "snippet": item.get("content"),
                "published_date": item.get("publishedDate"),
            }
        )

    return results
