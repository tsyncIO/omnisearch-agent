import re
from typing import List, Dict, Optional
import warnings

# Suppress DDGS rename warning if any
warnings.filterwarnings("ignore", category=RuntimeWarning)

def parse_web_search_tag(text: str) -> Optional[str]:
    """
    Parses <web_search>search query here</web_search> from model output.
    """
    pattern = r'<web_search>(.*?)</web_search>'
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        query = match.group(1).strip()
        if query:
            return query
    return None

def search_web_ddg(query: str, max_results: int = 4) -> List[Dict]:
    """
    Performs live web search using DuckDuckGo (free, no API key needed).
    Returns list of dicts with 'title', 'href', 'body'.
    """
    try:
        from duckduckgo_search import DDGS
        ddgs = DDGS()
        results = list(ddgs.text(query, max_results=max_results))
        formatted = []
        for r in results:
            formatted.append({
                "title": r.get("title", ""),
                "href": r.get("href", ""),
                "body": r.get("body", "")
            })
        return formatted
    except Exception as e:
        print(f"[Web Search Warning] Error searching '{query}': {e}")
        return []

def format_search_results_markdown(results: List[Dict], query: str) -> str:
    """
    Formats web search results into clean markdown cards with clickable sources.
    """
    if not results:
        return f"*No live web search results found for: `{query}`.*"
    
    md = f"### 🌐 Live Web Search Results (`{query}`)\n\n"
    for i, res in enumerate(results, 1):
        title = res.get("title", "Untitled Source")
        link = res.get("href", "#")
        snippet = res.get("body", "")
        md += f"**{i}. [{title}]({link})**  \n"
        md += f"> {snippet}\n\n"
    return md
