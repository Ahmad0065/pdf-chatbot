import os
import re
from tavily import TavilyClient

tavily_client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))

def clean_snippet(text: str) -> str:
    # Remove comment-thread junk, timestamps, "Anonymous said" patterns
    text = re.sub(r'(Anonymous said\.\.\.|#### \d+ comments?|Reply Delete|Posted by.*|\w+ \d{1,2}, \d{4} at \d{1,2}:\d{2}\s?[AP]M)', '', text)
    text = re.sub(r'\n{2,}', ' ', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()

def search_web(query: str, max_results: int = 3) -> str:
    try:
        results = tavily_client.search(
            query=query,
            max_results=max_results,
            search_depth="advanced",
            exclude_domains=["scribd.com", "pinterest.com", "quora.com"],   # better quality results, filters low-quality pages
        )
        snippets = []
        for r in results.get("results", []):
            cleaned = clean_snippet(r["content"])
            if len(cleaned) > 50:  # skip near-empty/junk results
                snippets.append(f"- {cleaned[:250]} (Source: {r['url']})")

        return "\n\n".join(snippets) if snippets else ""
    except Exception:
        return ""