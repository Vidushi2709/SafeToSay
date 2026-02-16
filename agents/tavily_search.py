"""
Tavily Search Integration for Medical Research
Provides research capabilities for all agents to gather evidence before answering.
"""

import os
from typing import Optional, List
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

# Check for Tavily API key
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

class SearchSource(BaseModel):
    """Represents a source from search results"""
    title: str = Field(description="Title of the source")
    url: str = Field(description="URL of the source")
    content: str = Field(description="Relevant content snippet from the source")
    score: float = Field(default=0.0, description="Relevance score")

class TavilySearchResult(BaseModel):
    """Structured result from Tavily search"""
    query: str = Field(description="The original search query")
    sources: List[SearchSource] = Field(default_factory=list, description="List of sources found")
    answer: Optional[str] = Field(default=None, description="AI-generated answer from Tavily (if available)")
    error: Optional[str] = Field(default=None, description="Error message if search failed")

def search_medical_literature(
    query: str,
    max_results: int = 5,
    search_depth: str = "advanced",
    include_domains: Optional[List[str]] = None,
    exclude_domains: Optional[List[str]] = None
) -> TavilySearchResult:
    """
    Search for medical literature and evidence using Tavily API.
    
    Args:
        query: The search query (should be medical/clinical focused)
        max_results: Maximum number of results to return
        search_depth: "basic" or "advanced" - advanced provides more comprehensive results
        include_domains: List of domains to prioritize (e.g., ["pubmed.ncbi.nlm.nih.gov", "who.int"])
        exclude_domains: List of domains to exclude
    
    Returns:
        TavilySearchResult with sources and optional AI-generated answer
    """
    
    if not TAVILY_API_KEY:
        return TavilySearchResult(
            query=query,
            sources=[],
            error="TAVILY_API_KEY not set. Please add it to your .env file."
        )
    
    try:
        from tavily import TavilyClient
        
        client = TavilyClient(api_key=TAVILY_API_KEY)
        
        # Default medical domains if not specified
        if include_domains is None:
            include_domains = [
                "pubmed.ncbi.nlm.nih.gov",
                "ncbi.nlm.nih.gov",
                "who.int",
                "cdc.gov",
                "mayoclinic.org",
                "uptodate.com",
                "medscape.com",
                "nih.gov",
                "cochrane.org",
                "bmj.com",
                "thelancet.com",
                "nejm.org"
            ]
        
        # Prepend "medical" or "clinical" context to query for better results
        enhanced_query = f"medical clinical evidence: {query}"
        
        response = client.search(
            query=enhanced_query,
            search_depth=search_depth,
            max_results=max_results,
            include_domains=include_domains if include_domains else None,
            exclude_domains=exclude_domains if exclude_domains else None,
            include_answer=True  # Get AI-generated answer
        )
        
        # Parse results into structured format
        sources = []
        for result in response.get("results", []):
            sources.append(SearchSource(
                title=result.get("title", "Unknown Title"),
                url=result.get("url", ""),
                content=result.get("content", ""),
                score=result.get("score", 0.0)
            ))
        
        return TavilySearchResult(
            query=query,
            sources=sources,
            answer=response.get("answer"),
            error=None
        )
        
    except ImportError:
        return TavilySearchResult(
            query=query,
            sources=[],
            error="tavily-python package not installed. Run: pip install tavily-python"
        )
    except Exception as e:
        return TavilySearchResult(
            query=query,
            sources=[],
            error=f"Tavily search failed: {str(e)}"
        )


def search_and_format_evidence(
    query: str,
    max_results: int = 5
) -> tuple[List[str], List[SearchSource]]:
    """
    Search for evidence and return both formatted evidence strings and source objects.
    
    Args:
        query: The clinical query to research
        max_results: Maximum number of results
    
    Returns:
        Tuple of (evidence_strings, source_objects)
        - evidence_strings: List of evidence snippets for answer generation
        - source_objects: List of SearchSource objects with full metadata
    """
    
    result = search_medical_literature(query, max_results=max_results)
    
    if result.error:
        print(f"[WARNING] Tavily search error: {result.error}")
        return [], []
    
    evidence_strings = []
    for source in result.sources:
        # Clean evidence content before including
        cleaned_content = _clean_evidence_content(source.content)
        if cleaned_content and len(cleaned_content) > 30:
            evidence_strings.append(cleaned_content)
    
    # If Tavily provided an AI answer, include it as the first evidence piece (cleaned)
    if result.answer:
        cleaned_answer = _clean_evidence_content(result.answer)
        if cleaned_answer:
            evidence_strings.insert(0, cleaned_answer)
    
    return evidence_strings, result.sources


def _clean_evidence_content(text: str) -> str:
    """Clean raw evidence text to remove artifacts, truncation markers, and noise."""
    import re
    if not text:
        return ""
    
    cleaned = text.strip()
    
    # Remove [Source: ...] tags
    cleaned = re.sub(r'\[Source:[^\]]*\]', '', cleaned)
    
    # Remove [Summary] prefix
    cleaned = re.sub(r'^\[Summary\]\s*', '', cleaned)
    
    # Remove [...] truncation markers and surrounding whitespace
    cleaned = re.sub(r'\s*\[\.{2,}\]\s*', ' ', cleaned)
    cleaned = re.sub(r'\s*\[…\]\s*', ' ', cleaned)
    
    # Remove trailing incomplete sentences (ending with [...] or cut off)
    cleaned = re.sub(r'\s*\[\.\.\.[^\]]*\]\s*$', '.', cleaned)
    
    # Remove reference numbers like [1], [2,3], [1-5]
    # but keep brackets that are part of medical content like [mg/dL]
    cleaned = re.sub(r'\[\d+(?:[,;\-–]\d+)*\]', '', cleaned)
    
    # Truncate at 500 chars per evidence piece to avoid overwhelming the model
    if len(cleaned) > 500:
        # Find sentence boundary near 500 chars
        boundary = cleaned[:500].rfind('. ')
        if boundary > 200:
            cleaned = cleaned[:boundary + 1]
        else:
            cleaned = cleaned[:500] + '...'
    
    # Collapse multiple spaces
    cleaned = re.sub(r'\s+', ' ', cleaned)
    
    return cleaned.strip()


def get_sources_for_display(sources: List[SearchSource]) -> List[dict]:
    """
    Convert SearchSource objects to display-friendly dictionaries for frontend.
    
    Args:
        sources: List of SearchSource objects
    
    Returns:
        List of dictionaries with source information for UI display
    """
    return [
        {
            "title": source.title,
            "url": source.url,
            "snippet": source.content[:200] + "..." if len(source.content) > 200 else source.content,
            "relevance": round(source.score * 100, 1) if source.score else None
        }
        for source in sources
    ]


# Test function
if __name__ == "__main__":
    print("Testing Tavily Search Integration\n")
    print("=" * 80)
    
    test_queries = [
        "Can beta blockers be given to patients with asthma?",
        "What is the mechanism of action of metformin?",
        "Contraindications for ACE inhibitors"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}\n")
        evidence, sources = search_and_format_evidence(query, max_results=3)
        
        if evidence:
            print(f"Found {len(sources)} sources:")
            for i, src in enumerate(sources, 1):
                print(f"  {i}. {src.title}")
                print(f"     URL: {src.url}")
                print(f"     Preview: {src.content[:100]}...")
        else:
            print("No evidence found (check TAVILY_API_KEY)")
        
        print("-" * 80)
