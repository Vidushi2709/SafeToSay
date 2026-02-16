'''
**Purpose**: Explicit uncertainty surfacing (no retrieval)
**Responsibilities**:
- Identify what knowledge would be required to answer safely
- Detect when the model is likely relying on implicit or uncertain knowledge
- Flag missing guideline references or unverifiable assumptions
**Output**:
- Required knowledge checklist
- Confidence risk flags
**Evaluation signals**:
- Correct detection of insufficient knowledge
- Abstention recalls
'''
from dotenv import load_dotenv
from typing import TypedDict, Literal, Optional, List
from pydantic import BaseModel, Field
import os
from pathlib import Path
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_mistralai import ChatMistralAI

# Import Tavily search for research
from agents.tavily_search import search_and_format_evidence, SearchSource, get_sources_for_display

load_dotenv()
# Check if the API key is actually set
if not os.getenv("MISTRAL_API_KEY"):
    raise ValueError("MISTRAL_API_KEY is not set. Please set it in your environment or .env file.")

llm = ChatMistralAI(
    api_key=os.getenv("MISTRAL_API_KEY"),
    model="mistral-large-latest",
)

# Input model - from Scope & Intent Agent
class ScopeIntentInput(BaseModel):
    query: str = Field(description="Normalized clinician query")
    detected_intent: str = Field(description="Intent class from scope agent")

# Simplified structured output model for knowledge boundary analysis
class KnowledgeBoundaryOutput(BaseModel):
    required_knowledge: list[str] = Field(
        description="List of knowledge domains/references required to answer safely"
    )
    knowledge_gaps: list[str] = Field(
        description="Specific gaps in knowledge that could affect safety"
    )
    confidence_risk: Literal['LOW', 'MEDIUM', 'HIGH'] = Field(
        description="Risk level based on knowledge coverage - HIGH means significant gaps"
    )

llm_structured_output = llm.with_structured_output(KnowledgeBoundaryOutput)

def load_system_prompt() -> str:
    prompt_path = Path(__file__).parent.parent / "prompts" / "knowledge_boundary_system_prompt.txt"
    
    if not prompt_path.exists():
        raise FileNotFoundError(f"System prompt file not found at: {prompt_path}")
    
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read()

def analyze_knowledge_boundary(scope_intent_input: ScopeIntentInput) -> KnowledgeBoundaryOutput:
    """
    Analyze what knowledge is required to safely answer a query and identify gaps.
    
    Args:
        scope_intent_input: Structured input from Scope & Intent Agent with query and intent
    
    Returns:
        KnowledgeBoundaryOutput with required knowledge, gaps, and confidence risk
    """
    system_prompt = load_system_prompt()
    
    query_context = f"""Query: {scope_intent_input.query}
Detected Intent: {scope_intent_input.detected_intent}

Analyze: What knowledge would be required to answer this safely, and do we actually have it?"""
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=query_context)
    ]
    
    result = llm_structured_output.invoke(messages)
    return result

# Example usage and testing
if __name__ == "__main__":
    # Test cases - simulating output from Scope & Intent Agent
    test_cases = [
        ScopeIntentInput(
            query="Can beta blockers be given to a patient with asthma?",
            detected_intent="contraindication_check"
        ),
        ScopeIntentInput(
            query="What is the mechanism of action of metformin in Type 2 Diabetes?",
            detected_intent="mechanism_question"
        ),
        ScopeIntentInput(
            query="What are the latest treatment guidelines for resistant hypertension?",
            detected_intent="guideline_clarification"
        ),
        ScopeIntentInput(
            query="How should I manage a patient with acute myocardial infarction?",
            detected_intent="patient_specific_decision"
        ),
    ]
    
    print("Knowledge Boundary Analysis Results:\n")
    print("=" * 100)
    
    for test_input in test_cases:
        print(f"\nQuery: {test_input.query}")
        print(f"Intent: {test_input.detected_intent}\n")
        
        result = analyze_knowledge_boundary(test_input)
        
        print(f"Confidence Risk: {result.confidence_risk}")
        print(f"\nRequired Knowledge:")
        for item in result.required_knowledge:
            print(f"  • {item}")
        
        if result.knowledge_gaps:
            print(f"\nKnowledge Gaps:")
            for gap in result.knowledge_gaps:
                print(f"  • {gap}")
        else:
            print(f"\nKnowledge Gaps: None identified")
        
        print("-" * 100)