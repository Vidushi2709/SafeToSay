'''
Scope & Intent Agent
**Purpose**: Early safety filtering
**Responsibilities**:
- Normalize clinician questions
- Detect scope violations (diagnosis, treatment, patient-specific requests)
**Output**:
- `IN_SCOPE` or `OUT_OF_SCOPE`
**Evaluation signals**:
- Scope classification accuracy
- False-negative safety escapes
'''

from langgraph.graph import StateGraph, START, END
from langchain.chat_models import init_chat_model
from os import getenv
from dotenv import load_dotenv
from typing import TypedDict, Annotated, operator, Literal
from pydantic import BaseModel, Field
import os
from pathlib import Path
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

load_dotenv()

# state schema for scope intent agent
class ScopeIntent(TypedDict):
    scope: Literal['IN_SCOPE', 'OUT_OF_SCOPE']
    intent: str
    

# Check if the API key is actually set
if not os.getenv("OPEN_ROUTER_API_KEY"):
    raise ValueError("OPEN_ROUTER_API_KEY is not set. Please set it in your environment or .env file.")

llm = ChatOpenAI(
    api_key=os.getenv("OPEN_ROUTER_API_KEY"),  # Make sure this is set in your environment or .env file
    base_url="https://openrouter.ai/api/v1",
    model="openai/gpt-4o-mini",
    default_headers={
    }
)

# Enhanced structured output model with reasoning
class ScopeIntentOutput(BaseModel):
    """Structured output for scope and intent classification"""
    scope_decision: Literal['IN_SCOPE', 'OUT_OF_SCOPE'] = Field(
        description="Binary decision on whether the query is within agent scope"
    )
    detected_intent: str = Field(
        description="Detected intent category (e.g., guideline_clarification, contraindication_check, diagnosis_request, treatment_recommendation, patient_specific_decision)"
    )
    risk_notes: list[str] = Field(
        description="List of risk flags if detected intent violates scope"
    )

llm_structured_output = llm.with_structured_output(ScopeIntentOutput)

def load_system_prompt() -> str:
    prompt_path = Path(__file__).parent.parent / "prompts" / "scope_intent_system_prompt.txt"
    
    if not prompt_path.exists():
        raise FileNotFoundError(f"System prompt file not found at: {prompt_path}")
    
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read()

def classify_query(user_query: str) -> ScopeIntentOutput:
    system_prompt = load_system_prompt()
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Classify this query:\n\n{user_query}")
    ]
    
    result = llm_structured_output.invoke(messages)
    return result

# Example usage and testing
if __name__ == "__main__":
    # Test cases
    test_queries = [
        "Can beta blockers be given to a patient with asthma?",  # IN_SCOPE - contraindication_check
        "My patient has symptoms of chest pain and shortness of breath, what should I do?",  # OUT_OF_SCOPE - patient_specific_decision
        "What is the mechanism of action of metformin?",  # IN_SCOPE - mechanism_question
        "I have a 45-year-old patient with hypertension, what medication should I prescribe?",  # OUT_OF_SCOPE - treatment_recommendation
        "What are the diagnostic criteria for Type 2 Diabetes?",  # IN_SCOPE - general_knowledge
    ]
    
    print("Scope & Intent Classification Results:\n")
    print("=" * 80)
    
    for query in test_queries:
        print(f"\nQuery: {query}\n")
        result = classify_query(query)
        print(f"Scope Decision: {result.scope_decision}")
        print(f"Detected Intent: {result.detected_intent}")
        print(f"Risk Notes: {result.risk_notes if result.risk_notes else 'None'}")
        print("-" * 80)