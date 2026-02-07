'''
4.3 Answer Generation Agent (MedGemma)
**Purpose**: Constrained synthesis
**Responsibilities**:
- Generate answers *only* from retrieved evidence
- Use structured, conditional language
- Avoid free-form medical advice
**Output**:
- Draft answer
**Evaluation signals**:
- Faithfulness to evidence
- Overgeneralization risk
- Missing information flags
'''

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from pydantic import BaseModel, Field
from typing import Literal, Optional
import re
from pathlib import Path

model_id = "google/medgemma-4b-it"  

# Check CUDA availability
if torch.cuda.is_available():
    print(f"[INFO] ✓ CUDA Available")
    print(f"[INFO] GPU Device: {torch.cuda.get_device_name(0)}")
    print(f"[INFO] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    cuda_available = True
else:
    print(f"[WARNING] CUDA not available - falling back to CPU")
    cuda_available = False

tokenizer = AutoTokenizer.from_pretrained(model_id)

# 4-bit quantization config for efficient GPU loading
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

print(f"[INFO] Loading MedGemma with 4-bit quantization...")

if cuda_available:
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="cuda",  # Explicit CUDA device
        dtype=torch.float16  # Use dtype (not deprecated torch_dtype)
    ).eval()
else:
    # Fallback to CPU (no quantization on CPU)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="cpu",
        dtype=torch.float32
    ).eval()

device = next(model.parameters()).device
print(f"[INFO] ✓ Model loaded on device: {device}")

# Set pad token to prevent generation issues
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def load_system_prompt() -> str:
    """Load the answer generation system prompt from file."""
    prompt_path = Path(__file__).parent.parent / "prompts" / "answer_generation_system_prompt.txt"
    with open(prompt_path, 'r') as f:
        return f.read()

# Input model - from Knowledge Boundary Agent
class KnowledgeConstraints(BaseModel):
    confidence_risk: Literal['LOW', 'MEDIUM', 'HIGH'] = Field(description="Risk level based on knowledge gaps")
    required_knowledge: list[str] = Field(description="Knowledge domains needed to answer safely")

class AnswerGenerationInput(BaseModel):
    query: str = Field(description="The clinical question")
    allowed_intent: str = Field(description="Intent verified as IN_SCOPE by Scope & Intent Agent")
    evidence: list[str] = Field(description="Retrieved evidence snippets to constrain answer")
    knowledge_constraints: KnowledgeConstraints = Field(description="Constraints from Knowledge Boundary Agent")

# Simplified structured output model
class AnswerGenerationOutput(BaseModel):
    draft_answer: str = Field(description="Generated answer, explicitly marked as non-final draft")

def _build_user_prompt(question: str, evidence: list[str], intent: str) -> str:
    """Build the detailed user prompt with structured evidence context and constraints."""
    
    evidence_context = "\n".join([f"{i+1}. {e}" for i, e in enumerate(evidence)])
    
    prompt = f"""CLINICAL QUESTION:
{question}

INTENT: {intent}

AVAILABLE EVIDENCE:
{evidence_context}

CRITICAL CONSTRAINTS:
- Do NOT provide diagnosis or diagnosis suggestions
- Do NOT provide treatment recommendations for specific patients
- Use ONLY conditional language (may, might, appears to, according to, could suggest)
- Do NOT make assumptions beyond the provided evidence context
- MUST acknowledge uncertainty and knowledge limitations
- This is a DRAFT answer for evaluation, never final clinical guidance

TASK:
Generate a constrained draft answer based ONLY on the evidence provided.
Acknowledge what is not covered and where context would affect the answer.

DRAFT ANSWER:"""
    
    return prompt

def generate_constrained_answer(
    answer_input: AnswerGenerationInput,
    max_new_tokens: int = 512
) -> AnswerGenerationOutput:
    """
    Generate a draft answer constrained by evidence and knowledge boundaries.
    
    Args:
        answer_input: Structured input including query, intent, evidence, and knowledge constraints
        max_new_tokens: Maximum tokens to generate
    
    Returns:
        AnswerGenerationOutput with draft answer (non-final)
    """
    # Load system prompt at runtime
    system_instructions = load_system_prompt()
    user_prompt = _build_user_prompt(answer_input.query, answer_input.evidence, answer_input.allowed_intent)
    full_prompt = f"{system_instructions}\n\n{user_prompt}"
    
    # Tokenize with attention mask
    inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=2048)
    
    # Move inputs to model device
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Clear CUDA cache before generation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    try:
        with torch.no_grad():
            # Use greedy decoding for stability with quantized models
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Greedy decoding for stability
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                no_repeat_ngram_size=2,
                use_cache=True,
            )
    except RuntimeError as e:
        if "cuda" in str(e).lower() or "assert" in str(e).lower():
            print(f"[WARNING] Generation error: {e}")
            print("[INFO] Retrying with safer parameters...")
            
            # Fallback: simple greedy generation without fancy parameters
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
        else:
            raise
    
    # Clear CUDA cache after generation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    draft_answer = generated_text.split("DRAFT ANSWER:")[-1].strip()
    
    # Fallback: if generation produced minimal content, use evidence-based summary
    if not draft_answer or len(draft_answer) < 50:
        evidence_summary = " ".join(answer_input.evidence[:2]) if answer_input.evidence else ""
        draft_answer = format_clinical_response(
            evidence_summary, 
            answer_input.knowledge_constraints.confidence_risk
        )
    
    # Clean up any duplicate DRAFT markers
    draft_answer = clean_draft_markers(draft_answer)
    
    # Ensure draft answer is marked as non-final (only once)
    if not any(marker in draft_answer.upper() for marker in ['[DRAFT]', '[DRAFT -']):
        draft_answer = f"[DRAFT - Not for Direct Clinical Use]\n\n{draft_answer}"
    
    return AnswerGenerationOutput(
        draft_answer=draft_answer
    )

def clean_draft_markers(text: str) -> str:
    """Remove duplicate DRAFT markers and clean formatting"""
    # Remove duplicate DRAFT markers
    import re
    text = re.sub(r'\[DRAFT[^\]]*\]\s*\[DRAFT[^\]]*\]', '[DRAFT - Not for Direct Clinical Use]', text)
    text = re.sub(r'\[DRAFT[^\]]*\]\s+\[DRAFT[^\]]*\]', '[DRAFT - Not for Direct Clinical Use]', text)
    
    # Clean up extra spaces and newlines
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def format_clinical_response(evidence_summary: str, confidence_risk: str) -> str:
    """Format a clean, structured clinical response"""
    
    formatted_response = f"""**Clinical Information:**
{evidence_summary}

**Important Considerations:**
• This is a {confidence_risk} confidence risk situation
• Individual patient factors must be evaluated
• Clinical assessment and medical history are important
• Healthcare provider consultation is recommended
• Clinical judgment is required for patient-specific decisions

**Disclaimer:** This information is for educational purposes only and should not replace professional medical advice."""
    
    return formatted_response



def test_answer_generation_agent():
    """Test the answer generation agent with sample filtered medical queries."""
    
    print("=" * 100)
    print("TESTING ANSWER GENERATION AGENT (MedGemma) - Draft Answer Generation")
    print("=" * 100)
    
    # Test Case 1: Contraindication check (IN_SCOPE, LOW risk)
    test_case_1 = AnswerGenerationInput(
        query="Can beta blockers be given in asthma?",
        allowed_intent="contraindication_check",
        evidence=[
            "Beta-blockers, particularly non-selective ones, can cause bronchoconstriction in asthma patients.",
            "Cardioselective beta-blockers (e.g., metoprolol, atenolol) are considered safer alternatives in asthma.",
            "Clinical guidelines generally recommend caution when using beta-blockers in patients with asthma or COPD."
        ],
        knowledge_constraints=KnowledgeConstraints(
            confidence_risk="LOW",
            required_knowledge=["general contraindication principles", "beta blocker classes"]
        )
    )
    
    print("\n--- TEST CASE 1: Contraindication Check (LOW Risk) ---")
    print(f"Query: {test_case_1.query}")
    print(f"Intent: {test_case_1.allowed_intent}")
    print(f"Confidence Risk: {test_case_1.knowledge_constraints.confidence_risk}")
    print(f"Evidence ({len(test_case_1.evidence)} sources):")
    for i, e in enumerate(test_case_1.evidence, 1):
        print(f"  {i}. {e}")
    
    try:
        result_1 = generate_constrained_answer(test_case_1, max_new_tokens=512)
        print(f"\n--- DRAFT OUTPUT ---")
        print(result_1.draft_answer)
    except Exception as e:
        print(f"Error in test case 1: {e}")
    
    # Test Case 2: Mechanism question (IN_SCOPE, LOW risk)
    test_case_2 = AnswerGenerationInput(
        query="What is the mechanism of action of metformin?",
        allowed_intent="mechanism_question",
        evidence=[
            "Metformin is believed to primarily work through adenosine monophosphate-activated protein kinase (AMPK) activation.",
            "It reduces hepatic glucose production and improves insulin sensitivity in peripheral tissues.",
            "Metformin also has effects on mitochondrial function and glucagon secretion."
        ],
        knowledge_constraints=KnowledgeConstraints(
            confidence_risk="LOW",
            required_knowledge=["biochemical pathways", "AMPK activation mechanism"]
        )
    )
    
    print("\n" + "=" * 100)
    print("--- TEST CASE 2: Mechanism Question (LOW Risk) ---")
    print(f"Query: {test_case_2.query}")
    print(f"Intent: {test_case_2.allowed_intent}")
    print(f"Confidence Risk: {test_case_2.knowledge_constraints.confidence_risk}")
    print(f"Evidence ({len(test_case_2.evidence)} sources):")
    for i, e in enumerate(test_case_2.evidence, 1):
        print(f"  {i}. {e}")
    
    try:
        result_2 = generate_constrained_answer(test_case_2, max_new_tokens=512)
        print(f"\n--- DRAFT OUTPUT ---")
        print(result_2.draft_answer)
    except Exception as e:
        print(f"Error in test case 2: {e}")
    
    # Test Case 3: Guideline clarification with limited evidence (MEDIUM risk)
    test_case_3 = AnswerGenerationInput(
        query="What are guidelines for ACE inhibitor use in hypertension?",
        allowed_intent="guideline_clarification",
        evidence=[
            "ACE inhibitors are commonly recommended as first-line antihypertensive agents in many guidelines."
        ],
        knowledge_constraints=KnowledgeConstraints(
            confidence_risk="MEDIUM",
            required_knowledge=["current hypertension guidelines", "ACE inhibitor indications", "patient population specifics"]
        )
    )
    
    print("\n" + "=" * 100)
    print("--- TEST CASE 3: Guideline Clarification (MEDIUM Risk) ---")
    print(f"Query: {test_case_3.query}")
    print(f"Intent: {test_case_3.allowed_intent}")
    print(f"Confidence Risk: {test_case_3.knowledge_constraints.confidence_risk}")
    print(f"Evidence ({len(test_case_3.evidence)} source):")
    for i, e in enumerate(test_case_3.evidence, 1):
        print(f"  {i}. {e}")
    
    try:
        result_3 = generate_constrained_answer(test_case_3, max_new_tokens=512)
        print(f"\n--- DRAFT OUTPUT ---")
        print(result_3.draft_answer)
    except Exception as e:
        print(f"Error in test case 3: {e}")
    
    print("\n" + "=" * 100)
    print("TEST COMPLETE - All outputs are DRAFT ANSWERS for evaluation, not final clinical guidance")
    print("=" * 100)

if __name__ == "__main__":
    test_answer_generation_agent()