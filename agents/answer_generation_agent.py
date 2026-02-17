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
from typing import Literal, Optional, List
import re
from pathlib import Path

# Import Tavily search for research
from agents.tavily_search import search_and_format_evidence, SearchSource, get_sources_for_display

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
        torch_dtype=torch.float16  # Fixed: use torch_dtype instead of deprecated dtype
    ).eval()
else:
    # Fallback to CPU (no quantization on CPU)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="cpu",
        torch_dtype=torch.float32  # Fixed: use torch_dtype instead of deprecated dtype
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

class SourceInfo(BaseModel):
    """Source information for citations"""
    title: str = Field(description="Title of the source")
    url: str = Field(description="URL of the source")
    snippet: str = Field(default="", description="Brief content snippet")

class AnswerGenerationInput(BaseModel):
    query: str = Field(description="The clinical question")
    allowed_intent: str = Field(description="Intent verified as IN_SCOPE by Scope & Intent Agent")
    evidence: list[str] = Field(description="Retrieved evidence snippets to constrain answer")
    knowledge_constraints: KnowledgeConstraints = Field(description="Constraints from Knowledge Boundary Agent")
    sources: Optional[List[SourceInfo]] = Field(default=None, description="Source metadata for citations")

# Structured output model with sources
class AnswerGenerationOutput(BaseModel):
    draft_answer: str = Field(description="Generated answer, explicitly marked as non-final draft")
    sources: List[dict] = Field(default_factory=list, description="Sources used in generating the answer")

def _clean_evidence_for_prompt(evidence: list[str]) -> list[str]:
    """Pre-clean evidence before feeding it into the model prompt."""
    cleaned = []
    for ev in evidence:
        e = re.sub(r'\[Source:[^\]]*\]', '', ev).strip()
        e = re.sub(r'^\[Summary\]\s*', '', e).strip()
        e = re.sub(r'\s*\[\.{2,}\]\s*', ' ', e).strip()
        e = re.sub(r'\s*\[…\]\s*', ' ', e).strip()
        e = re.sub(r'\[\d+(?:[,;\-–]\d+)*\]', '', e).strip()
        # Truncate long evidence pieces
        if len(e) > 400:
            boundary = e[:400].rfind('. ')
            if boundary > 150:
                e = e[:boundary + 1]
            else:
                e = e[:400].rstrip() + '...'
        e = re.sub(r'\s+', ' ', e).strip()
        if e and len(e) > 20:
            cleaned.append(e)
    return cleaned


def _build_user_prompt(question: str, evidence: list[str], intent: str) -> str:
    """Build the detailed user prompt with structured evidence context and constraints."""
    
    # Pre-clean evidence before including in prompt
    cleaned_evidence = _clean_evidence_for_prompt(evidence)
    evidence_context = "\n".join([f"{i+1}. {e}" for i, e in enumerate(cleaned_evidence)])
    
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
    
    # Extract only the generated answer, stripping the prompt echo
    draft_answer = _extract_answer_from_output(generated_text, full_prompt, answer_input)
    
    # Fallback: if generation produced minimal content, use evidence-based summary
    if not draft_answer or len(draft_answer) < 50:
        print(f"[INFO] Model generation was minimal ({len(draft_answer)} chars), creating evidence-based response")
        
        if answer_input.evidence:
            draft_answer = _synthesize_from_evidence(
                answer_input.query,
                answer_input.evidence,
                answer_input.knowledge_constraints.confidence_risk
            )
        else:
            draft_answer = format_clinical_response(
                "",
                answer_input.knowledge_constraints.confidence_risk
            )
    
    # Clean up any duplicate DRAFT markers
    draft_answer = clean_draft_markers(draft_answer)
    
    # No longer prepend DRAFT markers - frontend handles display cleanly
    
    # Extract sources from input if available
    sources = []
    if answer_input.sources:
        sources = [
            {
                "title": src.title,
                "url": src.url,
                "snippet": src.snippet if hasattr(src, 'snippet') else ""
            }
            for src in answer_input.sources
        ]

    return AnswerGenerationOutput(
        draft_answer=draft_answer,
        sources=sources
    )


def _extract_answer_from_output(generated_text: str, full_prompt: str, answer_input) -> str:
    """Extract only the actual answer from model output, stripping all prompt echoes."""
    import re
    
    # Strategy 1: Remove the exact prompt text if the model echoed it
    # The model output starts with the prompt, so strip it
    if full_prompt and generated_text.startswith(full_prompt[:200]):
        draft_answer = generated_text[len(full_prompt):].strip()
    else:
        # Strategy 2: Split on "DRAFT ANSWER:" and take the last part
        parts = generated_text.split("DRAFT ANSWER:")
        draft_answer = parts[-1].strip() if len(parts) > 1 else generated_text.strip()
    
    # Strategy 3: If the answer still contains system prompt markers, aggressively strip them
    # These are telltale signs the model echoed the prompt
    system_prompt_markers = [
        'CORE MISSION:', 'SOURCE CITATION REQUIREMENTS:', 'RESPONSE FORMATTING REQUIREMENTS:',
        'CORE SAFETY CONSTRAINTS:', 'RESPONSE PATTERNS BY QUERY TYPE:', 'QUALITY STANDARDS:',
        'PROHIBITED:', 'REQUIRED LANGUAGE PATTERNS:', 'You are MedGemma',
        'clinical information synthesis engine', 'CLINICAL QUESTION:', 'INTENT:', 
        'AVAILABLE EVIDENCE:', 'CRITICAL CONSTRAINTS:', 'TASK:',
        'NO SPECIFIC DIAGNOSIS', 'EVIDENCE-BASED:', 'QUALIFIED LANGUAGE:',
        'Factual/Educational Queries:', 'Medication/Treatment Queries:',
        'High-Risk Diagnostic Queries:', 'professional supervision recommended',
    ]
    
    # If multiple system prompt markers are found, try to find where real answer begins
    marker_count = sum(1 for m in system_prompt_markers if m.lower() in draft_answer.lower())
    if marker_count >= 3:
        # The model heavily echoed the prompt — try to find the actual answer
        # Look for the last occurrence of common prompt endings
        prompt_end_markers = [
            'DRAFT ANSWER:', 'Generate a constrained draft answer',
            'Acknowledge what is not covered', 'never final clinical guidance',
            'professional supervision recommended',
        ]
        last_pos = 0
        for marker in prompt_end_markers:
            pos = draft_answer.lower().rfind(marker.lower())
            if pos > last_pos:
                last_pos = pos + len(marker)
        
        if last_pos > 0:
            draft_answer = draft_answer[last_pos:].strip()
    
    # Final cleanup
    draft_answer = clean_draft_markers(draft_answer)
    return draft_answer


def clean_draft_markers(text: str) -> str:
    """Remove duplicate DRAFT markers, prompt artifacts, and clean formatting."""
    import re
    # Remove all DRAFT markers
    text = re.sub(r'\[DRAFT[^\]]*\]\s*', '', text)
    
    # Remove system prompt section headers that leaked through
    prompt_headers = [
        r'CORE MISSION:.*?(?=\n\n|$)',
        r'SOURCE CITATION REQUIREMENTS:.*?(?=\n\n|$)',
        r'RESPONSE FORMATTING REQUIREMENTS:.*?(?=\n\n|$)',
        r'CORE SAFETY CONSTRAINTS:.*?(?=\n\n|$)',
        r'RESPONSE PATTERNS BY QUERY TYPE:.*?(?=\n\n|$)',
        r'QUALITY STANDARDS:.*?(?=\n\n|$)',
        r'PROHIBITED:.*?(?=\n\n|$)',
        r'REQUIRED LANGUAGE PATTERNS:.*?(?=\n\n|$)',
        r'CLINICAL QUESTION:.*?(?=\n|$)',
        r'INTENT:.*?(?=\n|$)',
        r'AVAILABLE EVIDENCE:.*?(?=\n\n|$)',
        r'CRITICAL CONSTRAINTS:.*?(?=\n\n|$)',
        r'TASK:.*?(?=\n\n|$)',
        r'DRAFT ANSWER:\s*',
    ]
    for pattern in prompt_headers:
        text = re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove lines that are clearly prompt instructions (not answer content)
    instruction_patterns = [
        r'^.*You are MedGemma.*$',
        r'^.*clinical information synthesis engine.*$',
        r'^.*NO SPECIFIC DIAGNOSIS.*$',
        r'^.*EVIDENCE-BASED:.*$',
        r'^.*SAFETY DISCLAIMERS:.*$',
        r'^.*QUALIFIED LANGUAGE:.*$',
        r'^.*Do not diagnose individual patients.*$',
        r'^.*Base information on provided evidence.*$',
        r'^.*Include appropriate warnings.*$',
        r'^.*Use .may., .can., .typically..*rather than absolute.*$',
        r'^.*Generate comprehensive.*evidence-based.*$',
        r'^.*Support clinician decision-making.*$',
        r'^.*Cite sources when evidence.*$',
        r'^.*professional supervision recommended.*what.*$',
        r'^.*Example structure for medication.*$',
        r'^.*Vague, single-sentence responses.*$',
        r'^.*Absolute statements without clinical context.*$',
        r'^.*Patient-specific treatment decisions.*$',
        r'^.*Diagnostic conclusions for individual.*$',
        r'^.*requires clinical evaluation.*consult healthcare provider.*$',
        r'^.*individual factors may vary.*$',
        r'^.*\[Drug/Topic\].*Clinical Considerations.*$',
        r'^.*\[Specific risk \d\].*$',
        r'^.*\[Practical guidance \d\].*$',
        r'^.*\[Key safety disclaimer\].*$',
        r'^.*\[Key clinical concept\].*$',
        r'^.*\[Core concept\].*$',
        r'^.*\[Topic\].*Key Information.*$',
        r'^.*\[Drug/Treatment\].*and.*\[Condition\].*$',
    ]
    for pattern in instruction_patterns:
        text = re.sub(pattern, '', text, flags=re.MULTILINE | re.IGNORECASE)
    
    # Clean up excessive blank lines (3+ newlines -> 2)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Clean up multiple spaces on same line (but preserve newlines)
    text = re.sub(r'[^\S\n]+', ' ', text)
    
    # Remove leading/trailing whitespace on each line
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)
    
    # Remove excessive blank lines again after all cleanup
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()

def _synthesize_from_evidence(query: str, evidence: list[str], confidence_risk: str) -> str:
    """Synthesize a structured answer from evidence snippets instead of dumping them raw."""
    import re
    
    # Extract the topic from the query
    topic = query.strip().rstrip('?').strip()
    for prefix in ['what is', 'what are', 'define', 'explain', 'describe', 'tell me about']:
        if topic.lower().startswith(prefix):
            topic = topic[len(prefix):].strip()
            break
    topic = topic.strip().title()
    
    # Clean each evidence piece of artifacts
    cleaned_evidence = []
    for ev in evidence:
        clean = re.sub(r'\[Source:[^\]]*\]', '', ev).strip()
        clean = re.sub(r'^\[Summary\]\s*', '', clean).strip()
        clean = re.sub(r'\s*\[\.{2,}\]\s*', ' ', clean).strip()
        clean = re.sub(r'\[\d+(?:[,;\-–]\d+)*\]', '', clean).strip()
        if clean and len(clean) > 20:
            cleaned_evidence.append(clean)
    
    if not cleaned_evidence:
        return format_clinical_response("", confidence_risk)
    
    # Extract key facts by splitting on sentence boundaries
    all_sentences = []
    for ev in cleaned_evidence:
        sentences = re.split(r'(?<=[.!?])\s+', ev)
        for s in sentences:
            s = s.strip()
            if len(s) > 20 and s not in all_sentences:
                all_sentences.append(s)
    
    # Deduplicate similar sentences (keep first occurrence)
    unique_sentences = []
    seen_content = set()
    for s in all_sentences:
        # Create a simplified key for dedup
        key = re.sub(r'[^a-z0-9]', '', s.lower())[:60]
        if key not in seen_content:
            seen_content.add(key)
            unique_sentences.append(s)
    
    # Take the best sentences (up to 8) to form key points
    key_points = unique_sentences[:8]
    
    # Group into definition/overview and details
    overview = []
    details = []
    for s in key_points:
        if len(overview) < 3:
            overview.append(s)
        else:
            details.append(s)
    
    # Build structured response with blank lines between every element
    parts = []
    parts.append(f"**{topic} — Clinical Overview**")
    
    if overview:
        parts.append("**Overview:**")
        for s in overview:
            parts.append(s)
    
    if details:
        parts.append("**Key Clinical Points:**")
        for s in details:
            parts.append(s)
    
    parts.append("**Important Considerations:**")
    parts.append("Individual patient factors should be evaluated by a healthcare provider.")
    parts.append("Clinical context and medical history may affect applicability.")
    parts.append("This information is for educational purposes only.")
    
    parts.append("**Disclaimer:** This is general medical information and should not replace professional clinical advice. Always consult a qualified healthcare professional for patient-specific guidance.")
    
    # Join with double newlines so every element is its own paragraph
    return "\n\n".join(parts)


def format_clinical_response(evidence_summary: str, confidence_risk: str) -> str:
    """Format a clean, structured clinical response"""
    
    parts = []
    parts.append("**Clinical Information:**")
    if evidence_summary:
        parts.append(evidence_summary)
    
    parts.append("**Important Considerations:**")
    parts.append(f"This is a {confidence_risk} confidence risk situation.")
    parts.append("Individual patient factors must be evaluated.")
    parts.append("Clinical assessment and medical history are important.")
    parts.append("Healthcare provider consultation is recommended.")
    parts.append("Clinical judgment is required for patient-specific decisions.")
    
    parts.append("**Disclaimer:** This information is for educational purposes only and should not replace professional medical advice.")
    
    formatted_response = "\n\n".join(parts)
    
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