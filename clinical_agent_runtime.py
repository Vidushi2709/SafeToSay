'''
Clinical Agent Runtime (Orchestration Kernel)
**Purpose**: Sequential execution of agents with data flow control
**Architecture**: LangGraph-based state machine
**Responsibilities**:
- Execute agents in fixed order
- Pass only allowed outputs downstream
- Enforce data contracts between agents
- Run evaluation and decision gate
- Output: ANSWER | PARTIAL_ANSWER_WITH_WARNING | ABSTAIN
**Does NOT**:
- Generate text
- Do reasoning
- Judge safety
- Contain medical logic
**Each agent is a pure function node in the graph**
'''

from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Literal, List, Optional, Callable
from pydantic import BaseModel, Field
import json

# Import agent functions
from agents.scope_intent_agent import classify_query, ScopeIntentOutput
from agents.knowledge_boundary_agent import analyze_knowledge_boundary
from agents.knowledge_boundary_agent import ScopeIntentInput as KBInput
from agents.knowledge_boundary_agent import KnowledgeBoundaryOutput
from agents.answer_generation_agent import generate_constrained_answer, AnswerGenerationInput, AnswerGenerationOutput
from agents.answer_generation_agent import KnowledgeConstraints, SourceInfo
from agents.eval_agent import evaluate_answer, EvalInput, EvalOutput
from agents.decision_gate import decision_gate, EvalResult, GateDecision

# Import Tavily search for research
from agents.tavily_search import search_and_format_evidence, SearchSource, get_sources_for_display

# Runtime State Schema
class AgentRuntimeState(TypedDict):
    """State passed through the agent pipeline"""
    # Input
    clinical_query: str
    
    # Progress callback for step updates
    progress_callback: Optional[Callable]
    
    # Scope & Intent Agent output
    scope_decision: str
    detected_intent: str
    risk_notes: list[str]
    
    # Knowledge Boundary Agent output
    required_knowledge: list[str]
    knowledge_gaps: list[str]
    confidence_risk: str
    
    # Answer Generation Agent (requires evidence - must be provided externally)
    evidence: list[str]
    draft_answer: str
    
    # Sources for citations (from Tavily search)
    sources: list[dict]
    
    # Evaluation Agent output
    eval_scores: dict[str, int]
    eval_critical_failures: list[str]
    eval_rationale: str
    
    # Decision Gate output
    final_decision: str
    decision_reasoning: str

# ============================================================================
# SAFE REFUSAL MESSAGES - Used when queries are OUT_OF_SCOPE
# These are standardized, safe responses that never contain medical content
# ============================================================================

_REFUSAL_MESSAGES = {
    'diagnosis_request': (
        "I'm unable to provide a diagnosis. This system is designed for general "
        "medical education and knowledge queries only — not for diagnosing "
        "symptoms or conditions for any individual. Please consult a qualified "
        "healthcare professional for patient-specific evaluation."
    ),
    'treatment_recommendation': (
        "I'm unable to recommend treatments for specific patients. This system "
        "provides general medical knowledge only. For treatment decisions, "
        "please consult a qualified healthcare professional who can evaluate "
        "the individual clinical context."
    ),
    'patient_specific_decision': (
        "I'm unable to make patient-specific clinical decisions. This system "
        "is designed for general medical education queries only. For decisions "
        "about a specific patient, please consult a qualified healthcare provider."
    ),
    'emergency': (
        "If this is a medical emergency, please call emergency services "
        "immediately (e.g., 911 in the US). This system cannot provide "
        "emergency medical advice."
    ),
    'default': (
        "This query falls outside the scope of this system. I can only answer "
        "general medical knowledge and education questions — not provide "
        "diagnosis, treatment recommendations, or patient-specific advice. "
        "Please consult a qualified healthcare professional."
    ),
}

def _get_refusal_message(detected_intent: str, risk_notes: list[str]) -> str:
    """Get the appropriate safe refusal message based on intent and risk flags."""
    # Check for emergency first
    if any('emergency' in note.lower() for note in risk_notes):
        return _REFUSAL_MESSAGES['emergency']
    
    # Map intent to refusal message
    if detected_intent in _REFUSAL_MESSAGES:
        return _REFUSAL_MESSAGES[detected_intent]
    
    return _REFUSAL_MESSAGES['default']


def scope_intent_node(state: AgentRuntimeState) -> AgentRuntimeState:
    """Node 1: Classify query scope and intent"""
    progress_callback = state.get('progress_callback')
    if progress_callback:
        progress_callback({
            'step': 'scope_intent',
            'title': '🔍 Analyzing Query Scope',
            'status': 'running',
            'description': 'Checking if query is within medical scope and classifying intent'
        })
    
    print("\n[AGENT 1] Scope & Intent Classification")
    print(f"Query: {state['clinical_query']}")
    
    result = classify_query(state['clinical_query'])
    
    print(f"Decision: {result.scope_decision}")
    print(f"Intent: {result.detected_intent}")
    print(f"Risk Notes: {result.risk_notes}")
    
    # Only pass allowed outputs downstream
    state['scope_decision'] = result.scope_decision
    state['detected_intent'] = result.detected_intent
    state['risk_notes'] = result.risk_notes
    
    if progress_callback:
        progress_callback({
            'step': 'scope_intent',
            'title': '🔍 Query Scope Analysis',
            'status': 'completed',
            'description': f'Decision: {result.scope_decision} | Intent: {result.detected_intent}',
            'result': {
                'scope': result.scope_decision,
                'intent': result.detected_intent,
                'risk_notes': result.risk_notes
            }
        })
    
    return state


def scope_router(state: AgentRuntimeState) -> str:
    """
    Conditional router after scope classification.
    Routes OUT_OF_SCOPE queries directly to refusal, bypassing all downstream agents.
    This is the critical safety gate — no medical content is generated for out-of-scope queries.
    """
    scope = state.get('scope_decision', 'OUT_OF_SCOPE')  # Fail-closed: default to OUT_OF_SCOPE
    
    if scope == 'IN_SCOPE':
        print("[ROUTER] Query IN_SCOPE → proceeding to knowledge boundary analysis")
        return 'knowledge_boundary'
    else:
        print(f"[ROUTER] Query {scope} → routing to REFUSAL (no medical content will be generated)")
        return 'refusal'


def refusal_node(state: AgentRuntimeState) -> AgentRuntimeState:
    """
    Safe refusal node for OUT_OF_SCOPE queries.
    Generates a standardized rejection message without any medical content.
    This node is the ONLY output path for out-of-scope queries.
    """
    progress_callback = state.get('progress_callback')
    
    if progress_callback:
        progress_callback({
            'step': 'refusal',
            'title': '🚫 Query Out of Scope',
            'status': 'running',
            'description': 'Query classified as out of scope — generating safe refusal'
        })
    
    print("\n[REFUSAL] Generating safe refusal response")
    print(f"Intent: {state.get('detected_intent', 'unknown')}")
    print(f"Risk Notes: {state.get('risk_notes', [])}")
    
    # Generate safe refusal message
    refusal_message = _get_refusal_message(
        state.get('detected_intent', 'unknown'),
        state.get('risk_notes', [])
    )
    
    # Set all downstream state to reflect refusal
    state['draft_answer'] = refusal_message
    state['final_decision'] = 'ABSTAIN'
    state['decision_reasoning'] = (
        f"SCOPE GATE: Query classified as {state.get('scope_decision', 'OUT_OF_SCOPE')}. "
        f"Intent: {state.get('detected_intent', 'unknown')}. "
        f"Risk flags: {', '.join(state.get('risk_notes', []))}. "
        f"No medical content was generated."
    )
    state['confidence_risk'] = 'HIGH'
    state['required_knowledge'] = []
    state['knowledge_gaps'] = ['Query out of scope - refused at safety gate']
    state['eval_scores'] = {
        'evidence_support': 0,
        'missing_preconditions': 0,
        'overconfidence': 0,
        'contradictions': 0,
        'scope_violation': 0
    }
    state['eval_critical_failures'] = ['OUT_OF_SCOPE - query refused at safety gate']
    state['eval_rationale'] = 'Query was out of scope. No answer was generated or evaluated.'
    state['sources'] = []
    
    if progress_callback:
        progress_callback({
            'step': 'refusal',
            'title': '🚫 Query Refused — Out of Scope',
            'status': 'completed',
            'description': f'Query safely refused. Intent: {state.get("detected_intent", "unknown")}',
            'result': {
                'decision': 'ABSTAIN',
                'intent': state.get('detected_intent', 'unknown'),
                'risk_notes': state.get('risk_notes', [])
            }
        })
    
    print(f"[REFUSAL] Decision: ABSTAIN")
    print(f"[REFUSAL] Message: {refusal_message}")
    
    return state

def knowledge_boundary_node(state: AgentRuntimeState) -> AgentRuntimeState:
    """Node 2: Analyze knowledge boundaries"""
    progress_callback = state.get('progress_callback')
    if progress_callback:
        progress_callback({
            'step': 'knowledge_boundary',
            'title': '🧠 Knowledge Boundary Analysis',
            'status': 'running',
            'description': 'Assessing required knowledge and identifying potential gaps'
        })
    
    print("\n[AGENT 2] Knowledge Boundary Analysis")
    
    # Check if query is in scope before continuing
    if state['scope_decision'] != 'IN_SCOPE':
        print(f"Query OUT_OF_SCOPE - Skipping knowledge boundary analysis")
        state['confidence_risk'] = 'HIGH'
        state['required_knowledge'] = []
        state['knowledge_gaps'] = ['Query out of scope - cannot proceed']
        
        if progress_callback:
            progress_callback({
                'step': 'knowledge_boundary',
                'title': '🧠 Knowledge Boundary Analysis',
                'status': 'completed',
                'description': 'Skipped - Query out of scope',
                'result': {'confidence_risk': 'HIGH', 'reason': 'Out of scope'}
            })
        return state
    
    if progress_callback:
        progress_callback({
            'step': 'kb_analysis_preparing',
            'title': '📚 Analyzing Knowledge Requirements',
            'status': 'running',
            'description': 'Identifying medical domains and expertise needed'
        })
    
    # Create input for knowledge boundary agent
    kb_input = KBInput(
        query=state['clinical_query'],
        detected_intent=state['detected_intent']
    )
    
    if progress_callback:
        progress_callback({
            'step': 'kb_analysis_processing',
            'title': '🎯 Evaluating Knowledge Gaps',
            'status': 'running',
            'description': 'Identifying areas where additional evidence is needed'
        })
    
    result = analyze_knowledge_boundary(kb_input)
    
    print(f"Confidence Risk: {result.confidence_risk}")
    print(f"Required Knowledge: {result.required_knowledge}")
    print(f"Knowledge Gaps: {result.knowledge_gaps}")
    
    # Pass allowed outputs
    state['confidence_risk'] = result.confidence_risk
    state['required_knowledge'] = result.required_knowledge
    state['knowledge_gaps'] = result.knowledge_gaps
    
    if progress_callback:
        progress_callback({
            'step': 'knowledge_boundary',
            'title': '🧠 Knowledge Boundary Analysis',
            'status': 'completed',
            'description': f'Risk Level: {result.confidence_risk} | Gaps: {len(result.knowledge_gaps)}',
            'result': {
                'confidence_risk': result.confidence_risk,
                'required_knowledge': result.required_knowledge,
                'knowledge_gaps': result.knowledge_gaps
            }
        })
    
    return state

def answer_generation_node(state: AgentRuntimeState) -> AgentRuntimeState:
    """Node 3: Generate constrained answer
    
    SAFETY: This node has a redundant scope check. Even though the graph router
    should prevent OUT_OF_SCOPE queries from reaching here, this is defense-in-depth.
    """
    progress_callback = state.get('progress_callback')
    
    # =====================================================================
    # DEFENSE-IN-DEPTH: Redundant scope check
    # Even though the graph router should prevent this, we double-check here.
    # If scope is not IN_SCOPE, refuse to generate any medical content.
    # =====================================================================
    if state.get('scope_decision') != 'IN_SCOPE':
        print(f"[AGENT 3] SAFETY BLOCK: scope_decision={state.get('scope_decision')} — refusing to generate")
        refusal_message = _get_refusal_message(
            state.get('detected_intent', 'unknown'),
            state.get('risk_notes', [])
        )
        state['draft_answer'] = refusal_message
        state['final_decision'] = 'ABSTAIN'
        state['decision_reasoning'] = 'DEFENSE-IN-DEPTH: Answer generation blocked for out-of-scope query'
        if progress_callback:
            progress_callback({
                'step': 'answer_generation',
                'title': '🚫 Answer Generation Blocked',
                'status': 'completed',
                'description': 'Out-of-scope query — no medical content generated'
            })
        return state
    
    if progress_callback:
        progress_callback({
            'step': 'answer_generation',
            'title': '✍️ Generating Medical Response',
            'status': 'running',
            'description': f'Creating evidence-based response using {len(state["evidence"])} sources'
        })
    
    print("\n[AGENT 3] Answer Generation")
    
    if progress_callback:
        progress_callback({
            'step': 'answer_preparation',
            'title': '📝 Preparing Medical Context',
            'status': 'running',
            'description': 'Organizing evidence and applying medical formatting standards'
        })
    
    # Check confidence risk - high risk could warrant abstention
    if state['confidence_risk'] == 'HIGH':
        print(f"HIGH confidence risk - creating minimal draft")
    
    if progress_callback:
        progress_callback({
            'step': 'answer_sources',
            'title': '📚 Processing Source Citations',
            'status': 'running',
            'description': 'Preparing medical literature references for inclusion'
        })
    
    # Convert sources from state to SourceInfo objects if available
    source_infos = None
    if state.get('sources'):
        source_infos = [
            SourceInfo(
                title=src.get('title', 'Unknown'),
                url=src.get('url', ''),
                snippet=src.get('snippet', '')
            )
            for src in state['sources']
        ]
    
    # Create input for answer generation
    ag_input = AnswerGenerationInput(
        query=state['clinical_query'],
        allowed_intent=state['detected_intent'],
        evidence=state['evidence'],  # Evidence from Tavily or provided externally
        knowledge_constraints=KnowledgeConstraints(
            confidence_risk=state['confidence_risk'],
            required_knowledge=state['required_knowledge']
        ),
        sources=source_infos
    )
    
    if progress_callback:
        progress_callback({
            'step': 'answer_model_generation',
            'title': '🤖 Medical AI Processing',
            'status': 'running',
            'description': 'Generating clinical response using MedGemma language model'
        })
    
    result = generate_constrained_answer(ag_input)
    
    print(f"Draft Answer: {result.draft_answer[:100]}...")
    print(f"Sources: {len(result.sources)} sources available")
    
    # Pass draft answer
    state['draft_answer'] = result.draft_answer
    
    # FIX: Only overwrite sources if new sources were returned
    # Convert SourceInfo objects or dicts to dict format if needed
    if result.sources:
        state['sources'] = [
            {
                'title': src.get('title', '') if isinstance(src, dict) else getattr(src, 'title', ''),
                'url': src.get('url', '') if isinstance(src, dict) else getattr(src, 'url', ''),
                'snippet': src.get('snippet', '') if isinstance(src, dict) else getattr(src, 'snippet', '')
            }
            for src in result.sources
        ]
    # else: keep existing sources from Tavily search - don't overwrite with empty list
    
    if progress_callback:
        progress_callback({
            'step': 'answer_generation',
            'title': '✍️ Generating Medical Response',
            'status': 'completed',
            'description': f'Generated {len(result.draft_answer)} character response with {len(result.sources)} sources',
            'result': {
                'answer_length': len(result.draft_answer),
                'sources_count': len(result.sources)
            }
        })
    
    return state

def evaluation_node(state: AgentRuntimeState) -> AgentRuntimeState:
    """Node 4: Evaluate answer safety"""
    progress_callback = state.get('progress_callback')
    if progress_callback:
        progress_callback({
            'step': 'evaluation',
            'title': '🛡️ Safety Evaluation',
            'status': 'running',
            'description': 'Analyzing answer for safety, evidence support, and clinical appropriateness'
        })
    
    print("\n[AGENT 4] Evaluation (Safety Assessment)")
    
    eval_input = EvalInput(
        clinical_query=state['clinical_query'],
        generated_answer=state['draft_answer']
    )
    
    result = evaluate_answer(eval_input, progress_callback)
    
    print(f"Evidence Support: {result.scores.evidence_support}/5")
    print(f"Missing Preconditions: {result.scores.missing_preconditions}/5")
    print(f"Overconfidence: {result.scores.overconfidence}/5")
    print(f"Contradictions: {result.scores.contradictions}/5")
    print(f"Scope Violation: {result.scores.scope_violation}/5")
    print(f"Critical Failures: {result.critical_failures}")
    
    # Pass only allowed outputs
    state['eval_scores'] = {
        'evidence_support': result.scores.evidence_support,
        'missing_preconditions': result.scores.missing_preconditions,
        'overconfidence': result.scores.overconfidence,
        'contradictions': result.scores.contradictions,
        'scope_violation': result.scores.scope_violation
    }
    state['eval_critical_failures'] = result.critical_failures
    state['eval_rationale'] = result.rationale
    
    # FIX: Explicitly carry forward sources
    # LangGraph may drop keys that a node doesn't modify
    state['sources'] = state.get('sources', [])
    
    if progress_callback:
        avg_score = sum(state['eval_scores'].values()) / len(state['eval_scores'])
        progress_callback({
            'step': 'evaluation',
            'title': 'Safety Evaluation',
            'status': 'completed',
            'description': f'Average Score: {avg_score:.1f}/5 | Critical Issues: {len(result.critical_failures)}',
            'result': {
                'scores': state['eval_scores'],
                'critical_failures': result.critical_failures,
                'average_score': round(avg_score, 1)
            }
        })
    
    return state

def decision_gate_node(state: AgentRuntimeState) -> AgentRuntimeState:
    """Node 5: Deterministic decision gate"""
    progress_callback = state.get('progress_callback')
    if progress_callback:
        progress_callback({
            'step': 'decision_gate',
            'title': '⚖️ Final Decision',
            'status': 'running',
            'description': 'Making final decision based on evaluation scores'
        })
    
    print("\n[GATE] Deterministic Decision")
    
    # Create input for decision gate
    gate_input = EvalResult(
        scores=state['eval_scores'],
        critical_failures=state['eval_critical_failures']
    )
    
    result = decision_gate(gate_input)
    
    print(f"Decision: {result.decision}")
    print(f"Min Score: {result.min_score}/5")
    print(f"Avg Score: {result.avg_score:.2f}/5")
    print(f"Reasoning: {result.reasoning}")
    
    # Pass decision
    state['final_decision'] = result.decision
    state['decision_reasoning'] = result.reasoning
    
    # FIX: Explicitly carry forward sources
    state['sources'] = state.get('sources', [])
    
    if progress_callback:
        status_icon = '✅' if result.decision == 'ANSWER' else '⚠️' if result.decision == 'PARTIAL_ANSWER_WITH_WARNING' else '❌'
        progress_callback({
            'step': 'decision_gate',
            'title': f'⚖️ Final Decision: {result.decision}',
            'status': 'completed',
            'description': f'{status_icon} Min: {result.min_score}/5 | Avg: {result.avg_score:.1f}/5',
            'result': {
                'decision': result.decision,
                'reasoning': result.reasoning,
                'min_score': result.min_score,
                'avg_score': result.avg_score
            }
        })
    
    return state

def build_clinical_runtime_graph():
    """
    Build the LangGraph state machine for agent orchestration.
    
    ARCHITECTURE (with conditional safety routing):
    
        START → scope_intent → [ROUTER]
                                  ├── IN_SCOPE → knowledge_boundary → answer_generation → evaluation → decision_gate → END
                                  └── OUT_OF_SCOPE → refusal → END
    
    The router after scope_intent is the critical safety gate.
    OUT_OF_SCOPE queries NEVER reach the answer generation model.
    """
    
    # Create graph
    graph = StateGraph(AgentRuntimeState)
    
    # Add all nodes
    graph.add_node("scope_intent", scope_intent_node)
    graph.add_node("refusal", refusal_node)  # NEW: Safe refusal for out-of-scope
    graph.add_node("knowledge_boundary", knowledge_boundary_node)
    graph.add_node("answer_generation", answer_generation_node)
    graph.add_node("evaluation", evaluation_node)
    graph.add_node("decision_gate", decision_gate_node)
    
    # Start → scope classification
    graph.add_edge(START, "scope_intent")
    
    # CRITICAL: Conditional routing after scope classification
    # This is the safety gate — OUT_OF_SCOPE queries never reach answer generation
    graph.add_conditional_edges(
        "scope_intent",
        scope_router,
        {
            "knowledge_boundary": "knowledge_boundary",  # IN_SCOPE path
            "refusal": "refusal",                          # OUT_OF_SCOPE path
        }
    )
    
    # OUT_OF_SCOPE path: refusal → END (no further processing)
    graph.add_edge("refusal", END)
    
    # IN_SCOPE path: normal pipeline continues
    graph.add_edge("knowledge_boundary", "answer_generation")
    graph.add_edge("answer_generation", "evaluation")
    graph.add_edge("evaluation", "decision_gate")
    graph.add_edge("decision_gate", END)
    
    # Compile
    return graph.compile()

# Compile graph once at module level (stateless, reusable across requests)
_compiled_graph = build_clinical_runtime_graph()

def run_clinical_pipeline(
    clinical_query: str,
    evidence: list[str] = None,
    use_tavily_search: bool = True,
    progress_callback: callable = None,
) -> dict:
    """
    Execute the full clinical agent pipeline.
    
    SAFETY ARCHITECTURE:
    - Step 1: Scope classification (always runs)
    - Step 2: If OUT_OF_SCOPE → refusal node → safe rejection (no medical content)
    - Step 3: If IN_SCOPE → evidence search → answer generation → evaluation → decision gate
    
    OUT_OF_SCOPE queries NEVER reach the answer generation model.
    
    Args:
        clinical_query: The clinical question from the user
        evidence: Retrieved evidence snippets to constrain answer (optional)
        use_tavily_search: If True and no evidence provided, use Tavily to search for evidence
        progress_callback: Callback function to report progress updates
    
    Returns:
        Final decision output with reasoning and sources
    """
    
    print("CLINICAL AGENT RUNTIME - Sequential Pipeline Execution")
    print(f"\nClinical Query: {clinical_query}")
    
    # If no evidence provided and Tavily search is enabled, search for evidence
    sources = []
    if not evidence and use_tavily_search:
        if progress_callback:
            progress_callback({
                'step': 'research',
                'title': '🔎 Medical Literature Search',
                'status': 'running',
                'description': 'Searching trusted medical databases for relevant evidence'
            })
        
        if progress_callback:
            progress_callback({
                'step': 'research_connecting',
                'title': '🌐 Connecting to Medical Databases',
                'status': 'running',
                'description': 'Accessing PubMed, medical journals, and clinical databases'
            })
        
        print("\n[RESEARCH] Searching for evidence using Tavily...")
        
        if progress_callback:
            progress_callback({
                'step': 'research_querying',
                'title': '🔍 Executing Medical Search',
                'status': 'running',
                'description': f'Searching for: "{clinical_query}"'
            })
        
        evidence, search_sources = search_and_format_evidence(clinical_query, max_results=5)
        
        if progress_callback:
            progress_callback({
                'step': 'research_processing',
                'title': '📄 Processing Search Results',
                'status': 'running',
                'description': 'Filtering and ranking medical literature for relevance'
            })
        
        sources = get_sources_for_display(search_sources)
        print(f"[RESEARCH] Found {len(sources)} sources")
        
        if progress_callback:
            progress_callback({
                'step': 'research',
                'title': '🔎 Medical Literature Search',
                'status': 'completed',
                'description': f'Found {len(sources)} relevant medical sources',
                'result': {
                    'sources_found': len(sources),
                    'evidence_snippets': len(evidence)
                }
            })
    elif not evidence:
        evidence = []
    
    print(f"Evidence Sources: {len(evidence)}")
    
    # Use pre-compiled graph (built once at module level)
    agent_graph = _compiled_graph
    
    # Initialize state
    initial_state: AgentRuntimeState = {
        'clinical_query': clinical_query,
        'progress_callback': progress_callback,
        'scope_decision': '',
        'detected_intent': '',
        'risk_notes': [],
        'required_knowledge': [],
        'knowledge_gaps': [],
        'confidence_risk': '',
        'evidence': evidence,
        'draft_answer': '',
        'sources': sources,
        'eval_scores': {},
        'eval_critical_failures': [],
        'eval_rationale': '',
        'final_decision': '',
        'decision_reasoning': ''
    }
    
    # Execute pipeline
    final_state = agent_graph.invoke(initial_state)
    
    # =====================================================================
    # FINAL SAFETY ENFORCEMENT
    # Regardless of what happened in the pipeline, enforce these invariants:
    # 1. OUT_OF_SCOPE → always ABSTAIN
    # 2. ABSTAIN → draft_answer must be a safe refusal, not medical content
    # =====================================================================
    
    scope_decision = final_state.get('scope_decision', 'OUT_OF_SCOPE')
    final_decision = final_state.get('final_decision', 'ABSTAIN')
    
    # Invariant 1: OUT_OF_SCOPE must always result in ABSTAIN
    if scope_decision != 'IN_SCOPE' and final_decision != 'ABSTAIN':
        print(f"[SAFETY] INVARIANT VIOLATION: scope={scope_decision} but decision={final_decision}. Forcing ABSTAIN.")
        final_state['final_decision'] = 'ABSTAIN'
        final_state['decision_reasoning'] = (
            f"SAFETY OVERRIDE: Query was {scope_decision} but pipeline produced {final_decision}. "
            f"Forced to ABSTAIN."
        )
        final_state['draft_answer'] = _get_refusal_message(
            final_state.get('detected_intent', 'unknown'),
            final_state.get('risk_notes', [])
        )
    
    # Invariant 2: If ABSTAIN and scope was OUT_OF_SCOPE, ensure no medical content leaked
    if final_state.get('final_decision') == 'ABSTAIN' and scope_decision != 'IN_SCOPE':
        draft = final_state.get('draft_answer', '')
        # Check if the draft answer looks like it contains medical content instead of a refusal
        refusal_keywords = ['unable to', 'outside the scope', 'cannot provide', 'falls outside',
                           'consult a qualified', 'emergency services']
        is_refusal = any(kw in draft.lower() for kw in refusal_keywords)
        if not is_refusal and len(draft) > 100:
            # Medical content leaked through — replace with safe refusal
            print(f"[SAFETY] CONTENT LEAK DETECTED: ABSTAIN with medical content. Replacing with safe refusal.")
            final_state['draft_answer'] = _get_refusal_message(
                final_state.get('detected_intent', 'unknown'),
                final_state.get('risk_notes', [])
            )
    
    # DEBUG: Verify sources made it through
    print(f"\n[DEBUG] sources count in final_state: {len(final_state.get('sources', []))}")
    
    # Output results
    print("\n" + "=" * 100)
    print("PIPELINE RESULT")
    print("=" * 100)
    print(f"\nFinal Decision: {final_state['final_decision']}")
    print(f"Decision Reasoning: {final_state['decision_reasoning']}")
    
    if final_state['final_decision'] == 'ANSWER':
        print(f"\n✓ ANSWER RELEASED")
        print(f"Draft Answer:\n{final_state['draft_answer']}")
    elif final_state['final_decision'] == 'PARTIAL_ANSWER_WITH_WARNING':
        print(f"\n⚠ ANSWER WITH WARNING")
        print(f"Draft Answer:\n{final_state['draft_answer']}")
        print(f"\nWARNING: {final_state['decision_reasoning']}")
    else:  # ABSTAIN
        print(f"\n✗ ABSTAIN - Answer not released to clinician")
        print(f"Reason: {final_state['decision_reasoning']}")
    
    return {
        'final_decision': final_state['final_decision'],
        'draft_answer': final_state['draft_answer'],
        'decision_reasoning': final_state['decision_reasoning'],
        'eval_scores': final_state['eval_scores'],
        'eval_rationale': final_state['eval_rationale'],
        'scope_decision': final_state['scope_decision'],
        'detected_intent': final_state['detected_intent'],
        'confidence_risk': final_state['confidence_risk'],
        'sources': final_state.get('sources', [])
    }

if __name__ == "__main__":
    # Test run
    test_query = "Can beta blockers be given in asthma?"
    test_evidence = [
        "Beta-blockers, particularly non-selective ones, can cause bronchoconstriction in asthma patients.",
        "Cardioselective beta-blockers are considered safer alternatives in asthma.",
        "Clinical guidelines generally recommend caution when using beta-blockers in asthma or COPD."
    ]
    
    result = run_clinical_pipeline(test_query, test_evidence)
