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
    """Node 3: Generate constrained answer"""
    progress_callback = state.get('progress_callback')
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
    
    # Pass draft answer and sources
    state['draft_answer'] = result.draft_answer
    state['sources'] = result.sources
    
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
    """Build the LangGraph state machine for agent orchestration"""
    
    # Create graph
    graph = StateGraph(AgentRuntimeState)
    
    # Add nodes
    graph.add_node("scope_intent", scope_intent_node)
    graph.add_node("knowledge_boundary", knowledge_boundary_node)
    graph.add_node("answer_generation", answer_generation_node)
    graph.add_node("evaluation", evaluation_node)
    graph.add_node("decision_gate", decision_gate_node)
    
    # Define edges (fixed order)
    graph.add_edge(START, "scope_intent")
    graph.add_edge("scope_intent", "knowledge_boundary")
    graph.add_edge("knowledge_boundary", "answer_generation")
    graph.add_edge("answer_generation", "evaluation")
    graph.add_edge("evaluation", "decision_gate")
    graph.add_edge("decision_gate", END)
    
    # Compile
    return graph.compile()

def run_clinical_pipeline(
    clinical_query: str,
    evidence: list[str] = None,
    skip_strict_evaluation: bool = False,
    use_tavily_search: bool = True,
    progress_callback: callable = None,
) -> dict:
    """
    Execute the full clinical agent pipeline.
    
    Args:
        clinical_query: The clinical question from the user
        evidence: Retrieved evidence snippets to constrain answer (optional)
        skip_strict_evaluation: If True, skip evaluation agent for simple factual questions
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
    if skip_strict_evaluation:
        print(f"Evaluation Mode: LENIENT (simple factual question detected)")
    
    # Build graph
    agent_graph = build_clinical_runtime_graph()
    
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
    
    # For simple factual questions, override to ANSWER if we got a response
    if skip_strict_evaluation and final_state['draft_answer']:
        final_state['final_decision'] = 'ANSWER'
        final_state['decision_reasoning'] = 'Simple factual question - informational answer provided'
    
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
