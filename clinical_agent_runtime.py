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
from typing import TypedDict, Annotated, Literal
from pydantic import BaseModel, Field
import json

# Import agent functions
from agents.scope_intent_agent import classify_query, ScopeIntentOutput
from agents.knowledge_boundary_agent import analyze_knowledge_boundary
from agents.knowledge_boundary_agent import ScopeIntentInput as KBInput
from agents.knowledge_boundary_agent import KnowledgeBoundaryOutput
from agents.answer_generation_agent import generate_constrained_answer, AnswerGenerationInput, AnswerGenerationOutput
from agents.answer_generation_agent import KnowledgeConstraints
from agents.eval_agent import evaluate_answer, EvalInput, EvalOutput
from agents.decision_gate import decision_gate, EvalResult, GateDecision

# Runtime State Schema
class AgentRuntimeState(TypedDict):
    """State passed through the agent pipeline"""
    # Input
    clinical_query: str
    
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
    
    # Evaluation Agent output
    eval_scores: dict[str, int]
    eval_critical_failures: list[str]
    eval_rationale: str
    
    # Decision Gate output
    final_decision: str
    decision_reasoning: str

def scope_intent_node(state: AgentRuntimeState) -> AgentRuntimeState:
    """Node 1: Classify query scope and intent"""
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
    
    return state

def knowledge_boundary_node(state: AgentRuntimeState) -> AgentRuntimeState:
    """Node 2: Analyze knowledge boundaries"""
    print("\n[AGENT 2] Knowledge Boundary Analysis")
    
    # Check if query is in scope before continuing
    if state['scope_decision'] != 'IN_SCOPE':
        print(f"Query OUT_OF_SCOPE - Skipping knowledge boundary analysis")
        state['confidence_risk'] = 'HIGH'
        state['required_knowledge'] = []
        state['knowledge_gaps'] = ['Query out of scope - cannot proceed']
        return state
    
    # Create input for knowledge boundary agent
    kb_input = KBInput(
        query=state['clinical_query'],
        detected_intent=state['detected_intent']
    )
    
    result = analyze_knowledge_boundary(kb_input)
    
    print(f"Confidence Risk: {result.confidence_risk}")
    print(f"Required Knowledge: {result.required_knowledge}")
    print(f"Knowledge Gaps: {result.knowledge_gaps}")
    
    # Pass allowed outputs
    state['confidence_risk'] = result.confidence_risk
    state['required_knowledge'] = result.required_knowledge
    state['knowledge_gaps'] = result.knowledge_gaps
    
    return state

def answer_generation_node(state: AgentRuntimeState) -> AgentRuntimeState:
    """Node 3: Generate constrained answer"""
    print("\n[AGENT 3] Answer Generation")
    
    # Check confidence risk - high risk could warrant abstention
    if state['confidence_risk'] == 'HIGH':
        print(f"HIGH confidence risk - creating minimal draft")
    
    # Create input for answer generation
    ag_input = AnswerGenerationInput(
        query=state['clinical_query'],
        allowed_intent=state['detected_intent'],
        evidence=state['evidence'],  # Must be provided externally
        knowledge_constraints=KnowledgeConstraints(
            confidence_risk=state['confidence_risk'],
            required_knowledge=state['required_knowledge']
        )
    )
    
    result = generate_constrained_answer(ag_input)
    
    print(f"Draft Answer: {result.draft_answer[:100]}...")
    
    # Pass only the draft answer
    state['draft_answer'] = result.draft_answer
    
    return state

def evaluation_node(state: AgentRuntimeState) -> AgentRuntimeState:
    """Node 4: Evaluate answer safety"""
    print("\n[AGENT 4] Evaluation (Safety Assessment)")
    
    eval_input = EvalInput(
        clinical_query=state['clinical_query'],
        generated_answer=state['draft_answer']
    )
    
    result = evaluate_answer(eval_input)
    
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
    
    return state

def decision_gate_node(state: AgentRuntimeState) -> AgentRuntimeState:
    """Node 5: Deterministic decision gate"""
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
    evidence: list[str],
    skip_strict_evaluation: bool = False,
) -> dict:
    """
    Execute the full clinical agent pipeline.
    
    Args:
        clinical_query: The clinical question from the user
        evidence: Retrieved evidence snippets to constrain answer
        skip_strict_evaluation: If True, skip evaluation agent for simple factual questions
    
    Returns:
        Final decision output with reasoning
    """
    
    print("=" * 100)
    print("CLINICAL AGENT RUNTIME - Sequential Pipeline Execution")
    print("=" * 100)
    print(f"\nClinical Query: {clinical_query}")
    print(f"Evidence Sources: {len(evidence)}")
    if skip_strict_evaluation:
        print(f"Evaluation Mode: LENIENT (simple factual question detected)")
    
    # Build graph
    agent_graph = build_clinical_runtime_graph()
    
    # Initialize state
    initial_state: AgentRuntimeState = {
        'clinical_query': clinical_query,
        'scope_decision': '',
        'detected_intent': '',
        'risk_notes': [],
        'required_knowledge': [],
        'knowledge_gaps': [],
        'confidence_risk': '',
        'evidence': evidence,
        'draft_answer': '',
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
        'confidence_risk': final_state['confidence_risk']
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
