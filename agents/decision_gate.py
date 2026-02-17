'''
Decision Gate (Deterministic)
**Purpose**: Infrastructure-level decision making
**Responsibilities**:
- Apply hard rules to evaluation scores
- Make deterministic, auditable decisions
- No LLM, no vibes - pure logic
**Input**:
- Evaluation result with scores dictionary
**Output**:
- Decision: "ABSTAIN" | "PARTIAL_ANSWER_WITH_WARNING" | "ANSWER"
**Philosophy**:
- Explainable: Rules are clear and auditable
- Reproducible: Same scores always produce same decision
- Auditable: Every decision can be traced to specific thresholds
'''

from pydantic import BaseModel, Field
from typing import Literal

# Input model
class EvalResult(BaseModel):
    scores: dict[str, int] = Field(description="Dictionary of dimension scores (1-5)")
    critical_failures: list[str] = Field(description="List of critical failures from evaluation")

# Output model
class GateDecision(BaseModel):
    decision: Literal['ABSTAIN', 'PARTIAL_ANSWER_WITH_WARNING', 'ANSWER'] = Field(
        description="Deterministic gate decision"
    )
    reasoning: str = Field(description="Explanation of decision logic")
    min_score: int = Field(description="Minimum score across all dimensions")
    avg_score: float = Field(description="Average score across all dimensions")

def decision_gate(eval_result: EvalResult) -> GateDecision:
    """
    Deterministic decision gate based on evaluation scores.
    
    Rules (infrastructure logic, not vibes):
    1. If score is 0 OR all dimensions are 1 → ABSTAIN (complete failure)
    2. Else if min ≤ 2 OR average < 3.0 → PARTIAL_ANSWER_WITH_WARNING (concerns exist but answer useful)
    3. Else → ANSWER (safe to present)
    
    Args:
        eval_result: Evaluation result with scores and critical_failures
    
    Returns:
        GateDecision with decision, reasoning, and score metrics
    """
    scores = eval_result.scores
    critical_failures = eval_result.critical_failures
    
    # Calculate metrics
    min_score = min(scores.values())
    avg_score = sum(scores.values()) / len(scores)
    
    # Rule 0: ANY critical failure → immediate ABSTAIN regardless of scores
    # Critical failures represent safety-breaking issues that scores alone cannot capture
    if critical_failures:
        reasoning = (
            f"CRITICAL FAILURE DETECTED: {', '.join(critical_failures)}. "
            f"Automatic ABSTAIN triggered. Scores: {scores}"
        )
        return GateDecision(
            decision="ABSTAIN",
            reasoning=reasoning,
            min_score=min_score,
            avg_score=avg_score
        )
    
    # Rule 1: Complete failure - ALL dimensions are 1 or any dimension is 0 → ABSTAIN
    all_ones = all(score == 1 for score in scores.values())
    if min_score == 0 or all_ones:
        reasoning = (
            f"COMPLETE FAILURE: All scores are critically low. "
            f"Scores: {scores}."
        )
        return GateDecision(
            decision="ABSTAIN",
            reasoning=reasoning,
            min_score=min_score,
            avg_score=avg_score
        )
    
    # Rule 2: Significant concerns - low min OR low average → PARTIAL_ANSWER_WITH_WARNING
    if min_score <= 2 or avg_score < 3.0:
        reasoning = (
            f"CONCERNS DETECTED: Score concerns present (min: {min_score}/5, avg: {avg_score:.2f}/5). "
            f"Answer can be shown but clinician should review carefully. "
            f"Critical issues: {', '.join(critical_failures) if critical_failures else 'None'}"
        )
        return GateDecision(
            decision="PARTIAL_ANSWER_WITH_WARNING",
            reasoning=reasoning,
            min_score=min_score,
            avg_score=avg_score
        )
    
    # Rule 3: All other cases → ANSWER
    reasoning = (
        f"SAFE: Minimum score {min_score}/5, average {avg_score:.2f}/5. "
        f"Answer passes safety thresholds. Scores: {scores}"
    )
    return GateDecision(
        decision="ANSWER",
        reasoning=reasoning,
        min_score=min_score,
        avg_score=avg_score
    )

def test_decision_gate():
    """Test the decision gate with various score scenarios."""
    
    print("=" * 100)
    print("DECISION GATE TESTING (Deterministic Infrastructure)")
    print("=" * 100)
    
    # Test Case 1: Critical failure (score of 1)
    test_case_1 = EvalResult(
        scores={
            "evidence_support": 1,
            "missing_preconditions": 3,
            "overconfidence": 2,
            "contradictions": 4,
            "scope_violation": 5
        },
        critical_failures=["Unverifiable medical claim", "Absolute language"]
    )
    
    print("\n--- TEST CASE 1: Critical Safety Failure ---")
    print(f"Scores: {test_case_1.scores}")
    print(f"Critical Failures: {test_case_1.critical_failures}")
    
    result_1 = decision_gate(test_case_1)
    print(f"\nDecision: {result_1.decision}")
    print(f"Min Score: {result_1.min_score}/5")
    print(f"Avg Score: {result_1.avg_score:.2f}/5")
    print(f"Reasoning: {result_1.reasoning}")
    
    assert result_1.decision == "ABSTAIN", "Should ABSTAIN on score ≤ 2"
    print("✓ PASS")
    
    # Test Case 2: Marginal safety (average < 3.5)
    test_case_2 = EvalResult(
        scores={
            "evidence_support": 3,
            "missing_preconditions": 3,
            "overconfidence": 3,
            "contradictions": 3,
            "scope_violation": 4
        },
        critical_failures=[]
    )
    
    print("\n" + "=" * 100)
    print("--- TEST CASE 2: Marginal Safety (Avg < 3.5) ---")
    print(f"Scores: {test_case_2.scores}")
    print(f"Critical Failures: {test_case_2.critical_failures}")
    
    result_2 = decision_gate(test_case_2)
    print(f"\nDecision: {result_2.decision}")
    print(f"Min Score: {result_2.min_score}/5")
    print(f"Avg Score: {result_2.avg_score:.2f}/5")
    print(f"Reasoning: {result_2.reasoning}")
    
    assert result_2.decision == "PARTIAL_ANSWER_WITH_WARNING", "Should WARN when avg < 3.5"
    print("✓ PASS")
    
    # Test Case 3: Safe answer (all scores ≥ 3, average ≥ 3.5)
    test_case_3 = EvalResult(
        scores={
            "evidence_support": 5,
            "missing_preconditions": 4,
            "overconfidence": 5,
            "contradictions": 5,
            "scope_violation": 4
        },
        critical_failures=[]
    )
    
    print("\n" + "=" * 100)
    print("--- TEST CASE 3: Safe Answer (Avg ≥ 3.5) ---")
    print(f"Scores: {test_case_3.scores}")
    print(f"Critical Failures: {test_case_3.critical_failures}")
    
    result_3 = decision_gate(test_case_3)
    print(f"\nDecision: {result_3.decision}")
    print(f"Min Score: {result_3.min_score}/5")
    print(f"Avg Score: {result_3.avg_score:.2f}/5")
    print(f"Reasoning: {result_3.reasoning}")
    
    assert result_3.decision == "ANSWER", "Should ANSWER when avg ≥ 3.5"
    print("✓ PASS")
    
    # Test Case 4: Boundary case (exactly 2 on one dimension)
    test_case_4 = EvalResult(
        scores={
            "evidence_support": 2,
            "missing_preconditions": 4,
            "overconfidence": 4,
            "contradictions": 4,
            "scope_violation": 4
        },
        critical_failures=[]
    )
    
    print("\n" + "=" * 100)
    print("--- TEST CASE 4: Boundary Case (Min = 2) ---")
    print(f"Scores: {test_case_4.scores}")
    print(f"Critical Failures: {test_case_4.critical_failures}")
    
    result_4 = decision_gate(test_case_4)
    print(f"\nDecision: {result_4.decision}")
    print(f"Min Score: {result_4.min_score}/5")
    print(f"Avg Score: {result_4.avg_score:.2f}/5")
    print(f"Reasoning: {result_4.reasoning}")
    
    assert result_4.decision == "ABSTAIN", "Should ABSTAIN when min ≤ 2"
    print("✓ PASS")
    
    # Test Case 5: Boundary case (average exactly 3.6, above threshold)
    test_case_5 = EvalResult(
        scores={
            "evidence_support": 4,
            "missing_preconditions": 4,
            "overconfidence": 3,
            "contradictions": 4,
            "scope_violation": 3
        },
        critical_failures=[]
    )
    
    print("\n" + "=" * 100)
    print("--- TEST CASE 5: Boundary Case (Avg = 3.6, Above Threshold) ---")
    print(f"Scores: {test_case_5.scores}")
    print(f"Critical Failures: {test_case_5.critical_failures}")
    
    result_5 = decision_gate(test_case_5)
    print(f"\nDecision: {result_5.decision}")
    print(f"Min Score: {result_5.min_score}/5")
    print(f"Avg Score: {result_5.avg_score:.2f}/5")
    print(f"Reasoning: {result_5.reasoning}")
    
    assert result_5.decision == "ANSWER", "Should ANSWER when avg ≥ 3.5"
    print("✓ PASS")
    
    print("\n" + "=" * 100)
    print("ALL TESTS PASSED - Decision Gate Logic is Deterministic and Auditable")
    print("=" * 100)

if __name__ == "__main__":
    test_decision_gate()
