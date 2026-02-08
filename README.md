# Evaluation-Driven Clinical AI Agent

## Overview

This project implements a **clinical guideline question-answering AI agent system** where **evaluation is embedded into the runtime itself**, not applied after generation.

The system is designed to **answer only when evidence and safety conditions are satisfied** and to **explicitly abstain otherwise**. Safety emerges from structured agent checks and evaluation signals rather than blind trust in a single model or post-hoc filtering.

This work treats clinical AI as **infrastructure**, not a chatbot.

---

## Core Thesis

> **Reliable clinical AI should decide when *not* to answer.**

Instead of relying on learned safety classifiers or heuristic rules, this system uses:

* Multiple specialized agents with narrow responsibilities
* Runtime evaluation agents that audit decisions before output
* Explicit abstention logic that is transparent and auditable

Evaluation is not a metric at the end — **evaluation *is* the system**.

---

## Locked Use Case

### Task

Clinical **guideline-based Q&A**

### Target Users

* Junior doctors
* Nurses
* Physician assistants

### Allowed Queries

* Eligibility checks
* Contraindications
* Guideline clarifications

### Explicitly Disallowed

* Diagnosis
* Treatment recommendations
* Patient-specific decisions

These constraints keep the system realistic, safe, and empirically evaluable.

---

## System Architecture

```
User Query
   ↓
1. Scope & Intent Agent
   ↓
2. Knowledge Boundary Agent
   ↓
3. Answer Generation Agent (MedGemma)
   ↓
4. Evaluation Agent
   ↓
Final Decision Gate
   → ANSWER or ABSTAIN
```

### Agent Responsibilities

**1. Scope & Intent Agent**

* Determines whether the query is in-scope
* Classifies intent (e.g., contraindication check, eligibility)
* Rejects unsafe or disallowed requests early

**2. Knowledge Boundary Agent**

* Assesses whether the question can be answered safely using general guidelines
* Flags patient-specific or underspecified queries
* Prevents overreach

**3. Answer Generation Agent (MedGemma)**

* Generates guideline-based responses
* Does not decide safety or final output

**4. Evaluation Agent**

* Audits the generated answer for:

  * Safety
  * Evidence sufficiency
  * Scope alignment
* Produces structured evaluation signals

**Final Decision Gate**

* Combines evaluation signals
* Outputs either:

  * `"final_decision": "ANSWER"`
  * `"final_decision": "ABSTAIN"`

---

## Data Generation Strategy

* Create realistic clinician-style guideline queries
* Run each query through the full agent pipeline
* Log intermediate agent outputs and evaluation scores
* Manually audit a subset for quality and correctness

**Target dataset size:** 50–150 high-quality samples

---

## Evaluation Experiments

### Baselines

1. MedGemma (direct answer, no evaluation)
2. MedGemma + heuristic abstention rules
3. Full agent system with runtime evaluation (no retrieval)

### Metrics

* Unsafe answer rate
* Correct abstention rate
* False abstention rate
* Evidence mismatch detection accuracy

**Primary experimental result:**
How runtime evaluation changes system behavior.

---

## Why This Project Is About Evaluation

This system demonstrates:

* Runtime evaluation-in-the-loop
* Self-checking agent architectures
* Transparent abstention logic
* Auditable safety decisions

Rather than optimizing accuracy alone, the system optimizes **decision quality**.

---

## Immediate Next Steps

* Implement evaluation agent prompts
* Add decision gate logic
* Log evaluation scores per query
* Build abstention-first demo cases
* Write experiments and analysis

---

## Final Note

This project studies **how AI agents should decide not to answer**.

That framing aligns directly with current research directions in:

* LLM evaluation
* Agent safety
* Human-centered AI
* Clinical AI governance

---