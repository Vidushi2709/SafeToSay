# SafeToSay - Clinical AI Agent System

### A Multi-Agent Clinical Q&A System That Knows When *Not* to Answer

---

Most medical LLMs answer everything.

**MedGemma doesn’t.**

MedGemma is a multi-agent clinical guideline assistant that evaluates safety and evidence *before* responding. If a query is unsafe, out-of-scope, or insufficiently supported by evidence, the system **explicitly abstains**.

Evaluation isn’t a metric after generation —
**evaluation is the system.**

Built with LangGraph orchestration, MedGemma 4B for medical generation, Mistral AI for reasoning agents, and Tavily for real-time medical literature retrieval.

---

## 🎯 Problem

Clinical AI systems often:

* Over-answer
* Hallucinate evidence
* Provide patient-specific advice
* Fail to abstain under uncertainty

In medicine, **not answering can be safer than answering poorly.**

---

## 🧠 Core Idea

Instead of relying on a single model with safety prompting, MedGemma uses:

* ✅ Specialized agents with narrow responsibilities
* ✅ Runtime evaluation before release
* ✅ Deterministic decision gating
* ✅ Explicit abstention logic
* ✅ Real-time evidence retrieval

The system either:

* ✔️ Provides evidence-grounded guidance
* ⚠️ Returns a constrained partial answer
* ❌ Explicitly abstains

---

## 👥 Target Use Case

Guideline-based Q&A for healthcare professionals:

* Junior doctors
* Nurses
* Physician assistants

### ✅ Allowed

* Contraindications
* Eligibility criteria
* Guideline clarifications
* Medication interactions
* Protocol explanations

### ❌ Disallowed

* Diagnosis
* Treatment recommendations
* Patient-specific decisions
* Emergency medical advice

---

# 🏗️ System Architecture

```
User Query
     ↓
Scope & Intent Agent
     ↓
Knowledge Boundary Agent
     ↓
Medical Evidence Retrieval (Tavily)
     ↓
Answer Generation (MedGemma 4B)
     ↓
Evaluation Agent (5-metric audit)
     ↓
Deterministic Decision Gate
     ↓
ANSWER | PARTIAL | ABSTAIN
```

---

# 🤖 Agent Design

## 1️⃣ Scope & Intent Agent (Mistral AI)

Filters unsafe queries early.

Outputs:

* `IN_SCOPE` / `OUT_OF_SCOPE`
* Intent classification
* Risk flags

Example:

* "Can NSAIDs be given with aspirin?" → IN_SCOPE
* "Diagnose my chest pain" → OUT_OF_SCOPE

---

## 2️⃣ Knowledge Boundary Agent (Mistral AI)

Identifies knowledge gaps and confidence limits.

Outputs:

* Required domains
* Knowledge gaps
* Confidence risk (LOW / MEDIUM / HIGH)

Prevents overconfident generation when information is insufficient.

---

## 3️⃣ Answer Generation Agent (MedGemma 4B)

Generates structured, evidence-constrained clinical answers.

* Model: `google/medgemma-4b-it`
* 4-bit quantized (BitsAndBytes)
* Deterministic decoding
* Grounded in retrieved evidence

---

## 4️⃣ Evaluation Agent (Mistral AI)

Audits the draft before release.

Scores (1–5):

* Evidence Support
* Missing Preconditions
* Overconfidence
* Contradictions
* Scope Violation

Detects critical failures automatically.

---

## 5️⃣ Deterministic Decision Gate

Final logic layer:

* **ANSWER** → All scores ≥ 3
* **PARTIAL** → Minor weakness, no critical failure
* **ABSTAIN** → Any major failure or critical violation

This makes the system auditable and transparent.

---

# 🔎 Evidence Retrieval (Tavily)

Real-time medical search across:

* PubMed
* NIH
* WHO
* Mayo Clinic
* WebMD
* UpToDate
* Up to 5 sources per query
* Source URLs included in output
* Medical-domain filtering enabled

---

# 🛠️ Tech Stack

### Backend

* Python 3.10+
* FastAPI (SSE streaming)
* LangGraph (multi-agent orchestration)
* LangChain
* Mistral AI (reasoning agents)
* MedGemma 4B (generation)
* PyTorch + Transformers
* BitsAndBytes (4-bit quantization)
* Tavily (medical search)

### Frontend

* React 18
* Tailwind CSS
* Server-Sent Events (real-time agent tracing)

---

# ⚙️ Installation

### 1️⃣ Clone

```bash
git clone <repo-url>
cd SafeToSay
```

### 2️⃣ Backend Setup

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate (Windows)
pip install -r requirements.txt
```

Create `.env`:

```
MISTRAL_API_KEY=your_key
TAVILY_API_KEY=your_key
```

Run backend:

```bash
python api_server.py
```

---

### 3️⃣ Frontend Setup

```bash
cd frontend
npm install
npm start
```

---

# 💬 Demo Flow

Ask:

* "What are contraindications for beta-blockers?"
* "Can NSAIDs worsen kidney function?"
* "What are eligibility criteria for statin therapy?"

Watch the pipeline execute:

* Scope analysis
* Knowledge boundary check
* Evidence retrieval
* Draft generation
* Safety audit
* Final decision

If unsafe:

The system abstains clearly and transparently.

---

# 🎓 What This Demonstrates

* Runtime evaluation-in-the-loop
* Self-checking multi-agent systems
* Deterministic safety gating
* Evidence-grounded reasoning
* Transparent abstention logic

Rather than maximizing response rate, the system optimizes **decision quality and safety**.

---

# 🔒 Limitations

* Not for emergency decisions
* Not a replacement for clinical judgment
* Restricted to guideline-based Q&A
* Research prototype only

---

# 🧪 Why This Matters

In clinical AI, correctness is important.

But knowing when not to answer
may be more important.

MedGemma prioritizes safe abstention over unsafe confidence.

---
# ⭐ Contribution 

if you like it, feel free to take it, make it better. 

