# 🏥 MedGemma Clinical AI Agent System

## Overview

A **clinical guideline question-answering system** with **evaluation embedded into the runtime** not applied after generation. The system answers only when evidence and safety conditions are satisfied, and explicitly abstains otherwise.

Built with **LangGraph multi-agent orchestration**, **MedGemma 4B** for medical answer generation, **Mistral AI** for agent reasoning, and **Tavily Search** for real-time medical literature retrieval.

### 🎯 Core Thesis

> **Reliable clinical AI should decide when *not* to answer.**

Instead of relying on learned safety classifiers or heuristic rules, this system uses:

- ✅ Multiple specialized agents with narrow responsibilities
- ✅ Runtime evaluation agents that audit decisions before output
- ✅ Explicit abstention logic that is transparent and auditable
- ✅ Real-time evidence retrieval from trusted medical sources

**Evaluation is not a metric at the end — evaluation *is* the system.**

---

## 🎭 Use Case & Constraints

### 🎯 Task
Clinical **guideline-based Q&A** for healthcare professionals

### 👥 Target Users
- Junior doctors
- Nurses  
- Physician assistants

### ✅ Allowed Queries
- Eligibility checks
- Contraindications
- Guideline clarifications
- Medication interactions
- Clinical protocol questions

### ❌ Explicitly Disallowed
- Diagnosis
- Treatment recommendations
- Patient-specific decisions
- Emergency medical advice

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    User Query (via Chat UI)                 │
└──────────────────────┬──────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────┐
│  FastAPI Backend (Streaming SSE + Thread Management)         │
└──────────────────────┬──────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────┐
│              📚 Tavily Medical Search (Optional)             │
│       Retrieves evidence from PubMed, NIH, Mayo Clinic       │
└──────────────────────┬──────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────┐
│            LangGraph Agent Orchestration Pipeline            │
└─────────────────────┬───────────────────────────────────────┘
                      ↓
        ┌─────────────────────────────┐
        │  1️⃣  Scope & Intent Agent   │
        │     (Mistral AI)             │
        │  - IN_SCOPE / OUT_OF_SCOPE   │
        │  - Intent classification     │
        └──────────┬──────────────────┘
                   ↓
        ┌─────────────────────────────┐
        │  2️⃣  Knowledge Boundary     │
        │     Agent (Mistral AI)       │
        │  - Required knowledge        │
        │  - Confidence risk: L/M/H    │
        └──────────┬──────────────────┘
                   ↓
        ┌─────────────────────────────┐
        │  3️⃣  Answer Generation      │
        │     Agent (MedGemma 4B)      │
        │  - Draft answer generation   │
        │  - Evidence-constrained      │
        └──────────┬──────────────────┘
                   ↓
        ┌─────────────────────────────┐
        │  4️⃣  Evaluation Agent       │
        │     (Mistral AI)             │
        │  - Safety audit (5 metrics)  │
        │  - Critical failure detection│
        └──────────┬──────────────────┘
                   ↓
        ┌─────────────────────────────┐
        │  5️⃣  Decision Gate          │
        │     (Deterministic Logic)    │
        │  → ANSWER                    │
        │  → PARTIAL_ANSWER_WARNING    │
        │  → ABSTAIN                   │
        └──────────┬──────────────────┘
                   ↓
┌─────────────────────────────────────────────────────────────┐
│        Response + Sources + Rationale (Streamed to UI)      │
└─────────────────────────────────────────────────────────────┘
```

---

## 🧠 Agent Responsibilities

### 1️⃣ Scope & Intent Agent (Mistral AI)
**Purpose:** Filter unsafe queries early  
**Outputs:**
- `scope_decision`: `IN_SCOPE` | `OUT_OF_SCOPE`
- `detected_intent`: Classification (contraindication, eligibility, etc.)
- `risk_notes`: Early warning flags

**Example:**  
✅ "Can NSAIDs be given with aspirin?" → IN_SCOPE  
❌ "Diagnose my chest pain" → OUT_OF_SCOPE

---

### 2️⃣ Knowledge Boundary Agent (Mistral AI)
**Purpose:** Identify knowledge gaps and confidence limits  
**Outputs:**
- `required_knowledge`: Domains needed (pharmacology, cardiology, etc.)
- `knowledge_gaps`: Missing critical information
- `confidence_risk`: `LOW` | `MEDIUM` | `HIGH`

**Example:**  
🟡 "What's the dose for lisinopril?" → HIGH risk (patient-specific)  
🟢 "What are NSAIDs contraindications?" → LOW risk (general guideline)

---

### 3️⃣ Answer Generation Agent (MedGemma 4B)
**Purpose:** Generate evidence-based draft answer  
**Features:**
- Uses **4-bit quantization** for efficient GPU inference
- Constrained by retrieved Tavily evidence
- Structured medical response formatting
- Avoids overconfident language

**Technology:**
- Model: `google/medgemma-4b-it`
- Framework: Hugging Face Transformers + BitsAndBytes
- Device: CUDA (GPU) or CPU fallback

---

### 4️⃣ Evaluation Agent (Mistral AI)
**Purpose:** Audit answer safety before release  
**Outputs 5 Scores (1-5 scale):**
1. **Evidence Support:** Is answer grounded in evidence?
2. **Missing Preconditions:** Are critical context requirements stated?
3. **Overconfidence:** Does it claim certainty inappropriately?
4. **Contradictions:** Any internal logical conflicts?
5. **Scope Violation:** Does it exceed allowed question type?

**Critical Failures:** Auto-detected violations (e.g., unsupported diagnosis)

---

### 5️⃣ Decision Gate (Deterministic Logic)
**Purpose:** Make final release decision based on eval scores  
**Logic:**
- **ANSWER**: All scores ≥ 3, no critical failures
- **PARTIAL_ANSWER_WITH_WARNING**: Min score ≥ 2, avg ≥ 3, no critical failures
- **ABSTAIN**: Any score < 2 OR critical failure present

---

## 🔍 Tavily Medical Search Integration

Real-time evidence retrieval from trusted medical sources.

**Features:**
- 🔎 Searches PubMed, NIH, WHO, Mayo Clinic, WebMD, UpToDate
- 📚 Max 5 sources per query (configurable)
- 🔗 Source citations with URLs included in response
- ⚡ Medical domain filtering enabled

**Example Sources:**
```json
{
  "sources": [
    {
      "title": "Beta-blockers in asthma - PubMed",
      "url": "https://pubmed.ncbi.nlm.nih.gov/12345",
      "snippet": "Beta-blockers, particularly non-selective ones, can cause bronchoconstriction..."
    }
  ]
}
```

---

## 🛠️ Tech Stack

### Backend
- **Python 3.10+**
- **FastAPI** - API server with SSE streaming
- **LangGraph** - Agent orchestration state machine
- **LangChain** - LLM integration framework
- **Mistral AI** - Agent reasoning (scope, boundary, eval)
- **MedGemma 4B** - Medical answer generation
- **PyTorch** - Deep learning framework
- **Transformers** - HuggingFace model loading
- **BitsAndBytes** - 4-bit quantization
- **Tavily** - Medical literature search

### Frontend
- **React 18** - UI framework
- **Tailwind CSS** - Styling
- **Lucide Icons** - Icon library
- **Server-Sent Events (SSE)** - Real-time streaming

### Storage
- **JSON file-based** - Conversation thread storage

---

## 📁 Project Structure

```
medgemma/
├── agents/
│   ├── answer_generation_agent.py    # MedGemma 4B generation
│   ├── scope_intent_agent.py         # Query filtering
│   ├── knowledge_boundary_agent.py   # Confidence assessment
│   ├── eval_agent.py                 # Safety evaluation
│   ├── decision_gate.py              # Final decision logic
│   └── tavily_search.py              # Medical literature search
├── prompts/
│   ├── answer_generation_system_prompt.txt
│   ├── evaluation_system_prompt.txt
│   ├── knowledge_boundary_system_prompt.txt
│   └── scope_intent_system_prompt.txt
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── ChatInterface.js      # Main chat UI
│   │   │   ├── MessageList.js        # Message display
│   │   │   ├── ThreadList.js         # Conversation threads
│   │   │   ├── MessageInput.js       # User input
│   │   │   └── Header.js             # App header
│   │   ├── App.js
│   │   └── index.js
│   └── package.json
├── conversation_store/
│   └── threads_index.json            # Thread metadata storage
├── clinical_agent_runtime.py         # LangGraph orchestration
├── api_server.py                     # FastAPI server
├── requirements.txt                  # Python dependencies
├── start-system.ps1                  # Windows launcher
├── .env                              # Environment variables
└── README.md
```

---

## 📋 Prerequisites

### Required
- **Python 3.10+**
- **Node.js 16+** and npm
- **CUDA-capable GPU** (recommended) or CPU for inference
- **8GB+ RAM** (16GB recommended for GPU)

### API Keys Required
- **Mistral AI API Key** - For agent reasoning ([Get key](https://console.mistral.ai/))
- **Tavily API Key** - For medical search ([Get key](https://tavily.com/))

---

## 🚀 Installation & Setup

### 1️⃣ Clone Repository
```bash
git clone <your-repo-url>
cd medgemma
```

### 2️⃣ Backend Setup

#### Create Python Virtual Environment
```bash
python -m venv venv
```

#### Activate Virtual Environment
**Windows:**
```bash
venv\Scripts\activate
```

**Linux/Mac:**
```bash
source venv/bin/activate
```

#### Install Python Dependencies
```bash
pip install -r requirements.txt
```

This installs:
- FastAPI, Uvicorn (web server)
- LangGraph, LangChain (agent orchestration)
- PyTorch, Transformers (ML models)
- BitsAndBytes (quantization)
- Tavily (medical search)
- Mistral AI integration

### 3️⃣ Frontend Setup

```bash
cd frontend
npm install
cd ..
```

### 4️⃣ Environment Configuration

Create a `.env` file in the project root:

```env
# Required API Keys
MISTRAL_API_KEY=your_mistral_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here

# Optional Configuration
MISTRAL_MODEL=mistral-large-latest
MAX_TAVILY_RESULTS=5
```

**Get API Keys:**
- Mistral AI: https://console.mistral.ai/
- Tavily: https://tavily.com/

---

## ▶️ Running the System

### Option 1: Automated Launcher (Windows)

```powershell
.\start-system.ps1
```

This script:
1. Checks Python installation
2. Installs dependencies
3. Starts FastAPI backend on `http://localhost:8000`
4. Starts React frontend on `http://localhost:3000`
5. Opens browser automatically

### Option 2: Manual Start

#### Terminal 1 - Backend
```bash
# Activate virtual environment
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Start FastAPI server
python api_server.py
```

Backend runs on: `http://localhost:8000`  
API docs: `http://localhost:8000/docs`

#### Terminal 2 - Frontend
```bash
cd frontend
npm start
```

Frontend runs on: `http://localhost:3000`

---

## 💬 Usage Guide

### 1️⃣ Start New Conversation
- Click **"+ New Thread"** in sidebar
- Each thread maintains independent conversation history

### 2️⃣ Ask Medical Questions
Type questions like:
- "What are the contraindications for beta-blockers?"
- "Can aspirin and NSAIDs be taken together?"
- "What are the eligibility criteria for statin therapy?"

### 3️⃣ Real-Time Progress
Watch agent pipeline execution:
- 🔍 Query Scope Analysis
- 🧠 Knowledge Boundary Analysis
- 🔎 Medical Literature Search
- ✍️ Generating Response
- 🛡️ Safety Evaluation
- ⚖️ Final Decision

### 4️⃣ Review Response
- **Answer:** Main clinical response
- **Sources:** Clickable medical literature references
- **Decision:** ANSWER / PARTIAL_ANSWER_WITH_WARNING / ABSTAIN

### 5️⃣ Thread Management
- Switch between conversations in sidebar
- Delete threads with trash icon
- Threads auto-save and persist

---

## 🌐 API Documentation

### Base URL
```
http://localhost:8000/api/v1
```

### Endpoints

#### 1. Health Check
```http
GET /api/v1/health
```

**Response:**
```json
{
  "status": "healthy",
  "service": "Clinical Agent API"
}
```

#### 2. Create Thread
```http
POST /api/v1/threads
```

**Response:**
```json
"thread-uuid-here"
```

#### 3. List Threads
```http
GET /api/v1/threads
```

**Response:**
```json
[
  {
    "thread_id": "abc-123",
    "title": "Beta-blockers in asthma",
    "created_at": "2026-02-17T10:30:00",
    "updated_at": "2026-02-17T10:35:00",
    "message_count": 4
  }
]
```

#### 4. Get Thread Messages
```http
GET /api/v1/threads/{thread_id}/messages
```

**Response:**
```json
[
  {
    "role": "user",
    "content": "Can NSAIDs cause kidney damage?",
    "timestamp": "2026-02-17T10:30:00"
  },
  {
    "role": "assistant",
    "content": "Yes, NSAIDs can cause...",
    "timestamp": "2026-02-17T10:30:15",
    "sources": [...]
  }
]
```

#### 5. Send Message (Streaming)
```http
POST /api/v1/chat/stream
Content-Type: application/json

{
  "query": "What are contraindications for beta-blockers?",
  "thread_id": "abc-123"
}
```

**Response:** Server-Sent Events (SSE) stream

**Event Types:**
```javascript
// Progress update
data: {"type":"progress","progress":{"step":"scope_intent","status":"running"}}

// Source citations
data: {"type":"sources","sources":[{...}]}

// Answer tokens (word-by-word)
data: {"type":"token","token":"Beta-blockers "}

// Stream complete
data: {"type":"complete"}

// Error
data: {"type":"error","message":"Error details"}
```

#### 6. Delete Thread
```http
DELETE /api/v1/threads/{thread_id}
```

---

## ⚙️ Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `MISTRAL_API_KEY` | ✅ Yes | - | Mistral AI API key for agents |
| `TAVILY_API_KEY` | ✅ Yes | - | Tavily search API key |
| `MISTRAL_MODEL` | ❌ No | `mistral-large-latest` | Mistral model to use |
| `MAX_TAVILY_RESULTS` | ❌ No | `5` | Max search results |

### Model Configuration

**MedGemma Settings** (in `answer_generation_agent.py`):
- Model: `google/medgemma-4b-it`
- Quantization: 4-bit (BitsAndBytes NF4)
- Max tokens: 512
- Inference: Greedy decoding (deterministic)

**Mistral Settings** (in agent files):
- Model: `mistral-large-latest`
- Temperature: 0 (deterministic)
- JSON mode: Structured outputs

---

## 🧪 Development

### Running Backend Only
```bash
python api_server.py
```
Access API docs: http://localhost:8000/docs

### Running Frontend Only
```bash
cd frontend
npm start
```

### Testing CLI (Without Frontend)
```bash
python clinical_agent_runtime.py
```

### Adding New Agents
1. Create agent file in `agents/`
2. Define Pydantic input/output models
3. Implement agent function
4. Add to pipeline in `clinical_agent_runtime.py`
5. Update state schema in `AgentRuntimeState`

### Modifying Prompts
Edit files in `prompts/` directory:
- `scope_intent_system_prompt.txt`
- `knowledge_boundary_system_prompt.txt`
- `answer_generation_system_prompt.txt`
- `evaluation_system_prompt.txt`

---

## 🎯 System Decision Flow

```
User Query
    ↓
Scope Check → OUT_OF_SCOPE? → ABSTAIN ❌
    ↓ IN_SCOPE
Knowledge Boundary → HIGH Risk + No Evidence? → ABSTAIN ❌
    ↓ Proceed
Tavily Search (if enabled) → Retrieve Evidence
    ↓
MedGemma Generation → Draft Answer
    ↓
Evaluation → Score 5 Metrics (1-5)
    ↓
Decision Gate:
  - All scores ≥3, no critical failures → ANSWER ✅
  - Min ≥2, avg ≥3, no critical failures → PARTIAL_ANSWER ⚠️
  - Any score <2 OR critical failure → ABSTAIN ❌
```

---

## 🎓 Research Context

This system demonstrates:
- ✅ **Runtime evaluation-in-the-loop** (not post-hoc)
- ✅ **Self-checking agent architectures**
- ✅ **Transparent abstention logic**
- ✅ **Auditable safety decisions**
- ✅ **Evidence-grounded medical reasoning**

Rather than optimizing accuracy alone, the system optimizes **decision quality** and **safety**.

---

## 🔒 Safety & Limitations

### Safety Features
- ✅ Multi-agent checks before answering
- ✅ Explicit abstention on unsafe queries
- ✅ Evidence-grounded responses only
- ✅ Source citations for verification
- ✅ Conservative decision logic

### Known Limitations
- ⚠️ Not for emergency medical decisions
- ⚠️ Not a replacement for clinical judgment
- ⚠️ Restricted to guideline-based questions
- ⚠️ Requires human verification for critical decisions
- ⚠️ Model quantization may affect response quality

### Disclaimer
⚠️ **This system is for research and educational purposes only. It is NOT a medical device and should NOT be used for clinical decision-making without appropriate medical supervision.**

---

## 🤝 Contributing

Like it? Take it, use it, make it better

---