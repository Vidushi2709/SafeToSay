"""
FastAPI Server for Medical Clinical Agent with LangGraph Streaming & Thread Management
Supports:
- Thread-based conversation history
- Streaming responses
- Message persistence
"""

import os
import json
import uuid
from datetime import datetime
from typing import Optional, Generator, List, AsyncGenerator
from pathlib import Path
import asyncio
import traceback

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Lazy import of clinical pipeline (to avoid hanging on startup)
# from clinical_agent_runtime import run_clinical_pipeline

# Data models
class Message(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = None

class ThreadMessage(BaseModel):
    role: str
    content: str
    thread_id: str
    timestamp: datetime

class ThreadInfo(BaseModel):
    thread_id: str
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: int

class QueryRequest(BaseModel):
    query: str
    thread_id: Optional[str] = None
    evidence: List[str] = []  # Optional - if not provided, Tavily search will be used
    use_tavily_search: bool = True  # Enable Tavily search by default

class ThreadResponse(BaseModel):
    thread_id: str
    created_at: datetime
    messages: List[Message]

# Thread/Message storage
STORAGE_DIR = Path("./conversation_store")
STORAGE_DIR.mkdir(exist_ok=True)

class ConversationStore:
    """Simple file-based storage for conversations and threads"""
    
    def __init__(self, storage_dir: Path = STORAGE_DIR):
        self.storage_dir = storage_dir
        self.threads_file = storage_dir / "threads_index.json"
        self.ensure_initialized()
    
    def ensure_initialized(self):
        """Initialize storage if needed"""
        if not self.threads_file.exists():
            with open(self.threads_file, 'w') as f:
                json.dump({'threads': {}}, f)
    
    def create_thread(self, title: Optional[str] = None) -> str:
        """Create a new conversation thread"""
        thread_id = f"thread_{uuid.uuid4().hex[:12]}"
        
        with open(self.threads_file, 'r') as f:
            data = json.load(f)
        
        now = datetime.now().isoformat()
        data['threads'][thread_id] = {
            'title': title or f"Chat {len(data['threads']) + 1}",
            'created_at': now,
            'updated_at': now,
            'messages': []
        }
        
        with open(self.threads_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        return thread_id
    
    def get_all_threads(self) -> List[ThreadInfo]:
        """Get all threads sorted by most recent"""
        with open(self.threads_file, 'r') as f:
            data = json.load(f)
        
        threads = []
        for thread_id, thread_data in data['threads'].items():
            threads.append(ThreadInfo(
                thread_id=thread_id,
                title=thread_data['title'],
                created_at=datetime.fromisoformat(thread_data['created_at']),
                updated_at=datetime.fromisoformat(thread_data['updated_at']),
                message_count=len(thread_data['messages'])
            ))
        
        # Sort by most recent first
        return sorted(threads, key=lambda t: t.updated_at, reverse=True)
    
    def get_thread(self, thread_id: str) -> Optional[ThreadResponse]:
        """Get a specific thread with all messages"""
        with open(self.threads_file, 'r') as f:
            data = json.load(f)
        
        if thread_id not in data['threads']:
            return None
        
        thread_data = data['threads'][thread_id]
        messages = [
            Message(
                role=msg['role'],
                content=msg['content'],
                timestamp=datetime.fromisoformat(msg['timestamp'])
            )
            for msg in thread_data['messages']
        ]
        
        return ThreadResponse(
            thread_id=thread_id,
            created_at=datetime.fromisoformat(thread_data['created_at']),
            messages=messages
        )
    
    def add_message(self, thread_id: str, role: str, content: str) -> bool:
        """Add a message to a thread"""
        with open(self.threads_file, 'r') as f:
            data = json.load(f)
        
        if thread_id not in data['threads']:
            return False
        
        data['threads'][thread_id]['messages'].append({
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat()
        })
        
        data['threads'][thread_id]['updated_at'] = datetime.now().isoformat()
        
        with open(self.threads_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        return True
    
    def update_thread_title(self, thread_id: str, title: str) -> bool:
        """Update thread title (useful for auto-naming threads)"""
        with open(self.threads_file, 'r') as f:
            data = json.load(f)
        
        if thread_id not in data['threads']:
            return False
        
        data['threads'][thread_id]['title'] = title
        
        with open(self.threads_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        return True

# Initialize FastAPI and storage
app = FastAPI(title="Clinical Agent API", version="1.0.0")
store = ConversationStore()

def get_clinical_pipeline():
    """Lazy loading of clinical pipeline to avoid startup hangs"""
    try:
        from clinical_agent_runtime import run_clinical_pipeline
        return run_clinical_pipeline
    except Exception as e:
        print(f"ERROR loading clinical pipeline: {e}")
        raise

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def stream_response_generator(response_text: str) -> Generator[str, None, None]:
    """Generator to stream response token by token"""
    # Simulate streaming by sending chunks
    words = response_text.split()
    for i, word in enumerate(words):
        # Send word with space (except last word)
        chunk = word + (" " if i < len(words) - 1 else "")
        yield f"data: {json.dumps({'token': chunk})}\n\n"

def get_fallback_evidence(query: str) -> list[str]:
    """
    Provide minimal fallback evidence when Tavily search is disabled.
    In production with Tavily enabled, this function is not used.
    The clinical pipeline will use Tavily search to get real evidence.
    """
    # Return empty list - Tavily search will be used in the pipeline
    return []

def is_simple_factual_question(query: str) -> bool:
    """
    Detect simple factual/definitional questions that don't need strict evaluation.
    Examples: "What is diabetes?", "Define hypertension", "Explain what asthma is"
    """
    query_lower = query.lower().strip()
    
    # Pattern matching for definitional questions
    patterns = [
        query_lower.startswith("what is "),
        query_lower.startswith("what are "),
        query_lower.startswith("define "),
        query_lower.startswith("explain "),
        query_lower.startswith("describe "),
        " mean" in query_lower and ("do" in query_lower or "does" in query_lower),
        "difference between" in query_lower and "and" in query_lower,
    ]
    
    return any(patterns)

def stream_metadata_generator(metadata: dict) -> Generator[str, None, None]:
    """Send metadata about the response"""
    yield f"data: {json.dumps({'type': 'metadata', 'data': metadata})}\n\n"

# ===== Thread Management Endpoints =====

@app.get("/api/v1/threads", response_model=List[ThreadInfo])
async def list_threads():
    """Get all conversation threads"""
    return store.get_all_threads()

@app.post("/api/v1/threads", response_model=str)
async def create_thread(title: Optional[str] = None):
    """Create a new conversation thread"""
    thread_id = store.create_thread(title)
    return thread_id

@app.get("/api/v1/threads/{thread_id}", response_model=ThreadResponse)
async def get_thread(thread_id: str):
    """Get conversation history for a thread"""
    thread = store.get_thread(thread_id)
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")
    return thread

@app.delete("/api/v1/threads/{thread_id}")
async def delete_thread(thread_id: str):
    """Delete a thread"""
    with open(store.threads_file, 'r') as f:
        data = json.load(f)
    
    if thread_id not in data['threads']:
        raise HTTPException(status_code=404, detail="Thread not found")
    
    del data['threads'][thread_id]
    
    with open(store.threads_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    return {"message": "Thread deleted"}

# ===== Streaming Chat Endpoint =====

@app.post("/api/v1/chat/stream")
async def chat_stream(request: QueryRequest, background_tasks: BackgroundTasks):
    """
    Stream a response from the clinical agent with LangGraph
    
    Returns a streaming response with:
    1. Metadata about the decision process
    2. Streamed response tokens
    """
    
    # Create or use existing thread
    thread_id = request.thread_id or store.create_thread()
    
    # Store user message
    store.add_message(thread_id, "user", request.query)
    
    async def response_generator():
        try:
            # Check if this is a simple factual question
            is_simple = is_simple_factual_question(request.query)
            
            # Use provided evidence or let the pipeline use Tavily search
            evidence = request.evidence if request.evidence else None
            
            # Create progress callback for step-by-step updates
            def progress_callback(progress_data):
                # Send progress update to frontend
                yield f"data: {json.dumps({'type': 'progress', 'progress': progress_data})}\n\n"
            
            # Send initial status update
            yield f"data: {json.dumps({'type': 'status', 'message': 'Initializing clinical analysis pipeline...'})}\n\n"
            
            # Run the clinical pipeline in executor to avoid blocking
            loop = asyncio.get_event_loop()
            run_clinical_pipeline = get_clinical_pipeline()
            
            # We need to collect progress updates since they can't yield from executor
            progress_updates = []
            def sync_progress_callback(progress_data):
                progress_updates.append(progress_data)
            
            result = await loop.run_in_executor(
                None,
                lambda: run_clinical_pipeline(
                    clinical_query=request.query,
                    evidence=evidence,
                    skip_strict_evaluation=is_simple,
                    use_tavily_search=request.use_tavily_search,
                    progress_callback=sync_progress_callback
                )
            )
            
            # Send collected progress updates
            for progress_data in progress_updates:
                yield f"data: {json.dumps({'type': 'progress', 'progress': progress_data})}\n\n"
            
            # Send metadata first (includes sources)
            yield f"data: {json.dumps({'type': 'metadata', 'data': result})}\n\n"
            
            # Send sources separately for easy frontend access
            sources = result.get('sources', [])
            if sources:
                yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"
            
            # Stream the response
            response_text = result.get('draft_answer', '')
            if not response_text:
                response_text = f"Decision: {result.get('final_decision', 'ABSTAIN')}. {result.get('decision_reasoning', '')}"
            
            # Stream word by word
            words = response_text.split()
            for i, word in enumerate(words):
                chunk = word + (" " if i < len(words) - 1 else "")
                yield f"data: {json.dumps({'type': 'token', 'token': chunk})}\n\n"
            
            # Send completion
            yield f"data: {json.dumps({'type': 'complete'})}\n\n"
            
            # Store assistant message in background
            background_tasks.add_task(
                store.add_message,
                thread_id,
                "assistant",
                response_text
            )
            
            # Auto-name thread on first message
            thread = store.get_thread(thread_id)
            if thread and len(thread.messages) == 2:  # Just user + assistant
                title = request.query[:50] + ("..." if len(request.query) > 50 else "")
                background_tasks.add_task(store.update_thread_title, thread_id, title)
            
        except Exception as e:
            error_msg = f"Clinical pipeline error: {str(e)}"
            print(f"ERROR: {error_msg}")
            print(f"TRACEBACK: {traceback.format_exc()}")
            yield f"data: {json.dumps({'type': 'error', 'error': error_msg})}\n\n"
    
    return StreamingResponse(
        response_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Thread-ID": thread_id
        }
    )

@app.post("/api/v1/chat")
async def chat(request: QueryRequest):
    """
    Non-streaming chat endpoint (for compatibility)
    Useful for simple queries or when streaming is not needed
    """
    
    # Create or use existing thread
    thread_id = request.thread_id or store.create_thread()
    
    # Store user message
    store.add_message(thread_id, "user", request.query)
    
    try:
        # Check if this is a simple factual question
        is_simple = is_simple_factual_question(request.query)
        
        # Use provided evidence or let the pipeline use Tavily search
        evidence = request.evidence if request.evidence else None
        
        # Run the clinical pipeline (skip strict evaluation for simple questions)
        run_clinical_pipeline = get_clinical_pipeline()
        result = run_clinical_pipeline(
            clinical_query=request.query,
            evidence=evidence,
            skip_strict_evaluation=is_simple,
            use_tavily_search=request.use_tavily_search,
            progress_callback=None  # No real-time updates for non-streaming
        )
        
        response_text = result.get('draft_answer', '')
        if not response_text:
            response_text = f"Decision: {result.get('final_decision', 'ABSTAIN')}. {result.get('decision_reasoning', '')}"
        
        # Store assistant message
        store.add_message(thread_id, "assistant", response_text)
        
        # Auto-name thread on first message
        thread = store.get_thread(thread_id)
        if thread and len(thread.messages) == 2:  # Just user + assistant
            title = request.query[:50] + ("..." if len(request.query) > 50 else "")
            store.update_thread_title(thread_id, title)
        
        return {
            'thread_id': thread_id,
            'response': response_text,
            'sources': result.get('sources', []),  # Include sources in response
            **result
        }
    
    except Exception as e:
        print(f"ERROR in chat endpoint: {str(e)}")
        print(f"TRACEBACK: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Clinical pipeline error: {str(e)}")

# ===== Health Check =====

@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "service": "clinical_agent_api"}

if __name__ == "__main__":
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
