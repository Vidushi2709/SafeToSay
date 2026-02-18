import React, { useState, useEffect, useCallback } from 'react';
import LandingPage from './components/LandingPage';
import ChatInterface from './components/ChatInterface';
import ThreadList from './components/ThreadList';
import { ShieldCheck, PanelLeftClose, PanelLeft } from 'lucide-react';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const App = () => {
  const [view, setView] = useState('landing'); // 'landing' | 'chat'
  const [threads, setThreads] = useState([]);
  const [currentThread, setCurrentThread] = useState(null);
  const [sidebarOpen, setSidebarOpen] = useState(true);

  // ── Thread management ──────────────────────────
  const loadThreads = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/threads`);
      if (response.ok) {
        const data = await response.json();
        setThreads(data.threads || []);
      }
    } catch (error) {
      console.error('Error loading threads:', error);
    }
  }, []);

  useEffect(() => {
    if (view === 'chat') {
      loadThreads();
    }
  }, [view, loadThreads]);

  const handleSelectThread = (threadId) => {
    setCurrentThread(threadId);
  };

  const handleCreateThread = () => {
    setCurrentThread(null);
  };

  const handleDeleteThread = async (threadId) => {
    try {
      await fetch(`${API_BASE_URL}/threads/${threadId}`, { method: 'DELETE' });
      if (currentThread === threadId) {
        setCurrentThread(null);
      }
      loadThreads();
    } catch (error) {
      console.error('Error deleting thread:', error);
    }
  };

  const handleThreadCreated = (threadId) => {
    setCurrentThread(threadId);
    loadThreads();
  };

  // ── Landing → Chat transition ──────────────────
  const handleLaunch = () => {
    setView('chat');
  };

  const handleBackToLanding = () => {
    setView('landing');
  };

  // ── Render ────────────────────────────────────
  if (view === 'landing') {
    return <LandingPage onLaunch={handleLaunch} />;
  }

  return (
    <div className="h-screen flex flex-col bg-slate-50">
      {/* ── Top bar ── */}
      <header className="flex items-center justify-between px-4 py-2.5 bg-white border-b border-slate-200">
        <div className="flex items-center space-x-3">
          <button
            onClick={() => setSidebarOpen(!sidebarOpen)}
            className="p-1.5 rounded-lg hover:bg-slate-100 transition-colors text-slate-500"
            title={sidebarOpen ? 'Close sidebar' : 'Open sidebar'}
          >
            {sidebarOpen ? (
              <PanelLeftClose className="w-4 h-4" />
            ) : (
              <PanelLeft className="w-4 h-4" />
            )}
          </button>

          <button
            onClick={handleBackToLanding}
            className="flex items-center space-x-2 hover:opacity-80 transition-opacity"
          >
            <div className="w-6 h-6 rounded-md bg-gradient-to-br from-teal-600 to-teal-700 flex items-center justify-center">
              <ShieldCheck className="w-3.5 h-3.5 text-white" strokeWidth={2.5} />
            </div>
            <span className="text-sm font-bold text-slate-800 tracking-tight">SafeToSay</span>
          </button>
        </div>

        <p className="text-xs text-slate-400 hidden sm:block">
          Evaluation-Driven Clinical AI · Safety gates active
        </p>
      </header>

      {/* ── Main body ── */}
      <div className="flex-1 flex min-h-0">
        {/* Sidebar */}
        {sidebarOpen && (
          <aside className="w-64 flex-shrink-0 border-r border-slate-200 bg-white">
            <ThreadList
              threads={threads}
              currentThread={currentThread}
              onSelectThread={handleSelectThread}
              onCreateThread={handleCreateThread}
              onDeleteThread={handleDeleteThread}
            />
          </aside>
        )}

        {/* Chat area */}
        <main className="flex-1 flex flex-col min-h-0">
          <ChatInterface
            threadId={currentThread}
            onMessageSent={loadThreads}
            onThreadCreated={handleThreadCreated}
            apiBaseUrl={API_BASE_URL}
          />
        </main>
      </div>
    </div>
  );
};

export default App;
