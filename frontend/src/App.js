import React, { useState, useEffect } from 'react';
import ChatInterface from './components/ChatInterface';
import ThreadList from './components/ThreadList';
import Header from './components/Header';

const API_BASE_URL = 'http://localhost:8000/api/v1';

function App() {
  const [threads, setThreads] = useState([]);
  const [currentThread, setCurrentThread] = useState(() => {
    return localStorage.getItem('currentThread') || null;
  });
  const [sidebarVisible, setSidebarVisible] = useState(true);
  const [loading, setLoading] = useState(true);
  const [readingPanelVisible, setReadingPanelVisible] = useState(false);
  const [selectedMessage, setSelectedMessage] = useState(null);

  useEffect(() => {
    if (currentThread) {
      localStorage.setItem('currentThread', currentThread);
    } else {
      localStorage.removeItem('currentThread');
    }
  }, [currentThread]);

  useEffect(() => {
    fetchThreads();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const fetchThreads = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/threads`);
      if (response.ok) {
        const threadsData = await response.json();
        setThreads(threadsData);

        if (!currentThread && threadsData.length > 0) {
          setCurrentThread(threadsData[0].thread_id);
        } else if (currentThread && threadsData.length > 0) {
          const threadExists = threadsData.some(t => t.thread_id === currentThread);
          if (!threadExists) {
            setCurrentThread(threadsData[0].thread_id);
          }
        }
      }
    } catch (error) {
      console.error('Error fetching threads:', error);
    } finally {
      setLoading(false);
    }
  };

  const createNewThread = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/threads`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
      });

      if (response.ok) {
        const newThreadId = await response.json();
        setCurrentThread(newThreadId);
        await fetchThreads();
      }
    } catch (error) {
      console.error('Error creating new thread:', error);
    }
  };

  const deleteThread = async (threadId) => {
    try {
      const response = await fetch(`${API_BASE_URL}/threads/${threadId}`, {
        method: 'DELETE',
      });

      if (response.ok) {
        await fetchThreads();
        if (currentThread === threadId) {
          const remainingThreads = threads.filter(t => t.thread_id !== threadId);
          if (remainingThreads.length > 0) {
            setCurrentThread(remainingThreads[0].thread_id);
          } else {
            await createNewThread();
          }
        }
      }
    } catch (error) {
      console.error('Error deleting thread:', error);
    }
  };

  const onMessageSent = () => fetchThreads();

  const onThreadCreated = (newThreadId) => {
    setCurrentThread(newThreadId);
    fetchThreads();
  };

  const openReadingPanel = (message) => {
    setSelectedMessage(message);
    setReadingPanelVisible(true);
  };

  const closeReadingPanel = () => {
    setReadingPanelVisible(false);
    setSelectedMessage(null);
  };

  const renderStructuredContent = (content) => {
    if (!content) return null;
    const sections = content.split('\n\n').map(section => section.trim()).filter(Boolean);

    return sections.map((section, index) => {
      const isHeader = /^#{1,6}\s/.test(section) || /^\*\*.*\*\*$/.test(section) ||
        (section.length < 100 && section === section.toUpperCase() && /^[A-Z\s:.-]+$/.test(section));
      const isList = /^[-*•]\s/.test(section) || /^\d+\.\s/.test(section);
      const hasCitations = /\[\d+\]|\(\d{4}\)|doi:|PMID:|Source:/i.test(section);

      if (isHeader) {
        return (
          <h3 key={index} className="text-base font-semibold text-slate-900 mb-3 mt-4 first:mt-0">
            {section.replace(/^#+\s*|\*\*/g, '')}
          </h3>
        );
      }
      if (isList) {
        const items = section.split('\n').filter(Boolean);
        return (
          <ul key={index} className="list-disc list-inside space-y-2 mb-4 ml-4">
            {items.map((item, itemIndex) => (
              <li key={itemIndex} className="text-slate-700 text-sm">
                {item.replace(/^[-*•]\s*|^\d+\.\s*/, '')}
              </li>
            ))}
          </ul>
        );
      }
      if (hasCitations) {
        return (
          <div key={index} className="bg-teal-50 border-l-4 border-teal-300 p-4 mb-4 rounded-r">
            <p className="text-sm text-teal-800 whitespace-pre-line">{section}</p>
          </div>
        );
      }
      return (
        <p key={index} className="text-slate-700 mb-4 leading-relaxed whitespace-pre-line text-sm">
          {section}
        </p>
      );
    });
  };

  if (loading) {
    return (
      <div className="h-screen bg-slate-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-10 w-10 border-b-2 border-teal-600 mx-auto mb-4"></div>
          <p className="text-slate-500 text-sm font-medium">Loading SafeToSay...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="h-screen bg-slate-50 flex overflow-hidden">
      {/* Sidebar */}
      {sidebarVisible && (
        <div className="w-72 border-r border-slate-200 flex flex-col bg-white">
          <ThreadList
            threads={threads}
            currentThread={currentThread}
            onSelectThread={setCurrentThread}
            onCreateThread={createNewThread}
            onDeleteThread={deleteThread}
          />
        </div>
      )}

      {/* Main Content */}
      <div className="flex-1 flex flex-col min-h-0">
        <Header
          onToggleSidebar={() => setSidebarVisible(!sidebarVisible)}
          sidebarVisible={sidebarVisible}
        />

        <div className={`flex-1 overflow-hidden flex min-h-0 ${readingPanelVisible ? 'divide-x divide-slate-200' : ''}`}>
          <div className={`${readingPanelVisible ? 'w-1/2' : 'w-full'} flex flex-col min-h-0 transition-all duration-300`}>
            <ChatInterface
              threadId={currentThread}
              onMessageSent={onMessageSent}
              onThreadCreated={onThreadCreated}
              apiBaseUrl={API_BASE_URL}
              onOpenReadingPanel={openReadingPanel}
            />
          </div>

          {/* Reading Panel */}
          {readingPanelVisible && (
            <div className="w-1/2 bg-white flex flex-col">
              <div className="bg-slate-50 px-4 py-3 border-b flex items-center justify-between">
                <h3 className="font-medium text-slate-800 text-sm">Reading Panel</h3>
                <button
                  onClick={closeReadingPanel}
                  className="text-slate-400 hover:text-slate-600 transition-colors p-1 rounded-lg hover:bg-slate-100"
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>
              {selectedMessage && (
                <div className="flex-1 overflow-y-auto p-6">
                  <div className="prose prose-sm max-w-none">
                    <div className="mb-4">
                      <span className={`inline-block px-2.5 py-1 rounded-full text-xs font-medium ${selectedMessage.role === 'user'
                          ? 'bg-teal-100 text-teal-800'
                          : 'bg-slate-100 text-slate-700'
                        }`}>
                        {selectedMessage.role === 'user' ? 'Your Question' : 'AI Response'}
                      </span>
                      {selectedMessage.timestamp && (
                        <span className="text-xs text-slate-400 ml-2">
                          {new Date(selectedMessage.timestamp).toLocaleString()}
                        </span>
                      )}
                    </div>
                    <div className="structured-content">
                      {renderStructuredContent(selectedMessage.content)}
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
