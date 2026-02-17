import React, { useState, useEffect, useRef } from 'react';
import MessageList from './MessageList';
import MessageInput from './MessageInput';

const ChatInterface = ({ threadId, onMessageSent, onThreadCreated, apiBaseUrl, onOpenReadingPanel }) => {
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [streamingMessage, setStreamingMessage] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const [progressSteps, setProgressSteps] = useState([]);
  const eventSourceRef = useRef(null);
  const accumulatedMessageRef = useRef('');
  const pendingSourcesRef = useRef([]);

  // Scroll handled inside MessageList component now

  // Load thread messages when threadId changes
  useEffect(() => {
    if (threadId) {
      loadThreadMessages();
    } else {
      setMessages([]);
    }
    
    // Cleanup any active streaming
    const eventSource = eventSourceRef.current;
    return () => {
      if (eventSource) {
        eventSource.close();
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [threadId]);

  const loadThreadMessages = async () => {
    if (!threadId) return;
    
    try {
      setLoading(true);
      const response = await fetch(`${apiBaseUrl}/threads/${threadId}`);
      if (response.ok) {
        const threadData = await response.json();
        setMessages(threadData.messages || []);
      }
    } catch (error) {
      console.error('Error loading thread messages:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleSendMessage = async (messageText) => {
    if (!messageText.trim() || isStreaming) return;

    const userMessage = {
      role: 'user',
      content: messageText,
      timestamp: new Date().toISOString(),
    };

    // Add user message immediately
    setMessages(prev => [...prev, userMessage]);
    setIsStreaming(true);
    setStreamingMessage('');
    setProgressSteps([]);
    accumulatedMessageRef.current = '';

    try {
      // Send streaming request
      const response = await fetch(`${apiBaseUrl}/chat/stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: messageText,
          thread_id: threadId,
          evidence: [],
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      // Get thread ID from response headers (for new threads)
      const responseThreadId = response.headers.get('X-Thread-ID');
      if (responseThreadId && !threadId && onThreadCreated) {
        onThreadCreated(responseThreadId);
      }

      // Create reader for streaming
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';
        
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6));
              
              switch (data.type) {
                case 'status':
                  // Could show status in UI
                  break;
                  
                case 'progress':
                  // Update progress steps
                  setProgressSteps(prev => {
                    const existingIndex = prev.findIndex(step => step.step === data.progress.step);
                    if (existingIndex >= 0) {
                      // Update existing step
                      const updated = [...prev];
                      updated[existingIndex] = data.progress;
                      return updated;
                    } else {
                      // Add new step
                      return [...prev, data.progress];
                    }
                  });
                  break;
                  
                case 'sources':
                  pendingSourcesRef.current = data.sources || [];
                  break;
                  
                case 'token':
                  accumulatedMessageRef.current += data.token;
                  setStreamingMessage(accumulatedMessageRef.current);
                  break;
                  
                case 'complete':
                  // Finalize the assistant message with sources
                  setMessages(prev => [
                    ...prev,
                    {
                      role: 'assistant',
                      content: accumulatedMessageRef.current,
                      timestamp: new Date().toISOString(),
                      sources: pendingSourcesRef.current,
                    },
                  ]);
                  setStreamingMessage('');
                  setProgressSteps([]);
                  pendingSourcesRef.current = [];
                  setIsStreaming(false);
                  onMessageSent();
                  return;
                  
                case 'error':
                  console.error('Streaming error:', data.error);
                  setMessages(prev => [
                    ...prev,
                    {
                      role: 'assistant',
                      content: `Error: ${data.error}`,
                      timestamp: new Date().toISOString(),
                      isError: true,
                    },
                  ]);
                  setStreamingMessage('');
                  setProgressSteps([]);
                  setIsStreaming(false);
                  return;
                  
                default:
                  break;
              }
            } catch (e) {
              console.error('Error parsing SSE data:', e);
            }
          }
        }
      }

      // If we exit the loop without a complete event, finalize with what we have
      if (accumulatedMessageRef.current) {
        setMessages(prev => [
          ...prev,
          {
            role: 'assistant',
            content: accumulatedMessageRef.current,
            timestamp: new Date().toISOString(),
          },
        ]);
        setStreamingMessage('');
        setProgressSteps([]);
        setIsStreaming(false);
        onMessageSent();
      }
    } catch (error) {
      console.error('Error sending message:', error);
      setMessages(prev => [
        ...prev,
        {
          role: 'assistant',
          content: `Error: Failed to send message. ${error.message}`,
          timestamp: new Date().toISOString(),
          isError: true,
        },
      ]);
      setIsStreaming(false);
      setStreamingMessage('');
      setProgressSteps([]);
    }
  };

  const handleFollowUpClick = (question) => {
    if (!isStreaming) {
      handleSendMessage(question);
    }
  };

  if (loading) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto mb-2"></div>
          <p className="text-muted">Loading conversation...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 flex flex-col min-h-0 bg-light-50">
      <MessageList 
        messages={messages}
        streamingMessage={streamingMessage}
        isStreaming={isStreaming}
        progressSteps={progressSteps}
        onOpenReadingPanel={onOpenReadingPanel}
        onFollowUpClick={handleFollowUpClick}
      />
      <MessageInput
        onSendMessage={handleSendMessage}
        disabled={isStreaming}
        placeholder={isStreaming ? 'AI is responding...' : 'Ask a medical question...'}
      />
    </div>
  );
};

export default ChatInterface;
