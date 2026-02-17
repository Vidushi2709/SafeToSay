import React from 'react';
import { Plus, MessageSquare, Trash2 } from 'lucide-react';

const ThreadList = ({
  threads,
  currentThread,
  onSelectThread,
  onCreateThread,
  onDeleteThread,
}) => {
  const formatDate = (dateString) => {
    const date = new Date(dateString);
    const now = new Date();
    const diffTime = Math.abs(now - date);
    const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));
    
    if (diffDays === 1) {
      return 'Today';
    } else if (diffDays === 2) {
      return 'Yesterday';
    } else if (diffDays <= 7) {
      return `${diffDays - 1} days ago`;
    } else {
      return date.toLocaleDateString();
    }
  };

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="p-4 border-b border-light-400">
        <button
          onClick={onCreateThread}
          className="w-full flex items-center justify-center space-x-2 bg-primary hover:bg-primary-dark text-white py-2 px-4 rounded-lg transition-colors shadow-md hover:shadow-lg"
        >
          <Plus className="w-4 h-4" />
          <span>New Chat</span>
        </button>
      </div>

      {/* Thread List */}
      <div className="flex-1 overflow-y-auto custom-scrollbar">
        {threads.length === 0 ? (
          <div className="p-4 text-center text-muted">
            <MessageSquare className="w-12 h-12 mx-auto mb-2 text-light-500" />
            <p>No conversations yet</p>
            <p className="text-sm">Start a new chat to begin</p>
          </div>
        ) : (
          <div className="p-2">
            {threads.map((thread) => (
              <div
                key={thread.thread_id}
                className={`group relative p-3 mb-2 rounded-lg cursor-pointer transition-colors ${
                  currentThread === thread.thread_id
                    ? 'bg-primary-50 border border-primary-200'
                    : 'hover:bg-light border border-transparent'
                }`}
                onClick={() => onSelectThread(thread.thread_id)}
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1 min-w-0">
                    <h3 className="font-medium text-dark truncate">
                      {thread.title}
                    </h3>
                    <div className="flex items-center space-x-2 mt-1">
                      <span className="text-xs text-muted">
                        {formatDate(thread.updated_at)}
                      </span>
                      <span className="text-xs text-muted-light">•</span>
                      <span className="text-xs text-muted">
                        {thread.message_count} messages
                      </span>
                    </div>
                  </div>
                  
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      onDeleteThread(thread.thread_id);
                    }}
                    className="opacity-0 group-hover:opacity-100 p-1 hover:bg-red-100 rounded transition-all"
                    title="Delete conversation"
                  >
                    <Trash2 className="w-3 h-3 text-red-500" />
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
      
      {/* Footer */}
      <div className="p-4 border-t border-light-400 text-center text-xs text-muted">
        Medical AI Assistant v1.0
      </div>
    </div>
  );
};

export default ThreadList;
