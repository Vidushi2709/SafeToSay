import React from 'react';
import { Plus, MessageSquare, Trash2, ShieldCheck } from 'lucide-react';

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
    <div className="flex flex-col h-full bg-white">
      {/* Brand header */}
      <div className="p-4 pb-3">
        <div className="flex items-center space-x-2 mb-4">
          <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-teal-500 to-teal-700 flex items-center justify-center">
            <ShieldCheck className="w-4 h-4 text-white" strokeWidth={2.5} />
          </div>
          <span className="text-sm font-bold text-slate-800 tracking-tight">SafeToSay</span>
        </div>
        <button
          onClick={onCreateThread}
          className="w-full flex items-center justify-center space-x-2 bg-gradient-to-r from-teal-600 to-teal-700 hover:from-teal-700 hover:to-teal-800 text-white py-2.5 px-4 rounded-xl transition-all shadow-sm hover:shadow-md font-medium text-sm"
        >
          <Plus className="w-4 h-4" />
          <span>New Conversation</span>
        </button>
      </div>

      {/* Divider */}
      <div className="px-4">
        <div className="border-b border-slate-100" />
      </div>

      {/* Thread List */}
      <div className="flex-1 overflow-y-auto custom-scrollbar px-2 py-2">
        {threads.length === 0 ? (
          <div className="p-6 text-center">
            <div className="w-12 h-12 mx-auto mb-3 rounded-full bg-slate-100 flex items-center justify-center">
              <MessageSquare className="w-5 h-5 text-slate-400" />
            </div>
            <p className="text-sm font-medium text-slate-500">No conversations yet</p>
            <p className="text-xs text-slate-400 mt-1">Start a new chat to begin</p>
          </div>
        ) : (
          <div className="space-y-1">
            {threads.map((thread) => {
              const isActive = currentThread === thread.thread_id;
              return (
                <div
                  key={thread.thread_id}
                  className={`group relative px-3 py-2.5 rounded-xl cursor-pointer transition-all ${isActive
                      ? 'bg-teal-50 thread-item-active'
                      : 'hover:bg-slate-50'
                    }`}
                  onClick={() => onSelectThread(thread.thread_id)}
                >
                  <div className="flex items-start justify-between">
                    <div className="flex-1 min-w-0">
                      <h3 className={`text-sm font-medium truncate ${isActive ? 'text-teal-800' : 'text-slate-700'}`}>
                        {thread.title}
                      </h3>
                      <div className="flex items-center space-x-1.5 mt-0.5">
                        <span className="text-[11px] text-slate-400">
                          {formatDate(thread.updated_at)}
                        </span>
                        <span className="text-[11px] text-slate-300">·</span>
                        <span className="text-[11px] text-slate-400">
                          {thread.message_count} msg{thread.message_count !== 1 ? 's' : ''}
                        </span>
                      </div>
                    </div>

                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        onDeleteThread(thread.thread_id);
                      }}
                      className="opacity-0 group-hover:opacity-100 p-1.5 hover:bg-red-50 rounded-lg transition-all mt-0.5"
                      title="Delete conversation"
                    >
                      <Trash2 className="w-3.5 h-3.5 text-red-400 hover:text-red-500" />
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Footer */}
      <div className="p-3 border-t border-slate-100">
        <p className="text-[10px] text-slate-400 text-center tracking-wide">
          SafeToSay v1.0 · Evaluation-Driven Clinical AI
        </p>
      </div>
    </div>
  );
};

export default ThreadList;
