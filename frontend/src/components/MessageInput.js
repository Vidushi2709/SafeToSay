import React, { useState, useRef, useEffect } from 'react';
import { Send } from 'lucide-react';

const MessageInput = ({ onSendMessage, disabled, placeholder }) => {
  const [message, setMessage] = useState('');
  const textareaRef = useRef(null);

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 200)}px`;
    }
  }, [message]);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (message.trim() && !disabled) {
      onSendMessage(message.trim());
      setMessage('');
      if (textareaRef.current) {
        textareaRef.current.style.height = 'auto';
      }
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  return (
    <div className="border-t border-slate-200 bg-white px-4 py-3">
      <form onSubmit={handleSubmit} className="max-w-4xl mx-auto">
        <div className="flex items-end space-x-2 bg-slate-50 rounded-2xl border border-slate-200 focus-within:border-teal-400 focus-within:ring-2 focus-within:ring-teal-100 transition-all px-4 py-2">
          {/* Text input */}
          <textarea
            ref={textareaRef}
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={placeholder || 'Ask a medical question...'}
            disabled={disabled}
            rows={1}
            className="flex-1 resize-none bg-transparent py-1.5 text-sm text-slate-800 placeholder-slate-400 focus:outline-none disabled:opacity-50 leading-relaxed"
            style={{ maxHeight: '200px' }}
          />

          {/* Send button */}
          <button
            type="submit"
            disabled={!message.trim() || disabled}
            className={`flex-shrink-0 p-2.5 rounded-xl transition-all ${message.trim() && !disabled
                ? 'bg-gradient-to-r from-teal-600 to-teal-700 text-white shadow-sm hover:shadow-md hover:from-teal-700 hover:to-teal-800'
                : 'bg-slate-100 text-slate-300 cursor-not-allowed'
              }`}
            title="Send message"
          >
            <Send className="w-4 h-4" />
          </button>
        </div>

        {/* Helper text */}
        <p className="text-[11px] text-slate-400 text-center mt-1.5">
          Enter to send · Shift+Enter for new line · <span className="text-teal-500">Safety gates active on every query</span>
        </p>
      </form>
    </div>
  );
};

export default MessageInput;
