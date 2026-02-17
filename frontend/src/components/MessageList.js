import React, { useRef, useEffect } from 'react';
import { User, Bot, CheckCircle, Loader, AlertCircle, ExternalLink, ShieldCheck, ShieldAlert } from 'lucide-react';

const MessageList = ({ messages, streamingMessage, isStreaming, progressSteps = [], onOpenReadingPanel, onFollowUpClick }) => {
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, streamingMessage]);

  const formatTimestamp = (timestamp) => {
    if (!timestamp) return '';
    const date = new Date(timestamp);
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  /**
   * Clean raw response content — strip prompt artifacts, evidence dumps, and metadata.
   */
  const cleanContent = (content) => {
    if (!content) return '';
    let cleaned = content;

    cleaned = cleaned.replace(/\[DRAFT[^\]]*\]\s*/gi, '');

    const sectionHeaders = [
      'CORE MISSION', 'SOURCE CITATION REQUIREMENTS', 'RESPONSE FORMATTING REQUIREMENTS',
      'CORE SAFETY CONSTRAINTS', 'RESPONSE PATTERNS BY QUERY TYPE', 'QUALITY STANDARDS',
      'PROHIBITED', 'REQUIRED LANGUAGE PATTERNS', 'CLINICAL QUESTION', 'INTENT',
      'AVAILABLE EVIDENCE', 'CRITICAL CONSTRAINTS', 'TASK', 'DRAFT ANSWER'
    ];
    for (const header of sectionHeaders) {
      const re = new RegExp(`${header}:[\\s\\S]*?(?=\\n\\n[A-Z*#]|\\n\\n\\*\\*|$)`, 'gi');
      cleaned = cleaned.replace(re, '');
    }

    const instructionPatterns = [
      /^.*You are MedGemma.*$/gim,
      /^.*clinical information synthesis engine.*$/gim,
      /^.*NO SPECIFIC DIAGNOSIS.*$/gim,
      /^.*Do not diagnose individual patients.*$/gim,
      /^.*Base information on provided evidence.*$/gim,
      /^.*Include appropriate warnings.*$/gim,
      /^.*Generate comprehensive.*evidence-based.*$/gim,
      /^.*Support clinician decision-making.*$/gim,
      /^.*Cite sources when evidence.*$/gim,
      /^.*Use "may", "can", "typically".*$/gim,
      /^.*EVIDENCE-BASED:.*$/gim,
      /^.*SAFETY DISCLAIMERS:.*$/gim,
      /^.*QUALIFIED LANGUAGE:.*$/gim,
      /^.*Vague, single-sentence responses.*$/gim,
      /^.*Absolute statements without clinical context.*$/gim,
      /^.*Patient-specific treatment decisions.*$/gim,
      /^.*Diagnostic conclusions for individual.*$/gim,
      /^.*\[Drug\/Topic\].*Clinical Considerations.*$/gim,
      /^.*\[Specific risk \d\].*$/gim,
      /^.*\[Practical guidance \d\].*$/gim,
      /^.*\[Key safety disclaimer\].*$/gim,
      /^.*\[Key clinical concept\].*$/gim,
      /^.*\[Core concept\].*$/gim,
      /^.*\[Topic\].*Key Information.*$/gim,
      /^.*\[Drug\/Treatment\].*and.*\[Condition\].*$/gim,
      /^.*Example structure for medication.*$/gim,
      /^.*professional supervision recommended.*what.*$/gim,
      /^.*requires clinical evaluation.*consult healthcare provider.*$/gim,
      /^.*individual factors may vary.*$/gim,
      /^.*Responses should be 200-800 characters.*$/gim,
      /^.*Organize information logically.*$/gim,
      /^.*Balance detail with readability.*$/gim,
      /^.*Maintain professional tone.*$/gim,
      /^.*Balance thoroughness with.*clinical caution.*$/gim,
      /^.*Provide structured.*easy-to-read responses.*$/gim,
      /^.*Generate a constrained draft answer.*$/gim,
      /^.*Acknowledge what is not covered.*$/gim,
      /^.*never final clinical guidance.*$/gim,
      /^.*Do NOT provide diagnosis.*$/gim,
      /^.*Do NOT provide treatment recommendations.*$/gim,
      /^.*Do NOT make assumptions beyond.*$/gim,
      /^.*MUST acknowledge uncertainty.*$/gim,
      /^.*Use ONLY conditional language.*$/gim,
      /^.*This is a DRAFT answer.*$/gim,
    ];
    for (const pattern of instructionPatterns) {
      cleaned = cleaned.replace(pattern, '');
    }

    cleaned = cleaned.replace(/^\d+\.\s*\[Summary\].*(?:\n|$)/gim, '');
    cleaned = cleaned.replace(/\[Source:[^\]]*\]/gi, '');
    cleaned = cleaned.replace(/\[Summary\]\s*/gi, '');
    cleaned = cleaned.replace(/\s*\[\.{2,}\]\s*/g, ' ');
    cleaned = cleaned.replace(/\s*\[…\]\s*/g, ' ');
    cleaned = cleaned.replace(/\[(\d+(?:[,;\-–]\d+)*)\]/g, '');
    cleaned = cleaned.replace(/^INFO:.*$/gm, '');
    cleaned = cleaned.replace(/^WARNING:.*$/gm, '');
    cleaned = cleaned.replace(/^ERROR:.*$/gm, '');
    cleaned = cleaned.replace(/\n{3,}/g, '\n\n');
    cleaned = cleaned.replace(/[^\S\n]+/g, ' ');

    return cleaned.trim();
  };

  /**
   * Render structured text with proper formatting.
   */
  const renderStructuredText = (rawContent) => {
    const content = cleanContent(rawContent);
    if (!content) return <p className="text-slate-400 italic text-sm">No response content.</p>;

    const lines = content.split('\n');

    // Parse lines into tokens
    const tokens = [];
    for (let i = 0; i < lines.length; i++) {
      const trimmed = lines[i].trim();
      if (!trimmed) continue;

      const headerMatch = trimmed.match(/^(#{1,4})\s+(.+)/);
      if (headerMatch) {
        tokens.push({ type: 'heading', level: headerMatch[1].length, text: headerMatch[2].replace(/\*\*/g, ''), idx: i });
        continue;
      }

      const boldLineMatch = trimmed.match(/^\*\*(.+?)\*\*:?\s*$/);
      if (boldLineMatch && trimmed.length < 120) {
        const isTitle = /overview|summary/i.test(boldLineMatch[1]) || tokens.length === 0;
        tokens.push({ type: 'heading', level: isTitle ? 1 : 2, text: boldLineMatch[1], idx: i });
        continue;
      }

      const bulletMatch = trimmed.match(/^[-*•]\s+(.+)/);
      if (bulletMatch) { tokens.push({ type: 'bullet', text: bulletMatch[1], idx: i }); continue; }
      const numMatch = trimmed.match(/^\d+\.\s+(.+)/);
      if (numMatch) { tokens.push({ type: 'bullet', text: numMatch[1], idx: i }); continue; }

      const calloutMatch = trimmed.match(/^\*?\*?(Important|Note|Critical|Warning|Disclaimer)\*?\*?:\s*(.+)/i);
      if (calloutMatch) {
        tokens.push({ type: 'callout', label: calloutMatch[1], text: calloutMatch[2], idx: i });
        continue;
      }

      tokens.push({ type: 'para', text: trimmed, idx: i });
    }

    // Group tokens into sections
    const sections = [];
    let current = { heading: null, children: [] };

    for (const tok of tokens) {
      if (tok.type === 'heading') {
        if (current.heading || current.children.length > 0) {
          sections.push(current);
        }
        current = { heading: tok, children: [] };
      } else {
        current.children.push(tok);
      }
    }
    if (current.heading || current.children.length > 0) sections.push(current);

    // Render sections
    return sections.map((section, sIdx) => {
      const children = [];
      let bulletBuf = [];
      let bulletKey = 0;

      const flushBullets = () => {
        if (bulletBuf.length > 0) {
          bulletBuf.forEach((b, j) => {
            children.push(
              <div key={`bl-${sIdx}-${bulletKey}-${j}`} className="flex items-start gap-2.5 text-sm text-slate-700 mb-2">
                <span className="mt-1.5 h-1.5 w-1.5 rounded-full bg-teal-400 flex-shrink-0" />
                <span className="leading-relaxed">{renderInlineFormatting(b)}</span>
              </div>
            );
          });
          bulletKey++;
          bulletBuf = [];
        }
      };

      for (const tok of section.children) {
        if (tok.type === 'bullet') {
          bulletBuf.push(tok.text);
          continue;
        }
        flushBullets();

        if (tok.type === 'callout') {
          const ct = tok.label.toLowerCase();
          const color = ct === 'disclaimer'
            ? 'bg-slate-50 border-slate-300 text-slate-600'
            : ct === 'critical' || ct === 'warning'
              ? 'bg-red-50 border-red-300 text-red-800'
              : 'bg-amber-50 border-amber-300 text-amber-800';
          children.push(
            <div key={`co-${tok.idx}`} className={`${color} border-l-4 px-3 py-2 rounded-r text-sm mt-2`}>
              <span className="font-semibold">{tok.label}:</span>{' '}
              {renderInlineFormatting(tok.text)}
            </div>
          );
        } else {
          children.push(
            <p key={`p-${tok.idx}`} className="text-sm text-slate-700 leading-relaxed">
              {renderInlineFormatting(tok.text)}
            </p>
          );
        }
      }
      flushBullets();

      const isTitle = section.heading && section.heading.level <= 1;

      return (
        <div
          key={`sec-${sIdx}`}
          className={
            isTitle
              ? 'mb-3'
              : 'bg-slate-50/70 rounded-xl border border-slate-100 p-4 mb-3'
          }
        >
          {section.heading && (
            isTitle ? (
              <h3 className="text-[15px] font-bold text-slate-900 mb-2 pb-1.5 border-b-2 border-teal-200">
                {section.heading.text}
              </h3>
            ) : (
              <h4 className="text-sm font-semibold text-teal-700 mb-2 flex items-center gap-1.5">
                <span className="w-1 h-4 rounded-full bg-teal-400 flex-shrink-0" />
                {section.heading.text}
              </h4>
            )
          )}
          <div className="space-y-2">{children}</div>
        </div>
      );
    });
  };

  /**
   * Handle inline formatting: **bold**, *italic*
   */
  const renderInlineFormatting = (text) => {
    if (!text) return null;
    const parts = text.split(/(\*\*[^*]+\*\*)/g);
    return parts.map((part, i) => {
      const boldMatch = part.match(/^\*\*(.+)\*\*$/);
      if (boldMatch) {
        return <strong key={i} className="font-semibold text-slate-900">{boldMatch[1]}</strong>;
      }
      return <span key={i}>{part}</span>;
    });
  };

  return (
    <div className="flex-1 min-h-0 overflow-y-scroll custom-scrollbar p-4 space-y-4 bg-background">
      {/* Empty state */}
      {messages.length === 0 && !isStreaming && (
        <div className="flex flex-col items-center justify-center h-full text-center px-4">
          <div className="w-20 h-20 bg-gradient-to-br from-teal-100 to-teal-50 rounded-2xl flex items-center justify-center mb-5 shadow-sm">
            <ShieldCheck className="w-10 h-10 text-teal-600" strokeWidth={1.5} />
          </div>
          <h2 className="text-2xl font-bold text-slate-900 mb-1.5 tracking-tight">
            SafeToSay
          </h2>
          <p className="text-slate-500 max-w-md text-sm leading-relaxed mb-1">
            Evidence-based medical knowledge with multi-layered safety evaluation.
          </p>
          <p className="text-xs text-slate-400 mb-8">
            Every response passes through scope classification, knowledge boundary analysis, and safety gates.
          </p>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5 max-w-lg w-full">
            <ExamplePrompt text="What are the symptoms of type 2 diabetes?" onClick={onFollowUpClick} />
            <ExamplePrompt text="How do beta-blockers work?" onClick={onFollowUpClick} />
            <ExamplePrompt text="Chest pain differential diagnosis" onClick={onFollowUpClick} />
            <ExamplePrompt text="What is amoxicillin used for?" onClick={onFollowUpClick} />
          </div>
        </div>
      )}

      {/* Messages */}
      {messages.map((message, index) => (
        <MessageBubble
          key={index}
          message={message}
          formatTimestamp={formatTimestamp}
          renderStructuredText={renderStructuredText}
          onFollowUpClick={onFollowUpClick}
          isLatest={index === messages.length - 1}
        />
      ))}

      {/* Progress steps during streaming */}
      {progressSteps.length > 0 && isStreaming && (
        <div className="flex items-start space-x-3 mb-4 message-animate">
          <div className="flex-shrink-0 w-8 h-8 rounded-full bg-teal-100 flex items-center justify-center">
            <Bot className="w-4 h-4 text-teal-600" />
          </div>
          <div className="flex-1 max-w-3xl">
            <div className="bg-teal-50 rounded-2xl rounded-tl-sm px-4 py-3 shadow-sm border border-teal-100">
              <div className="space-y-2.5">
                <div className="text-xs font-semibold text-teal-800 uppercase tracking-wider mb-1">Processing Pipeline</div>
                {progressSteps.map((step, index) => (
                  <ProgressStep key={step.step} step={step} index={index} />
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Streaming message */}
      {isStreaming && (
        <div className="flex items-start space-x-3 message-animate">
          <div className="flex-shrink-0 w-8 h-8 rounded-full bg-teal-100 flex items-center justify-center">
            <Bot className="w-4 h-4 text-teal-600" />
          </div>
          <div className="flex-1 max-w-3xl">
            <div className="bg-white rounded-2xl rounded-tl-sm px-4 py-3 shadow-sm border border-slate-200">
              {streamingMessage ? (
                <div className="text-slate-800">
                  {renderStructuredText(streamingMessage)}
                  <span className="inline-block w-1.5 h-4 bg-teal-500 ml-1 rounded-sm typing-indicator" />
                </div>
              ) : (
                <div className="flex items-center space-x-2 text-slate-400">
                  <div className="flex space-x-1">
                    <span className="w-2 h-2 bg-teal-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                    <span className="w-2 h-2 bg-teal-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                    <span className="w-2 h-2 bg-teal-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                  </div>
                  <span className="text-sm">Thinking...</span>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      <div ref={bottomRef} />
    </div>
  );
};

const MessageBubble = ({ message, formatTimestamp, renderStructuredText, onFollowUpClick, isLatest }) => {
  const isUser = message.role === 'user';
  const isError = message.isError;
  const isRefusal = message.isRefusal;
  const sources = message.sources || [];

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} message-animate`}>
      <div className={`flex items-start gap-3 max-w-[80%] ${isUser ? 'flex-row-reverse' : 'flex-row'}`}>
        {/* Avatar */}
        <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${isUser
            ? 'bg-gradient-to-br from-teal-600 to-teal-700'
            : isError
              ? 'bg-red-100'
              : isRefusal
                ? 'bg-amber-100'
                : 'bg-teal-100'
          }`}>
          {isUser ? (
            <User className="w-4 h-4 text-white" />
          ) : isRefusal ? (
            <ShieldAlert className="w-4 h-4 text-amber-600" />
          ) : (
            <Bot className={`w-4 h-4 ${isError ? 'text-red-500' : 'text-teal-600'}`} />
          )}
        </div>

        {/* Message content */}
        <div className="flex flex-col">
          <div className={`rounded-2xl px-4 py-3 shadow-sm relative ${isUser
              ? 'bg-gradient-to-br from-teal-600 to-teal-700 text-white rounded-tr-sm'
              : isError
                ? 'bg-red-50 text-red-700 rounded-tl-sm border border-red-200'
                : isRefusal
                  ? 'bg-amber-50 text-amber-800 rounded-tl-sm border border-amber-200'
                  : 'bg-white text-slate-800 rounded-tl-sm border border-slate-200'
            }`}>
            {isUser ? (
              <p className="whitespace-pre-wrap text-sm">{message.content}</p>
            ) : (
              <div className="bot-message-content">{renderStructuredText(message.content)}</div>
            )}

            {/* Sources / Citations inside bubble */}
            {!isUser && sources.length > 0 && (
              <div className="mt-4 pt-3 border-t border-slate-100">
                <h5 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2 flex items-center gap-1">
                  <ExternalLink className="w-3 h-3" />
                  Sources & Citations
                </h5>
                <div className="space-y-1.5">
                  {sources.map((source, idx) => (
                    <a
                      key={idx}
                      href={source.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="flex items-start gap-2 p-2 rounded-lg bg-slate-50 hover:bg-teal-50 border border-slate-100 hover:border-teal-200 transition-colors group"
                    >
                      <span className="flex-shrink-0 w-5 h-5 rounded-full bg-teal-100 text-teal-700 text-xs font-bold flex items-center justify-center mt-0.5">
                        {idx + 1}
                      </span>
                      <div className="min-w-0 flex-1">
                        <p className="text-xs font-medium text-slate-700 group-hover:text-teal-700 truncate">
                          {source.title}
                        </p>
                        {source.snippet && (
                          <p className="text-xs text-slate-400 mt-0.5 line-clamp-2">
                            {source.snippet}
                          </p>
                        )}
                      </div>
                      <ExternalLink className="w-3 h-3 text-slate-300 group-hover:text-teal-500 flex-shrink-0 mt-1" />
                    </a>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Timestamp */}
          {message.timestamp && (
            <p className={`text-[11px] text-slate-400 mt-1 ${isUser ? 'text-right' : 'text-left'}`}>
              {formatTimestamp(message.timestamp)}
            </p>
          )}
        </div>
      </div>
    </div>
  );
};

const ExamplePrompt = ({ text, onClick }) => (
  <div
    className="bg-white border border-slate-200 rounded-xl p-3 text-sm text-slate-500 hover:border-teal-300 hover:text-teal-700 hover:bg-teal-50/50 cursor-pointer transition-all group"
    onClick={() => onClick && onClick(text)}
    role="button"
    tabIndex={0}
    onKeyDown={(e) => e.key === 'Enter' && onClick && onClick(text)}
  >
    <span className="group-hover:translate-x-0.5 inline-block transition-transform">{text}</span>
  </div>
);

const ProgressStep = ({ step, index }) => {
  const getStatusIcon = () => {
    switch (step.status) {
      case 'completed':
        return <CheckCircle className="w-4 h-4 text-teal-500" />;
      case 'running':
        return <Loader className="w-4 h-4 text-teal-600 animate-spin" />;
      case 'error':
        return <AlertCircle className="w-4 h-4 text-red-500" />;
      default:
        return <div className="w-4 h-4 border-2 border-slate-300 rounded-full" />;
    }
  };

  const getStatusColor = () => {
    switch (step.status) {
      case 'completed':
        return 'text-teal-700';
      case 'running':
        return 'text-teal-800';
      case 'error':
        return 'text-red-700';
      default:
        return 'text-slate-500';
    }
  };

  return (
    <div className="flex items-start space-x-2.5 progress-animate">
      <div className="mt-0.5">
        {getStatusIcon()}
      </div>
      <div className="flex-1 min-w-0">
        <div className={`text-sm font-medium ${getStatusColor()}`}>
          {step.title}
        </div>
        <div className="text-xs text-slate-500 mt-0.5">
          {step.description}
        </div>
        {step.result && step.status === 'completed' && (
          <div className="text-xs text-slate-400 mt-0.5 font-mono">
            {typeof step.result === 'object' ? JSON.stringify(step.result, null, 2) : step.result}
          </div>
        )}
      </div>
    </div>
  );
};

export default MessageList;
