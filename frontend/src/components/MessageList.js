import React, { useRef, useEffect } from 'react';
import { User, Bot, CheckCircle, Loader, AlertCircle, ExternalLink } from 'lucide-react';

const MessageList = ({ messages, streamingMessage, isStreaming, progressSteps = [], onOpenReadingPanel, onFollowUpClick }) => {
  const bottomRef = useRef(null);

  // Auto-scroll to bottom when messages change or streaming updates
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, streamingMessage]);

  const formatTimestamp = (timestamp) => {
    if (!timestamp) return '';
    const date = new Date(timestamp);
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  /**
   * Clean the raw response content — strip prompt artifacts, evidence dumps,
   * DRAFT markers, server logs, and other metadata that shouldn't be shown.
   */
  const cleanContent = (content) => {
    if (!content) return '';

    let cleaned = content;

    // Remove [DRAFT ...] markers
    cleaned = cleaned.replace(/\[DRAFT[^\]]*\]\s*/gi, '');

    // ── System prompt section headers & blocks ──
    const sectionHeaders = [
      'CORE MISSION', 'SOURCE CITATION REQUIREMENTS', 'RESPONSE FORMATTING REQUIREMENTS',
      'CORE SAFETY CONSTRAINTS', 'RESPONSE PATTERNS BY QUERY TYPE', 'QUALITY STANDARDS',
      'PROHIBITED', 'REQUIRED LANGUAGE PATTERNS', 'CLINICAL QUESTION', 'INTENT',
      'AVAILABLE EVIDENCE', 'CRITICAL CONSTRAINTS', 'TASK', 'DRAFT ANSWER'
    ];
    for (const header of sectionHeaders) {
      // Remove header + content until next double-newline section or end
      const re = new RegExp(`${header}:[\\s\\S]*?(?=\\n\\n[A-Z*#]|\\n\\n\\*\\*|$)`, 'gi');
      cleaned = cleaned.replace(re, '');
    }

    // ── Lines that are clearly prompt instructions, not answer content ──
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

    // Remove numbered evidence blocks like "1. [Summary] ..." or raw evidence lines
    cleaned = cleaned.replace(/^\d+\.\s*\[Summary\].*(?:\n|$)/gim, '');

    // Remove "[Source: ...]" inline tags
    cleaned = cleaned.replace(/\[Source:[^\]]*\]/gi, '');

    // Remove "[Summary]" prefix tags
    cleaned = cleaned.replace(/\[Summary\]\s*/gi, '');

    // Remove [...] and […] truncation markers
    cleaned = cleaned.replace(/\s*\[\.{2,}\]\s*/g, ' ');
    cleaned = cleaned.replace(/\s*\[…\]\s*/g, ' ');

    // Remove bare reference numbers like [1], [2,3], [1-5] but keep [mg/dL] etc.
    cleaned = cleaned.replace(/\[(\d+(?:[,;\-–]\d+)*)\]/g, '');

    // Remove server log lines
    cleaned = cleaned.replace(/^INFO:.*$/gm, '');
    cleaned = cleaned.replace(/^WARNING:.*$/gm, '');
    cleaned = cleaned.replace(/^ERROR:.*$/gm, '');

    // Collapse excessive blank lines
    cleaned = cleaned.replace(/\n{3,}/g, '\n\n');

    // Collapse multiple spaces (but not newlines)
    cleaned = cleaned.replace(/[^\S\n]+/g, ' ');

    return cleaned.trim();
  };

  /**
   * Render structured text with proper formatting for markdown-like content.
   * Groups content under each heading into visually distinct section blocks.
   */
  const renderStructuredText = (rawContent) => {
    const content = cleanContent(rawContent);
    if (!content) return <p className="text-gray-400 italic">No response content.</p>;

    const lines = content.split('\n');

    // ── First pass: parse lines into typed tokens ──
    const tokens = [];
    for (let i = 0; i < lines.length; i++) {
      const trimmed = lines[i].trim();
      if (!trimmed) continue;

      // ## / ### markdown header
      const headerMatch = trimmed.match(/^(#{1,4})\s+(.+)/);
      if (headerMatch) {
        tokens.push({ type: 'heading', level: headerMatch[1].length, text: headerMatch[2].replace(/\*\*/g, ''), idx: i });
        continue;
      }

      // **Bold Header:** or **Bold Header** (short line acts as section title)
      const boldLineMatch = trimmed.match(/^\*\*(.+?)\*\*:?\s*$/);
      if (boldLineMatch && trimmed.length < 120) {
        // Treat "— Clinical Overview" style as level-1, others as level-2
        const isTitle = /overview|summary/i.test(boldLineMatch[1]) || tokens.length === 0;
        tokens.push({ type: 'heading', level: isTitle ? 1 : 2, text: boldLineMatch[1], idx: i });
        continue;
      }

      // Bullet / numbered list item
      const bulletMatch = trimmed.match(/^[-*•]\s+(.+)/);
      if (bulletMatch) { tokens.push({ type: 'bullet', text: bulletMatch[1], idx: i }); continue; }
      const numMatch = trimmed.match(/^\d+\.\s+(.+)/);
      if (numMatch) { tokens.push({ type: 'bullet', text: numMatch[1], idx: i }); continue; }

      // Callout: Important / Note / Disclaimer / Warning / Critical
      const calloutMatch = trimmed.match(/^\*?\*?(Important|Note|Critical|Warning|Disclaimer)\*?\*?:\s*(.+)/i);
      if (calloutMatch) {
        tokens.push({ type: 'callout', label: calloutMatch[1], text: calloutMatch[2], idx: i });
        continue;
      }

      // Regular paragraph
      tokens.push({ type: 'para', text: trimmed, idx: i });
    }

    // ── Second pass: group tokens into sections (heading → body) ──
    const sections = [];  // { heading?, children[] }
    let current = { heading: null, children: [] };

    for (const tok of tokens) {
      if (tok.type === 'heading') {
        // Flush previous section
        if (current.heading || current.children.length > 0) {
          sections.push(current);
        }
        current = { heading: tok, children: [] };
      } else {
        current.children.push(tok);
      }
    }
    if (current.heading || current.children.length > 0) sections.push(current);

    // ── Third pass: render each section as a distinct visual block ──
    return sections.map((section, sIdx) => {
      const children = [];
      let bulletBuf = [];
      let bulletKey = 0;

      const flushBullets = () => {
        if (bulletBuf.length > 0) {
          // Render each bullet as its own spaced paragraph block
          bulletBuf.forEach((b, j) => {
            children.push(
              <div key={`bl-${sIdx}-${bulletKey}-${j}`} className="flex items-start gap-2 text-sm text-gray-700 mb-3">
                <span className="mt-1.5 h-1.5 w-1.5 rounded-full bg-pink-400 flex-shrink-0" />
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
            ? 'bg-gray-50 border-gray-300 text-gray-700'
            : ct === 'critical' || ct === 'warning'
              ? 'bg-red-50 border-red-300 text-red-800'
              : 'bg-amber-50 border-amber-300 text-amber-800';
          children.push(
            <div key={`co-${tok.idx}`} className={`${color} border-l-4 px-3 py-2 rounded-r text-sm`}>
              <span className="font-semibold">{tok.label}:</span>{' '}
              {renderInlineFormatting(tok.text)}
            </div>
          );
        } else {
          // paragraph
          children.push(
            <p key={`p-${tok.idx}`} className="text-sm text-gray-700 leading-relaxed">
              {renderInlineFormatting(tok.text)}
            </p>
          );
        }
      }
      flushBullets();

      // Determine heading style
      const isTitle = section.heading && section.heading.level <= 1;

      return (
        <div
          key={`sec-${sIdx}`}
          className={
            isTitle
              ? 'mb-4'                                          // title — no card, just spacing
              : 'bg-gray-50 rounded-lg border border-gray-100 p-4 mb-3'  // section card
          }
        >
          {section.heading && (
            isTitle ? (
              <h3 className="text-base font-bold text-gray-900 mb-2 pb-1 border-b border-pink-200">
                {section.heading.text}
              </h3>
            ) : (
              <h4 className="text-sm font-semibold text-pink-600 mb-2">
                {section.heading.text}
              </h4>
            )
          )}
          <div className="space-y-3">{children}</div>
        </div>
      );
    });
  };

  /**
   * Handle inline formatting: **bold**, *italic*
   */
  const renderInlineFormatting = (text) => {
    if (!text) return null;
    // Split on **bold** patterns
    const parts = text.split(/(\*\*[^*]+\*\*)/g);
    return parts.map((part, i) => {
      const boldMatch = part.match(/^\*\*(.+)\*\*$/);
      if (boldMatch) {
        return <strong key={i} className="font-semibold text-gray-900">{boldMatch[1]}</strong>;
      }
      return <span key={i}>{part}</span>;
    });
  };

  return (
    <div className="flex-1 min-h-0 overflow-y-scroll custom-scrollbar p-4 space-y-4 bg-background">
      {messages.length === 0 && !isStreaming && (
        <div className="flex flex-col items-center justify-center h-full text-center">
          <div className="w-16 h-16 bg-primary/10 rounded-full flex items-center justify-center mb-4">
            <Bot className="w-8 h-8 text-primary" />
          </div>
          <h2 className="text-xl font-semibold text-dark mb-2">
            Clinical Agent Assistant
          </h2>
          <p className="text-muted max-w-md">
            Ask me any medical questions. I'll provide evidence-based responses
            with citations and clear reasoning.
          </p>
          <div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-3 max-w-lg">
            <ExamplePrompt text="What are the symptoms of type 2 diabetes?" />
            <ExamplePrompt text="How do beta-blockers work?" />
            <ExamplePrompt text="What's the treatment for hypertension?" />
            <ExamplePrompt text="What is amoxicillin used for?" />
          </div>
        </div>
      )}

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
        <div className="flex items-start space-x-3 mb-4">
          <div className="flex-shrink-0 w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center">
            <Bot className="w-4 h-4 text-primary" />
          </div>
          <div className="flex-1 max-w-3xl">
            <div className="bg-blue-50 rounded-2xl rounded-tl-sm px-4 py-3 shadow-sm border border-blue-100">
              <div className="space-y-3">
                <div className="text-sm font-medium text-blue-900 mb-2">Processing your query...</div>
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
        <div className="flex items-start space-x-3">
          <div className="flex-shrink-0 w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center">
            <Bot className="w-4 h-4 text-primary" />
          </div>
          <div className="flex-1 max-w-3xl">
            <div className="bg-light rounded-2xl rounded-tl-sm px-4 py-3 shadow-sm border border-gray-100">
              {streamingMessage ? (
                <div className="text-dark">
                  {renderStructuredText(streamingMessage)}
                  <span className="inline-block w-2 h-4 bg-primary ml-1 typing-indicator" />
                </div>
              ) : (
                <div className="flex items-center space-x-2 text-muted">
                  <div className="flex space-x-1">
                    <span className="w-2 h-2 bg-primary rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                    <span className="w-2 h-2 bg-primary rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                    <span className="w-2 h-2 bg-primary rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                  </div>
                  <span className="text-sm">Thinking...</span>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Scroll anchor */}
      <div ref={bottomRef} />
    </div>
  );
};

const MessageBubble = ({ message, formatTimestamp, renderStructuredText, onFollowUpClick, isLatest }) => {
  const isUser = message.role === 'user';
  const isError = message.isError;
  const sources = message.sources || [];

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div className={`flex items-start gap-3 max-w-[80%] ${isUser ? 'flex-row-reverse' : 'flex-row'}`}>
        {/* Avatar */}
        <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${
          isUser ? 'bg-primary' : isError ? 'bg-red-100' : 'bg-primary/10'
        }`}>
          {isUser ? (
            <User className="w-4 h-4 text-white" />
          ) : (
            <Bot className={`w-4 h-4 ${isError ? 'text-red-500' : 'text-primary'}`} />
          )}
        </div>

        {/* Message content */}
        <div className="flex flex-col">
          <div className={`rounded-2xl px-4 py-3 shadow-sm relative ${
            isUser 
              ? 'bg-primary text-white rounded-tr-sm' 
              : isError
                ? 'bg-red-50 text-red-700 rounded-tl-sm border border-red-200'
                : 'bg-white text-dark rounded-tl-sm border border-gray-200'
          }`}>
            {isUser ? (
              <p className="whitespace-pre-wrap text-sm">{message.content}</p>
            ) : (
              <div className="bot-message-content">{renderStructuredText(message.content)}</div>
            )}

            {/* Sources / Citations inside bubble */}
            {!isUser && sources.length > 0 && (
              <div className="mt-4 pt-3 border-t border-gray-100">
                <h5 className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2 flex items-center gap-1">
                  <ExternalLink className="w-3 h-3" />
                  Sources & Citations
                </h5>
                <div className="space-y-2">
                  {sources.map((source, idx) => (
                    <a
                      key={idx}
                      href={source.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="flex items-start gap-2 p-2 rounded-lg bg-gray-50 hover:bg-pink-50 border border-gray-100 hover:border-pink-200 transition-colors group"
                    >
                      <span className="flex-shrink-0 w-5 h-5 rounded-full bg-pink-100 text-pink-600 text-xs font-bold flex items-center justify-center mt-0.5">
                        {idx + 1}
                      </span>
                      <div className="min-w-0 flex-1">
                        <p className="text-xs font-medium text-gray-800 group-hover:text-pink-700 truncate">
                          {source.title}
                        </p>
                        {source.snippet && (
                          <p className="text-xs text-gray-500 mt-0.5 line-clamp-2">
                            {source.snippet}
                          </p>
                        )}
                        <p className="text-xs text-pink-500 mt-0.5 truncate opacity-0 group-hover:opacity-100 transition-opacity">
                          {source.url}
                        </p>
                      </div>
                      <ExternalLink className="w-3 h-3 text-gray-400 group-hover:text-pink-500 flex-shrink-0 mt-1" />
                    </a>
                  ))}
                </div>
              </div>
            )}
          </div>
          
          {/* Timestamp */}
          {message.timestamp && (
            <p className={`text-xs text-muted mt-1 ${isUser ? 'text-right' : 'text-left'}`}>
              {formatTimestamp(message.timestamp)}
            </p>
          )}
        </div>
      </div>
    </div>
  );
};

const ExamplePrompt = ({ text }) => (
  <div className="bg-white border border-gray-200 rounded-lg p-3 text-sm text-muted hover:border-primary hover:text-primary cursor-pointer transition-colors">
    {text}
  </div>
);

const ProgressStep = ({ step, index }) => {
  const getStatusIcon = () => {
    switch (step.status) {
      case 'completed':
        return <CheckCircle className="w-4 h-4 text-green-500" />;
      case 'running':
        return <Loader className="w-4 h-4 text-blue-500 animate-spin" />;
      case 'error':
        return <AlertCircle className="w-4 h-4 text-red-500" />;
      default:
        return <div className="w-4 h-4 border-2 border-gray-300 rounded-full" />;
    }
  };

  const getStatusColor = () => {
    switch (step.status) {
      case 'completed':
        return 'text-green-700';
      case 'running':
        return 'text-blue-700';
      case 'error':
        return 'text-red-700';
      default:
        return 'text-gray-600';
    }
  };

  return (
    <div className="flex items-start space-x-3">
      <div className="mt-0.5">
        {getStatusIcon()}
      </div>
      <div className="flex-1 min-w-0">
        <div className={`text-sm font-medium ${getStatusColor()}`}>
          {step.title}
        </div>
        <div className="text-xs text-gray-600 mt-1">
          {step.description}
        </div>
        {step.result && step.status === 'completed' && (
          <div className="text-xs text-gray-500 mt-1">
            {typeof step.result === 'object' ? JSON.stringify(step.result, null, 2) : step.result}
          </div>
        )}
      </div>
    </div>
  );
};

export default MessageList;
