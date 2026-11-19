import React, { useState, useRef, useEffect } from 'react';
import { Send, Sparkles, Building, Compass, AlertCircle, Square } from 'lucide-react';
import { ChatMessage, ChatResponse } from '../types';
import { MessageBubble } from './MessageBubble';
import { LoadingDots } from './LoadingDots';

interface ChatContainerProps {
  messages: ChatMessage[];
  responses: Record<number, ChatResponse>;
  isLoading: boolean;
  isStreaming?: boolean;
  streamingStatus?: string | null;
  error: string | null;
  onSendMessage: (content: string) => void;
  onStopGenerating?: () => void;
  onRetry?: (messageIndex: number) => void;
  isRetrying?: boolean;
  sourceFilter: 'all' | 'darglobal' | 'wasalt';
}

const SUGGESTED_QUERIES = [
  {
    title: 'Branded Developments',
    query: 'What branded residences and luxury projects does DarGlobal offer in Dubai and Oman?',
    icon: Building,
  },
  {
    title: 'Investment & Heritage',
    query: 'Can you provide DarGlobal company details, listing information, and key markets?',
    icon: Compass,
  },
  {
    title: 'Domestic Properties',
    query: 'What types of residential properties and locations are listed on Wasalt in Saudi Arabia?',
    icon: Sparkles,
  },
];

export const ChatContainer: React.FC<ChatContainerProps> = ({
  messages,
  responses,
  isLoading,
  isStreaming = false,
  streamingStatus = null,
  error,
  onSendMessage,
  onStopGenerating,
  onRetry,
  isRetrying = false,
  sourceFilter,
}) => {
  const [input, setInput] = useState('');
  const messageRefs = useRef<(HTMLDivElement | null)[]>([]);
  const lastScrolledIndex = useRef<number>(-1);
  const loadingRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (messages.length === 0) {
      lastScrolledIndex.current = -1;
      return;
    }

    const lastIdx = messages.length - 1;
    if (lastIdx > lastScrolledIndex.current) {
      lastScrolledIndex.current = lastIdx;
      const targetEl = messageRefs.current[lastIdx];
      if (targetEl && typeof targetEl.scrollIntoView === 'function') {
        const isUser = messages[lastIdx].role === 'user';
        targetEl.scrollIntoView({
          behavior: 'smooth',
          block: isUser ? 'nearest' : 'start',
        });
      }
    }
  }, [messages]);

  useEffect(() => {
    if (isLoading && loadingRef.current && typeof loadingRef.current.scrollIntoView === 'function') {
      loadingRef.current.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  }, [isLoading]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const trimmed = input.trim();
    if (!trimmed || isLoading) return;
    onSendMessage(trimmed);
    setInput('');
  };

  const isAwaitingFirstToken =
    isLoading &&
    (messages.length === 0 || messages[messages.length - 1].role === 'user');

  return (
    <div className="flex-1 min-h-0 flex flex-col max-w-4xl w-full mx-auto px-4 lg:px-6 relative overflow-hidden">
      {/* Messages Scroll Area */}
      <div className="flex-1 min-h-0 overflow-y-auto pr-1 space-y-4 pb-4 scrollbar-thin">
        {messages.length === 0 ? (
          <div className="h-full flex flex-col items-center justify-center text-center max-w-xl mx-auto px-4 py-8">
            <div className="w-10 h-10 rounded-xl bg-surface-subtle border border-surface-border flex items-center justify-center mb-4">
              <Sparkles className="w-5 h-5 text-luxury-400" />
            </div>
            <h2 className="text-xl font-medium text-white mb-2 tracking-tight">
              Real Estate Intelligence Assistant
            </h2>
            <p className="text-sm text-[#A1A1AA] mb-8 leading-relaxed">
              Inquire about DarGlobal's international luxury portfolio, company information,
              investment details, and Wasalt's domestic Saudi property listings.
            </p>

            <div className="w-full grid grid-cols-1 sm:grid-cols-3 gap-2.5 text-left">
              {SUGGESTED_QUERIES.map((item, idx) => {
                const Icon = item.icon;
                return (
                  <button
                    key={idx}
                    type="button"
                    onClick={() => onSendMessage(item.query)}
                    className="p-3.5 rounded-xl bg-surface-card hover:bg-surface-subtle border border-surface-border hover:border-[#3F3F46] text-left transition-all group"
                  >
                    <Icon className="w-4 h-4 text-[#A1A1AA] group-hover:text-luxury-400 mb-2 transition-colors" />
                    <p className="text-xs font-medium text-[#F5F5F5] group-hover:text-white mb-1">
                      {item.title}
                    </p>
                    <p className="text-[11px] text-[#A1A1AA] line-clamp-2 leading-relaxed">
                      {item.query}
                    </p>
                  </button>
                );
              })}
            </div>
          </div>
        ) : (
          messages.map((msg, index) => {
            const resp = responses[index];
            const isLastStreaming =
              isStreaming &&
              index === messages.length - 1 &&
              msg.role === 'assistant';

            return (
              <div
                key={index}
                ref={(el) => {
                  messageRefs.current[index] = el;
                }}
              >
                <MessageBubble
                  message={msg}
                  citations={resp?.citations}
                  propertyCards={resp?.property_cards}
                  grounded={resp?.grounded}
                  dataMode={resp?.data_mode}
                  provenanceWarning={resp?.provenance_warning}
                  errorCode={resp?.error_code}
                  retryAfter={resp?.retry_after}
                  disclaimer={resp?.disclaimer}
                  isStreaming={isLastStreaming}
                  model={resp?.model}
                  onRetry={() => onRetry?.(index)}
                  isRetrying={isRetrying}
                />
              </div>
            );
          })
        )}

        {isAwaitingFirstToken && (
          <div ref={loadingRef} className="flex items-center space-x-3 my-4">
            <div className="w-7 h-7 rounded-lg bg-surface-subtle border border-surface-border flex items-center justify-center">
              <Sparkles className="w-3.5 h-3.5 text-luxury-400 animate-pulse" />
            </div>
            <div className="flex items-center space-x-2.5 bg-surface-card border border-surface-border px-3.5 py-2 rounded-xl text-xs text-[#A1A1AA]">
              <LoadingDots />
              <span className="text-[#A1A1AA] text-xs">
                {streamingStatus || 'Searching the knowledge base...'}
              </span>
            </div>
          </div>
        )}

        {error && messages.length === 0 && (
          <div className="p-3 rounded-lg bg-red-500/10 border border-red-500/20 flex items-center space-x-2.5 text-red-400 text-xs my-4">
            <AlertCircle className="w-4 h-4 text-red-400 flex-shrink-0" />
            <span>{error}</span>
          </div>
        )}
      </div>

      {/* Pinned Fixed Input Form Bar */}
      <div className="flex-shrink-0 pt-2 pb-3 sm:pb-4 border-t border-surface-border/40 bg-surface-base/95 backdrop-blur-sm z-10 w-full pb-[max(0.75rem,env(safe-area-inset-bottom))]">
        <form
          onSubmit={handleSubmit}
          className="relative bg-surface-card border border-surface-border rounded-xl p-1.5 focus-within:border-[#52525B] shadow-lg transition-all"
        >
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder={`Ask anything about ${
              sourceFilter === 'darglobal'
                ? 'DarGlobal luxury developments...'
                : sourceFilter === 'wasalt'
                ? 'Wasalt properties in Saudi Arabia...'
                : 'DarGlobal luxury projects or Wasalt listings...'
            }`}
            className="w-full bg-transparent px-3 py-2 pr-20 text-sm text-[#F5F5F5] placeholder-[#71717A] focus:outline-none"
            disabled={isLoading && !isStreaming}
          />
          {isStreaming || isLoading ? (
            <button
              type="button"
              onClick={onStopGenerating}
              className="absolute right-2 top-1/2 -translate-y-1/2 px-2.5 py-1.5 rounded-lg bg-red-950/40 hover:bg-red-900/60 border border-red-500/30 text-red-400 hover:text-red-300 flex items-center space-x-1.5 text-xs font-medium transition-all shadow-sm"
              aria-label="Stop generating response"
            >
              <Square className="w-3 h-3 fill-current" />
              <span>Stop</span>
            </button>
          ) : (
            <button
              type="submit"
              disabled={!input.trim() || isLoading}
              className="absolute right-2 top-1/2 -translate-y-1/2 p-2 rounded-lg bg-surface-subtle hover:bg-[#27272A] border border-surface-border disabled:opacity-30 disabled:hover:bg-surface-subtle text-[#F5F5F5] transition-all"
              aria-label="Send message"
            >
              <Send className="w-3.5 h-3.5" />
            </button>
          )}
        </form>
      </div>
    </div>
  );
};

