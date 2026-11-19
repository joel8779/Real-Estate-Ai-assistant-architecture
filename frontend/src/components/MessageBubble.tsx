import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Sparkles, ShieldCheck, AlertTriangle, AlertCircle, RotateCcw } from 'lucide-react';
import { ChatMessage, Citation, PropertyCardData } from '../types';
import { CitationCard } from './CitationCard';
import { PropertyCard } from './PropertyCard';

interface MessageBubbleProps {
  message: ChatMessage;
  citations?: Citation[];
  propertyCards?: PropertyCardData[];
  grounded?: boolean;
  dataMode?: 'verified' | 'demonstration' | 'conversational';
  provenanceWarning?: string | null;
  errorCode?: string | null;
  retryAfter?: number | null;
  disclaimer?: string;
  isStreaming?: boolean;
  onRetry?: () => void;
  isRetrying?: boolean;
  model?: string;
  provider?: 'openrouter' | 'ollama';
  fallbackActive?: boolean;
  fallbackReason?: 'quota_exhausted' | 'provider_outage';
  fallbackNotice?: string | null;
}

const getModelDisplayName = (modelStr?: string, provider?: string): string => {
  if (!modelStr) return 'Nemotron 3 Ultra';
  const lower = modelStr.toLowerCase();
  if (lower.includes('nemotron')) {
    return 'Nemotron 3 Ultra';
  }
  if (lower.includes('qwen') || provider === 'ollama') {
    return 'Local Fallback: Qwen 2.5 1.5B';
  }
  const shortName = modelStr.split('/').pop() || modelStr;
  return `Fallback: ${shortName}`;
};

export const MessageBubble: React.FC<MessageBubbleProps> = ({
  message,
  citations = [],
  propertyCards = [],
  grounded = true,
  dataMode,
  provenanceWarning,
  errorCode,
  disclaimer,
  isStreaming = false,
  onRetry,
  isRetrying = false,
  model,
  provider,
  fallbackActive,
  fallbackReason,
  fallbackNotice,
}) => {
  const isUser = message.role === 'user';

  if (isUser) {
    return (
      <div className="flex justify-end my-4">
        <div className="max-w-2xl bg-surface-subtle border border-surface-border text-[#F5F5F5] rounded-xl px-4 py-2.5 text-sm shadow-sm">
          <p className="whitespace-pre-wrap leading-relaxed">{message.content}</p>
        </div>
      </div>
    );
  }

  const isFailedGeneration = Boolean(errorCode);

  return (
    <div className="flex justify-start my-6 group">
      <div className="flex space-x-3.5 max-w-3xl w-full">
        {/* Assistant Avatar */}
        <div className="w-7 h-7 rounded-lg bg-surface-subtle border border-surface-border flex items-center justify-center flex-shrink-0 mt-0.5">
          <Sparkles className="w-3.5 h-3.5 text-luxury-400" />
        </div>

        <div className="flex-1 min-w-0 space-y-3.5">
          {/* Header Metadata */}
          <div className="flex items-center justify-between pb-1 text-xs text-[#A1A1AA]">
            <div className="flex items-center space-x-2">
              <span className="font-medium text-[#F5F5F5]">DarGlobal Knowledge Engine</span>
              {model && !isFailedGeneration && dataMode !== 'conversational' && (
                <span
                  data-testid="model-badge"
                  className={`text-[10px] px-1.5 py-0.5 rounded border font-mono ${
                    fallbackActive || (model && !model.toLowerCase().includes('nemotron'))
                      ? 'bg-amber-500/10 border-amber-500/20 text-amber-300'
                      : 'bg-surface-subtle border-surface-border text-[#A1A1AA]'
                  }`}
                  title={
                    fallbackNotice ||
                    (fallbackActive
                      ? `Local fallback active: ${fallbackReason === 'quota_exhausted' ? 'Daily quota limit reached' : 'Provider outage'}`
                      : undefined)
                  }
                >
                  {getModelDisplayName(model, provider)}
                </span>
              )}
            </div>
            {isFailedGeneration ? (
              <span className="flex items-center space-x-1 text-red-400 bg-red-500/10 px-2 py-0.5 rounded text-[11px] border border-red-500/20 font-medium">
                <AlertCircle className="w-3 h-3" />
                <span>Service Notice</span>
              </span>
            ) : dataMode === 'conversational' || (!provenanceWarning && citations.length === 0 && !model) ? null : dataMode === 'verified' ? (
              <span className="flex items-center space-x-1 text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded text-[11px] border border-emerald-500/20">
                <ShieldCheck className="w-3 h-3" />
                <span>{grounded ? 'Verified Public Grounding' : 'Partial Grounding'}</span>
              </span>
            ) : provenanceWarning ? (
              <span className="flex items-center space-x-1 text-amber-400 bg-amber-500/10 px-2 py-0.5 rounded text-[11px] border border-amber-500/20">
                <AlertTriangle className="w-3 h-3" />
                <span>Prototype Dataset</span>
              </span>
            ) : null}
          </div>

          {/* Compact Error Treatment within Assistant Message */}
          {isFailedGeneration ? (
            <div className="p-3.5 rounded-xl bg-red-500/10 border border-red-500/20 text-xs">
              <div className="flex items-start space-x-2.5">
                <AlertCircle className="w-4 h-4 text-red-400 flex-shrink-0 mt-0.5" />
                <div className="flex-1 space-y-2.5">
                  <p className="leading-relaxed text-red-200 text-xs font-normal">
                    {message.content}
                  </p>
                  {onRetry && (
                    <button
                      type="button"
                      onClick={onRetry}
                      disabled={isRetrying || isStreaming}
                      className="inline-flex items-center space-x-1.5 px-2.5 py-1.5 rounded-lg bg-surface-subtle hover:bg-[#27272A] border border-surface-border hover:border-[#3F3F46] text-[#F5F5F5] hover:text-white transition-all text-xs font-medium disabled:opacity-40 shadow-sm"
                    >
                      <RotateCcw className={`w-3 h-3 text-luxury-400 ${isRetrying ? 'animate-spin' : ''}`} />
                      <span>{isRetrying ? 'Retrying...' : 'Try again'}</span>
                    </button>
                  )}
                </div>
              </div>
            </div>
          ) : (
            <>
              {/* Visible Provenance Warning Banner (Only for successful demonstration answers) */}
              {provenanceWarning && (
                <div className="flex items-start space-x-2.5 p-2.5 rounded-lg bg-amber-500/5 border border-amber-500/20 text-amber-300 text-xs">
                  <AlertTriangle className="w-3.5 h-3.5 text-amber-400 flex-shrink-0 mt-0.5" />
                  <div className="flex-1">
                    <span className="font-medium block text-amber-300 mb-0.5 text-[11px]">
                      Research Dataset Notice
                    </span>
                    <p className="leading-relaxed text-[11px] text-amber-200/90">
                      {provenanceWarning}
                    </p>
                  </div>
                </div>
              )}

              {/* Markdown-rendered Answer */}
              <div className="prose prose-invert prose-sm max-w-none text-[#F5F5F5] leading-relaxed text-sm">
                {message.content ? (
                  <ReactMarkdown
                    remarkPlugins={[remarkGfm]}
                    components={{
                      p: ({ children }) => <p className="mb-3 last:mb-0 leading-relaxed text-[#F5F5F5]">{children}</p>,
                      strong: ({ children }) => <strong className="font-semibold text-white">{children}</strong>,
                      em: ({ children }) => <em className="italic text-[#E4E4E7]">{children}</em>,
                      h1: ({ children }) => <h1 className="text-base font-semibold text-white mt-4 mb-2 first:mt-0">{children}</h1>,
                      h2: ({ children }) => <h2 className="text-sm font-semibold text-white mt-3 mb-1.5 first:mt-0">{children}</h2>,
                      h3: ({ children }) => <h3 className="text-xs font-semibold text-[#F5F5F5] mt-2 mb-1 first:mt-0 uppercase tracking-wide">{children}</h3>,
                      ul: ({ children }) => <ul className="list-disc list-outside pl-5 mb-3 space-y-1 text-[#F5F5F5]">{children}</ul>,
                      ol: ({ children }) => <ol className="list-decimal list-outside pl-5 mb-3 space-y-1 text-[#F5F5F5]">{children}</ol>,
                      li: ({ children }) => <li className="leading-relaxed pl-0.5">{children}</li>,
                      blockquote: ({ children }) => (
                        <blockquote className="border-l-2 border-surface-border pl-3 my-2 text-[#A1A1AA] italic">
                          {children}
                        </blockquote>
                      ),
                      a: ({ href, children }) => (
                        <a
                          href={href}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-luxury-400 hover:text-luxury-300 underline underline-offset-2 transition-colors"
                        >
                          {children}
                        </a>
                      ),
                      code: ({ className, children, ...props }) => {
                        const isInline = !className && typeof children === 'string' && !children.includes('\n');
                        if (isInline) {
                          return (
                            <code className="bg-surface-subtle border border-surface-border text-[#D4AF37] px-1.5 py-0.5 rounded text-[11px] font-mono">
                              {children}
                            </code>
                          );
                        }
                        return (
                          <pre className="bg-surface-base border border-surface-border p-3 rounded-lg overflow-x-auto my-2 text-xs font-mono text-[#F5F5F5]">
                            <code className={className} {...props}>
                              {children}
                            </code>
                          </pre>
                        );
                      },
                      table: ({ children }) => (
                        <div className="overflow-x-auto my-3">
                          <table className="min-w-full text-xs text-left border border-surface-border rounded-lg overflow-hidden">
                            {children}
                          </table>
                        </div>
                      ),
                      thead: ({ children }) => (
                        <thead className="bg-surface-subtle text-[#A1A1AA] uppercase border-b border-surface-border">
                          {children}
                        </thead>
                      ),
                      th: ({ children }) => <th className="px-3 py-2 font-medium">{children}</th>,
                      tbody: ({ children }) => (
                        <tbody className="divide-y divide-surface-border text-[#F5F5F5]">
                          {children}
                        </tbody>
                      ),
                      td: ({ children }) => <td className="px-3 py-2">{children}</td>,
                    }}
                  >
                    {message.content}
                  </ReactMarkdown>
                ) : isStreaming ? (
                  <span className="text-xs text-[#A1A1AA] italic">Synthesizing answer...</span>
                ) : null}
                {isStreaming && (
                  <span
                    data-testid="streaming-cursor"
                    className="inline-block w-1.5 h-4 ml-1 bg-luxury-400 animate-pulse align-middle"
                    aria-label="Generating response..."
                  />
                )}
              </div>

              {/* Structured Property Results Carousel / Grid (Only when valid) */}
              {propertyCards.length > 0 && (
                <div className="pt-2">
                  <h4 className="text-xs font-medium text-[#A1A1AA] uppercase tracking-wider mb-2">
                    Related Properties ({propertyCards.length})
                  </h4>
                  <div className="flex space-x-3 overflow-x-auto pb-2 scrollbar-thin">
                    {propertyCards.map((prop, idx) => (
                      <PropertyCard key={idx} property={prop} />
                    ))}
                  </div>
                </div>
              )}

              {/* Grounded Source Citations (Only displayed when answer is successfully generated) */}
              {citations.length > 0 && (
                <div className="pt-2 border-t border-surface-border/60">
                  <h4 className="text-xs font-medium text-[#A1A1AA] mb-2">
                    Sources ({citations.length})
                  </h4>
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                    {citations.map((cite, idx) => (
                      <CitationCard key={idx} citation={cite} />
                    ))}
                  </div>
                </div>
              )}

              {/* Legal Disclaimer */}
              {disclaimer && (
                <p className="text-[11px] text-[#71717A] italic border-l-2 border-surface-border pl-2.5 py-0.5">
                  {disclaimer}
                </p>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
};
