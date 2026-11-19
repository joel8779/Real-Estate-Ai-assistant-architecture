import React, { useState, useEffect, useRef } from 'react';
import { Navbar } from './components/Navbar';
import { ChatContainer } from './components/ChatContainer';
import { ChatMessage, ChatResponse } from './types';
import { streamMessage, checkHealth } from './api/client';

export const App: React.FC = () => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [responses, setResponses] = useState<Record<number, ChatResponse>>({});
  const [isLoading, setIsLoading] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [isRetrying, setIsRetrying] = useState(false);
  const [streamingStatus, setStreamingStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [sourceFilter, setSourceFilter] = useState<'all' | 'darglobal' | 'wasalt'>('all');
  const [isBackendHealthy, setIsBackendHealthy] = useState<boolean | null>(null);

  const abortControllerRef = useRef<AbortController | null>(null);

  // Poll backend health on initial mount
  useEffect(() => {
    let mounted = true;
    const probe = async () => {
      try {
        await checkHealth();
        if (mounted) setIsBackendHealthy(true);
      } catch {
        if (mounted) setIsBackendHealthy(false);
      }
    };
    probe();
    const interval = setInterval(probe, 15000);
    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, []);

  const handleStopGenerating = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    setIsStreaming(false);
    setIsLoading(false);
    setIsRetrying(false);
    setStreamingStatus(null);

    setMessages((prev) => {
      if (prev.length === 0) return prev;
      const lastIdx = prev.length - 1;
      if (prev[lastIdx].role === 'assistant') {
        const updated = [...prev];
        const content = updated[lastIdx].content;
        if (!content.includes('*(Generation stopped by user)*')) {
          updated[lastIdx] = {
            ...updated[lastIdx],
            content: content
              ? `${content}\n\n*(Generation stopped by user)*`
              : '*(Generation stopped by user)*',
          };
        }
        return updated;
      }
      return prev;
    });
  };

  const handleSendMessage = async (content: string) => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }

    const controller = new AbortController();
    abortControllerRef.current = controller;

    const userMessage: ChatMessage = { role: 'user', content };
    const updatedMessages = [...messages, userMessage];

    const assistantIndex = updatedMessages.length;
    const initialAssistantMsg: ChatMessage = { role: 'assistant', content: '' };
    const messagesWithPlaceholder = [...updatedMessages, initialAssistantMsg];

    setMessages(messagesWithPlaceholder);
    setIsLoading(true);
    setIsStreaming(true);
    setStreamingStatus('Searching the knowledge base...');
    setError(null);

    setResponses((prev) => ({
      ...prev,
      [assistantIndex]: {
        answer: '',
        citations: [],
        property_cards: [],
        confidence_score: 1.0,
        grounded: true,
        data_mode: undefined,
        provenance_warning: null,
        error_code: null,
        retry_after: null,
        disclaimer: 'This information is for general inquiry purposes only.',
      },
    }));

    try {
      await streamMessage(
        {
          messages: updatedMessages,
          source_filter: sourceFilter,
        },
        {
          onStatus: (statusPayload) => {
            setStreamingStatus(statusPayload.message);
          },
          onToken: (token) => {
            setMessages((prev) => {
              const copy = [...prev];
              if (copy[assistantIndex]) {
                copy[assistantIndex] = {
                  ...copy[assistantIndex],
                  content: copy[assistantIndex].content + token,
                };
              }
              return copy;
            });
            setResponses((prev) => {
              const existing = prev[assistantIndex];
              if (!existing) return prev;
              return {
                ...prev,
                [assistantIndex]: {
                  ...existing,
                  answer: existing.answer + token,
                  error_code: null,
                },
              };
            });
          },
          onCitations: (citationsPayload) => {
            setResponses((prev) => {
              const existing = prev[assistantIndex];
              if (!existing) return prev;
              return {
                ...prev,
                [assistantIndex]: {
                  ...existing,
                  citations: citationsPayload.citations || [],
                  property_cards: citationsPayload.property_cards || [],
                  grounded: citationsPayload.grounded ?? true,
                  data_mode: citationsPayload.data_mode || existing.data_mode,
                  provenance_warning:
                    citationsPayload.provenance_warning ?? existing.provenance_warning,
                  model: citationsPayload.model || existing.model,
                  error_code: null,
                },
              };
            });
          },
          onComplete: (completePayload) => {
            setIsStreaming(false);
            setIsLoading(false);
            setStreamingStatus(null);
            setResponses((prev) => {
              const existing = prev[assistantIndex];
              if (!existing) return prev;
              return {
                ...prev,
                [assistantIndex]: {
                  ...existing,
                  grounded: completePayload.grounded ?? existing.grounded,
                  data_mode: completePayload.data_mode || existing.data_mode,
                  provenance_warning:
                    completePayload.provenance_warning ?? existing.provenance_warning,
                  latency_ms: completePayload.latency_ms,
                  stage_latencies: completePayload.stage_latencies,
                  model: completePayload.model || existing.model,
                  disclaimer: completePayload.disclaimer || existing.disclaimer,
                  error_code: null,
                },
              };
            });
          },
          onError: (errorPayload) => {
            setIsStreaming(false);
            setIsLoading(false);
            setStreamingStatus(null);

            setResponses((prev) => {
              const existing = prev[assistantIndex];
              if (!existing) return prev;
              return {
                ...prev,
                [assistantIndex]: {
                  ...existing,
                  error_code: errorPayload.code || errorPayload.error_code || 'ERROR',
                  retry_after: errorPayload.retry_after,
                  citations: [], // Do NOT show citations on failed generation!
                  property_cards: [],
                  grounded: false,
                  provenance_warning: null, // Do NOT show demo warning on provider error screen!
                },
              };
            });

            setMessages((prev) => {
              const copy = [...prev];
              if (copy[assistantIndex]) {
                copy[assistantIndex] = {
                  ...copy[assistantIndex],
                  content: errorPayload.message,
                };
              }
              return copy;
            });

            // Keep global error cleared to avoid duplicate error banners
            setError(null);
          },
        },
        controller.signal
      );
    } catch (err: any) {
      if (!controller.signal.aborted) {
        setResponses((prev) => {
          const existing = prev[assistantIndex];
          if (!existing) return prev;
          return {
            ...prev,
            [assistantIndex]: {
              ...existing,
              error_code: 'CLIENT_ERROR',
              citations: [],
              property_cards: [],
              grounded: false,
              provenance_warning: null,
            },
          };
        });
        setMessages((prev) => {
          const copy = [...prev];
          if (copy[assistantIndex]) {
            copy[assistantIndex] = {
              ...copy[assistantIndex],
              content: err.message || 'The AI service is temporarily unavailable. Please try again.',
            };
          }
          return copy;
        });
        setError(null);
      }
    } finally {
      setIsStreaming(false);
      setIsLoading(false);
      setStreamingStatus(null);
      if (abortControllerRef.current === controller) {
        abortControllerRef.current = null;
      }
    }
  };

  const handleRetry = async (assistantIndex: number) => {
    if (isLoading || isStreaming || isRetrying) return;

    // Find the user query corresponding to this assistant message
    const userMessage = messages[assistantIndex - 1];
    if (!userMessage || userMessage.role !== 'user') return;

    setIsRetrying(true);
    setIsLoading(true);
    setIsStreaming(true);
    setStreamingStatus('Searching the knowledge base...');
    setError(null);

    // Reset this assistant message in-place without creating a duplicate!
    setMessages((prev) => {
      const copy = [...prev];
      if (copy[assistantIndex]) {
        copy[assistantIndex] = { role: 'assistant', content: '' };
      }
      return copy;
    });

    setResponses((prev) => ({
      ...prev,
      [assistantIndex]: {
        answer: '',
        citations: [],
        property_cards: [],
        confidence_score: 1.0,
        grounded: true,
        data_mode: undefined,
        provenance_warning: null,
        error_code: null,
        retry_after: null,
        disclaimer: 'This information is for general inquiry purposes only.',
      },
    }));

    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    const controller = new AbortController();
    abortControllerRef.current = controller;

    try {
      // Use conversation messages up to the user message
      const contextMessages = messages.slice(0, assistantIndex);
      await streamMessage(
        {
          messages: contextMessages,
          source_filter: sourceFilter,
        },
        {
          onStatus: (statusPayload) => {
            setStreamingStatus(statusPayload.message);
          },
          onToken: (token) => {
            setMessages((prev) => {
              const copy = [...prev];
              if (copy[assistantIndex]) {
                copy[assistantIndex] = {
                  ...copy[assistantIndex],
                  content: copy[assistantIndex].content + token,
                };
              }
              return copy;
            });
            setResponses((prev) => {
              const existing = prev[assistantIndex];
              if (!existing) return prev;
              return {
                ...prev,
                [assistantIndex]: {
                  ...existing,
                  answer: existing.answer + token,
                  error_code: null,
                },
              };
            });
          },
          onCitations: (citationsPayload) => {
            setResponses((prev) => {
              const existing = prev[assistantIndex];
              if (!existing) return prev;
              return {
                ...prev,
                [assistantIndex]: {
                  ...existing,
                  citations: citationsPayload.citations || [],
                  property_cards: citationsPayload.property_cards || [],
                  grounded: citationsPayload.grounded ?? true,
                  data_mode: citationsPayload.data_mode || existing.data_mode,
                  provenance_warning:
                    citationsPayload.provenance_warning ?? existing.provenance_warning,
                  model: citationsPayload.model || existing.model,
                  error_code: null,
                },
              };
            });
          },
          onComplete: (completePayload) => {
            setIsStreaming(false);
            setIsLoading(false);
            setIsRetrying(false);
            setStreamingStatus(null);
            setResponses((prev) => {
              const existing = prev[assistantIndex];
              if (!existing) return prev;
              return {
                ...prev,
                [assistantIndex]: {
                  ...existing,
                  grounded: completePayload.grounded ?? existing.grounded,
                  data_mode: completePayload.data_mode || existing.data_mode,
                  provenance_warning:
                    completePayload.provenance_warning ?? existing.provenance_warning,
                  latency_ms: completePayload.latency_ms,
                  stage_latencies: completePayload.stage_latencies,
                  model: completePayload.model || existing.model,
                  disclaimer: completePayload.disclaimer || existing.disclaimer,
                  error_code: null,
                },
              };
            });
          },
          onError: (errorPayload) => {
            setIsStreaming(false);
            setIsLoading(false);
            setIsRetrying(false);
            setStreamingStatus(null);

            setResponses((prev) => {
              const existing = prev[assistantIndex];
              if (!existing) return prev;
              return {
                ...prev,
                [assistantIndex]: {
                  ...existing,
                  error_code: errorPayload.code || errorPayload.error_code || 'ERROR',
                  retry_after: errorPayload.retry_after,
                  citations: [],
                  property_cards: [],
                  grounded: false,
                  provenance_warning: null,
                },
              };
            });

            setMessages((prev) => {
              const copy = [...prev];
              if (copy[assistantIndex]) {
                copy[assistantIndex] = {
                  ...copy[assistantIndex],
                  content: errorPayload.message,
                };
              }
              return copy;
            });
          },
        },
        controller.signal
      );
    } catch (err: any) {
      if (!controller.signal.aborted) {
        setResponses((prev) => {
          const existing = prev[assistantIndex];
          if (!existing) return prev;
          return {
            ...prev,
            [assistantIndex]: {
              ...existing,
              error_code: 'CLIENT_ERROR',
              citations: [],
              property_cards: [],
              grounded: false,
              provenance_warning: null,
            },
          };
        });
        setMessages((prev) => {
          const copy = [...prev];
          if (copy[assistantIndex]) {
            copy[assistantIndex] = {
              ...copy[assistantIndex],
              content: err.message || 'The AI service is temporarily unavailable. Please try again.',
            };
          }
          return copy;
        });
      }
    } finally {
      setIsStreaming(false);
      setIsLoading(false);
      setIsRetrying(false);
      setStreamingStatus(null);
      if (abortControllerRef.current === controller) {
        abortControllerRef.current = null;
      }
    }
  };

  const handleClearChat = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    setMessages([]);
    setResponses({});
    setError(null);
    setIsLoading(false);
    setIsStreaming(false);
    setIsRetrying(false);
    setStreamingStatus(null);
  };

  return (
    <div className="h-[100dvh] max-h-[100dvh] overflow-hidden bg-surface-base text-gray-100 flex flex-col font-sans">
      <Navbar
        sourceFilter={sourceFilter}
        setSourceFilter={setSourceFilter}
        isBackendHealthy={isBackendHealthy}
        onClearChat={handleClearChat}
      />
      <main className="flex-1 min-h-0 flex flex-col overflow-hidden">
        <ChatContainer
          messages={messages}
          responses={responses}
          isLoading={isLoading}
          isStreaming={isStreaming}
          isRetrying={isRetrying}
          streamingStatus={streamingStatus}
          error={error}
          onSendMessage={handleSendMessage}
          onStopGenerating={handleStopGenerating}
          onRetry={handleRetry}
          sourceFilter={sourceFilter}
        />
      </main>
    </div>
  );
};

export default App;
