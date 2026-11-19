import {
  ChatRequest,
  ChatResponse,
  HealthStatus,
  ReadinessStatus,
  StreamStatusPayload,
  StreamCitationsPayload,
  StreamCompletePayload,
  StreamErrorPayload,
} from '../types';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '';

export class ApiError extends Error {
  constructor(
    public code: string,
    message: string,
    public status: number,
    public details?: any
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

export interface StreamHandlers {
  onStatus?: (status: StreamStatusPayload) => void;
  onToken?: (token: string) => void;
  onCitations?: (data: StreamCitationsPayload) => void;
  onComplete?: (data: StreamCompletePayload) => void;
  onError?: (error: StreamErrorPayload) => void;
}

export async function sendMessage(payload: ChatRequest): Promise<ChatResponse> {
  const url = `${API_BASE_URL}/api/v1/chat`;
  const response = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
    },
    body: JSON.stringify(payload),
  });

  const data = await response.json();

  if (!response.ok) {
    const errorInfo = data.error || { code: 'HTTP_ERROR', message: `Request failed with status ${response.status}` };
    throw new ApiError(errorInfo.code, errorInfo.message, response.status, errorInfo.details);
  }

  return data as ChatResponse;
}

export async function streamMessage(
  payload: ChatRequest,
  handlers: StreamHandlers,
  signal?: AbortSignal
): Promise<void> {
  const url = `${API_BASE_URL}/api/v1/chat/stream`;

  let response: Response;
  try {
    response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'text/event-stream',
      },
      body: JSON.stringify(payload),
      signal,
    });
  } catch (err: any) {
    if (signal?.aborted) {
      return;
    }
    const message = err.message || 'Network connection failed.';
    handlers.onError?.({
      code: 'NETWORK_ERROR',
      message,
    });
    return;
  }

  if (!response.ok) {
    let errCode = 'HTTP_ERROR';
    let errMsg = `Request failed with status ${response.status}`;
    try {
      const errJson = await response.json();
      if (errJson.error) {
        errCode = errJson.error.code || errCode;
        errMsg = errJson.error.message || errMsg;
      }
    } catch {
      // ignore JSON parse error for plain text responses
    }
    handlers.onError?.({
      code: errCode,
      message: errMsg,
    });
    return;
  }

  if (!response.body) {
    handlers.onError?.({
      code: 'NO_RESPONSE_BODY',
      message: 'Server returned empty response body.',
    });
    return;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder('utf-8');
  let buffer = '';

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      let boundaryIndex: number;
      while ((boundaryIndex = buffer.indexOf('\n\n')) !== -1) {
        const rawMessage = buffer.slice(0, boundaryIndex).trim();
        buffer = buffer.slice(boundaryIndex + 2);

        if (!rawMessage) continue;

        let eventType = 'message';
        let dataStr = '';

        const lines = rawMessage.split('\n');
        for (const line of lines) {
          if (line.startsWith('event:')) {
            eventType = line.slice(6).trim();
          } else if (line.startsWith('data:')) {
            dataStr = line.slice(5).trim();
          }
        }

        if (!dataStr) continue;

        try {
          const parsed = JSON.parse(dataStr);
          switch (eventType) {
            case 'status':
              handlers.onStatus?.(parsed as StreamStatusPayload);
              break;
            case 'token':
              if (parsed.token) {
                handlers.onToken?.(parsed.token);
              }
              break;
            case 'citations':
              handlers.onCitations?.(parsed as StreamCitationsPayload);
              break;
            case 'complete':
              handlers.onComplete?.(parsed as StreamCompletePayload);
              break;
            case 'error':
              handlers.onError?.(parsed as StreamErrorPayload);
              break;
            default:
              if (parsed.token) {
                handlers.onToken?.(parsed.token);
              }
              break;
          }
        } catch (e) {
          console.warn('Failed to parse SSE JSON payload:', dataStr, e);
        }
      }
    }
  } catch (err: any) {
    if (signal?.aborted) {
      return;
    }
    handlers.onError?.({
      code: 'STREAM_READ_ERROR',
      message: err.message || 'Stream connection was interrupted.',
    });
  }
}

export async function checkHealth(): Promise<HealthStatus> {
  const response = await fetch(`${API_BASE_URL}/healthz`);
  if (!response.ok) {
    throw new Error(`Health check failed with status ${response.status}`);
  }
  return response.json();
}

export async function checkReadiness(): Promise<ReadinessStatus> {
  const response = await fetch(`${API_BASE_URL}/readyz`);
  return response.json();
}
