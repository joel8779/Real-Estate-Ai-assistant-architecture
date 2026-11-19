export type MessageRole = 'user' | 'assistant';

export interface ChatMessage {
  role: MessageRole;
  content: string;
}

export interface Citation {
  title: string;
  url: string;
  source: 'darglobal' | 'wasalt';
  category: string;
  snippet: string;
  provenance?: 'verified_source' | 'recruiter_provided' | 'demonstration_unverified';
}

export interface PropertyCardData {
  title: string;
  developer?: string;
  location: string;
  property_type?: string;
  price?: string;
  bedrooms?: string;
  amenities: string[];
  url: string;
  image_url?: string;
}

export interface ChatRequest {
  messages: ChatMessage[];
  source_filter?: 'all' | 'darglobal' | 'wasalt';
  category_filter?: string;
  session_id?: string;
}

export interface ChatResponse {
  answer: string;
  citations: Citation[];
  property_cards: PropertyCardData[];
  confidence_score: number;
  grounded: boolean;
  data_mode?: 'verified' | 'demonstration' | 'conversational';
  provenance_warning?: string | null;
  error_code?: string | null;
  request_id?: string | null;
  latency_ms?: number;
  stage_latencies?: Record<string, number>;
  retry_after?: number | null;
  model?: string;
  provider?: 'openrouter' | 'ollama';
  fallback_active?: boolean;
  fallback_reason?: 'quota_exhausted' | 'provider_outage';
  fallback_notice?: string | null;
  disclaimer: string;
}

export interface HealthStatus {
  status: string;
  timestamp: string;
  service: string;
  version: string;
  environment: string;
}

export interface ReadinessStatus {
  status: string;
  timestamp: string;
  checks: Record<string, any>;
}

export interface StreamStatusPayload {
  stage: 'searching' | 'preparing' | 'generating';
  message: string;
}

export interface StreamTokenPayload {
  token: string;
}

export interface StreamCitationsPayload {
  citations: Citation[];
  property_cards: PropertyCardData[];
  grounded: boolean;
  data_mode?: 'verified' | 'demonstration' | 'conversational';
  provenance_warning?: string | null;
  model?: string;
  provider?: 'openrouter' | 'ollama';
  fallback_active?: boolean;
  fallback_reason?: 'quota_exhausted' | 'provider_outage';
  fallback_notice?: string | null;
}

export interface StreamCompletePayload {
  request_id?: string;
  latency_ms?: number;
  ttft_ms?: number;
  generation_ms?: number;
  total_tokens?: number;
  model?: string;
  provider?: 'openrouter' | 'ollama';
  fallback_active?: boolean;
  fallback_reason?: 'quota_exhausted' | 'provider_outage';
  fallback_notice?: string | null;
  grounded?: boolean;
  data_mode?: 'verified' | 'demonstration' | 'conversational';
  provenance_warning?: string | null;
  stage_latencies?: Record<string, number>;
  disclaimer?: string;
}

export interface StreamErrorPayload {
  code: string;
  error_code?: string;
  message: string;
  retry_after?: number | null;
  citations?: Citation[];
  property_cards?: PropertyCardData[];
  data_mode?: 'verified' | 'demonstration' | 'conversational';
  provenance_warning?: string | null;
}
