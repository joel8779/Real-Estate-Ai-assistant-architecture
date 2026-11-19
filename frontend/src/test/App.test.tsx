import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import App from '../App';
import * as apiClient from '../api/client';

describe('DarGlobal & Wasalt Assistant UI', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.spyOn(apiClient, 'checkHealth').mockResolvedValue({
      status: 'healthy',
      timestamp: '2026-11-18T18:00:00Z',
      service: 'DarGlobal Knowledge Assistant',
      version: '1.0.0',
      environment: 'development',
    });
  });

  it('renders application brand, header, and suggested queries', async () => {
    render(<App />);

    expect(screen.getByText('DarGlobal')).toBeInTheDocument();
    expect(screen.getByText('Wasalt')).toBeInTheDocument();
    expect(screen.getByText(/Real Estate Intelligence Assistant/i)).toBeInTheDocument();
    expect(screen.getByText('Branded Developments')).toBeInTheDocument();
    expect(screen.getByText('Domestic Properties')).toBeInTheDocument();

    await waitFor(() => {
      expect(screen.getByText('Online')).toBeInTheDocument();
    });
  });

  it('allows user to switch source filters', async () => {
    render(<App />);

    await waitFor(() => {
      expect(screen.getByText('Online')).toBeInTheDocument();
    });

    const darGlobalBtn = screen.getByRole('button', { name: 'DarGlobal' });
    const wasaltBtn = screen.getByRole('button', { name: 'Wasalt' });
    const allBtn = screen.getByRole('button', { name: 'All Sources' });

    expect(allBtn).toBeInTheDocument();
    fireEvent.click(darGlobalBtn);
    expect(darGlobalBtn).toHaveClass('bg-luxury-500/20');

    fireEvent.click(wasaltBtn);
    expect(wasaltBtn).toHaveClass('bg-emerald-500/20');
  });

  it('submits a query and displays assistant response with citations in verified mode via streaming', async () => {
    vi.spyOn(apiClient, 'streamMessage').mockImplementation(async (_payload, handlers) => {
      handlers.onStatus?.({ stage: 'searching', message: 'Searching the knowledge base...' });
      handlers.onToken?.('DarGlobal offers Trump International, Aida in Oman, and Urban Oasis.');
      handlers.onCitations?.({
        citations: [
          {
            title: 'Aida, Oman Overview',
            url: 'https://darglobal.co.uk/projects/aida',
            source: 'darglobal',
            category: 'project',
            snippet: 'Aida is a premier mixed-use luxury development.',
            provenance: 'verified_source',
          },
        ],
        property_cards: [],
        grounded: true,
        data_mode: 'verified',
        provenance_warning: null,
      });
      handlers.onComplete?.({
        latency_ms: 120,
        grounded: true,
        data_mode: 'verified',
        disclaimer: 'Verified public record.',
      });
    });

    render(<App />);

    const input = screen.getByPlaceholderText(/Ask anything about/i);
    fireEvent.change(input, { target: { value: 'What projects are in Oman?' } });

    const submitBtn = screen.getByRole('button', { name: /Send message/i });
    fireEvent.click(submitBtn);

    await waitFor(() => {
      expect(
        screen.getByText(/DarGlobal offers Trump International, Aida in Oman/i)
      ).toBeInTheDocument();
    });

    expect(screen.getByText('Aida, Oman Overview')).toBeInTheDocument();
    expect(screen.getByText(/Verified Public Grounding/i)).toBeInTheDocument();
  });

  it('displays visible warning banner and demo badge when in demonstration mode', async () => {
    vi.spyOn(apiClient, 'streamMessage').mockImplementation(async (_payload, handlers) => {
      handlers.onToken?.('This is a sample answer about luxury villas in Spain.');
      handlers.onCitations?.({
        citations: [
          {
            title: 'Tierra Viva by Lamborghini',
            url: 'https://darglobal.co.uk/projects/tierra-viva',
            source: 'darglobal',
            category: 'project',
            snippet: 'Villas in Benahavis Marbella.',
            provenance: 'demonstration_unverified',
          },
        ],
        property_cards: [],
        grounded: true,
        data_mode: 'demonstration',
        provenance_warning:
          'This prototype uses sample property information prepared for demonstration purposes. Property details, pricing and availability have not been independently verified. Please consult official sources for current information.',
      });
      handlers.onComplete?.({
        latency_ms: 150,
        grounded: true,
        data_mode: 'demonstration',
        provenance_warning:
          'This prototype uses sample property information prepared for demonstration purposes. Property details, pricing and availability have not been independently verified. Please consult official sources for current information.',
      });
    });

    render(<App />);

    const input = screen.getByPlaceholderText(/Ask anything about/i);
    fireEvent.change(input, { target: { value: 'Tell me about Tierra Viva' } });

    const submitBtn = screen.getByRole('button', { name: /Send message/i });
    fireEvent.click(submitBtn);

    await waitFor(() => {
      expect(
        screen.getByText(/This is a sample answer about luxury villas in Spain/i)
      ).toBeInTheDocument();
    });

    expect(screen.getByText(/Prototype Dataset/i)).toBeInTheDocument();
    expect(screen.getByText(/Research Dataset Notice/i)).toBeInTheDocument();
    expect(
      screen.getByText(/This prototype uses sample property information prepared for demonstration purposes/i)
    ).toBeInTheDocument();
  });

  it('allows stopping streaming generation via Stop button', async () => {
    let capturedSignal: AbortSignal | undefined;
    vi.spyOn(apiClient, 'streamMessage').mockImplementation(async (_payload, handlers, signal) => {
      capturedSignal = signal;
      handlers.onToken?.('First token chunk...');
      // wait without completing immediately
      await new Promise((resolve) => setTimeout(resolve, 500));
    });

    render(<App />);

    const input = screen.getByPlaceholderText(/Ask anything about/i);
    fireEvent.change(input, { target: { value: 'Explain investment options' } });
    fireEvent.click(screen.getByRole('button', { name: /Send message/i }));

    await waitFor(() => {
      expect(screen.getByText(/First token chunk.../i)).toBeInTheDocument();
    });

    const stopBtn = screen.getByRole('button', { name: /Stop generating response/i });
    expect(stopBtn).toBeInTheDocument();
    fireEvent.click(stopBtn);

    expect(capturedSignal?.aborted).toBe(true);
    await waitFor(() => {
      expect(screen.getByText(/Generation stopped by user/i)).toBeInTheDocument();
    });
  });

  it('clears conversation when Clear button is clicked', async () => {
    vi.spyOn(apiClient, 'streamMessage').mockImplementation(async (_payload, handlers) => {
      handlers.onToken?.('Temporary message that will be cleared.');
      handlers.onComplete?.({
        latency_ms: 100,
        grounded: true,
        data_mode: 'verified',
      });
    });

    render(<App />);

    const input = screen.getByPlaceholderText(/Ask anything about/i);
    fireEvent.change(input, { target: { value: 'Hello' } });
    fireEvent.click(screen.getByRole('button', { name: /Send message/i }));

    await waitFor(() => {
      expect(screen.getByText(/Temporary message that will be cleared/i)).toBeInTheDocument();
    });

    const clearBtn = screen.getByRole('button', { name: /Clear/i });
    fireEvent.click(clearBtn);

    expect(screen.queryByText(/Temporary message that will be cleared/i)).not.toBeInTheDocument();
    expect(screen.getByText(/Real Estate Intelligence Assistant/i)).toBeInTheDocument();
  });

  it('renders single error box with Try again button and suppresses citations/demo warning on 429 error', async () => {
    vi.spyOn(apiClient, 'streamMessage').mockImplementation(async (_payload, handlers) => {
      handlers.onError?.({
        code: 'LLM_RATE_LIMITED',
        error_code: 'LLM_RATE_LIMITED',
        message: 'The AI service is temporarily busy. Please try again in 45 seconds.',
        retry_after: 45,
      });
    });

    render(<App />);

    const input = screen.getByPlaceholderText(/Ask anything about/i);
    fireEvent.change(input, { target: { value: 'Tell me about AIDA' } });
    fireEvent.click(screen.getByRole('button', { name: /Send message/i }));

    await waitFor(() => {
      expect(screen.getByText(/The AI service is temporarily busy. Please try again in 45 seconds./i)).toBeInTheDocument();
    });

    // Check "Try again" button exists
    const retryBtn = screen.getByRole('button', { name: /Try again/i });
    expect(retryBtn).toBeInTheDocument();

    // Verify citations and demo warnings are NOT displayed
    expect(screen.queryByText(/Sources & Evidence/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/Demonstration Dataset Notice/i)).not.toBeInTheDocument();

    // Verify no duplicate error banner exists
    const errorTexts = screen.getAllByText(/The AI service is temporarily busy/i);
    expect(errorTexts).toHaveLength(1);
  });

  it('retries request in-place without creating duplicate assistant bubbles', async () => {
    let callCount = 0;
    vi.spyOn(apiClient, 'streamMessage').mockImplementation(async (_payload, handlers) => {
      callCount++;
      if (callCount === 1) {
        handlers.onError?.({
          code: 'LLM_RATE_LIMITED',
          error_code: 'LLM_RATE_LIMITED',
          message: 'The AI service is temporarily busy. Please wait a moment and try again.',
          retry_after: null,
        });
      } else {
        handlers.onToken?.('Retried answer successfully loaded.');
        handlers.onComplete?.({
          latency_ms: 120,
          grounded: true,
          data_mode: 'verified',
        });
      }
    });

    render(<App />);

    const input = screen.getByPlaceholderText(/Ask anything about/i);
    fireEvent.change(input, { target: { value: 'What is AIDA?' } });
    fireEvent.click(screen.getByRole('button', { name: /Send message/i }));

    await waitFor(() => {
      expect(screen.getByText(/The AI service is temporarily busy/i)).toBeInTheDocument();
    });

    const retryBtn = screen.getByRole('button', { name: /Try again/i });
    fireEvent.click(retryBtn);

    await waitFor(() => {
      expect(screen.getByText(/Retried answer successfully loaded./i)).toBeInTheDocument();
    });

    // Verify error is gone and there is only 1 user message and 1 assistant message
    expect(screen.queryByText(/The AI service is temporarily busy/i)).not.toBeInTheDocument();
    expect(screen.getByText('What is AIDA?')).toBeInTheDocument();
    expect(screen.getByText('Retried answer successfully loaded.')).toBeInTheDocument();
  });

  it('displays Nemotron badge when primary model generates the response', async () => {
    vi.spyOn(apiClient, 'streamMessage').mockImplementation(async (_payload, handlers) => {
      handlers.onToken?.('Response from primary model.');
      handlers.onComplete?.({
        model: 'nvidia/nemotron-3-ultra-550b-a55b:free',
        grounded: true,
      });
    });

    render(<App />);

    const input = screen.getByPlaceholderText(/Ask anything about/i);
    fireEvent.change(input, { target: { value: 'Query primary model' } });
    fireEvent.click(screen.getByRole('button', { name: /Send message/i }));

    await waitFor(() => {
      expect(screen.getByText('Response from primary model.')).toBeInTheDocument();
    });

    const badge = screen.getByTestId('model-badge');
    expect(badge).toHaveTextContent('Nemotron 3 Ultra');
  });

  it('displays Fallback badge when a fallback model generates the response', async () => {
    vi.spyOn(apiClient, 'streamMessage').mockImplementation(async (_payload, handlers) => {
      handlers.onToken?.('Response from fallback model.');
      handlers.onComplete?.({
        model: 'meta-llama/llama-3.3-70b-instruct:free',
        grounded: true,
      });
    });

    render(<App />);

    const input = screen.getByPlaceholderText(/Ask anything about/i);
    fireEvent.change(input, { target: { value: 'Query fallback model' } });
    fireEvent.click(screen.getByRole('button', { name: /Send message/i }));

    await waitFor(() => {
      expect(screen.getByText('Response from fallback model.')).toBeInTheDocument();
    });

    const badge = screen.getByTestId('model-badge');
    expect(badge).toHaveTextContent('Fallback: llama-3.3-70b-instruct:free');
    expect(badge).toHaveClass('text-amber-300');
  });
});
