// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { renderHook, act, waitFor } from '@testing-library/react'
import { vi, describe, test, expect, beforeEach, afterEach } from 'vitest'
import { useDeepResearch } from './use-deep-research'

// ============================================================
// Mock store state and actions
// ============================================================

const mockUpdateDeepResearchStatus = vi.fn()
const mockCompleteDeepResearch = vi.fn()
const mockAddDeepResearchCitation = vi.fn()
const mockSetReportContent = vi.fn()
const mockAddThinkingStep = vi.fn(() => 'step-1')
const mockAppendToThinkingStep = vi.fn()
const mockCompleteThinkingStep = vi.fn()
const mockSetCurrentStatus = vi.fn()
const mockSetStreaming = vi.fn()
const mockSetDeepResearchTodos = vi.fn()
const mockStopDeepResearchTodos = vi.fn()
const mockStopAllDeepResearchSpinners = vi.fn()
const mockSetDeepResearchLastEventId = vi.fn()
const mockAddDeepResearchLLMStep = vi.fn(() => 'llm-step-1')
const mockAppendToDeepResearchLLMStep = vi.fn()
const mockCompleteDeepResearchLLMStep = vi.fn()
const mockAddDeepResearchAgent = vi.fn(() => 'agent-1')
const mockAddDeepResearchAgentWithId = vi.fn(() => 'agent-1')
const mockCompleteDeepResearchAgent = vi.fn()
const mockAddDeepResearchToolCall = vi.fn(() => 'tool-call-1')
const mockCompleteDeepResearchToolCall = vi.fn()
const mockAddDeepResearchFile = vi.fn()
const mockAddAgentResponse = vi.fn()
const mockAddErrorCard = vi.fn()
const mockPatchConversationMessage = vi.fn()
const mockPersistDeepResearchToSession = vi.fn()
const mockAddDeepResearchBanner = vi.fn()
const mockSetStreamLoaded = vi.fn()

let mockStoreState = {
  deepResearchJobId: null as string | null,
  deepResearchLastEventId: null as string | null,
  isDeepResearchStreaming: false,
  deepResearchStatus: null as string | null,
  reportContent: '',
  deepResearchLLMSteps: [] as unknown[],
  deepResearchToolCalls: [] as unknown[],
  deepResearchCitations: [] as unknown[],
  deepResearchOwnerConversationId: 'test-conv-123',
  currentConversation: { id: 'test-conv-123' } as { id: string } | null,
  activeDeepResearchMessageId: null as string | null,
  currentUserMessageId: 'user-msg-1' as string | null,
}

vi.mock('../store', () => ({
  useChatStore: Object.assign(
    vi.fn((selector?: (s: any) => any) => {
      const state = {
        ...mockStoreState,
        updateDeepResearchStatus: mockUpdateDeepResearchStatus,
        completeDeepResearch: mockCompleteDeepResearch,
        addDeepResearchCitation: mockAddDeepResearchCitation,
        setReportContent: mockSetReportContent,
        addThinkingStep: mockAddThinkingStep,
        appendToThinkingStep: mockAppendToThinkingStep,
        completeThinkingStep: mockCompleteThinkingStep,
        setCurrentStatus: mockSetCurrentStatus,
        setStreaming: mockSetStreaming,
        setDeepResearchTodos: mockSetDeepResearchTodos,
        stopDeepResearchTodos: mockStopDeepResearchTodos,
        stopAllDeepResearchSpinners: mockStopAllDeepResearchSpinners,
        setDeepResearchLastEventId: mockSetDeepResearchLastEventId,
        addDeepResearchLLMStep: mockAddDeepResearchLLMStep,
        appendToDeepResearchLLMStep: mockAppendToDeepResearchLLMStep,
        completeDeepResearchLLMStep: mockCompleteDeepResearchLLMStep,
        addDeepResearchAgent: mockAddDeepResearchAgent,
        addDeepResearchAgentWithId: mockAddDeepResearchAgentWithId,
        completeDeepResearchAgent: mockCompleteDeepResearchAgent,
        addDeepResearchToolCall: mockAddDeepResearchToolCall,
        completeDeepResearchToolCall: mockCompleteDeepResearchToolCall,
        addDeepResearchFile: mockAddDeepResearchFile,
        addAgentResponse: mockAddAgentResponse,
        patchConversationMessage: mockPatchConversationMessage,
        persistDeepResearchToSession: mockPersistDeepResearchToSession,
        addDeepResearchBanner: mockAddDeepResearchBanner,
        setStreamLoaded: mockSetStreamLoaded,
      }
      return selector ? selector(state) : state
    }),
    {
      getState: vi.fn(() => ({
        ...mockStoreState,
        addErrorCard: mockAddErrorCard,
        addDeepResearchBanner: mockAddDeepResearchBanner,
        stopAllDeepResearchSpinners: mockStopAllDeepResearchSpinners,
        patchConversationMessage: mockPatchConversationMessage,
        completeDeepResearch: mockCompleteDeepResearch,
        setStreaming: mockSetStreaming,
        setStreamLoaded: mockSetStreamLoaded,
      })),
      setState: vi.fn((updater: (state: typeof mockStoreState) => Partial<typeof mockStoreState>) => {
        if (typeof updater === 'function') {
          const updates = updater(mockStoreState)
          Object.assign(mockStoreState, updates)
        }
      }),
    }
  ),
}))

// Mock layout store
const mockOpenRightPanel = vi.fn()
const mockSetResearchPanelTab = vi.fn()

vi.mock('@/features/layout/store', () => ({
  useLayoutStore: vi.fn((selector?: (s: any) => any) => {
    const state = {
      openRightPanel: mockOpenRightPanel,
      setResearchPanelTab: mockSetResearchPanelTab,
    }
    return selector ? selector(state) : state
  }),
}))

// Mock auth hook
vi.mock('@/adapters/auth', () => ({
  useAuth: vi.fn(() => ({
    idToken: 'mock-id-token',
  })),
}))

// Mock backend health check
const mockCheckBackendHealthCached = vi.fn<() => Promise<boolean>>().mockResolvedValue(false)
vi.mock('@/shared/hooks/use-backend-health', () => ({
  checkBackendHealthCached: () => mockCheckBackendHealthCached(),
  invalidateHealthCache: vi.fn(),
}))

// ============================================================
// Mock deep research client
// ============================================================

interface MockClient {
  connect: ReturnType<typeof vi.fn>
  disconnect: ReturnType<typeof vi.fn>
  isConnected: ReturnType<typeof vi.fn>
  getLastEventId: ReturnType<typeof vi.fn>
  callbacks: Record<string, (...args: unknown[]) => void>
}

let mockClient: MockClient | null = null
const mockCreateDeepResearchClient = vi.fn((options: { callbacks: Record<string, (...args: unknown[]) => void> }) => {
  mockClient = {
    connect: vi.fn(),
    disconnect: vi.fn(),
    isConnected: vi.fn(() => false),
    getLastEventId: vi.fn(() => null),
    callbacks: options.callbacks,
  }
  return mockClient
})

const mockCancelJob = vi.fn()
const mockGetJobStatus = vi.fn<() => Promise<{ status: string }>>().mockResolvedValue({ status: 'running' })
const mockGetJobReport = vi.fn<() => Promise<{ has_report: boolean; report?: string }>>().mockResolvedValue({ has_report: false })

vi.mock('@/adapters/api', () => ({
  createDeepResearchClient: (options: { callbacks: Record<string, (...args: unknown[]) => void> }) =>
    mockCreateDeepResearchClient(options),
  cancelJob: (...args: unknown[]) => mockCancelJob(...args),
  getJobStatus: () => mockGetJobStatus(),
  getJobReport: () => mockGetJobReport(),
}))

import { useChatStore } from '../store'

// ============================================================
// Tests
// ============================================================

describe('useDeepResearch', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.useFakeTimers()
    mockClient = null
    mockStoreState = {
      deepResearchJobId: null,
      deepResearchLastEventId: null,
      isDeepResearchStreaming: false,
      deepResearchStatus: null,
      reportContent: '',
      deepResearchLLMSteps: [],
      deepResearchToolCalls: [],
      deepResearchCitations: [],
      deepResearchOwnerConversationId: 'test-conv-123',
      currentConversation: { id: 'test-conv-123' },
      activeDeepResearchMessageId: null,
      currentUserMessageId: 'user-msg-1',
    }
    vi.mocked(useChatStore).getState = vi.fn(() => ({
      ...mockStoreState,
      addErrorCard: mockAddErrorCard,
      persistDeepResearchToSession: mockPersistDeepResearchToSession,
      addDeepResearchBanner: mockAddDeepResearchBanner,
      stopAllDeepResearchSpinners: mockStopAllDeepResearchSpinners,
    })) as unknown as typeof useChatStore.getState
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  describe('initial state', () => {
    test('returns correct initial values when no job is active', () => {
      const { result } = renderHook(() => useDeepResearch())

      expect(result.current.isStreaming).toBe(false)
      expect(result.current.jobId).toBeNull()
      expect(result.current.status).toBeNull()
      expect(result.current.isTimedOut).toBe(false)
      expect(typeof result.current.disconnect).toBe('function')
      expect(typeof result.current.reconnect).toBe('function')
      expect(typeof result.current.cancelCurrentJob).toBe('function')
    })

    test('returns streaming state from store', () => {
      mockStoreState.isDeepResearchStreaming = true
      mockStoreState.deepResearchJobId = 'job-123'
      mockStoreState.deepResearchStatus = 'running'

      const { result } = renderHook(() => useDeepResearch())

      expect(result.current.isStreaming).toBe(true)
      expect(result.current.jobId).toBe('job-123')
      expect(result.current.status).toBe('running')
    })
  })

  describe('auto-connect behavior', () => {
    test('connects when job ID and streaming flag are set', async () => {
      mockStoreState.deepResearchJobId = 'job-456'
      mockStoreState.isDeepResearchStreaming = true

      renderHook(() => useDeepResearch())

      await act(async () => { await advanceAndFlush(60) })

      expect(mockCreateDeepResearchClient).toHaveBeenCalledWith(
        expect.objectContaining({
          jobId: 'job-456',
          authToken: 'mock-id-token',
        })
      )
      expect(mockClient?.connect).toHaveBeenCalled()
    })

    test('opens research panel when connecting', async () => {
      mockStoreState.deepResearchJobId = 'job-456'
      mockStoreState.isDeepResearchStreaming = true

      renderHook(() => useDeepResearch())

      await act(async () => { await advanceAndFlush(60) })

      expect(mockSetResearchPanelTab).toHaveBeenCalledWith('tasks')
      expect(mockOpenRightPanel).toHaveBeenCalledWith('research')
    })

    test('always connects from the beginning (no lastEventId)', async () => {
      mockStoreState.deepResearchJobId = 'job-456'
      mockStoreState.deepResearchLastEventId = 'event-789'
      mockStoreState.isDeepResearchStreaming = true

      renderHook(() => useDeepResearch())

      await act(async () => { await advanceAndFlush(60) })

      expect(mockCreateDeepResearchClient).toHaveBeenCalledWith(
        expect.objectContaining({
          jobId: 'job-456',
        })
      )
      expect(mockCreateDeepResearchClient).not.toHaveBeenCalledWith(
        expect.objectContaining({
          lastEventId: expect.anything(),
        })
      )
    })

    test('does not connect when job ID is null', () => {
      mockStoreState.deepResearchJobId = null
      mockStoreState.isDeepResearchStreaming = true

      renderHook(() => useDeepResearch())

      expect(mockCreateDeepResearchClient).not.toHaveBeenCalled()
    })

    test('does not connect when streaming is false', () => {
      mockStoreState.deepResearchJobId = 'job-456'
      mockStoreState.isDeepResearchStreaming = false

      renderHook(() => useDeepResearch())

      expect(mockCreateDeepResearchClient).not.toHaveBeenCalled()
    })
  })

  describe('disconnect behavior', () => {
    test('disconnect calls client disconnect', async () => {
      mockStoreState.deepResearchJobId = 'job-456'
      mockStoreState.isDeepResearchStreaming = true
      mockStoreState.deepResearchStatus = 'submitted'

      const { result } = renderHook(() => useDeepResearch())

      // Advance past the 50ms deferred connect so the client is created
      await act(async () => { await advanceAndFlush(60) })

      act(() => {
        result.current.disconnect()
      })

      expect(mockClient?.disconnect).toHaveBeenCalled()
    })

    test('disconnects on unmount', async () => {
      mockStoreState.deepResearchJobId = 'job-456'
      mockStoreState.isDeepResearchStreaming = true
      mockStoreState.deepResearchStatus = 'submitted'

      const { unmount } = renderHook(() => useDeepResearch())

      // Advance past the 50ms deferred connect so the client is created
      await act(async () => { await advanceAndFlush(60) })

      unmount()

      expect(mockClient?.disconnect).toHaveBeenCalled()
    })
  })

  describe('reconnect behavior', () => {
    test('reconnect creates new connection with last event ID', async () => {
      mockStoreState.deepResearchJobId = 'job-456'
      mockStoreState.isDeepResearchStreaming = true
      mockStoreState.deepResearchStatus = 'submitted'

      const { result } = renderHook(() => useDeepResearch())

      // Advance past the 50ms deferred connect so the client is created
      await act(async () => { await advanceAndFlush(60) })

      // Simulate getting a last event ID
      mockClient!.getLastEventId.mockReturnValue('event-123')

      // Disconnect first
      act(() => {
        result.current.disconnect()
      })

      // Mock client as disconnected
      mockClient!.isConnected.mockReturnValue(false)

      // Clear mocks to track reconnect call
      mockCreateDeepResearchClient.mockClear()

      act(() => {
        result.current.reconnect()
      })

      expect(mockCreateDeepResearchClient).toHaveBeenCalled()
    })
  })

  describe('cancelCurrentJob', () => {
    test('calls cancel API and schedules fallback timer', async () => {
      mockStoreState.deepResearchJobId = 'job-456'
      mockStoreState.isDeepResearchStreaming = true
      mockCancelJob.mockResolvedValue({ cancelled: true })

      const { result } = renderHook(() => useDeepResearch())

      await act(async () => {
        await result.current.cancelCurrentJob()
      })

      expect(mockCancelJob).toHaveBeenCalledWith('job-456', 'mock-id-token')

      // Fallback should NOT have fired yet (only 0ms elapsed)
      expect(mockCompleteDeepResearch).not.toHaveBeenCalled()
    })

    test('fallback cleans up locally if SSE does not deliver interrupted status', async () => {
      mockStoreState.deepResearchJobId = 'job-456'
      mockStoreState.isDeepResearchStreaming = true
      mockStoreState.deepResearchOwnerConversationId = 'test-conv-123'
      mockStoreState.activeDeepResearchMessageId = 'msg-1'
      mockCancelJob.mockResolvedValue({ cancelled: true })

      const consoleWarnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {})

      const { result } = renderHook(() => useDeepResearch())

      await act(async () => {
        await result.current.cancelCurrentJob()
      })

      // Advance past the 5s fallback timeout
      await act(async () => {
        vi.advanceTimersByTime(5000)
      })

      // Fallback should have run optimistic cleanup
      expect(mockStopAllDeepResearchSpinners).toHaveBeenCalled()
      expect(mockCompleteDeepResearch).toHaveBeenCalled()
      expect(mockSetStreaming).toHaveBeenCalledWith(false)
      expect(mockAddDeepResearchBanner).toHaveBeenCalledWith('cancelled', 'job-456', 'test-conv-123')
      expect(consoleWarnSpy).toHaveBeenCalledWith(
        expect.stringContaining('Cancel fallback'),
        expect.any(Number),
        expect.stringContaining('ms')
      )

      consoleWarnSpy.mockRestore()
    })

    test('fallback is a no-op if SSE already handled cleanup', async () => {
      mockStoreState.deepResearchJobId = 'job-456'
      mockStoreState.isDeepResearchStreaming = true
      mockCancelJob.mockResolvedValue({ cancelled: true })

      const { result } = renderHook(() => useDeepResearch())

      await act(async () => {
        await result.current.cancelCurrentJob()
      })

      // Simulate SSE delivering the status before fallback fires
      mockStoreState.isDeepResearchStreaming = false

      await act(async () => {
        vi.advanceTimersByTime(5000)
      })

      // Should NOT have called cleanup since streaming was already false
      expect(mockCompleteDeepResearch).not.toHaveBeenCalled()
      expect(mockAddDeepResearchBanner).not.toHaveBeenCalled()
    })

    test('does nothing when no job ID', async () => {
      mockStoreState.deepResearchJobId = null
      mockStoreState.isDeepResearchStreaming = false

      const { result } = renderHook(() => useDeepResearch())

      await act(async () => {
        await result.current.cancelCurrentJob()
      })

      expect(mockCancelJob).not.toHaveBeenCalled()
    })

    test('handles cancel errors gracefully', async () => {
      mockStoreState.deepResearchJobId = 'job-456'
      mockStoreState.isDeepResearchStreaming = true
      mockCancelJob.mockRejectedValue(new Error('Network error'))

      const consoleErrorSpy = vi.spyOn(console, 'error').mockImplementation(() => {})

      const { result } = renderHook(() => useDeepResearch())

      await act(async () => {
        await result.current.cancelCurrentJob()
      })

      expect(consoleErrorSpy).toHaveBeenCalledWith('Failed to cancel job:', expect.any(Error))

      consoleErrorSpy.mockRestore()
    })
  })

  describe('timeout detection', () => {
    // Skip: This test times out in CI due to waitFor interaction with fake timers
    // TODO: Fix fake timer interaction with waitFor or use a different approach
    test.skip('sets timeout warning after no events for 60 seconds', async () => {
      mockStoreState.deepResearchJobId = 'job-456'
      mockStoreState.isDeepResearchStreaming = true

      const { result } = renderHook(() => useDeepResearch())

      expect(result.current.isTimedOut).toBe(false)

      // Advance timers past the timeout threshold (60s) plus check interval (10s)
      act(() => {
        vi.advanceTimersByTime(70000)
      })

      await waitFor(() => {
        expect(result.current.isTimedOut).toBe(true)
      })
    })

    test('clears timeout on unmount', () => {
      mockStoreState.deepResearchJobId = 'job-456'
      mockStoreState.isDeepResearchStreaming = true

      const { unmount, result } = renderHook(() => useDeepResearch())

      unmount()

      // Advancing timers after unmount should not cause issues
      act(() => {
        vi.advanceTimersByTime(70000)
      })

      // Result should still be from before unmount
      expect(result.current.isTimedOut).toBe(false)
    })
  })

  /**
   * Advance fake timers and flush microtasks in one step.
   * vi.advanceTimersByTimeAsync processes microtasks between timer callbacks,
   * avoiding the hang that occurs with `setTimeout(r, 0)` under fake timers.
   */
  const advanceAndFlush = (ms: number) => vi.advanceTimersByTimeAsync(ms)

  /**
   * Helper: render hook, advance timers to connect in live (non-buffered) mode.
   *
   * Sets deepResearchStatus to 'submitted' so the hook's connect() computes
   * isReconnect=false → bufferReplay=false → buf.active=false.
   * This means SSE callbacks execute directly (live path) instead of buffering,
   * which is what the callback tests need.
   */
  const setupConnectedHook = async (overrides?: Partial<typeof mockStoreState>) => {
    if (overrides) Object.assign(mockStoreState, overrides)
    mockStoreState.deepResearchJobId = mockStoreState.deepResearchJobId || 'job-456'
    mockStoreState.isDeepResearchStreaming = true
    // 'submitted' makes isReconnect=false, disabling the replay buffer
    mockStoreState.deepResearchStatus = mockStoreState.deepResearchStatus || 'submitted'
    const hook = renderHook(() => useDeepResearch())
    // Advance past the 50ms StrictMode defer + flush microtasks
    await act(async () => { await advanceAndFlush(60) })
    vi.clearAllMocks()
    return hook
  }

  /**
   * Helper: render hook, advance timers to connect in replay-buffer mode.
   *
   * Sets deepResearchStatus to 'running' so the hook treats the connection
   * as a restored job and buffers historical events until stream.mode="live".
   */
  const setupBufferedHook = async (overrides?: Partial<typeof mockStoreState>) => {
    if (overrides) Object.assign(mockStoreState, overrides)
    mockStoreState.deepResearchJobId = mockStoreState.deepResearchJobId || 'job-456'
    mockStoreState.isDeepResearchStreaming = true
    mockStoreState.deepResearchStatus = mockStoreState.deepResearchStatus || 'running'
    const hook = renderHook(() => useDeepResearch())
    await act(async () => { await advanceAndFlush(60) })
    vi.clearAllMocks()
    return hook
  }

  describe('SSE callbacks', () => {
    test('onStreamStart sets status to researching', async () => {
      await setupConnectedHook()

      act(() => {
        mockClient?.callbacks.onStreamStart?.('job-456')
      })

      expect(mockSetCurrentStatus).toHaveBeenCalledWith('researching')
    })

    test('does not flush a replay buffer after the active job changes', async () => {
      await setupBufferedHook({
        deepResearchJobId: 'job-old',
        deepResearchStatus: 'running',
      })

      act(() => {
        mockClient?.callbacks.onOutputUpdate?.('Old job report', 'final_report')
        mockClient?.callbacks.onCitationUpdate?.('https://old.example', 'Old citation', true)
      })

      mockStoreState.deepResearchJobId = 'job-new'

      act(() => {
        mockClient?.callbacks.onStreamMode?.('live')
      })

      expect(useChatStore.setState).not.toHaveBeenCalled()
      expect(mockSetCurrentStatus).not.toHaveBeenCalledWith('researching')
    })

    test('onJobStatus success completes research and patches message', async () => {
      await setupConnectedHook({
        reportContent: 'Test report',
        activeDeepResearchMessageId: 'msg-123',
      })

      vi.mocked(useChatStore).getState = vi.fn(() => ({
        ...mockStoreState,
        reportContent: 'Test report',
        deepResearchLLMSteps: [],
        deepResearchToolCalls: [],
        deepResearchCitations: [],
        addErrorCard: mockAddErrorCard,
        deepResearchOwnerConversationId: 'test-conv-123',
        activeDeepResearchMessageId: 'msg-123',
      })) as unknown as typeof useChatStore.getState

      act(() => {
        mockClient?.callbacks.onJobStatus?.('success', undefined)
      })

      expect(mockSetCurrentStatus).toHaveBeenCalledWith('complete')
      expect(mockCompleteDeepResearch).toHaveBeenCalled()
      expect(mockAddDeepResearchBanner).toHaveBeenCalledWith(
        'success',
        'job-456',
        'test-conv-123',
        expect.objectContaining({
          totalTokens: expect.any(Number),
          toolCallCount: expect.any(Number),
        })
      )
      expect(mockPatchConversationMessage).toHaveBeenCalledWith(
        'test-conv-123',
        'msg-123',
        expect.objectContaining({
          deepResearchJobStatus: 'success',
          isDeepResearchActive: false,
        })
      )
    })

    test('records a real responseCompletedAt only when the terminal event arrives', async () => {
      await setupConnectedHook({
        reportContent: 'Test report',
        activeDeepResearchMessageId: 'msg-dur',
      })

      vi.mocked(useChatStore).getState = vi.fn(() => ({
        ...mockStoreState,
        reportContent: 'Test report',
        deepResearchLLMSteps: [],
        deepResearchToolCalls: [],
        deepResearchCitations: [],
        addErrorCard: mockAddErrorCard,
        deepResearchOwnerConversationId: 'test-conv-123',
        activeDeepResearchMessageId: 'msg-dur',
      })) as unknown as typeof useChatStore.getState

      act(() => {
        mockClient?.callbacks.onWorkflowStart?.('researcher-agent', 'query', 'event-1', 'agent-1')
      })

      const midRunPatch = mockPatchConversationMessage.mock.calls.find(
        ([, , patch]) => (patch as Record<string, unknown>).responseCompletedAt !== undefined
      )
      expect(midRunPatch).toBeUndefined()

      act(() => {
        mockClient?.callbacks.onJobStatus?.('success', undefined)
      })

      const terminalPatch = mockPatchConversationMessage.mock.calls.at(-1)?.[2] as Record<string, unknown>
      expect(terminalPatch.deepResearchJobStatus).toBe('success')
      expect(terminalPatch.responseCompletedAt).toBeInstanceOf(Date)
    })

    test('onJobStatus failure stops todos and shows error', async () => {
      await setupConnectedHook()

      act(() => {
        mockClient?.callbacks.onJobStatus?.('failure', 'Something went wrong')
      })

      expect(mockSetCurrentStatus).toHaveBeenCalledWith('error')
      expect(mockStopAllDeepResearchSpinners).toHaveBeenCalled()
      expect(mockCompleteDeepResearch).toHaveBeenCalled()
      expect(mockAddErrorCard).toHaveBeenCalledWith(
        'agent.deep_research_failed',
        'Something went wrong'
      )
    })

    test('onJobStatus interrupted by user shows cancelled banner', async () => {
      await setupConnectedHook()

      act(() => {
        mockClient?.callbacks.onJobStatus?.('interrupted', 'cancelled by user')
      })

      expect(mockAddDeepResearchBanner).toHaveBeenCalledWith(
        'cancelled',
        'job-456',
        'test-conv-123'
      )
      expect(mockAddErrorCard).not.toHaveBeenCalled()
    })

    test('onJobStatus interrupted for non-user reason shows failure banner', async () => {
      await setupConnectedHook()

      act(() => {
        mockClient?.callbacks.onJobStatus?.('interrupted', 'worker lost during reconnect')
      })

      expect(mockAddDeepResearchBanner).toHaveBeenCalledWith(
        'failure',
        'job-456',
        'test-conv-123'
      )
      expect(mockAddErrorCard).toHaveBeenCalledWith(
        'agent.deep_research_failed',
        'worker lost during reconnect'
      )
    })

    test('onJobStatus interrupted without error shows fallback failure', async () => {
      await setupConnectedHook()

      expect(() => {
        act(() => {
          mockClient?.callbacks.onJobStatus?.('interrupted', undefined)
        })
      }).not.toThrow()

      expect(mockAddDeepResearchBanner).toHaveBeenCalledWith(
        'failure',
        'job-456',
        'test-conv-123'
      )
      expect(mockAddErrorCard).toHaveBeenCalledWith(
        'agent.deep_research_failed',
        'Research was interrupted before completion.'
      )
    })

    test('onWorkflowStart adds thinking step and agent', async () => {
      await setupConnectedHook()

      act(() => {
        mockClient?.callbacks.onWorkflowStart?.('test-workflow', 'test input', 'event-1', 'agent-123')
      })

      expect(mockAddThinkingStep).toHaveBeenCalledWith(
        expect.objectContaining({
          category: 'agents',
          functionName: 'test-workflow',
          displayName: 'test-workflow',
          isDeepResearch: true,
        })
      )
      expect(mockAddDeepResearchAgentWithId).toHaveBeenCalledWith('agent-123', {
        name: 'test-workflow',
        input: 'test input',
      })
    })

    test('onWorkflowEnd completes thinking step and agent', async () => {
      await setupConnectedHook()

      // Start workflow first
      act(() => {
        mockClient?.callbacks.onWorkflowStart?.('test-workflow', 'test input', 'event-1', 'agent-123')
      })

      act(() => {
        mockClient?.callbacks.onWorkflowEnd?.('test-workflow', 'test output', 'event-2', 'agent-123')
      })

      expect(mockCompleteThinkingStep).toHaveBeenCalled()
      expect(mockCompleteDeepResearchAgent).toHaveBeenCalledWith('agent-123', 'test output')
    })

    test('onLLMStart adds LLM step', async () => {
      await setupConnectedHook()

      act(() => {
        mockClient?.callbacks.onLLMStart?.('gpt-4', 'test-workflow')
      })

      expect(mockAddThinkingStep).toHaveBeenCalledWith(
        expect.objectContaining({
          category: 'agents',
          functionName: 'llm:gpt-4',
          isDeepResearch: true,
        })
      )
      expect(mockAddDeepResearchLLMStep).toHaveBeenCalledWith({
        name: 'gpt-4',
        workflow: 'test-workflow',
        content: '',
      })
    })

    test('onLLMChunk appends to LLM step', async () => {
      await setupConnectedHook()

      // Start LLM first
      act(() => {
        mockClient?.callbacks.onLLMStart?.('gpt-4', 'test-workflow')
      })

      act(() => {
        mockClient?.callbacks.onLLMChunk?.('Hello ')
      })

      expect(mockAppendToThinkingStep).toHaveBeenCalledWith('step-1', 'Hello ')
      expect(mockAppendToDeepResearchLLMStep).toHaveBeenCalledWith('llm-step-1', 'Hello ')
    })

    test('onLLMEnd completes LLM step with thinking and usage', async () => {
      await setupConnectedHook()

      // Start LLM first
      act(() => {
        mockClient?.callbacks.onLLMStart?.('gpt-4', 'test-workflow')
      })

      act(() => {
        mockClient?.callbacks.onLLMEnd?.(
          'Final output',
          'Thinking about this...',
          { input_tokens: 100, output_tokens: 50 }
        )
      })

      expect(mockCompleteThinkingStep).toHaveBeenCalled()
      expect(mockCompleteDeepResearchLLMStep).toHaveBeenCalledWith(
        'llm-step-1',
        'Thinking about this...',
        { input_tokens: 100, output_tokens: 50 }
      )
    })

    test('onToolStart adds tool call', async () => {
      await setupConnectedHook()

      act(() => {
        mockClient?.callbacks.onToolStart?.('web_search', { query: 'test' }, 'test-workflow', 'event-1', 'agent-123')
      })

      expect(mockSetCurrentStatus).toHaveBeenCalledWith('searching')
      expect(mockAddThinkingStep).toHaveBeenCalledWith(
        expect.objectContaining({
          category: 'tools',
          functionName: 'web_search',
          isDeepResearch: true,
        })
      )
      expect(mockAddDeepResearchToolCall).toHaveBeenCalledWith({
        name: 'web_search',
        input: { query: 'test' },
        workflow: 'test-workflow',
        agentId: 'agent-123',
      })
    })

    test('onToolEnd completes tool call', async () => {
      await setupConnectedHook()

      // Start tool first
      act(() => {
        mockClient?.callbacks.onToolStart?.('web_search', { query: 'test' }, 'test-workflow', 'event-1', 'agent-123')
      })

      act(() => {
        mockClient?.callbacks.onToolEnd?.('web_search', 'Search results...', 'event-2')
      })

      expect(mockCompleteThinkingStep).toHaveBeenCalled()
      expect(mockCompleteDeepResearchToolCall).toHaveBeenCalledWith('tool-call-1', 'Search results...')
      expect(mockSetCurrentStatus).toHaveBeenCalledWith('researching')
    })

    test('onTodoUpdate sets todos in store', async () => {
      await setupConnectedHook()

      const todos = [
        { id: '1', content: 'Task 1', status: 'pending' as const },
        { id: '2', content: 'Task 2', status: 'in_progress' as const },
      ]

      act(() => {
        mockClient?.callbacks.onTodoUpdate?.(todos)
      })

      expect(mockSetDeepResearchTodos).toHaveBeenCalledWith(todos)
    })

    test('onTodoUpdate ignores workflow-scoped sub-agent todos', async () => {
      await setupConnectedHook()

      const todos = [
        { id: '1', content: 'Review sources', status: 'in_progress' as const },
      ]

      act(() => {
        mockClient?.callbacks.onTodoUpdate?.(todos, 'deep_research_agent')
      })

      expect(mockSetDeepResearchTodos).not.toHaveBeenCalled()
    })

    test('replay buffer restores root-level todos after refresh', async () => {
      await setupBufferedHook()

      const todos = [
        { id: '1', content: 'Replay task', status: 'pending' as const },
      ]

      act(() => {
        mockClient?.callbacks.onTodoUpdate?.(todos)
        mockClient?.callbacks.onStreamMode?.('live')
      })

      const replayCommit = vi.mocked(useChatStore.setState).mock.calls[0]?.[0]
      expect(replayCommit).toEqual(expect.any(Function))

      const updates = (replayCommit as unknown as (state: { currentStatus: string }) => Record<string, unknown>)({
        currentStatus: 'researching',
      })
      expect(updates.deepResearchTodos).toEqual([
        { id: 'todo-0-replay-task', content: 'Replay task', status: 'pending' },
      ])
    })

    test('replay buffer keeps a start-only workflow agent running after refresh', async () => {
      await setupBufferedHook()

      act(() => {
        mockClient?.callbacks.onWorkflowStart?.(
          'researcher-agent',
          'Research query',
          'event-1',
          'researcher-1'
        )
        mockClient?.callbacks.onStreamMode?.('live')
      })

      const replayCommit = vi.mocked(useChatStore.setState).mock.calls[0]?.[0]
      const updates = (replayCommit as unknown as (state: { currentStatus: string }) => Record<string, unknown>)({
        currentStatus: 'researching',
      })
      const agents = updates.deepResearchAgents as Array<Record<string, unknown>>

      expect(agents).toHaveLength(1)
      expect(agents[0]).toMatchObject({
        id: 'researcher-1',
        name: 'researcher-agent',
        status: 'running',
      })
      expect(agents[0]).not.toHaveProperty('completedAt')
    })

    test('replay buffer completes a workflow agent only after workflow.end', async () => {
      await setupBufferedHook()

      act(() => {
        mockClient?.callbacks.onWorkflowStart?.(
          'researcher-agent',
          'Research query',
          'event-1',
          'researcher-1'
        )
        mockClient?.callbacks.onWorkflowEnd?.(
          'researcher-agent',
          'Research notes',
          'event-2',
          'researcher-1'
        )
        mockClient?.callbacks.onStreamMode?.('live')
      })

      const replayCommit = vi.mocked(useChatStore.setState).mock.calls[0]?.[0]
      const updates = (replayCommit as unknown as (state: { currentStatus: string }) => Record<string, unknown>)({
        currentStatus: 'researching',
      })
      const agents = updates.deepResearchAgents as Array<Record<string, unknown>>

      expect(agents[0]).toMatchObject({
        id: 'researcher-1',
        status: 'complete',
        output: 'Research notes',
        completedAt: expect.any(Date),
      })
    })

    test('replay buffer ignores workflow-scoped sub-agent todos after refresh', async () => {
      await setupBufferedHook()

      const todos = [
        { id: '1', content: 'Sub-agent task', status: 'pending' as const },
      ]

      act(() => {
        mockClient?.callbacks.onTodoUpdate?.(todos, 'researcher-agent')
        mockClient?.callbacks.onStreamMode?.('live')
      })

      const replayCommit = vi.mocked(useChatStore.setState).mock.calls[0]?.[0]
      expect(replayCommit).toEqual(expect.any(Function))

      const updates = (replayCommit as unknown as (state: { currentStatus: string }) => Record<string, unknown>)({
        currentStatus: 'researching',
      })
      expect(updates).not.toHaveProperty('deepResearchTodos')
    })

    test('replay buffer merges file content with a later metadata-only update', async () => {
      await setupBufferedHook()

      act(() => {
        // Legacy content event, then a durable metadata-only event for the same file.
        // Replay must merge (not overwrite) so the earlier content survives.
        mockClient?.callbacks.onFileUpdate?.({ filename: 'chart.png', content: 'BASE64DATA' })
        mockClient?.callbacks.onFileUpdate?.({ filename: 'chart.png', artifactId: 'art-1', mimeType: 'image/png' })
        mockClient?.callbacks.onStreamMode?.('live')
      })

      const replayCommit = vi.mocked(useChatStore.setState).mock.calls[0]?.[0]
      expect(replayCommit).toEqual(expect.any(Function))

      const updates = (replayCommit as unknown as (state: { currentStatus: string }) => Record<string, unknown>)({
        currentStatus: 'researching',
      })
      const files = updates.deepResearchFiles as Array<Record<string, unknown>>
      expect(files).toHaveLength(1)
      expect(files[0]).toMatchObject({
        filename: 'chart.png',
        content: 'BASE64DATA',
        artifactId: 'art-1',
        mimeType: 'image/png',
      })
    })

    test('replay buffer keeps interleaved same-tool outputs id-keyed to each worker', async () => {
      await setupBufferedHook()

      act(() => {
        mockClient?.callbacks.onToolStart?.('web_search', { q: 'a' }, 'researcher-agent', 'e1', 'researcher-1', false)
        mockClient?.callbacks.onToolStart?.('web_search', { q: 'b' }, 'researcher-agent', 'e2', 'researcher-2', false)
        mockClient?.callbacks.onToolEnd?.('web_search', 'result A', 'e3', 'researcher-1')
        mockClient?.callbacks.onToolEnd?.('web_search', 'result B', 'e4', 'researcher-2')
        mockClient?.callbacks.onStreamMode?.('live')
      })

      const replayCommit = vi.mocked(useChatStore.setState).mock.calls[0]?.[0]
      const updates = (replayCommit as unknown as (state: { currentStatus: string }) => Record<string, unknown>)({
        currentStatus: 'researching',
      })
      const toolCalls = updates.deepResearchToolCalls as Array<Record<string, unknown>>
      const rowA = toolCalls.find((t) => t.agentId === 'researcher-1')
      const rowB = toolCalls.find((t) => t.agentId === 'researcher-2')

      expect(rowA?.input).toEqual({ q: 'a' })
      expect(rowB?.input).toEqual({ q: 'b' })
      expect(String(rowA?.output)).toContain('result A')
      expect(String(rowA?.output)).not.toContain('result B')
      expect(String(rowB?.output)).toContain('result B')
      expect(String(rowB?.output)).not.toContain('result A')
    })

    test('replay buffer id-keys same-tool outputs even when the ends arrive out of order', async () => {
      await setupBufferedHook()

      act(() => {
        mockClient?.callbacks.onToolStart?.('web_search', { q: 'a' }, 'researcher-agent', 'e1', 'researcher-1', false)
        mockClient?.callbacks.onToolStart?.('web_search', { q: 'b' }, 'researcher-agent', 'e2', 'researcher-2', false)
        mockClient?.callbacks.onToolEnd?.('web_search', 'result B', 'e4', 'researcher-2')
        mockClient?.callbacks.onToolEnd?.('web_search', 'result A', 'e3', 'researcher-1')
        mockClient?.callbacks.onStreamMode?.('live')
      })

      const replayCommit = vi.mocked(useChatStore.setState).mock.calls[0]?.[0]
      const updates = (replayCommit as unknown as (state: { currentStatus: string }) => Record<string, unknown>)({
        currentStatus: 'researching',
      })
      const toolCalls = updates.deepResearchToolCalls as Array<Record<string, unknown>>
      const rowA = toolCalls.find((t) => t.agentId === 'researcher-1')
      const rowB = toolCalls.find((t) => t.agentId === 'researcher-2')

      expect(String(rowA?.output)).toContain('result A')
      expect(String(rowB?.output)).toContain('result B')
    })

    test('replay buffer keeps interleaved same-model LLM thinking id-keyed to each worker', async () => {
      await setupBufferedHook()

      act(() => {
        mockClient?.callbacks.onLLMStart?.('gpt-4', 'researcher-agent', 'researcher-1')
        mockClient?.callbacks.onLLMChunk?.('chunk-1')
        mockClient?.callbacks.onLLMStart?.('gpt-4', 'researcher-agent', 'researcher-2')
        mockClient?.callbacks.onLLMChunk?.('chunk-2')
        mockClient?.callbacks.onLLMEnd?.('out A', 'thinking A', { input_tokens: 1, output_tokens: 2 }, 'gpt-4', 'researcher-1')
        mockClient?.callbacks.onLLMEnd?.('out B', 'thinking B', { input_tokens: 3, output_tokens: 4 }, 'gpt-4', 'researcher-2')
        mockClient?.callbacks.onStreamMode?.('live')
      })

      const replayCommit = vi.mocked(useChatStore.setState).mock.calls[0]?.[0]
      const updates = (replayCommit as unknown as (state: { currentStatus: string }) => Record<string, unknown>)({
        currentStatus: 'researching',
      })
      const llmSteps = updates.deepResearchLLMSteps as Array<Record<string, unknown>>
      const rowFor = (content: string) => llmSteps.find((s) => s.content === content)

      expect(rowFor('chunk-1')?.thinking).toBe('thinking A')
      expect(rowFor('chunk-1')?.usage).toEqual({ input_tokens: 1, output_tokens: 2 })
      expect(rowFor('chunk-2')?.thinking).toBe('thinking B')
      expect(rowFor('chunk-2')?.usage).toEqual({ input_tokens: 3, output_tokens: 4 })
    })

    test('onCitationUpdate adds citation to store', async () => {
      await setupConnectedHook()

      act(() => {
        mockClient?.callbacks.onCitationUpdate?.('https://example.com', 'Citation content', true)
      })

      expect(mockAddDeepResearchCitation).toHaveBeenCalledWith(
        'https://example.com',
        'Citation content',
        true,
        undefined
      )
    })

    test('onCitationUpdate forwards citation verification to store', async () => {
      await setupConnectedHook()
      const verification = { verified: false, confidence: 0.3, matchKind: 'host_only' }

      act(() => {
        mockClient?.callbacks.onCitationUpdate?.('https://example.com/guess', 'Guess', true, verification)
      })

      expect(mockAddDeepResearchCitation).toHaveBeenCalledWith(
        'https://example.com/guess',
        'Guess',
        true,
        verification
      )
    })

    test('onFileUpdate adds file to store', async () => {
      await setupConnectedHook()

      act(() => {
        mockClient?.callbacks.onFileUpdate?.({ filename: 'report.md', content: '# Report content' })
      })

      expect(mockAddDeepResearchFile).toHaveBeenCalledWith({
        filename: 'report.md',
        content: '# Report content',
      })
    })

    test('onFileUpdate sets status to writing when report.md is received', async () => {
      await setupConnectedHook()

      act(() => {
        mockClient?.callbacks.onFileUpdate?.({ filename: 'report.md', content: '# Final report' })
      })

      expect(mockSetCurrentStatus).toHaveBeenCalledWith('writing')
    })

    test('onFileUpdate sets status to writing for path ending in report.md', async () => {
      await setupConnectedHook()

      act(() => {
        mockClient?.callbacks.onFileUpdate?.({ filename: 'artifacts/report.md', content: '# Final report' })
      })

      expect(mockSetCurrentStatus).toHaveBeenCalledWith('writing')
    })

    test('onFileUpdate does not set writing status for non-report files', async () => {
      await setupConnectedHook()

      act(() => {
        mockClient?.callbacks.onFileUpdate?.({ filename: 'notes.md', content: '# Some notes' })
      })

      expect(mockAddDeepResearchFile).toHaveBeenCalledWith({
        filename: 'notes.md',
        content: '# Some notes',
      })
      expect(mockSetCurrentStatus).not.toHaveBeenCalledWith('writing')
    })

    test('onOutputUpdate sets report content for final_report output', async () => {
      await setupConnectedHook()

      act(() => {
        mockClient?.callbacks.onOutputUpdate?.('Report content here', 'final_report')
      })

      expect(mockSetReportContent).toHaveBeenCalledWith('Report content here', 'final_report')
      expect(mockSetCurrentStatus).toHaveBeenCalledWith('writing')
    })

    test('onOutputUpdate ignores uncategorized output so failed jobs do not fill the report tab with JSON', async () => {
      await setupConnectedHook()

      act(() => {
        mockClient?.callbacks.onOutputUpdate?.('{"status":"failure","error":"worker failed"}')
      })

      expect(mockSetReportContent).not.toHaveBeenCalled()
      expect(mockSetCurrentStatus).not.toHaveBeenCalledWith('writing')
    })

    test('onOutputUpdate ignores research notes because the report tab only shows final reports', async () => {
      await setupConnectedHook()

      act(() => {
        mockClient?.callbacks.onOutputUpdate?.('Draft research notes', 'research_notes')
      })

      expect(mockSetReportContent).not.toHaveBeenCalled()
      expect(mockSetCurrentStatus).not.toHaveBeenCalledWith('writing')
    })

    test('onComplete does not throw in live mode', async () => {
      await setupConnectedHook()

      // In live mode (buf.active=false), onComplete is a no-op.
      // Just verify it doesn't throw.
      act(() => {
        mockClient?.callbacks.onComplete?.()
      })
    })

    test('onError logs error and performs full cleanup when backend is unreachable', async () => {
      await setupConnectedHook({
        activeDeepResearchMessageId: 'msg-123',
        reportContent: 'Partial report',
      })

      mockCheckBackendHealthCached.mockResolvedValue(false)
      vi.mocked(useChatStore).getState = vi.fn(() => ({
        ...mockStoreState,
        isDeepResearchStreaming: true,
        deepResearchOwnerConversationId: 'test-conv-123',
        activeDeepResearchMessageId: 'msg-123',
        reportContent: 'Partial report',
        addErrorCard: mockAddErrorCard,
        stopAllDeepResearchSpinners: mockStopAllDeepResearchSpinners,
        patchConversationMessage: mockPatchConversationMessage,
        addDeepResearchBanner: mockAddDeepResearchBanner,
        completeDeepResearch: mockCompleteDeepResearch,
        setStreaming: mockSetStreaming,
        setStreamLoaded: mockSetStreamLoaded,
      })) as unknown as typeof useChatStore.getState

      const consoleWarnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {})
      const consoleErrorSpy = vi.spyOn(console, 'error').mockImplementation(() => {})

      const testError = new Error('Connection lost')

      await act(async () => {
        await mockClient?.callbacks.onError?.(testError)
      })

      expect(consoleWarnSpy).toHaveBeenCalledWith('Deep research SSE error:', 'Connection lost')
      expect(consoleErrorSpy).toHaveBeenCalledWith('Deep research SSE failed (backend unreachable):', testError)
      expect(mockSetCurrentStatus).toHaveBeenCalledWith('error')
      expect(mockAddErrorCard).toHaveBeenCalledWith(
        'agent.deep_research_failed',
        'Connection lost',
        testError.stack
      )

      expect(mockPatchConversationMessage).toHaveBeenCalledWith(
        'test-conv-123',
        'msg-123',
        expect.objectContaining({
          deepResearchJobStatus: 'failure',
          isDeepResearchActive: false,
          showViewReport: true,
        })
      )
      expect(mockAddDeepResearchBanner).toHaveBeenCalledWith('failure', 'job-456', 'test-conv-123')
      expect(mockStopAllDeepResearchSpinners).toHaveBeenCalled()
      expect(mockClient?.disconnect).toHaveBeenCalled()
      expect(mockSetStreamLoaded).toHaveBeenCalledWith(true)
      expect(mockCompleteDeepResearch).toHaveBeenCalled()
      expect(mockSetStreaming).toHaveBeenCalledWith(false)

      consoleWarnSpy.mockRestore()
      consoleErrorSpy.mockRestore()
    })

    test('onError surfaces error when backend is reachable (no longer silently swallowed)', async () => {
      await setupConnectedHook()

      mockCheckBackendHealthCached.mockResolvedValue(true)
      vi.mocked(useChatStore).getState = vi.fn(() => ({
        ...mockStoreState,
        isDeepResearchStreaming: true,
        addErrorCard: mockAddErrorCard,
        stopAllDeepResearchSpinners: mockStopAllDeepResearchSpinners,
      })) as unknown as typeof useChatStore.getState

      const consoleWarnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {})

      await act(async () => {
        await mockClient?.callbacks.onError?.(new Error('Transient error'))
      })

      // Errors are now surfaced instead of silently swallowed when backend is healthy
      expect(mockAddErrorCard).toHaveBeenCalledWith(
        'connection.failed',
        'Transient error',
        expect.any(String)
      )
      expect(mockCompleteDeepResearch).toHaveBeenCalled()
      expect(mockSetStreaming).toHaveBeenCalledWith(false)

      consoleWarnSpy.mockRestore()
    })

    test('onError skips cleanup when research already in terminal state', async () => {
      await setupConnectedHook()

      mockCheckBackendHealthCached.mockResolvedValue(false)
      vi.mocked(useChatStore).getState = vi.fn(() => ({
        ...mockStoreState,
        isDeepResearchStreaming: false,
        deepResearchStatus: 'failure',
        addErrorCard: mockAddErrorCard,
        stopAllDeepResearchSpinners: mockStopAllDeepResearchSpinners,
      })) as unknown as typeof useChatStore.getState

      const consoleWarnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {})

      await act(async () => {
        await mockClient?.callbacks.onError?.(new Error('Late error'))
      })

      expect(mockAddErrorCard).not.toHaveBeenCalled()
      expect(mockCompleteDeepResearch).not.toHaveBeenCalled()

      consoleWarnSpy.mockRestore()
    })

    test('onDisconnect does not throw in live mode', async () => {
      await setupConnectedHook()

      // In live mode (buf.active=false), onDisconnect is a no-op.
      // Just verify it doesn't throw.
      act(() => {
        mockClient?.callbacks.onDisconnect?.()
      })
    })
  })

  describe('formatDuration and formatTokens (integration)', () => {
    test('shows formatted stats on job success', async () => {
      await setupConnectedHook({
        reportContent: 'Test report',
        activeDeepResearchMessageId: 'msg-123',
      })

      vi.mocked(useChatStore).getState = vi.fn(() => ({
        ...mockStoreState,
        reportContent: 'Test report',
        deepResearchLLMSteps: [
          { usage: { input_tokens: 500, output_tokens: 200 } },
          { usage: { input_tokens: 300, output_tokens: 150 } },
        ],
        deepResearchToolCalls: [{ id: '1' }, { id: '2' }, { id: '3' }],
        deepResearchCitations: [
          { isCited: true },
          { isCited: true },
          { isCited: false },
        ],
        addErrorCard: mockAddErrorCard,
        deepResearchOwnerConversationId: 'test-conv-123',
        activeDeepResearchMessageId: 'msg-123',
      })) as unknown as typeof useChatStore.getState

      // Simulate stream start to set start time
      act(() => {
        mockClient?.callbacks.onStreamStart?.('job-456')
      })

      // Advance time by 2 minutes
      act(() => {
        vi.advanceTimersByTime(120000)
      })

      act(() => {
        mockClient?.callbacks.onJobStatus?.('success', undefined)
      })

      // Check that stats are included in the banner call
      // totalTokens = 500 + 200 + 300 + 150 = 1150
      // toolCallCount = 3
      expect(mockAddDeepResearchBanner).toHaveBeenCalledWith(
        'success',
        'job-456',
        'test-conv-123',
        {
          totalTokens: 1150,
          toolCallCount: 3,
        }
      )
      expect(mockPatchConversationMessage).toHaveBeenCalledWith(
        'test-conv-123',
        'msg-123',
        expect.objectContaining({
          deepResearchJobStatus: 'success',
        })
      )
    })
  })
})
