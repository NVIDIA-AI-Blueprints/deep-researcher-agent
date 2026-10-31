// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * useLoadJobData Hook
 *
 * Loads deep research job data (report, citations, todos, tool calls, etc.)
 * either from the report API endpoint or by replaying the SSE stream.
 *
 * Use cases:
 * - "View Report" button clicks to load data on-demand
 * - Session restoration when reconnecting to completed jobs
 * - Importing historical job data
 *
 * Two primary methods:
 * 1. `loadReport(jobId)` - Quick fetch of just the report text via REST API
 * 2. `importJobStream(jobId)` - Full replay of SSE stream to get all artifacts
 *    (citations, todos, tool calls, agents, files, etc.)
 */

'use client'

import { useState, useCallback, useRef } from 'react'
import {
  getJobReport,
  getJobStatus,
  getJobState,
  createDeepResearchClient,
  type DeepResearchClient,
  type DeepResearchJobStatus,
  type FileArtifactUpdate,
  type TodoItem,
} from '@/adapters/api'
import { useChatStore } from '../store'
import type { CitationMatchKind } from '../types'
import {
  getDeepResearchJobLoadErrorDetails,
  getDeepResearchJobLoadFailureKind,
} from '../lib/deep-research-errors'
import { useAuth } from '@/adapters/auth'
import { useLayoutStore } from '@/features/layout/store'
import type { ResearchPanelTab } from '@/features/layout/types'
import { normalizeDeepResearchTodos } from '../lib/deep-research-todos'
import { createResearchCorrelator } from '../lib/deep-research-correlation'

const EXPIRED_REPORT_MESSAGE = 'This research report is no longer available.'
const BACKEND_UNREACHABLE_MESSAGE = 'The backend is not reachable. Start the backend and try again.'
const STREAM_BACKED_RESEARCH_TABS = new Set<ResearchPanelTab>(['tasks', 'thinking'])

interface JobLoadScope {
  jobId: string
  conversationId: string | null
  requiresJobMatch: boolean
}

const conversationHasJob = (
  conversation: ReturnType<typeof useChatStore.getState>['currentConversation'],
  jobId: string
): boolean => {
  return Boolean(conversation?.messages.some((m) => m.deepResearchJobId === jobId))
}

const createJobLoadScope = (jobId: string): JobLoadScope => {
  const state = useChatStore.getState()
  const currentConversation = state.currentConversation

  if (conversationHasJob(currentConversation, jobId)) {
    return { jobId, conversationId: currentConversation?.id ?? null, requiresJobMatch: true }
  }

  if (state.deepResearchJobId === jobId) {
    return {
      jobId,
      conversationId: state.deepResearchOwnerConversationId ?? currentConversation?.id ?? null,
      requiresJobMatch: true,
    }
  }

  const matchingConversation = state.conversations.find((conversation) =>
    conversation.messages.some((message) => message.deepResearchJobId === jobId)
  )

  if (matchingConversation) {
    return { jobId, conversationId: matchingConversation.id, requiresJobMatch: true }
  }

  // Tests and a few legacy entry points can load by job ID before the message
  // has been persisted. In that case we still bind to the current session ID
  // so switching sessions aborts the eventual replay commit.
  return { jobId, conversationId: currentConversation?.id ?? null, requiresJobMatch: false }
}

const isJobLoadScopeCurrent = (scope: JobLoadScope): boolean => {
  const state = useChatStore.getState()

  if (scope.conversationId && state.currentConversation?.id !== scope.conversationId) {
    return false
  }

  if (!scope.requiresJobMatch) {
    return true
  }

  return (
    state.deepResearchJobId === scope.jobId ||
    conversationHasJob(state.currentConversation, scope.jobId)
  )
}

export interface LoadJobDataOptions {
  /**
   * Whether to stream the full job to get all artifacts (citations, todos, tool calls, etc.)
   * If false, only fetches the final report via REST API.
   * @default false
   */
  streamFullJob?: boolean
}

export interface UseLoadJobDataReturn {
  /**
   * Load just the report text via REST API (fast, minimal data)
   * Use when you only need the final report content
   */
  loadReport: (jobId: string) => Promise<void>

  /**
   * Import the full job stream to get all artifacts
   * Replays the SSE stream from the beginning to populate:
   * - Report content
   * - Citations (referenced and cited sources)
   * - Todos/tasks
   * - Tool calls with inputs/outputs
   * - Agent/workflow executions
   * - File artifacts
   * - LLM thought traces
   *
   * Use when you need the complete research context, not just the report
   * Opens report tab after completion
   */
  importJobStream: (jobId: string) => Promise<void>

  /**
   * Import stream data only - does NOT change panel tab
   * Use when loading stream data for an already-open tab (e.g., Tasks/Thinking/Citations)
   */
  importStreamOnly: (jobId: string) => Promise<void>

  /**
   * Legacy method - calls either loadReport or importJobStream based on options
   * @deprecated Use loadReport or importJobStream directly for clarity
   */
  loadJobData: (jobId: string, options?: LoadJobDataOptions) => Promise<void>

  /**
   * Open a research panel tab and ensure the minimum data required for that tab is loaded.
   * Report uses the cheap report endpoint; detail tabs use full stream replay.
   */
  loadResearchPanelTab: (jobId: string, tab: ResearchPanelTab) => Promise<void>

  /** Whether data is currently being loaded */
  isLoading: boolean

  /** Error message if loading failed */
  error: string | null

  /** Clear any error state */
  clearError: () => void
}

/**
 * Hook for loading deep research job data on-demand
 *
 * Can either:
 * 1. Fetch just the report via REST API (fast, minimal data)
 * 2. Replay the full SSE stream to get all artifacts (comprehensive)
 */
export const useLoadJobData = (): UseLoadJobDataReturn => {
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const clientRef = useRef<DeepResearchClient | null>(null)

  const { idToken } = useAuth()
  const setReportContent = useChatStore((s) => s.setReportContent)
  const addDeepResearchToolCall = useChatStore((s) => s.addDeepResearchToolCall)
  const completeDeepResearchToolCall = useChatStore((s) => s.completeDeepResearchToolCall)
  const clearDeepResearch = useChatStore((s) => s.clearDeepResearch)
  const setCurrentStatus = useChatStore((s) => s.setCurrentStatus)
  const setLoadedJobId = useChatStore((s) => s.setLoadedJobId)
  const setStreamLoaded = useChatStore((s) => s.setStreamLoaded)
  const updateDeepResearchStatus = useChatStore((s) => s.updateDeepResearchStatus)
  const stopAllDeepResearchSpinners = useChatStore((s) => s.stopAllDeepResearchSpinners)
  const addErrorCard = useChatStore((s) => s.addErrorCard)
  const completeDeepResearch = useChatStore((s) => s.completeDeepResearch)
  const setStreaming = useChatStore((s) => s.setStreaming)
  const patchConversationMessage = useChatStore((s) => s.patchConversationMessage)
  const addDeepResearchBanner = useChatStore((s) => s.addDeepResearchBanner)

  const openRightPanel = useLayoutStore((s) => s.openRightPanel)
  const setResearchPanelTab = useLayoutStore((s) => s.setResearchPanelTab)

  const clearError = useCallback(() => {
    setError(null)
  }, [])

  const syncMissingJobToFailureState = useCallback(
    (jobId: string): void => {
      const state = useChatStore.getState()
      const conversation = state.currentConversation
      if (!conversation) return

      const trackingMessage = [...conversation.messages]
        .reverse()
        .find((m) => m.messageType === 'agent_response' && m.deepResearchJobId === jobId)

      let hadCompletedReport =
        Boolean(trackingMessage?.deepResearchReportExpired) ||
        Boolean(
          trackingMessage?.deepResearchJobStatus === 'success' &&
          (trackingMessage.showViewReport || trackingMessage.reportContent?.trim())
        )

      if (!hadCompletedReport) {
        hadCompletedReport = conversation.messages.some(
          (m) =>
            m.messageType === 'deep_research_banner' &&
            m.deepResearchBannerData?.jobId === jobId &&
            m.deepResearchBannerData?.bannerType === 'success'
        )
      }

      if (trackingMessage?.id) {
        patchConversationMessage(conversation.id, trackingMessage.id, {
          deepResearchJobStatus: 'failure',
          isDeepResearchActive: false,
          showViewReport: false,
          deepResearchReportExpired: hadCompletedReport,
        })
      }

      const hasTerminalBanner = conversation.messages.some(
        (m) =>
          m.messageType === 'deep_research_banner' &&
          m.deepResearchBannerData?.jobId === jobId &&
          ['success', 'failure', 'cancelled', 'expired'].includes(
            m.deepResearchBannerData?.bannerType || ''
          )
      )

      if (hadCompletedReport || !hasTerminalBanner) {
        addDeepResearchBanner(hadCompletedReport ? 'expired' : 'failure', jobId, conversation.id)
      }
    },
    [patchConversationMessage, addDeepResearchBanner]
  )

  /**
   * Load job data using REST API (report only)
   */
  const _loadReportOnly = useCallback(
    async (jobId: string): Promise<boolean> => {
      const response = await getJobReport(jobId, idToken || undefined)

      if (response.has_report && response.report) {
        setReportContent(response.report, 'final_report')
        return true
      }

      return false
    },
    [idToken, setReportContent]
  )

  /**
   * Load job state for additional artifacts (tool calls, outputs)
   * This is faster than streaming but provides less data than full stream replay
   */
  const loadJobState = useCallback(
    async (jobId: string, scope: JobLoadScope): Promise<void> => {
      try {
        const stateResponse = await getJobState(jobId, idToken || undefined)

        if (stateResponse.has_state && stateResponse.artifacts) {
          if (!isJobLoadScopeCurrent(scope)) return

          const { tools, outputs } = stateResponse.artifacts

          tools?.forEach(
            (tool: { name: string; input?: Record<string, unknown>; output?: string; is_sandbox?: boolean }) => {
              const toolCallId = addDeepResearchToolCall({
                name: tool.name,
                input: tool.input,
                workflow: undefined,
                isSandbox: tool.is_sandbox,
              })
              if (tool.output) {
                completeDeepResearchToolCall(toolCallId, tool.output)
              }
            }
          )

          outputs?.forEach((output: { type: string; content: string; output_category?: string }) => {
            if (output.type === 'report' || output.output_category === 'final_report') {
              setReportContent(output.content, 'final_report')
            }
          })
        }
      } catch (stateError) {
        console.warn('Failed to load job state:', stateError)
      }
    },
    [idToken, addDeepResearchToolCall, completeDeepResearchToolCall, setReportContent]
  )

  /**
   * Load job data using REST APIs (report + state) - fast approach
   * Fetches both report and state in parallel for speed
   */
  const loadJobDataFast = useCallback(
    async (jobId: string, scope: JobLoadScope): Promise<void> => {
      const [reportResult] = await Promise.allSettled([
        getJobReport(jobId, idToken || undefined),
        loadJobState(jobId, scope),
      ])

      if (!isJobLoadScopeCurrent(scope)) return

      if (
        reportResult.status === 'fulfilled' &&
        reportResult.value.has_report &&
        reportResult.value.report
      ) {
        setReportContent(reportResult.value.report, 'final_report')
      }
    },
    [idToken, loadJobState, setReportContent]
  )

  /**
   * Stream the full job from the beginning to get all artifacts.
   * Buffers ALL events in memory and commits to the store in a single
   * setState call when the stream ends, preventing hundreds of individual
   * set() calls that cause render storms and Aw Snap crashes.
   */
  const streamFullJob = useCallback(
    (jobId: string, scope: JobLoadScope): Promise<void> => {
      return new Promise((resolve, reject) => {
        let idCounter = 0

        // Accumulation buffer — everything stays here until the stream ends
        const buffer = {
          agents: new Map<
            string,
            { name: string; input?: string; output?: string; ended: boolean }
          >(),
          llmSteps: new Map<
            string,
            {
              name: string
              workflow?: string
              content: string
              thinking?: string
              usage?: { input_tokens: number; output_tokens: number }
            }
          >(),
          toolCalls: new Map<
            string,
            {
              name: string
              input?: Record<string, unknown>
              output?: string
              workflow?: string
              agentId?: string
              isSandbox?: boolean
            }
          >(),
          todos: null as TodoItem[] | null,
          citations: [] as Array<{
            url: string
            content: string
            isCited: boolean
            confidence?: number
            matchKind?: CitationMatchKind
          }>,
          files: new Map<string, FileArtifactUpdate>(), // filename -> latest event (deduped)
          reportContent: null as string | null,
        }

        const correlator = createResearchCorrelator({
          hasUserMessage: () => false,
          addThinkingStep: () => '',
          appendToThinkingStep: () => undefined,
          completeThinkingStep: () => undefined,
          addAgent: (agentId, agent) => {
            if (!buffer.agents.has(agentId)) {
              buffer.agents.set(agentId, { name: agent.name, input: agent.input, ended: false })
            }
            return agentId
          },
          completeAgent: (agentId, output) => {
            const agent = buffer.agents.get(agentId)
            if (agent) {
              agent.output = output
              agent.ended = true
            }
          },
          addToolCall: (toolCall) => {
            const id = `tool-${idCounter++}`
            buffer.toolCalls.set(id, {
              name: toolCall.name,
              input: toolCall.input,
              workflow: toolCall.workflow,
              agentId: toolCall.agentId,
              isSandbox: toolCall.isSandbox,
            })
            return id
          },
          completeToolCall: (toolCallId, output) => {
            const toolCall = buffer.toolCalls.get(toolCallId)
            if (toolCall) toolCall.output = output ? JSON.stringify(output) : undefined
          },
          addLLMStep: (step) => {
            const id = `llm-${idCounter++}`
            buffer.llmSteps.set(id, { name: step.name, workflow: step.workflow, content: step.content })
            return id
          },
          appendLLMStep: (stepId, chunk) => {
            const step = buffer.llmSteps.get(stepId)
            if (step) step.content += chunk
          },
          completeLLMStep: (stepId, thinking, usage) => {
            const step = buffer.llmSteps.get(stepId)
            if (step) {
              step.thinking = thinking
              step.usage = usage
            }
          },
        })

        /**
         * Convert buffer to store-compatible arrays and write everything
         * in a single useChatStore.setState() call.
         */
        const commitToStore = (): boolean => {
          if (!isJobLoadScopeCurrent(scope)) {
            return false
          }

          const now = new Date()

          const agents = Array.from(buffer.agents.entries()).map(([id, a]) => ({
            id,
            name: a.name,
            input: a.input,
            output: a.output,
            status: a.ended ? ('complete' as const) : ('running' as const),
            startedAt: now,
            ...(a.ended && { completedAt: now }),
          }))

          const llmSteps = Array.from(buffer.llmSteps.entries()).map(([id, s]) => ({
            id,
            name: s.name,
            workflow: s.workflow,
            content: s.content,
            thinking: s.thinking,
            usage: s.usage,
            isComplete: true,
            timestamp: now,
          }))

          const toolCalls = Array.from(buffer.toolCalls.entries()).map(([id, t]) => ({
            id,
            name: t.name,
            input: t.input,
            output: t.output,
            workflow: t.workflow,
            agentId: t.agentId,
            isSandbox: t.isSandbox,
            status: 'complete' as const,
            timestamp: now,
          }))

          const citations = buffer.citations.map((c, idx) => ({
            id: `citation-${idx}`,
            url: c.url,
            content: c.content,
            isCited: c.isCited,
            confidence: c.confidence,
            matchKind: c.matchKind,
            timestamp: now,
          }))

          const files = Array.from(buffer.files.values()).map((file, idx) => ({
            id: `file-${idx}`,
            ...file,
            timestamp: now,
          }))

          const todos = buffer.todos ? normalizeDeepResearchTodos(buffer.todos) : undefined

          useChatStore.setState((state) => ({
            ...(buffer.reportContent !== null && {
              reportContent: buffer.reportContent,
              reportContentCategory: 'final_report' as const,
            }),
            ...(todos && { deepResearchTodos: todos }),
            ...(agents.length > 0 && { deepResearchAgents: agents }),
            ...(llmSteps.length > 0 && { deepResearchLLMSteps: llmSteps }),
            ...(toolCalls.length > 0 && { deepResearchToolCalls: toolCalls }),
            ...(citations.length > 0 && { deepResearchCitations: citations }),
            ...(files.length > 0 && { deepResearchFiles: files }),
            currentStatus: buffer.reportContent !== null ? 'complete' : state.currentStatus,
          }))
          return true
        }

        if (clientRef.current) {
          clientRef.current.disconnect()
          clientRef.current = null
        }

        let client: DeepResearchClient | null = null
        const disconnectReplayClient = (): void => {
          if (!client) return
          client.disconnect()
          if (clientRef.current === client) {
            clientRef.current = null
          }
        }

        client = createDeepResearchClient({
          jobId,
          authToken: idToken || undefined,
          callbacks: {
            onStreamStart: () => {
              if (!isJobLoadScopeCurrent(scope)) return
              setCurrentStatus('researching')
            },

            onJobStatus: (status: DeepResearchJobStatus, statusError?: string) => {
              if (status === 'success' || status === 'failure' || status === 'interrupted') {
                disconnectReplayClient()
                commitToStore()

                if (status === 'failure' && statusError) {
                  reject(new Error(statusError))
                } else {
                  resolve()
                }
              }
            },

            onWorkflowStart: (name, input, _eventId, agentId) => {
              if (!agentId) return
              correlator.onWorkflowStart(agentId, name, input)
            },

            onWorkflowEnd: (name, output, _eventId, agentId) => {
              if (!agentId) return
              correlator.onWorkflowEnd(agentId, name, output)
            },

            onLLMStart: (name, workflow, agentId) => {
              correlator.onLLMStart(agentId, name, workflow)
            },

            onLLMChunk: (chunk) => {
              correlator.onLLMChunk(chunk)
            },

            onLLMEnd: (_output, thinking, usage, name, agentId) => {
              correlator.onLLMEnd(agentId, name, thinking, usage)
            },

            onToolStart: (name, input, workflow, _eventId, agentId, isSandbox) => {
              if (name === 'task') return
              correlator.onToolStart(agentId, name, input, workflow, isSandbox)
            },

            onToolEnd: (name, output, _eventId, agentId) => {
              if (name === 'task') return
              correlator.onToolEnd(agentId, name, output)
            },

            onTodoUpdate: (todos: TodoItem[], workflow?: string) => {
              if (workflow) return
              buffer.todos = todos
            },

            onCitationUpdate: (url, content, isCited, confidence, matchKind) => {
              buffer.citations.push({
                url,
                content,
                isCited: isCited ?? false,
                confidence,
                matchKind,
              })
            },

            onFileUpdate: (file) => {
              // Merge like the live store: a later metadata-only event must not drop
              // content from an earlier event for the same filename during replay.
              const prev = buffer.files.get(file.filename)
              buffer.files.set(file.filename, prev ? { ...prev, ...file } : file)
            },

            onOutputUpdate: (content, outputCategory) => {
              if (outputCategory !== 'final_report') return
              buffer.reportContent = content
            },

            onComplete: () => {
              commitToStore()
              resolve()
            },

            onError: (err) => {
              console.error('Stream error while loading job data:', err)
              commitToStore()
              reject(err)
            },

            onDisconnect: () => {
              commitToStore()
              resolve()
            },
          },
        })

        clientRef.current = client
        client.connect()
      })
    },
    [idToken, setCurrentStatus]
  )

  /**
   * Main function to load job data
   * Checks ephemeral cache first - if data exists, just opens the panel
   * Otherwise fetches from backend
   */
  const loadJobData = useCallback(
    async (jobId: string, options: LoadJobDataOptions = {}): Promise<void> => {
      const { streamFullJob: shouldStreamFull = false } = options
      const scope = createJobLoadScope(jobId)

      // Check ephemeral cache first - if we have data for this job, just show it
      const currentState = useChatStore.getState()
      const hasReportData =
        currentState.deepResearchJobId === jobId &&
        currentState.reportContent &&
        currentState.reportContent.trim().length > 0

      // For stream requests, also check if stream is already loaded
      const hasStreamData =
        currentState.deepResearchJobId === jobId && currentState.deepResearchStreamLoaded

      // If we have what we need, just open the panel
      if (hasReportData && (!shouldStreamFull || hasStreamData)) {
        setResearchPanelTab('report')
        openRightPanel('research')
        return
      }

      setIsLoading(true)
      setError(null)

      try {
        const statusResponse = await getJobStatus(jobId, idToken || undefined)
        const jobStatus = statusResponse.status

        if (!isJobLoadScopeCurrent(scope)) return

        if (jobStatus !== 'success' && jobStatus !== 'failure' && jobStatus !== 'interrupted') {
          throw new Error(`Job is still ${jobStatus}. Cannot load data from incomplete job.`)
        }

        clearDeepResearch()
        updateDeepResearchStatus(jobStatus)

        if (shouldStreamFull) {
          await streamFullJob(jobId, scope)
          if (!isJobLoadScopeCurrent(scope)) return
          setStreamLoaded(true)
        } else {
          await loadJobDataFast(jobId, scope)
          if (!isJobLoadScopeCurrent(scope)) return
        }

        // Defensive cleanup: loaded data may have stale 'running' items
        // if the backend never sent completion events. Only treat as
        // successful for success jobs; interrupted/failed jobs should
        // leave un-attempted tasks as 'stopped'.
        stopAllDeepResearchSpinners(jobStatus === 'success')

        // Set job ID for cache tracking (so subsequent clicks show cached data)
        setLoadedJobId(jobId)

        setResearchPanelTab('report')
        openRightPanel('research')
      } catch (err) {
        if (!isJobLoadScopeCurrent(scope)) return

        const failureKind = getDeepResearchJobLoadFailureKind(err)
        const errorDetails = getDeepResearchJobLoadErrorDetails(err)
        const errorMessage =
          failureKind === 'unavailable'
            ? EXPIRED_REPORT_MESSAGE
            : failureKind === 'backend_unreachable'
              ? BACKEND_UNREACHABLE_MESSAGE
              : err instanceof Error
                ? err.message
                : 'Failed to load job data'
        setError(errorMessage)
        if (failureKind === 'unavailable') {
          syncMissingJobToFailureState(jobId)
          stopAllDeepResearchSpinners()
          completeDeepResearch()
          setStreaming(false)
        } else if (failureKind === 'backend_unreachable') {
          addErrorCard('connection.failed', errorMessage, errorDetails)
        } else {
          console.error('Failed to load job data:', err)
          addErrorCard('agent.deep_research_load_failed', errorMessage)
          stopAllDeepResearchSpinners()
          completeDeepResearch()
          setStreaming(false)
        }
      } finally {
        setIsLoading(false)
      }
    },
    [
      idToken,
      clearDeepResearch,
      loadJobDataFast,
      streamFullJob,
      setLoadedJobId,
      setStreamLoaded,
      updateDeepResearchStatus,
      stopAllDeepResearchSpinners,
      setResearchPanelTab,
      openRightPanel,
      addErrorCard,
      completeDeepResearch,
      setStreaming,
      syncMissingJobToFailureState,
    ]
  )

  /**
   * Public method: Load report + state via REST APIs (fast)
   */
  const loadReport = useCallback(
    async (jobId: string): Promise<void> => {
      await loadJobData(jobId, { streamFullJob: false })
    },
    [loadJobData]
  )

  /**
   * Public method: Import full job stream (slow but comprehensive)
   * Opens report tab after completion
   */
  const importJobStream = useCallback(
    async (jobId: string): Promise<void> => {
      await loadJobData(jobId, { streamFullJob: true })
    },
    [loadJobData]
  )

  /**
   * Import stream data only - does NOT change panel tab
   * Use when loading stream data for an already-open tab (e.g., Tasks/Thinking/Citations)
   * Checks ephemeral cache first to avoid duplicate API calls
   * Silently returns if job is still in progress (active SSE will populate data)
   */
  const importStreamOnly = useCallback(
    async (jobId: string): Promise<void> => {
      const scope = createJobLoadScope(jobId)

      // Check if stream is already loaded for this job
      const currentState = useChatStore.getState()
      if (currentState.deepResearchJobId === jobId && currentState.deepResearchStreamLoaded) {
        return
      }

      setIsLoading(true)
      setError(null)

      try {
        const statusResponse = await getJobStatus(jobId, idToken || undefined)
        const jobStatus = statusResponse.status

        if (!isJobLoadScopeCurrent(scope)) return

        if (jobStatus !== 'success' && jobStatus !== 'failure' && jobStatus !== 'interrupted') {
          // Job is still in progress - silently return (live SSE will populate data)
          // This is expected when opening tabs for active jobs
          setIsLoading(false)
          return
        }

        clearDeepResearch()
        updateDeepResearchStatus(jobStatus)
        await streamFullJob(jobId, scope)
        if (!isJobLoadScopeCurrent(scope)) return
        // Defensive cleanup: loaded data may have stale 'running' items.
        // Only mark as successful completion for success jobs; interrupted/failed
        // jobs should leave un-attempted tasks as 'stopped'.
        stopAllDeepResearchSpinners(jobStatus === 'success')
        setStreamLoaded(true)
        setLoadedJobId(jobId)
      } catch (err) {
        if (!isJobLoadScopeCurrent(scope)) return

        const failureKind = getDeepResearchJobLoadFailureKind(err)
        const errorDetails = getDeepResearchJobLoadErrorDetails(err)
        const errorMessage =
          failureKind === 'unavailable'
            ? EXPIRED_REPORT_MESSAGE
            : failureKind === 'backend_unreachable'
              ? BACKEND_UNREACHABLE_MESSAGE
              : err instanceof Error
                ? err.message
                : 'Failed to load stream data'
        setError(errorMessage)
        if (failureKind === 'unavailable') {
          syncMissingJobToFailureState(jobId)
          stopAllDeepResearchSpinners()
          completeDeepResearch()
          setStreaming(false)
        } else if (failureKind === 'backend_unreachable') {
          addErrorCard('connection.failed', errorMessage, errorDetails)
        } else {
          console.error('Failed to load stream data:', err)
          addErrorCard('agent.deep_research_load_failed', errorMessage)
          stopAllDeepResearchSpinners()
          completeDeepResearch()
          setStreaming(false)
        }
      } finally {
        setIsLoading(false)
      }
    },
    [
      idToken,
      clearDeepResearch,
      streamFullJob,
      stopAllDeepResearchSpinners,
      setStreamLoaded,
      setLoadedJobId,
      updateDeepResearchStatus,
      syncMissingJobToFailureState,
      addErrorCard,
      completeDeepResearch,
      setStreaming,
    ]
  )

  /**
   * Shared tab-loading policy for all ResearchPanel entry points.
   *
   * This keeps "View Report", banner actions, and direct tab clicks aligned:
   * - Report tab fetches the final report quickly via /report.
   * - Tasks/Thinking hydrate rich details by replaying /stream.
   */
  const loadResearchPanelTab = useCallback(
    async (jobId: string, tab: ResearchPanelTab): Promise<void> => {
      setResearchPanelTab(tab)
      openRightPanel('research')

      const currentState = useChatStore.getState()

      if (tab === 'report') {
        const hasReportForJob =
          currentState.deepResearchJobId === jobId && currentState.reportContent.trim().length > 0
        const isLiveReportForJob =
          currentState.deepResearchJobId === jobId && currentState.isDeepResearchStreaming

        if (hasReportForJob || isLiveReportForJob) {
          return
        }

        await loadJobData(jobId, { streamFullJob: false })
        return
      }

      if (STREAM_BACKED_RESEARCH_TABS.has(tab)) {
        const hasStreamForJob =
          currentState.deepResearchJobId === jobId && currentState.deepResearchStreamLoaded
        const isLiveStreamForJob =
          currentState.deepResearchJobId === jobId && currentState.isDeepResearchStreaming

        if (hasStreamForJob || isLiveStreamForJob) {
          return
        }

        await importStreamOnly(jobId)
      }
    },
    [loadJobData, importStreamOnly, setResearchPanelTab, openRightPanel]
  )

  return {
    loadReport,
    importJobStream,
    importStreamOnly,
    loadJobData,
    loadResearchPanelTab,
    isLoading,
    error,
    clearError,
  }
}
