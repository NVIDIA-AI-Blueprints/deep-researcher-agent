// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * useDeepResearch Hook
 *
 * Manages the SSE connection lifecycle for deep research jobs.
 * Automatically connects when a job ID is set in the store,
 * routes events to appropriate UI components, and handles reconnection.
 * Includes timeout detection for hung jobs.
 */

'use client'

import { useEffect, useRef, useCallback, useState } from 'react'
import { useShallow } from 'zustand/react/shallow'
import {
  createDeepResearchClient,
  cancelJob,
  type DeepResearchClient,
  type DeepResearchJobStatus,
  type FileArtifactUpdate,
  type TodoItem,
} from '@/adapters/api'
import { useChatStore } from '../store'
import type { CitationMatchKind } from '../types'
import { useAuth } from '@/adapters/auth'
import { useLayoutStore } from '@/features/layout/store'
import { checkBackendHealthCached } from '@/shared/hooks/use-backend-health'
import { isLikelyAuthRelatedTransportError, isDeepResearchReplayCompleteMode } from '../lib/transport-auth-signals'
import { normalizeDeepResearchTodos } from '../lib/deep-research-todos'
import { createResearchCorrelator, type ResearchCorrelator } from '../lib/deep-research-correlation'

/** Timeout in milliseconds before showing a warning (60 seconds) */
const TIMEOUT_WARNING_MS = 60000
/** How often to check for timeouts (10 seconds) */
const TIMEOUT_CHECK_INTERVAL_MS = 10000
/** Fallback timeout after cancel POST succeeds: if SSE doesn't deliver
 *  job.status "interrupted" within this window, clean up locally so the
 *  UI never stays stuck in a streaming state. */
const CANCEL_FALLBACK_TIMEOUT_MS = 5000
const USER_CANCELLED_ERROR_MARKER = 'cancelled by user'

const isUserCancelledStatus = (
  status: DeepResearchJobStatus,
  error?: string
): boolean => (
  status === 'interrupted' &&
  error?.toLowerCase().includes(USER_CANCELLED_ERROR_MARKER) === true
)

interface UseDeepResearchReturn {
  /** Whether deep research is currently streaming */
  isStreaming: boolean
  /** Current job ID */
  jobId: string | null
  /** Current job status */
  status: DeepResearchJobStatus | null
  /** Whether we're showing a timeout warning (no events received for too long) */
  isTimedOut: boolean
  /** Manually disconnect from the stream */
  disconnect: () => void
  /** Manually reconnect to the stream (uses last event ID) */
  reconnect: () => void
  /** Cancel the current job (useful for hung jobs) */
  cancelCurrentJob: () => Promise<void>
}

/**
 * Hook for managing deep research SSE streaming
 *
 * Automatically:
 * - Connects when deepResearchJobId is set in the store
 * - Routes SSE events to appropriate store actions
 * - Updates UI state (report, citations, thinking steps)
 * - Handles completion and errors
 */
export const useDeepResearch = (): UseDeepResearchReturn => {
  // Refs for SSE client lifecycle
  const clientRef = useRef<DeepResearchClient | null>(null)
  const connectRef = useRef<((jobId: string, bufferReplay?: boolean) => void) | null>(null)
  const lastEventTimeRef = useRef<number>(Date.now())
  const timeoutIntervalRef = useRef<NodeJS.Timeout | null>(null)
  const cancelFallbackRef = useRef<NodeJS.Timeout | null>(null)
  const researchStartTimeRef = useRef<number | null>(null)


  // State for timeout warning
  const [isTimedOut, setIsTimedOut] = useState(false)

  // Auth token for authenticated requests
  // Note: idToken is used for backend auth, not accessToken
  const { idToken, authRequired, error: authError } = useAuth()

  // Chat store — reactive state only
  const { deepResearchJobId, isDeepResearchStreaming, deepResearchStatus } =
    useChatStore(useShallow((s) => ({
      deepResearchJobId: s.deepResearchJobId,
      isDeepResearchStreaming: s.isDeepResearchStreaming,
      deepResearchStatus: s.deepResearchStatus,
    })))

  // Actions — stable references, won't trigger re-renders
  const updateDeepResearchStatus = useChatStore((s) => s.updateDeepResearchStatus)
  const completeDeepResearch = useChatStore((s) => s.completeDeepResearch)
  const addDeepResearchCitation = useChatStore((s) => s.addDeepResearchCitation)
  const setReportContent = useChatStore((s) => s.setReportContent)
  const addThinkingStep = useChatStore((s) => s.addThinkingStep)
  const appendToThinkingStep = useChatStore((s) => s.appendToThinkingStep)
  const completeThinkingStep = useChatStore((s) => s.completeThinkingStep)
  const setCurrentStatus = useChatStore((s) => s.setCurrentStatus)
  const setStreaming = useChatStore((s) => s.setStreaming)
  const setDeepResearchTodos = useChatStore((s) => s.setDeepResearchTodos)
  const stopAllDeepResearchSpinners = useChatStore((s) => s.stopAllDeepResearchSpinners)
  const addDeepResearchLLMStep = useChatStore((s) => s.addDeepResearchLLMStep)
  const appendToDeepResearchLLMStep = useChatStore((s) => s.appendToDeepResearchLLMStep)
  const completeDeepResearchLLMStep = useChatStore((s) => s.completeDeepResearchLLMStep)
  const addDeepResearchAgentWithId = useChatStore((s) => s.addDeepResearchAgentWithId)
  const completeDeepResearchAgent = useChatStore((s) => s.completeDeepResearchAgent)
  const addDeepResearchToolCall = useChatStore((s) => s.addDeepResearchToolCall)
  const completeDeepResearchToolCall = useChatStore((s) => s.completeDeepResearchToolCall)
  const addDeepResearchFile = useChatStore((s) => s.addDeepResearchFile)
  const patchConversationMessage = useChatStore((s) => s.patchConversationMessage)
  const addDeepResearchBanner = useChatStore((s) => s.addDeepResearchBanner)
  const setStreamLoaded = useChatStore((s) => s.setStreamLoaded)

  /**
   * Check if the current session owns the active deep research stream.
   * This prevents SSE events from mutating the wrong session.
   */
  const isOwnerActive = useCallback((expectedJobId?: string): boolean => {
    const state = useChatStore.getState()
    return Boolean(
      state.isDeepResearchStreaming &&
      (!expectedJobId || state.deepResearchJobId === expectedJobId) &&
      state.deepResearchOwnerConversationId &&
      state.currentConversation?.id === state.deepResearchOwnerConversationId
    )
  }, [])

  // Layout store for opening research panel
  const openRightPanel = useLayoutStore((s) => s.openRightPanel)
  const setResearchPanelTab = useLayoutStore((s) => s.setResearchPanelTab)

  const correlatorRef = useRef<ResearchCorrelator | null>(null)

  /**
   * Reset the timeout tracker - called when we receive any live event.
   */
  const resetTimeout = useCallback(() => {
    lastEventTimeRef.current = Date.now()
    setIsTimedOut(false)
  }, [])

  /**
   * Classify a deep research stream failure as auth-related or generic.
   * Used when the backend is healthy but the SSE stream errored,
   * which typically means the auth cookie or token drifted.
   */
  const getDeepResearchStreamFailure = useCallback(
    (message: string, details?: string): { code: string; message: string; details?: string } => {
      if (!authRequired) {
        return { code: 'connection.failed', message, details }
      }
      if (authError === 'RefreshAccessTokenError' || isLikelyAuthRelatedTransportError(message)) {
        return {
          code: 'auth.session_expired',
          message: 'Your session has expired. Please sign in again to continue.',
          details,
        }
      }
      return { code: 'connection.failed', message, details }
    },
    [authRequired, authError]
  )

  /**
   * Create and connect to the SSE stream
   */
  /**
   * Connect to the SSE stream from the beginning.
   *
   * Single connection, two internal phases:
   * 1. Buffer phase: all replayed events accumulate in plain JS objects (zero store writes).
   *    After 500ms of silence the buffer is flushed in one useChatStore.setState() call.
   * 2. Live phase: subsequent events go straight to individual store actions (fine for low volume).
   */
  const connect = useCallback(
    (jobId: string, bufferReplay = false) => {
      if (clientRef.current) {
        clientRef.current.disconnect()
        clientRef.current = null
      }

      // ---------- inline buffer for replay phase ----------
      // bufferReplay=true on page-refresh reconnect: buffer ALL replayed events
      // with zero store writes (like streamFullJob). The backend sends a
      // stream.mode event with mode="live" when replay is done. On that signal,
      // flush the buffer in one setState and switch to per-event live streaming.
      // A safety timeout (30s) flushes if the signal never arrives.
      // bufferReplay=false for fresh new jobs: events go straight to store.
      const SAFETY_TIMEOUT_MS = 30000
      const buf = {
        active: bufferReplay,
        timer: null as NodeJS.Timeout | null,
        idCounter: 0,
        agents: new Map<
          string,
          { name: string; input?: string; output?: string; ended: boolean }
        >(),
        llmSteps: new Map<string, { name: string; workflow?: string; content: string; thinking?: string; usage?: { input_tokens: number; output_tokens: number } }>(),
        toolCalls: new Map<string, { name: string; input?: Record<string, unknown>; output?: string; workflow?: string; agentId?: string; isSandbox?: boolean }>(),
        todos: null as TodoItem[] | null,
        citations: [] as Array<{
          url: string
          content: string
          isCited: boolean
          confidence?: number
          matchKind?: CitationMatchKind
        }>,
        files: new Map<string, FileArtifactUpdate>(),
        reportContent: null as string | null,
      }

      const correlator = createResearchCorrelator({
        hasUserMessage: () => !buf.active && Boolean(useChatStore.getState().currentUserMessageId),
        addThinkingStep,
        appendToThinkingStep,
        completeThinkingStep,
        addAgent: (agentId, agent) => {
          if (buf.active) {
            if (!buf.agents.has(agentId)) {
              buf.agents.set(agentId, { name: agent.name, input: agent.input, ended: false })
            }
            return agentId
          }
          return addDeepResearchAgentWithId(agentId, agent)
        },
        completeAgent: (agentId, output) => {
          if (buf.active) {
            const agent = buf.agents.get(agentId)
            if (agent) {
              agent.output = output
              agent.ended = true
            }
            return
          }
          completeDeepResearchAgent(agentId, output)
        },
        addToolCall: (toolCall) => {
          if (buf.active) {
            const id = `tool-${buf.idCounter++}`
            buf.toolCalls.set(id, {
              name: toolCall.name,
              input: toolCall.input,
              workflow: toolCall.workflow,
              agentId: toolCall.agentId,
              isSandbox: toolCall.isSandbox,
            })
            return id
          }
          return addDeepResearchToolCall(toolCall)
        },
        completeToolCall: (toolCallId, output) => {
          if (buf.active) {
            const toolCall = buf.toolCalls.get(toolCallId)
            if (toolCall) toolCall.output = output ? JSON.stringify(output) : undefined
            return
          }
          completeDeepResearchToolCall(toolCallId, output)
        },
        addLLMStep: (step) => {
          if (buf.active) {
            const id = `llm-${buf.idCounter++}`
            buf.llmSteps.set(id, { name: step.name, workflow: step.workflow, content: step.content })
            return id
          }
          return addDeepResearchLLMStep(step)
        },
        appendLLMStep: (stepId, chunk) => {
          if (buf.active) {
            const step = buf.llmSteps.get(stepId)
            if (step) step.content += chunk
            return
          }
          appendToDeepResearchLLMStep(stepId, chunk)
        },
        completeLLMStep: (stepId, thinking, usage) => {
          if (buf.active) {
            const step = buf.llmSteps.get(stepId)
            if (step) {
              step.thinking = thinking
              step.usage = usage
            }
            return
          }
          completeDeepResearchLLMStep(stepId, thinking, usage)
        },
      })
      correlatorRef.current = correlator
      resetTimeout()

      /** The stream may outlive the selected job; guard every delayed flush by job ID. */
      const isActiveJob = (): boolean => isOwnerActive(jobId)

      const deactivateBuffer = (): void => {
        buf.active = false
        if (buf.timer) { clearTimeout(buf.timer); buf.timer = null }
      }

      /** Flush buffer to store in one setState, deactivate buffer, switch to live. */
      const flushBuffer = (): boolean => {
        if (!buf.active) return true
        if (!isActiveJob()) {
          deactivateBuffer()
          return false
        }
        deactivateBuffer()

        const now = new Date()
        const agents = Array.from(buf.agents.entries()).map(([id, a]) => ({
          id,
          name: a.name,
          input: a.input,
          output: a.output,
          status: a.ended ? ('complete' as const) : ('running' as const),
          startedAt: now,
          ...(a.ended && { completedAt: now }),
        }))
        const llmSteps = Array.from(buf.llmSteps.entries()).map(([id, s]) => ({ id, name: s.name, workflow: s.workflow, content: s.content, thinking: s.thinking, usage: s.usage, isComplete: true, timestamp: now }))
        const toolCalls = Array.from(buf.toolCalls.entries()).map(([id, t]) => ({ id, name: t.name, input: t.input, output: t.output, workflow: t.workflow, agentId: t.agentId, isSandbox: t.isSandbox, status: 'complete' as const, timestamp: now }))
        const citations = buf.citations.map((c, i) => ({
          id: `citation-${i}`,
          url: c.url,
          content: c.content,
          isCited: c.isCited,
          confidence: c.confidence,
          matchKind: c.matchKind,
          timestamp: now,
        }))
        const files = Array.from(buf.files.values()).map((file, i) => ({ id: `file-${i}`, ...file, timestamp: now }))
        const todos = buf.todos ? normalizeDeepResearchTodos(buf.todos) : undefined

        useChatStore.setState((state) => ({
          ...(buf.reportContent !== null && {
            reportContent: buf.reportContent,
            reportContentCategory: 'final_report' as const,
          }),
          ...(todos && todos.length > 0 && { deepResearchTodos: todos }),
          ...(agents.length > 0 && { deepResearchAgents: agents }),
          ...(llmSteps.length > 0 && { deepResearchLLMSteps: llmSteps }),
          ...(toolCalls.length > 0 && { deepResearchToolCalls: toolCalls }),
          ...(citations.length > 0 && { deepResearchCitations: citations }),
          ...(files.length > 0 && { deepResearchFiles: files }),
          currentStatus: buf.reportContent !== null ? 'writing' : state.currentStatus,
        }))

        return true
      }

      // Safety timeout: flush if the backend never sends the live signal
      if (bufferReplay) {
        buf.timer = setTimeout(flushBuffer, SAFETY_TIMEOUT_MS)
      }

      // Create SSE client — callbacks check buf.active to decide buffer vs real-time
      const client = createDeepResearchClient({
        jobId,
        authToken: idToken || undefined,
        callbacks: {
          onStreamStart: () => {
            if (buf.active) return
            if (!isActiveJob()) return
            resetTimeout()
            researchStartTimeRef.current = Date.now()
            setCurrentStatus('researching')
          },

          onStreamMode: (mode) => {
            if (isDeepResearchReplayCompleteMode(mode) && buf.active) {
              if (!flushBuffer()) return
              setCurrentStatus('researching')
            }
          },

          onJobStatus: (status, error) => {
            if (buf.active) flushBuffer()
            if (!isActiveJob()) return
            resetTimeout()
            // Clear the cancel-fallback timer — the SSE stream delivered
            // the terminal status so optimistic cleanup is unnecessary.
            if (cancelFallbackRef.current) {
              clearTimeout(cancelFallbackRef.current)
              cancelFallbackRef.current = null
            }
            updateDeepResearchStatus(status)

            const state = useChatStore.getState()
            const ownerConvId = state.deepResearchOwnerConversationId
            const messageId = state.activeDeepResearchMessageId

            if (status === 'success') {
              setCurrentStatus('complete')
              const { reportContent: currentReport, deepResearchLLMSteps, deepResearchToolCalls } = state
              const totalTokens = deepResearchLLMSteps.reduce((sum, step) => sum + (step.usage?.input_tokens || 0) + (step.usage?.output_tokens || 0), 0)
              const toolCallCount = deepResearchToolCalls.length
              const hasReport = Boolean(currentReport?.trim())

              if (ownerConvId && messageId) {
                patchConversationMessage(ownerConvId, messageId, {
                  content: '',
                  deepResearchJobStatus: 'success',
                  isDeepResearchActive: false,
                  showViewReport: hasReport,
                  responseCompletedAt: new Date(),
                })
              }
              addDeepResearchBanner('success', jobId, ownerConvId || undefined, { totalTokens, toolCallCount })
              researchStartTimeRef.current = null
              stopAllDeepResearchSpinners(true)
              setStreamLoaded(true)
              completeDeepResearch()
              setStreaming(false)
            } else if (status === 'failure' || status === 'interrupted') {
              setCurrentStatus('error')
              stopAllDeepResearchSpinners()
              const hasReport = Boolean(state.reportContent?.trim())
              const isUserCancelled = isUserCancelledStatus(status, error)

              if (ownerConvId && messageId) {
                patchConversationMessage(ownerConvId, messageId, {
                  content: '',
                  deepResearchJobStatus: status,
                  isDeepResearchActive: false,
                  showViewReport: hasReport,
                  responseCompletedAt: new Date(),
                })
              }
              addDeepResearchBanner(isUserCancelled ? 'cancelled' : 'failure', jobId, ownerConvId || undefined)
              researchStartTimeRef.current = null
              clientRef.current?.disconnect()
              setStreamLoaded(true)
              completeDeepResearch()
              setStreaming(false)
              if (error && !isUserCancelled) {
                const { addErrorCard } = useChatStore.getState()
                addErrorCard('agent.deep_research_failed', error)
              } else if (status === 'interrupted' && !isUserCancelled) {
                const { addErrorCard } = useChatStore.getState()
                addErrorCard('agent.deep_research_failed', 'Research was interrupted before completion.')
              }
            }
          },

          onHeartbeat: () => {
            if (buf.active) return
            if (!isActiveJob()) return
            resetTimeout()
          },

          onWorkflowStart: (name, input, eventId, agentId) => {
            const id = agentId || eventId || `agent-${buf.idCounter++}`
            if (!buf.active) {
              if (!isActiveJob()) return
              resetTimeout()
            }
            correlator.onWorkflowStart(id, name, input)
          },

          onWorkflowEnd: (name, output, eventId, agentId) => {
            if (!buf.active && !isActiveJob()) return
            correlator.onWorkflowEnd(agentId || eventId, name, output)
          },

          onLLMStart: (name, workflow, agentId) => {
            if (!buf.active && !isActiveJob()) return
            correlator.onLLMStart(agentId, name, workflow)
          },

          onLLMChunk: (chunk) => {
            if (!buf.active) {
              if (!isActiveJob()) return
              resetTimeout()
            }
            correlator.onLLMChunk(chunk)
          },

          onLLMEnd: (_output, thinking, usage, name, agentId) => {
            if (!buf.active && !isActiveJob()) return
            correlator.onLLMEnd(agentId, name, thinking, usage)
          },

          onToolStart: (name, input, workflow, _eventId, agentId, isSandbox) => {
            if (name === 'task') return
            if (!buf.active) {
              if (!isActiveJob()) return
              resetTimeout(); setCurrentStatus('searching')
            }
            correlator.onToolStart(agentId, name, input, workflow, isSandbox)
          },

          onToolEnd: (name, output, _eventId, agentId) => {
            if (name === 'task') return
            if (!buf.active && !isActiveJob()) return
            correlator.onToolEnd(agentId, name, output)
            if (!buf.active) setCurrentStatus('researching')
          },

          onTodoUpdate: (todos: TodoItem[], workflow?: string) => {
            // Workflow-scoped todo artifacts belong to sub-agent-local plans.
            // The Tasks tab shows only the top-level research todo list.
            if (workflow) return
            if (buf.active) { buf.todos = todos; return }
            if (!isActiveJob()) return
            resetTimeout(); setDeepResearchTodos(todos)

          },

          onCitationUpdate: (url, content, isCited, confidence, matchKind) => {
            if (buf.active) {
              buf.citations.push({
                url,
                content,
                isCited: isCited ?? false,
                confidence,
                matchKind,
              })
              return
            }
            if (!isActiveJob()) return
            resetTimeout()
            addDeepResearchCitation(url, content, isCited, confidence, matchKind)
          },

          onFileUpdate: (file) => {
            if (buf.active) {
              // Merge like the live store (addDeepResearchFile): a later metadata-only
              // event must not drop content from an earlier event for the same filename.
              const prev = buf.files.get(file.filename)
              buf.files.set(file.filename, prev ? { ...prev, ...file } : file)
              return
            }
            if (!isActiveJob()) return
            resetTimeout(); addDeepResearchFile(file)
            // report.md artifact arrives 1-2 min before the final_report output event —
            // use it as an early signal to switch the UI to "writing" status.
            if (file.filename.endsWith('report.md')) {
              setCurrentStatus('writing')
            }
          },

          onOutputUpdate: (content, outputCategory, _workflow) => {
            // Only the explicit final_report artifact belongs in the Report tab.
            // Draft, research_notes, and uncategorized output can be partial JSON
            // from failed or cancelled workflow paths.
            if (outputCategory !== 'final_report') return
            if (buf.active) {
              buf.reportContent = content
              return
            }
            if (!isActiveJob()) return
            setReportContent(content, 'final_report')
            setCurrentStatus('writing')
          },

          onComplete: () => {
            if (buf.active) flushBuffer()
          },

          onError: async (error) => {
            console.warn('Deep research SSE error:', error.message)
            if (buf.active) flushBuffer()
            if (!isActiveJob()) return
            const { isDeepResearchStreaming, deepResearchStatus } = useChatStore.getState()
            if (isDeepResearchStreaming && deepResearchStatus !== 'interrupted' && deepResearchStatus !== 'failure') {
              const backendUp = await checkBackendHealthCached()

              const errorInfo = backendUp
                ? getDeepResearchStreamFailure(error.message, error.stack)
                : { code: 'agent.deep_research_failed' as const, message: error.message, details: error.stack }

              console.error(
                backendUp
                  ? 'Deep research SSE failed while backend remained reachable:'
                  : 'Deep research SSE failed (backend unreachable):',
                error
              )
              setCurrentStatus('error')

              const state = useChatStore.getState()
              const ownerConvId = state.deepResearchOwnerConversationId
              const messageId = state.activeDeepResearchMessageId
              const hasReport = Boolean(state.reportContent?.trim())

              if (ownerConvId && messageId) {
                patchConversationMessage(ownerConvId, messageId, {
                  content: '',
                  deepResearchJobStatus: 'failure',
                  isDeepResearchActive: false,
                  showViewReport: hasReport,
                  responseCompletedAt: new Date(),
                })
              }

              state.addErrorCard(errorInfo.code as Parameters<typeof state.addErrorCard>[0], errorInfo.message, errorInfo.details)
              addDeepResearchBanner('failure', jobId, ownerConvId || undefined)
              stopAllDeepResearchSpinners()
              clientRef.current?.disconnect()
              setStreamLoaded(true)
              completeDeepResearch()
              setStreaming(false)
            }
          },

          onDisconnect: () => {
            if (buf.active) flushBuffer()
          },
        },
      })

      clientRef.current = client
      client.connect()
    },
    [
      idToken, resetTimeout, isOwnerActive, updateDeepResearchStatus, completeDeepResearch,
      addDeepResearchCitation, setReportContent, addThinkingStep, appendToThinkingStep,
      completeThinkingStep, setCurrentStatus, setDeepResearchTodos, stopAllDeepResearchSpinners,
      addDeepResearchLLMStep, appendToDeepResearchLLMStep,
      completeDeepResearchLLMStep, addDeepResearchAgentWithId, completeDeepResearchAgent,
      addDeepResearchToolCall, completeDeepResearchToolCall, addDeepResearchFile,
      patchConversationMessage, addDeepResearchBanner, setStreaming, setStreamLoaded,
      getDeepResearchStreamFailure,
    ]
  )

  // Keep ref in sync so the effect always uses the latest connect without re-triggering
  connectRef.current = connect

  /**
   * Disconnect from the SSE stream
   */
  const disconnect = useCallback(() => {
    if (clientRef.current) {
      clientRef.current.disconnect()
      clientRef.current = null
    }
  }, [])

  /**
   * Reconnect to the SSE stream from the beginning
   */
  const reconnect = useCallback(() => {
    if (deepResearchJobId && !clientRef.current?.isConnected()) {
      connectRef.current?.(deepResearchJobId, true)
    }
  }, [deepResearchJobId])

  /**
   * Cancel the current job (useful for hung jobs)
   */
  const cancelCurrentJob = useCallback(async () => {
    if (!deepResearchJobId) return
    const cancelledJobId = deepResearchJobId

    try {
      await cancelJob(cancelledJobId, idToken || undefined)
      setIsTimedOut(false)

      // Fallback: if the SSE stream is broken or stalled and never delivers
      // the job.status: "interrupted" event, clean up locally after a short
      // grace period so the UI doesn't stay stuck in "streaming" state.
      // If the SSE event arrives in time, onJobStatus clears this timer.
      if (cancelFallbackRef.current) clearTimeout(cancelFallbackRef.current)
      cancelFallbackRef.current = setTimeout(() => {
        cancelFallbackRef.current = null
        const state = useChatStore.getState()
        if (!state.isDeepResearchStreaming || state.deepResearchJobId !== cancelledJobId) {
          return // SSE already handled cleanup — nothing to do
        }
        console.warn(
          '[DeepResearch] Cancel fallback: SSE did not deliver interrupted status within',
          CANCEL_FALLBACK_TIMEOUT_MS,
          'ms. Cleaning up locally.'
        )
        const ownerConvId = state.deepResearchOwnerConversationId
        const messageId = state.activeDeepResearchMessageId
        const hasReport = Boolean(state.reportContent?.trim())
        if (ownerConvId && messageId) {
          patchConversationMessage(ownerConvId, messageId, {
            content: '',
            deepResearchJobStatus: 'interrupted',
            isDeepResearchActive: false,
            showViewReport: hasReport,
            responseCompletedAt: new Date(),
          })
        }
        addDeepResearchBanner('cancelled', cancelledJobId, ownerConvId || undefined)
        stopAllDeepResearchSpinners()
        clientRef.current?.disconnect()
        clientRef.current = null
        setStreamLoaded(true)
        completeDeepResearch()
        setStreaming(false)
      }, CANCEL_FALLBACK_TIMEOUT_MS)
    } catch (error) {
      console.error('Failed to cancel job:', error)
    }
  }, [deepResearchJobId, idToken, patchConversationMessage, addDeepResearchBanner, stopAllDeepResearchSpinners, completeDeepResearch, setStreaming, setStreamLoaded])

  /**
   * Auto-connect when job ID changes
   * Uses lastEventId from store for reconnection scenarios (session restore, tab reopen)
   */
  useEffect(() => {
    // Capture values at effect start to detect stale effects
    const effectJobId = deepResearchJobId
    const effectStreaming = isDeepResearchStreaming
    let connectTimeout: NodeJS.Timeout | null = null
    let cancelled = false

    if (effectJobId && effectStreaming) {
      // Verify state hasn't changed before connecting (prevents race conditions)
      const currentState = useChatStore.getState()
      if (currentState.deepResearchJobId !== effectJobId || !currentState.isDeepResearchStreaming) {
        return // State changed, don't connect
      }

      // Defer connect by 50ms so React StrictMode cleanup can cancel it.
      // StrictMode sequence: mount1-effect → mount1-cleanup → mount2-effect.
      // The cleanup clears the timeout, preventing mount1 from ever connecting.
      // Only mount2's deferred connect actually fires.
      connectTimeout = setTimeout(async () => {
        if (cancelled) return

        // Determine if this is a reconnection (page refresh) or a fresh job start.
        // Fresh jobs (status 'submitted') use per-event store writes for live updates.
        // Reconnections (status 'running') buffer historical events then flush once
        // when the backend sends stream.mode: "live".
        const isReconnect = useChatStore.getState().deepResearchStatus !== 'submitted'
        connectRef.current?.(effectJobId, isReconnect)

        setResearchPanelTab('tasks')
        openRightPanel('research')

        // Start timeout check interval
        timeoutIntervalRef.current = setInterval(() => {
          const timeSinceLastEvent = Date.now() - lastEventTimeRef.current
          if (timeSinceLastEvent > TIMEOUT_WARNING_MS) {
            setIsTimedOut(true)
          }
        }, TIMEOUT_CHECK_INTERVAL_MS)
      }, 50)

      // Session persistence is now handled by debounced resetTimeout()
      // (fires 2s after each event instead of fixed 10s interval)
    }

    return () => {
      cancelled = true
      // Cancel the deferred connect if it hasn't fired yet
      if (connectTimeout) clearTimeout(connectTimeout)
      // Cleanup on unmount or job ID change
      disconnect()
      // Clear timeout interval
      if (timeoutIntervalRef.current) {
        clearInterval(timeoutIntervalRef.current)
        timeoutIntervalRef.current = null
      }
      // Clear cancel fallback timer
      if (cancelFallbackRef.current) {
        clearTimeout(cancelFallbackRef.current)
        cancelFallbackRef.current = null
      }
      setIsTimedOut(false)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps -- connectRef avoids re-triggering on token refresh; store actions are stable refs
  }, [deepResearchJobId, isDeepResearchStreaming, disconnect, setResearchPanelTab, openRightPanel])

  return {
    isStreaming: isDeepResearchStreaming,
    jobId: deepResearchJobId,
    status: deepResearchStatus,
    isTimedOut,
    disconnect,
    reconnect,
    cancelCurrentJob,
  }
}
