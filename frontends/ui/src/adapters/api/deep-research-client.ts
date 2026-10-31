// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Deep Research SSE Client
 *
 * Handles Server-Sent Events streaming for deep research async jobs.
 * Uses native EventSource for proper SSE protocol support including
 * event types, event IDs, and automatic reconnection.
 *
 * @see docs/api.md - Deep Research API (Async Jobs) section
 */

import { apiConfig } from './config'
import { artifactContentPath } from '@/shared/utils/artifact-url'

// ============================================================
// Types
// ============================================================

/** Job status values */
export type DeepResearchJobStatus = 'submitted' | 'running' | 'success' | 'failure' | 'interrupted'

/** SSE event types from the deep research stream */
export type DeepResearchEventType =
  | 'stream.start'
  | 'stream.mode'
  | 'job.status'
  | 'job.heartbeat'
  | 'workflow.start'
  | 'workflow.end'
  | 'llm.start'
  | 'llm.chunk'
  | 'llm.end'
  | 'tool.start'
  | 'tool.end'
  | 'artifact.update'

/** Artifact types in artifact.update events */
export type ArtifactType = 'todo' | 'citation_source' | 'citation_use' | 'file' | 'output'

/** Base SSE event structure */
export interface DeepResearchSSEEvent {
  event: DeepResearchEventType
  id?: string
  timestamp?: string
}

/** stream.start event */
export interface StreamStartEvent extends DeepResearchSSEEvent {
  event: 'stream.start'
  job_id: string
}

/** job.status event */
export interface JobStatusEvent extends DeepResearchSSEEvent {
  event: 'job.status'
  data: {
    status: DeepResearchJobStatus
    error?: string
  }
}

/** workflow.start event */
export interface WorkflowStartEvent extends DeepResearchSSEEvent {
  event: 'workflow.start'
  data: {
    name: string
    data?: {
      input?: string
    }
  }
}

/** workflow.end event */
export interface WorkflowEndEvent extends DeepResearchSSEEvent {
  event: 'workflow.end'
  data: {
    name: string
    data?: {
      output?: string
    }
  }
}

/** llm.start event */
export interface LLMStartEvent extends DeepResearchSSEEvent {
  event: 'llm.start'
  data: {
    name: string
    metadata?: {
      workflow?: string
    }
  }
}

/** llm.chunk event */
export interface LLMChunkEvent extends DeepResearchSSEEvent {
  event: 'llm.chunk'
  data: {
    chunk: string
  }
}

/** llm.end event */
export interface LLMEndEvent extends DeepResearchSSEEvent {
  event: 'llm.end'
  data: {
    output: string
    metadata?: {
      thinking?: string
      usage?: {
        prompt_tokens: number
        completion_tokens: number
      }
    }
  }
}

/** tool.start event */
export interface ToolStartEvent extends DeepResearchSSEEvent {
  event: 'tool.start'
  data: {
    name: string
    data?: {
      input?: Record<string, unknown>
    }
    metadata?: {
      workflow?: string
      agent_id?: string
    }
  }
}

/** tool.end event */
export interface ToolEndEvent extends DeepResearchSSEEvent {
  event: 'tool.end'
  data: {
    name: string
    data?: {
      output?: string
    }
  }
}

/** Todo item in artifact.update */
export interface TodoItem {
  id: string
  content: string
  status: 'pending' | 'in_progress' | 'completed' | 'cancelled'
}

/** File update emitted by artifact.update. Durable generated files carry metadata, not bytes. */
export interface FileArtifactUpdate {
  filename: string
  content?: string
  artifactId?: string
  contentUrl?: string
  kind?: string
  mimeType?: string
  sizeBytes?: number
  sha256?: string
  title?: string
  caption?: string
  inline?: boolean
}

/** artifact.update event */
export interface ArtifactUpdateEvent extends DeepResearchSSEEvent {
  event: 'artifact.update'
  data: {
    id: string
    timestamp: string
    data: {
      type: ArtifactType
      content?: string | TodoItem[]
      url?: string // For citation_source and citation_use types
      content_url?: string
      file_path?: string
      artifact_id?: string
      job_id?: string
      kind?: string
      mime_type?: string
      size_bytes?: number
      sha256?: string
      title?: string
      caption?: string
      inline?: boolean
      // Issue #210: confidence scoring fields (only present on citation_use events)
      confidence?: number
      match_kind?: string
    }
    metadata?: {
      workflow?: string
    }
  }
}

/** Union of all SSE event types */
export type DeepResearchEvent =
  | StreamStartEvent
  | JobStatusEvent
  | WorkflowStartEvent
  | WorkflowEndEvent
  | LLMStartEvent
  | LLMChunkEvent
  | LLMEndEvent
  | ToolStartEvent
  | ToolEndEvent
  | ArtifactUpdateEvent

// ============================================================
// Callbacks
// ============================================================

/** Callbacks for deep research SSE events */
export interface DeepResearchCallbacks {
  /** Called when stream connection is established */
  onStreamStart?: (jobId: string) => void
  /** Called when stream mode changes (e.g., replay → live) */
  onStreamMode?: (mode: string) => void
  /** Called on job status updates */
  onJobStatus?: (status: DeepResearchJobStatus, error?: string) => void
  /** Called on workflow events */
  onWorkflowStart?: (name: string, input?: string, eventId?: string, agentId?: string) => void
  onWorkflowEnd?: (name: string, output?: string, eventId?: string, agentId?: string) => void
  /** Called on LLM events */
  onLLMStart?: (name: string, workflow?: string, agentId?: string) => void
  onLLMChunk?: (chunk: string) => void
  onLLMEnd?: (
    output: string,
    thinking?: string,
    usage?: { input_tokens: number; output_tokens: number },
    name?: string,
    agentId?: string
  ) => void
  /** Called on tool events */
  onToolStart?: (name: string, input?: Record<string, unknown>, workflow?: string, eventId?: string, agentId?: string, isSandbox?: boolean) => void
  onToolEnd?: (name: string, output?: string, eventId?: string, agentId?: string, isSandbox?: boolean) => void
  /** Called on artifact updates */
  onTodoUpdate?: (todos: TodoItem[], workflow?: string) => void
  onCitationUpdate?: (
    url: string,
    content: string,
    isCited?: boolean,
    confidence?: number,
    matchKind?: CitationMatchKind
  ) => void
  onFileUpdate?: (file: FileArtifactUpdate) => void
  onOutputUpdate?: (content: string, outputCategory?: string, workflow?: string) => void
  /** Called on job heartbeat (confirms job is alive during long operations) */
  onHeartbeat?: (uptimeSeconds: number) => void
  /** Called when job completes successfully */
  onComplete?: () => void
  /** Called on errors */
  onError?: (error: Error) => void
  /** Called when connection is lost */
  onDisconnect?: () => void
}

// ============================================================
// Utilities
// ============================================================

/** Supported citation verification match kinds (Issue #210) */
export const MATCH_KINDS = [
  'exact',
  'truncation',
  'prefix',
  'child-path',
  'query-subset',
  'host-only',
  'unmatched',
] as const

export type CitationMatchKind = (typeof MATCH_KINDS)[number]

export const VALID_MATCH_KINDS: ReadonlySet<string> = new Set(MATCH_KINDS)

export const isCitationMatchKind = (value: unknown): value is CitationMatchKind =>
  typeof value === 'string' && VALID_MATCH_KINDS.has(value)

/**
 * Normalize tool input from the backend SSE stream.
 *
 * The backend's `_trim_tool_input` converts large dict inputs to Python `str()`
 * repr when they exceed 500 characters. For example:
 *   "{'query': 'search term', 'params': {'max_results': 10, ...}"
 *
 * This is NOT valid JSON (single quotes, Python booleans/None). The frontend
 * expects `Record<string, unknown>` for tool input, and downstream code calls
 * `JSON.stringify(input, null, 2)` to display it — which double-quotes a raw
 * string, producing confusing output like `"\"{'query': ...\""`
 *
 * This utility normalizes the input at the adapter boundary so all downstream
 * consumers receive a proper object:
 *   1. If input is already an object → pass through unchanged
 *   2. If input is a JSON string → parse it back to an object
 *   3. If input is a Python repr string → wrap in { _raw, _truncated } sentinel
 */
const normalizeToolInput = (input: unknown): Record<string, unknown> | undefined => {
  if (input == null) return undefined

  // Already a proper object — pass through
  if (typeof input === 'object' && !Array.isArray(input)) {
    return input as Record<string, unknown>
  }

  if (typeof input === 'string') {
    // Try JSON parse first (handles the case where backend used json.dumps)
    try {
      const parsed = JSON.parse(input)
      if (typeof parsed === 'object' && parsed !== null && !Array.isArray(parsed)) {
        return parsed as Record<string, unknown>
      }
    } catch {
      // Not valid JSON — likely a Python repr string from str(dict)
    }

    // Python repr or other non-JSON string — wrap in sentinel object
    // so downstream JSON.stringify produces clean output
    return { _raw: input, _truncated: true }
  }

  // Unexpected primitive (number, boolean) — wrap for safety
  return { _raw: String(input) }
}

// ============================================================
// Deep Research SSE Client
// ============================================================

export interface DeepResearchStreamOptions {
  /** Job ID to stream */
  jobId: string
  /** Event callbacks */
  callbacks: DeepResearchCallbacks
  /** Last event ID for reconnection (optional) */
  lastEventId?: string
  /** Auth token for authenticated requests */
  authToken?: string
}

export interface DeepResearchClient {
  /** Connect to the SSE stream */
  connect: () => void
  /** Disconnect from the SSE stream */
  disconnect: () => void
  /** Check if connected */
  isConnected: () => boolean
  /** Get the last received event ID (for reconnection) */
  getLastEventId: () => string | null
}

/**
 * Create a deep research SSE client
 *
 * Uses native EventSource for proper SSE protocol support.
 * Handles event types, reconnection, and routes events to callbacks.
 */
/** Max consecutive reconnection failures before surfacing an error to the caller */
const MAX_RECONNECT_ATTEMPTS = 5

export const createDeepResearchClient = (options: DeepResearchStreamOptions): DeepResearchClient => {
  const { jobId, callbacks, lastEventId, authToken } = options

  let eventSource: EventSource | null = null
  let lastReceivedEventId: string | null = lastEventId || null
  let isTerminated = false
  let reconnectAttempts = 0

  /**
   * Build the stream URL with optional last event ID for reconnection
   * Uses local API route in browser to avoid CORS issues
   */
  const buildStreamUrl = (): string => {
    // Use local API route in browser, direct backend URL on server
    const isBrowser = typeof window !== 'undefined'
    const baseUrl = isBrowser ? '' : apiConfig.baseUrl

    let url = `${baseUrl}/api/jobs/async/job/${jobId}/stream`

    if (lastReceivedEventId) {
      url += `/${lastReceivedEventId}`
    }

    // Add auth token as query param if provided (EventSource doesn't support headers)
    if (authToken) {
      url += `?token=${encodeURIComponent(authToken)}`
    }

    return url
  }

  /**
   * Parse SSE event data
   */
  const parseEventData = (data: string): unknown => {
    try {
      return JSON.parse(data)
    } catch {
      return data
    }
  }

  /**
   * Handle incoming SSE message
   *
   * SSE events have a nested structure where the actual payload is inside a 'data' property:
   * {
   *   "id": "uuid",
   *   "name": "workflow-name",
   *   "timestamp": "...",
   *   "data": { actual payload },
   *   "metadata": { workflow context }
   * }
   */
  const handleMessage = (event: MessageEvent, eventType: string) => {
    // Track last event ID for reconnection
    if (event.lastEventId) {
      lastReceivedEventId = event.lastEventId
    }

    const rawData = parseEventData(event.data)

    switch (eventType) {
      case 'stream.start': {
        const streamData = rawData as { job_id: string }
        callbacks.onStreamStart?.(streamData.job_id || jobId)
        break
      }

      case 'stream.mode': {
        const modeData = rawData as { mode?: string }
        if (modeData.mode) {
          callbacks.onStreamMode?.(modeData.mode)
        }
        break
      }

      case 'job.heartbeat': {
        const heartbeatData = rawData as { data?: { uptime_seconds?: number }; uptime_seconds?: number }
        const uptimeSeconds = heartbeatData.data?.uptime_seconds ?? heartbeatData.uptime_seconds ?? 0
        callbacks.onHeartbeat?.(uptimeSeconds)
        break
      }

      case 'job.status': {
        // job.status wraps status in data property
        const statusWrapper = rawData as { data?: { status: DeepResearchJobStatus; error?: string }; status?: DeepResearchJobStatus; error?: string }
        const statusData = statusWrapper.data || statusWrapper
        callbacks.onJobStatus?.(statusData.status!, statusData.error)

        // Check for terminal states — close EventSource immediately to prevent
        // auto-reconnection loops (EventSource reconnects on its own if left open)
        if (statusData.status === 'success') {
          isTerminated = true
          eventSource?.close()
          callbacks.onComplete?.()
        } else if (statusData.status === 'failure' || statusData.status === 'interrupted') {
          isTerminated = true
          eventSource?.close()
          // Only call onError for actual failures, not user-initiated cancellations
          const isUserCancelled = statusData.status === 'interrupted' && statusData.error?.toLowerCase().includes('cancelled by user')
          if (!isUserCancelled && statusData.error) {
            callbacks.onError?.(new Error(statusData.error || `Job ${statusData.status}`))
          }
        }
        break
      }

      case 'workflow.start': {
        // workflow events have nested structure: { id, name, timestamp, data: { input }, metadata: { agent_id } }
        const workflowData = rawData as { id?: string; name: string; data?: { input?: string }; metadata?: { agent_id?: string } }
        callbacks.onWorkflowStart?.(workflowData.name, workflowData.data?.input, workflowData.id, workflowData.metadata?.agent_id)
        break
      }

      case 'workflow.end': {
        const workflowData = rawData as { id?: string; name: string; data?: { output?: string }; metadata?: { agent_id?: string } }
        callbacks.onWorkflowEnd?.(workflowData.name, workflowData.data?.output, workflowData.id, workflowData.metadata?.agent_id)
        break
      }

      case 'llm.start': {
        const llmData = rawData as { name: string; metadata?: { workflow?: string; agent_id?: string } }
        callbacks.onLLMStart?.(llmData.name, llmData.metadata?.workflow, llmData.metadata?.agent_id)
        break
      }

      case 'llm.chunk': {
        const chunkData = rawData as { chunk?: string; data?: { chunk?: string } }
        const chunk = chunkData.chunk || chunkData.data?.chunk || ''
        callbacks.onLLMChunk?.(chunk)
        break
      }

      case 'llm.end': {
        // llm.end has nested structure: { id, name, timestamp, metadata: { thinking, usage, agent_id }, data: { output } }
        const endData = rawData as {
          name?: string
          data?: { output?: string }
          metadata?: {
            thinking?: string
            agent_id?: string
            usage?: { input_tokens?: number; output_tokens?: number; prompt_tokens?: number; completion_tokens?: number }
          }
        }
        const output = endData.data?.output || ''
        const thinking = endData.metadata?.thinking
        // Handle both naming conventions for usage (input_tokens/output_tokens or prompt_tokens/completion_tokens)
        const rawUsage = endData.metadata?.usage
        const usage = rawUsage
          ? {
              input_tokens: rawUsage.input_tokens ?? rawUsage.prompt_tokens ?? 0,
              output_tokens: rawUsage.output_tokens ?? rawUsage.completion_tokens ?? 0,
            }
          : undefined
        callbacks.onLLMEnd?.(output, thinking, usage, endData.name, endData.metadata?.agent_id)
        break
      }

      case 'tool.start': {
        // tool events have nested structure: { id, name, timestamp, data: { input }, metadata: { workflow, agent_id, sandbox } }
        // Note: data.input may be a Python repr string when the backend trims large inputs via str()
        const toolData = rawData as {
          id?: string
          name: string
          data?: { input?: unknown }
          metadata?: { workflow?: string; agent_id?: string; sandbox?: boolean }
        }
        const normalizedInput = normalizeToolInput(toolData.data?.input)
        callbacks.onToolStart?.(toolData.name, normalizedInput, toolData.metadata?.workflow, toolData.id, toolData.metadata?.agent_id, Boolean(toolData.metadata?.sandbox))
        break
      }

      case 'tool.end': {
        const toolData = rawData as { id?: string; name: string; data?: { output?: string }; metadata?: { agent_id?: string; sandbox?: boolean } }
        callbacks.onToolEnd?.(toolData.name, toolData.data?.output, toolData.id, toolData.metadata?.agent_id, Boolean(toolData.metadata?.sandbox))
        break
      }

      case 'artifact.update': {
        // artifact.update has nested structure: { id, timestamp, data: { type, content, url?, output_category? }, metadata?: { workflow } }
        // Reuse the exported ArtifactUpdateEvent data contract (durable metadata fields
        // included) instead of re-declaring a narrower shape; `path`/`output_category` are
        // the two extra keys this parser also reads.
        type ArtifactData = ArtifactUpdateEvent['data']['data'] & { path?: string; output_category?: string }
        const artifactWrapper = rawData as { data?: ArtifactData; metadata?: { workflow?: string } } & Partial<ArtifactData>
        // Handle both nested (data.type) and flat (type) structures
        const artifactData = (artifactWrapper.data || artifactWrapper) as ArtifactData
        const artifactWorkflow = artifactWrapper.metadata?.workflow

        switch (artifactData.type) {
          case 'todo':
            callbacks.onTodoUpdate?.(artifactData.content as TodoItem[], artifactWorkflow)
            break
          case 'citation_source':
            // citation_source = "Referenced" sources (discovered during search)
            callbacks.onCitationUpdate?.(artifactData.url || '', artifactData.content as string, false)
            break
          case 'citation_use': {
            // citation_use = "Cited" sources (actually used in the report)
            // Issue #210: runtime validation for untyped SSE boundary fields
            const rawConfidence = artifactData.confidence
            const rawMatchKind = artifactData.match_kind
            const validConfidence =
              typeof rawConfidence === 'number' &&
              Number.isFinite(rawConfidence) &&
              rawConfidence >= 0 &&
              rawConfidence <= 1
                ? rawConfidence
                : undefined
            const validMatchKind: CitationMatchKind | undefined =
              isCitationMatchKind(rawMatchKind) ? rawMatchKind : undefined

            callbacks.onCitationUpdate?.(
              artifactData.url || '',
              artifactData.content as string,
              true,
              validConfidence,
              validMatchKind
            )
            break
          }
          case 'file': {
            // Generated artifacts carry durable metadata; legacy text-file events carry content.
            const artifactId = artifactData.artifact_id
            // Fall back to artifactId (not a shared 'unknown') when no path/url is present, so two
            // distinct pathless artifacts don't collapse onto one filename key downstream.
            const filePath = artifactData.file_path || artifactData.path || artifactData.url
            const fileName = filePath ? filePath.split('/').pop() || filePath : artifactId || 'unknown'
            callbacks.onFileUpdate?.({
              filename: fileName,
              // content is typed as string | TodoItem[]; only a string is a real file body.
              content: typeof artifactData.content === 'string' ? artifactData.content : undefined,
              artifactId,
              contentUrl: artifactId ? artifactContentPath(jobId, artifactId) : artifactData.content_url || artifactData.url,
              kind: artifactData.kind,
              mimeType: artifactData.mime_type,
              sizeBytes: artifactData.size_bytes,
              sha256: artifactData.sha256,
              title: artifactData.title,
              caption: artifactData.caption,
              inline: artifactData.inline,
            })
            break
          }
          case 'output':
            callbacks.onOutputUpdate?.(artifactData.content as string, artifactData.output_category, artifactWorkflow)
            break
          default:
            if (process.env.NODE_ENV === 'development') {
              console.warn(`[SSE:artifact.update] Unknown artifact type: ${artifactData.type}`)
            }
        }
        break
      }

      default:
        // Unknown event type - log in dev
        if (process.env.NODE_ENV === 'development') {
          console.warn(`[SSE] Unknown event type: ${eventType}`, rawData)
        }
        break
    }
  }

  /**
   * Connect to the SSE stream
   */
  const connect = () => {
    if (eventSource) {
      return // Already connected
    }

    isTerminated = false
    const url = buildStreamUrl()
    eventSource = new EventSource(url)

    // Handle connection open
    eventSource.onopen = () => {
      // Connection established (or re-established after reconnect) — reset counter
      reconnectAttempts = 0
    }

    // Handle generic messages (fallback)
    eventSource.onmessage = (event) => {
      handleMessage(event, 'message')
    }

    // Register handlers for each known event type
    const eventTypes: DeepResearchEventType[] = [
      'stream.start',
      'stream.mode',
      'job.status',
      'job.heartbeat',
      'workflow.start',
      'workflow.end',
      'llm.start',
      'llm.chunk',
      'llm.end',
      'tool.start',
      'tool.end',
      'artifact.update',
    ]

    eventTypes.forEach((eventType) => {
      eventSource?.addEventListener(eventType, (event) => {
        handleMessage(event as MessageEvent, eventType)
      })
    })

    // Handle errors
    // EventSource fires onerror on *any* connection issue, then auto-reconnects
    // (readyState transitions to CONNECTING). This is normal SSE behaviour, not a
    // fatal error. We only surface an error to the caller after the reconnection
    // retry threshold is exceeded or the browser has given up (readyState CLOSED).
    eventSource.onerror = () => {
      if (isTerminated) {
        // Expected disconnection after terminal state — close to stop auto-reconnect
        eventSource?.close()
        eventSource = null
        return
      }

      if (eventSource?.readyState === EventSource.CLOSED) {
        // Browser gave up reconnecting — treat as a real disconnect
        callbacks.onDisconnect?.()
        eventSource = null
      } else if (eventSource?.readyState === EventSource.CONNECTING) {
        // EventSource is auto-reconnecting — this is expected behaviour.
        // Only escalate to an error after repeated consecutive failures.
        reconnectAttempts++
        if (reconnectAttempts <= MAX_RECONNECT_ATTEMPTS) {
          if (process.env.NODE_ENV === 'development') {
            console.warn(`[SSE] Reconnecting (attempt ${reconnectAttempts}/${MAX_RECONNECT_ATTEMPTS})…`)
          }
          return // let EventSource retry on its own
        }
        // Too many consecutive reconnection failures — give up
        eventSource?.close()
        eventSource = null
        callbacks.onError?.(
          new Error(`SSE connection failed after ${reconnectAttempts} reconnection attempts`)
        )
      } else {
        // Unexpected state (e.g. OPEN) — should not happen per SSE spec, but
        // handle defensively so the caller is never left in an unknown state.
        const readyState = eventSource?.readyState
        eventSource?.close()
        eventSource = null
        callbacks.onError?.(
          new Error(`SSE connection error in unexpected readyState: ${readyState ?? 'unknown'}`)
        )
      }
    }
  }

  /**
   * Disconnect from the SSE stream
   */
  const disconnect = () => {
    if (eventSource) {
      isTerminated = true
      eventSource.close()
      eventSource = null
    }
  }

  /**
   * Check if connected
   */
  const isConnected = (): boolean => {
    return eventSource !== null && eventSource.readyState === EventSource.OPEN
  }

  /**
   * Get the last received event ID for reconnection
   */
  const getLastEventId = (): string | null => {
    return lastReceivedEventId
  }

  return {
    connect,
    disconnect,
    isConnected,
    getLastEventId,
  }
}

// ============================================================
// REST API Functions (for non-streaming operations)
// ============================================================

/**
 * Get the base URL for deep research API
 * Uses local API route in browser to avoid CORS issues
 */
const getDeepResearchBaseUrl = (): string => {
  const isBrowser = typeof window !== 'undefined'
  return isBrowser ? '/api/jobs/async' : `${apiConfig.baseUrl}/v1/jobs/async`
}

const getDeepResearchErrorDetails = async (response: Response): Promise<string | null> => {
  const responseText = await response.text().catch(() => '')
  if (!responseText) return null

  try {
    const parsed = JSON.parse(responseText) as {
      error?: {
        code?: unknown
        message?: unknown
      }
    }
    const code = typeof parsed.error?.code === 'string' ? parsed.error.code : ''
    const message = typeof parsed.error?.message === 'string' ? parsed.error.message : ''
    return [code, message].filter(Boolean).join(': ') || responseText
  } catch {
    return responseText
  }
}

const throwDeepResearchApiError = async (response: Response, context: string): Promise<never> => {
  const details = await getDeepResearchErrorDetails(response)
  throw new Error(`${context}: ${response.status}${details ? ` - ${details}` : ''}`)
}

/** Get job status */
export const getJobStatus = async (
  jobId: string,
  authToken?: string
): Promise<{ job_id: string; status: DeepResearchJobStatus; error: string | null }> => {
  const url = `${getDeepResearchBaseUrl()}/job/${jobId}`
  const headers: HeadersInit = {
    'Content-Type': 'application/json',
  }

  if (authToken) {
    headers['Authorization'] = `Bearer ${authToken}`
  }

  const response = await fetch(url, { headers })

  if (!response.ok) {
    await throwDeepResearchApiError(response, 'Failed to get job status')
  }

  return response.json()
}

/** Get job report */
export const getJobReport = async (
  jobId: string,
  authToken?: string
): Promise<{ job_id: string; has_report: boolean; report: string | null }> => {
  const url = `${getDeepResearchBaseUrl()}/job/${jobId}/report`
  const headers: HeadersInit = {
    'Content-Type': 'application/json',
  }

  if (authToken) {
    headers['Authorization'] = `Bearer ${authToken}`
  }

  const response = await fetch(url, { headers })

  if (!response.ok) {
    await throwDeepResearchApiError(response, 'Failed to get job report')
  }

  return response.json()
}

/** Cancel a running job */
export const cancelJob = async (
  jobId: string,
  authToken?: string
): Promise<{ cancelled: boolean }> => {
  const url = `${getDeepResearchBaseUrl()}/job/${jobId}/cancel`
  const headers: HeadersInit = {
    'Content-Type': 'application/json',
  }

  if (authToken) {
    headers['Authorization'] = `Bearer ${authToken}`
  }

  const response = await fetch(url, {
    method: 'POST',
    headers,
  })

  if (!response.ok) {
    await throwDeepResearchApiError(response, 'Failed to cancel job')
  }

  return response.json()
}

/** Job state/artifacts response */
export interface JobStateResponse {
  job_id: string
  has_state: boolean
  state: Record<string, unknown> | null
  artifacts: {
    tools: Array<{
      name: string
      input?: Record<string, unknown>
      output?: string
      timestamp?: string
    }>
    outputs: Array<{
      type: string
      content: string
      output_category?: string
      timestamp?: string
    }>
  } | null
}

/** Get job state/artifacts (for catching up on missed events) */
export const getJobState = async (
  jobId: string,
  authToken?: string
): Promise<JobStateResponse> => {
  const url = `${getDeepResearchBaseUrl()}/job/${jobId}/state`
  const headers: HeadersInit = {
    'Content-Type': 'application/json',
  }

  if (authToken) {
    headers['Authorization'] = `Bearer ${authToken}`
  }

  const response = await fetch(url, { headers })

  if (!response.ok) {
    await throwDeepResearchApiError(response, 'Failed to get job state')
  }

  return response.json()
}
