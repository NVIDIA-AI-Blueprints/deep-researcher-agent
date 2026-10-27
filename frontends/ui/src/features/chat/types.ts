// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Chat Feature Types
 *
 * Type definitions for chat messages, conversations, and state.
 */

/** Message role types */
export type MessageRole = 'user' | 'assistant' | 'system'

/** Message types for different display purposes */
export type MessageType =
  | 'user'
  | 'assistant'
  | 'status'
  | 'prompt'
  | 'agent_response'
  | 'file'
  | 'file_upload_status'
  | 'error'
  | 'deep_research_banner'

/** Deep research banner types for status notifications */
export type DeepResearchBannerType = 'starting' | 'success' | 'failure' | 'cancelled' | 'expired'

/** File upload status types for banner messages */
export type FileUploadStatusType = 'uploaded' | 'pending_warning'

/** Status types for status card messages */
export type StatusType =
  | 'thinking'
  | 'searching'
  | 'planning'
  | 'researching'
  | 'writing'
  | 'complete'
  | 'data_source_added'
  | 'data_source_removed'
  | 'error'

/** File status for file operations */
export type FileStatus = 'uploading' | 'ingesting' | 'success' | 'deleted' | 'error'

/** Error codes using dot-notation for extensibility */
export type ErrorCode =
  // Connection errors
  | 'connection.lost'
  | 'connection.failed'
  | 'connection.timeout'
  // Auth errors
  | 'auth.session_expired'
  | 'auth.unauthorized'
  // Agent errors
  | 'agent.response_failed'
  | 'agent.response_interrupted'
  | 'agent.deep_research_failed'
  | 'agent.deep_research_load_failed'
  // System errors
  | 'system.unknown'

/** Prompt types for agent prompts requiring user response */
export type PromptType = 'clarification' | 'approval' | 'choice' | 'text-input' | 'plan_approval'

/** File card data for file messages */
export interface FileCardData {
  fileName: string
  fileSize?: number
  fileStatus: FileStatus
  progress?: number
  errorMessage?: string
}

/** Error card data for error messages */
export interface ErrorCardData {
  errorCode: ErrorCode
  errorMessage?: string
  errorDetails?: string
}

/** File upload status data for banner messages */
export interface FileUploadStatusData {
  /** Type of status: uploaded (files uploaded, ingesting) or ingested (ready to use) */
  type: FileUploadStatusType
  /** Number of files in the batch */
  fileCount: number
  /** Job ID to prevent duplicate banners */
  jobId: string
}

/** Deep research banner data for status notifications */
export interface DeepResearchBannerData {
  /** Type of banner: starting, success, failure, cancellation, or expiry */
  bannerType: DeepResearchBannerType
  /** Job ID for identification */
  jobId: string
  /** Total tokens used (for success banner) */
  totalTokens?: number
  /** Number of tool calls (for success banner) */
  toolCallCount?: number
}

/** Individual chat message */
export interface ChatMessage {
  id: string
  role: MessageRole
  content: string
  timestamp: Date
  /** Type of message for routing to correct display */
  messageType?: MessageType
  /** Whether this message is still streaming */
  isStreaming?: boolean
  /** Intermediate thinking steps from the agent */
  intermediateSteps?: IntermediateStep[]
  /** Status type for status messages */
  statusType?: StatusType
  /** Prompt type for prompt messages */
  promptType?: PromptType
  /** Prompt ID for HITL routing (may differ from message ID) */
  promptId?: string
  /** Parent message ID for HITL response routing */
  promptParentId?: string
  /** Input type for HITL prompts */
  promptInputType?: 'text' | 'multiple_choice' | 'binary_choice' | 'approval' | 'notification'
  /** Options for choice prompts */
  promptOptions?: string[]
  /** Placeholder for text input prompts */
  promptPlaceholder?: string
  /** User's response to prompt */
  promptResponse?: string
  /** Whether the prompt has been responded to */
  isPromptResponded?: boolean
  /** File card data for file messages */
  fileData?: FileCardData
  /** Error card data for error messages */
  errorData?: ErrorCardData
  /** File upload status data for banner messages */
  fileUploadStatusData?: FileUploadStatusData
  /** Deep research banner data for status notifications */
  deepResearchBannerData?: DeepResearchBannerData
  /** Whether to show "View Report" button on agent responses */
  showViewReport?: boolean
  /** Real completion timestamp for a deep research turn, recorded when the terminal job event arrives (start timestamp stays in `timestamp`) */
  responseCompletedAt?: Date

  // Session persistence fields (embedded in messages for localStorage persistence)

  /** Thinking steps that occurred during processing (for user messages) */
  thinkingSteps?: ThinkingStep[]
  /** Report content shown in ResearchPanel (for agent_response messages) */
  reportContent?: string
  /** Citations/sources used (for agent_response messages) */
  citations?: CitationSource[]

  // ResearchPanel persistence fields (for agent_response messages)

  /** Plan messages shown in chat/HITL restore flows */
  planMessages?: PlanMessage[]
  /** Task todos shown in TasksTab */
  deepResearchTodos?: DeepResearchTodo[]
  /** LLM steps shown in ThoughtTracesTab */
  deepResearchLLMSteps?: DeepResearchLLMStep[]
  /** Agent steps shown in AgentsTab */
  deepResearchAgents?: DeepResearchAgent[]
  /** Tool calls shown in ToolCallsTab */
  deepResearchToolCalls?: DeepResearchToolCall[]
  /** File artifacts shown in FilesTab */
  deepResearchFiles?: DeepResearchFile[]

  // Deep research job persistence fields (for session restoration across tab close/reopen)

  /** Deep research job ID for session restoration */
  deepResearchJobId?: string
  /** Last SSE event ID received (for reconnection to running jobs) */
  deepResearchLastEventId?: string
  /** Job status at time of save (submitted, running, success, failure, interrupted) */
  deepResearchJobStatus?: DeepResearchJobStatus
  /** True when a completed report's backend job/report can no longer be loaded. */
  deepResearchReportExpired?: boolean
  /** Whether this message has active (streaming) deep research - used for UI state */
  isDeepResearchActive?: boolean
  /** Data sources that were enabled when this message was sent (for display in thinking panel) */
  enabledDataSources?: string[]
  /** Files that were available when this message was sent (for display in thinking panel) */
  messageFiles?: Array<{ id: string; fileName: string }>
  /** Final-response model selected for this specific turn. */
  selectedModel?: string
}

/** Intermediate thinking step from agent */
export interface IntermediateStep {
  id: string
  name: string
  status: 'in_progress' | 'complete' | 'error'
  content: string
  timestamp: Date
}

/** Categories for intermediate step tabs in the thinking panel */
export type IntermediateStepCategory = 'tasks' | 'agents' | 'tools'

/** Thinking step for the Details Panel Thinking tab */
export interface ThinkingStep {
  /** Unique identifier for this step */
  id: string
  /** ID of the user message that triggered this thinking step */
  userMessageId: string
  /** Category for tab routing (tasks, agents, tools) */
  category: IntermediateStepCategory
  /** Raw function name from backend (e.g., "web_search_tool") */
  functionName: string
  /** Human-readable display name (e.g., "Web Search Tool") */
  displayName: string
  /** Content/output of this step (parsed payload) */
  content: string
  /** Raw payload from backend for debugging */
  rawPayload?: string
  /** When this step started */
  timestamp: Date
  /** When this step completed (Function Complete received) */
  completedAt?: Date
  /** Whether this step is complete (Function Complete received) */
  isComplete: boolean
  /** Lifecycle status derived from the NAT step status (drives trace colors) */
  status?: 'running' | 'success' | 'error'
  /** Short human summary of the step's tool input (for the trace secondary line) */
  argSummary?: string
  /** Whether this step is from deep research (for Research Panel routing) */
  isDeepResearch?: boolean
  /** True when backend name is "Function Start: ..." (top-level workflow step); false for model/tool sub-calls (indented) */
  isTopLevel?: boolean
}

/** Conversation/Session */
export interface Conversation {
  id: string
  /** Owner of this session - used to filter sessions by user */
  userId: string
  title: string
  messages: ChatMessage[]
  createdAt: Date
  updatedAt: Date
  /** Per-session enabled data source IDs (persisted across refresh) */
  enabledDataSourceIds?: string[]
}

/** Pending human interaction from agent */
export interface PendingInteraction {
  /** Unique ID for this interaction */
  id: string
  /** Parent message ID for response */
  parentId: string
  /** Type of input expected */
  inputType: 'text' | 'multiple_choice' | 'binary_choice' | 'approval' | 'notification'
  /** Prompt text */
  text: string
  /** Options for choice prompts */
  options?: string[]
  /** Default value for text input */
  defaultValue?: string
}

/** Deep research job status (from SSE stream) */
export type DeepResearchJobStatus = 'submitted' | 'running' | 'success' | 'failure' | 'interrupted'

/** Citation source from deep research */
export interface CitationSource {
  confidence: number
  matchKind: string
  id: string
  url: string
  content: string
  timestamp: Date
  /** Whether this source was actually cited in the report (vs just referenced/discovered) */
  isCited?: boolean
}

/** Plan message for chat/HITL display and restore flows */
export interface PlanMessage {
  id: string
  /** The text content (clarification question, plan preview, etc.) */
  text: string
  /** Input type expected from user (if any) */
  inputType?: 'text' | 'multiple_choice' | 'binary_choice' | 'approval' | 'notification'
  /** Placeholder for text input */
  placeholder?: string
  /** Whether input is required */
  required?: boolean
  /** User's response to this message (if applicable) */
  userResponse?: string
  /** When this message was received */
  timestamp: Date
}

/** Todo item status from deep research SSE */
export type DeepResearchTodoStatus = 'pending' | 'in_progress' | 'completed' | 'stopped'

/** Todo item from deep research (artifact.update with type: "todo") */
export interface DeepResearchTodo {
  /** Unique identifier (generated from content hash) */
  id: string
  /** Task content/description */
  content: string
  /** Current status */
  status: DeepResearchTodoStatus
}

/** LLM step for ThoughtTracesTab (from llm.start/end) */
export interface DeepResearchLLMStep {
  /** Unique identifier */
  id: string
  /** LLM model name */
  name: string
  /** Parent workflow (metadata.workflow) */
  workflow?: string
  /** Streaming/final output content */
  content: string
  /** Chain-of-thought reasoning (metadata.thinking) */
  thinking?: string
  /** Token usage (from llm.end metadata.usage) */
  usage?: { input_tokens: number; output_tokens: number }
  /** When LLM started */
  timestamp: Date
  /** Whether LLM call is complete */
  isComplete: boolean
}

/** Agent step for AgentsTab (from workflow.start/end) */
export interface DeepResearchAgent {
  /** Unique identifier */
  id: string
  /** Agent/workflow name (e.g., "planner-agent", "researcher-agent") */
  name: string
  /** Input provided to agent (data.input) */
  input?: string
  /** Output from agent (data.output) */
  output?: string
  /** Current execution status */
  status: 'running' | 'complete' | 'error'
  /** When agent started */
  startedAt: Date
  /** When agent completed */
  completedAt?: Date
}

/** Tool call for ToolCallsTab (from tool.start/end) */
export interface DeepResearchToolCall {
  /** Unique identifier */
  id: string
  /** Tool name (e.g., "tavily_web_search", "write_file") */
  name: string
  /** Structured tool input arguments */
  input?: Record<string, unknown>
  /** Tool output/result (after tool.end) */
  output?: string
  /** Parent workflow that invoked the tool */
  workflow?: string
  /** Parent agent ID that invoked the tool (for grouping under agents) */
  agentId?: string
  /** Whether this tool call executed against the sandbox runtime */
  isSandbox?: boolean
  /** Current execution status */
  status: 'running' | 'complete' | 'error'
  /** When tool was called */
  timestamp: Date
}

/** File artifact for FilesTab (from artifact.update type: "file") */
export interface DeepResearchFile {
  /** Unique identifier */
  id: string
  /** File name/path */
  filename: string
  /** Inline text content for legacy text-file events */
  content?: string
  /** Durable generated-artifact identifier */
  artifactId?: string
  /** Authenticated endpoint for the generated bytes */
  contentUrl?: string
  kind?: string
  mimeType?: string
  sizeBytes?: number
  sha256?: string
  title?: string
  caption?: string
  inline?: boolean
  /** When file was created/updated */
  timestamp: Date
}

/** Chat state for Zustand store */
export interface ChatState {
  /** Current authenticated user ID - used for filtering sessions */
  currentUserId: string | null
  /** Current active conversation */
  currentConversation: Conversation | null
  /** All conversations for the sessions sidebar (includes all users) */
  conversations: Conversation[]
  /** Whether a message is currently streaming */
  isStreaming: boolean
  /** Whether we're waiting for the first response token */
  isLoading: boolean
  /** ID of the current user message being processed (for associating thinking steps) */
  currentUserMessageId: string | null
  /** Thinking steps for the Details Panel - Thinking tab */
  thinkingSteps: ThinkingStep[]
  /** ID of the currently active thinking step (for appending content) */
  activeThinkingStepId: string | null
  /** Content for the Details Panel - Report tab */
  reportContent: string
  /** Category of the current report content (distinguishes intermediate notes from final report) */
  reportContentCategory: 'research_notes' | 'final_report' | null
  /** Current status type (for status indicators) */
  currentStatus: StatusType | null
  /** Pending interaction requiring user response (for HITL) */
  pendingInteraction: PendingInteraction | null
  /** Transient callback for responding to HITL interactions (registered by InputArea, not persisted) */
  respondToInteractionFn: ((response: string) => void) | null

  // Deep research SSE state
  /** Current deep research job ID (null when not active) */
  deepResearchJobId: string | null
  /** Last SSE event ID received (for reconnection) */
  deepResearchLastEventId: string | null
  /** Whether deep research SSE is currently streaming */
  isDeepResearchStreaming: boolean
  /** Current deep research job status */
  deepResearchStatus: DeepResearchJobStatus | null
  /** Conversation ID that owns the current deep research stream (for session isolation) */
  deepResearchOwnerConversationId: string | null
  /** Message ID of the originating deep research message (for patching on completion) */
  activeDeepResearchMessageId: string | null
  /** Citations collected during deep research */
  deepResearchCitations: CitationSource[]
  /** Todo items from deep research (from artifact.update with type: "todo") */
  deepResearchTodos: DeepResearchTodo[]
  /** LLM steps for ThoughtTracesTab (from llm.start/end events) */
  deepResearchLLMSteps: DeepResearchLLMStep[]
  /** Agent steps for AgentsTab (from workflow.start/end events) */
  deepResearchAgents: DeepResearchAgent[]
  /** Tool calls for ToolCallsTab (from tool.start/end events) */
  deepResearchToolCalls: DeepResearchToolCall[]
  /** File artifacts for FilesTab (from artifact.update type: "file" events) */
  deepResearchFiles: DeepResearchFile[]
  /** Whether the full stream data (artifacts, tool calls, etc.) has been loaded for current job */
  deepResearchStreamLoaded: boolean

  // Plan state (for chat/HITL restore flows)
  /** Messages for clarification questions, plan previews, and approvals. */
  planMessages: PlanMessage[]
}

/** Chat actions for Zustand store */
export interface ChatActions {
  /** Set the current authenticated user ID */
  setCurrentUser: (userId: string | null) => void
  /** Get conversations filtered by current user */
  getUserConversations: () => Conversation[]
  /** Create a new conversation for the current user */
  createConversation: () => Conversation
  /** Start a new unsaved session draft; persisted only after first interaction. */
  startNewSessionDraft: () => void
  /** Ensure a session exists, creating one if needed. Returns session ID or undefined if no user. */
  ensureSession: () => string | undefined
  /** Select a conversation (only if owned by current user) */
  selectConversation: (conversationId: string) => void
  /** Add a user message to the current conversation */
  addUserMessage: (
    content: string,
    metadata?: {
      enabledDataSources?: string[]
      messageFiles?: Array<{ id: string; fileName: string }>
      selectedModel?: string
    }
  ) => ChatMessage
  /** Start streaming an assistant response */
  startAssistantMessage: () => ChatMessage
  /** Append content to the streaming assistant message */
  appendToAssistantMessage: (content: string) => void
  /** Complete the streaming assistant message */
  completeAssistantMessage: () => void
  /** Set loading state */
  setLoading: (loading: boolean) => void
  /** Set streaming state */
  setStreaming: (streaming: boolean) => void
  /** Delete a conversation */
  deleteConversation: (conversationId: string) => void
  /** Delete all conversations for the current user */
  deleteAllConversations: () => void
  /** Update conversation title */
  updateConversationTitle: (conversationId: string, title: string) => void
  /** Persist enabled data source IDs to the current conversation for per-session storage */
  saveDataSourcesToConversation: (ids: string[]) => void

  // New actions for thinking/report content and status/prompts

  /** Add a new thinking step to the Details Panel with category and metadata */
  addThinkingStep: (step: Omit<ThinkingStep, 'id' | 'timestamp' | 'userMessageId'>) => string
  /** Get thinking steps filtered by user message ID */
  getThinkingStepsForMessage: (userMessageId: string) => ThinkingStep[]
  /** Append content to an existing thinking step */
  appendToThinkingStep: (stepId: string, content: string) => void
  /** Mark a thinking step as complete */
  completeThinkingStep: (stepId: string) => void
  /** Mark a thinking step as failed (status: 'error') */
  failThinkingStep: (stepId: string) => void
  /** Update or complete a thinking step by function name (for "Function Complete" messages) */
  updateThinkingStepByFunctionName: (
    functionName: string,
    content: string,
    isComplete: boolean
  ) => void
  /** Find a thinking step by function name */
  findThinkingStepByFunctionName: (functionName: string) => ThinkingStep | undefined
  /** Set the report content with optional category */
  setReportContent: (content: string, category?: 'research_notes' | 'final_report') => void
  /** Clear all thinking steps (for new request) */
  clearThinkingSteps: () => void
  /** Clear report content (for new request) */
  clearReportContent: () => void
  /** Add a status card message to the conversation */
  addStatusCard: (type: StatusType, message?: string) => void
  /** Set current status type */
  setCurrentStatus: (status: StatusType | null) => void
  /** Add an agent prompt message to the conversation */
  addAgentPrompt: (
    type: PromptType,
    content: string,
    options?: string[],
    placeholder?: string,
    promptId?: string,
    parentId?: string,
    inputType?: 'text' | 'multiple_choice' | 'binary_choice' | 'approval' | 'notification'
  ) => void
  /** Respond to a prompt */
  respondToPrompt: (messageId: string, response: string) => void

  // Actions for agent responses and HITL

  /** Add an agent response message to the chat (for short answers) */
  addAgentResponse: (content: string, showViewReport?: boolean) => void
  /** Add an agent response with additional metadata - returns the created message ID */
  addAgentResponseWithMeta: (
    content: string,
    showViewReport: boolean,
    meta: Partial<ChatMessage>
  ) => string
  /** Patch a specific message in a conversation */
  patchConversationMessage: (
    conversationId: string,
    messageId: string,
    patch: Partial<ChatMessage>
  ) => void
  /** Set pending interaction requiring user response */
  setPendingInteraction: (interaction: PendingInteraction | null) => void
  /** Clear pending interaction (after user responds) */
  clearPendingInteraction: () => void
  /** Register the respondToInteraction callback (called by InputArea on mount) */
  setRespondToInteractionFn: (fn: ((response: string) => void) | null) => void

  // Actions for file and error cards

  /** Add a file card message to the conversation */
  addFileCard: (data: FileCardData) => void
  /** Update file card status (for progress updates) */
  updateFileCard: (messageId: string, data: Partial<FileCardData>) => void
  /** Add an error card message to the conversation */
  addErrorCard: (code: ErrorCode, message?: string, details?: string) => void
  /** Dismiss an error card */
  dismissErrorCard: (messageId: string) => void
  /** Dismiss all connection error cards (connection.*) from the current conversation */
  dismissConnectionErrors: () => void

  // Actions for file upload status banners

  /**
   * Add a file upload status banner to a conversation.
   * If sessionId is provided, adds to that specific conversation.
   * Otherwise, adds to the current conversation.
   */
  addFileUploadStatusCard: (
    type: FileUploadStatusType,
    fileCount: number,
    jobId: string,
    sessionId?: string
  ) => void

  /**
   * Remove the pending_warning file upload status message from the current conversation.
   * Used when user acknowledges the warning by re-submitting.
   */
  removeFileUploadWarning: () => void

  /**
   * Add a deep research banner (starting, success, or failure) to a conversation.
   * If conversationId is provided, adds to that specific conversation.
   * Otherwise, adds to the current conversation.
   * For 'starting' banners, job metadata is automatically set for session restoration.
   */
  addDeepResearchBanner: (
    bannerType: DeepResearchBannerType,
    jobId: string,
    conversationId?: string,
    stats?: { totalTokens?: number; toolCallCount?: number }
  ) => void

  // Deep research SSE actions

  /** Start deep research streaming with a job ID and optional originating message ID */
  startDeepResearch: (jobId: string, messageId?: string) => void
  /** Update deep research job status */
  updateDeepResearchStatus: (status: DeepResearchJobStatus) => void
  /** Update the last received SSE event ID (for reconnection) */
  setDeepResearchLastEventId: (eventId: string | null) => void
  /** Persist current deep research state to sessionStorage (for page refresh) */
  persistDeepResearchToSession: () => void
  /** Complete deep research (clears streaming state, keeps content) */
  completeDeepResearch: () => void
  /** Save current deep research progress to conversation (for session switching) */
  saveDeepResearchProgress: () => void
  /** Reconnect to an in-progress job after page refresh (running/submitted only) */
  reconnectToActiveJob: () => Promise<void>
  /** Clean up orphaned 'starting' banners by polling job status via REST */
  cleanupOrphanedStartingBanners: () => Promise<void>
  /**
   * Refresh persisted deep-research job metadata without deleting the chat
   * session. Missing backend jobs are represented as expired report state when
   * the session previously had a completed report.
   */
  refreshDeepResearchSessionStatuses: () => Promise<void>
  /** Add a citation from deep research (isCited=true for citation_use, false for citation_source) */
  addDeepResearchCitation: (url: string, content: string, isCited?: boolean) => void
  /** Set the full todo list from deep research (replaces existing) */
  setDeepResearchTodos: (todos: Array<{ content: string; status: string }>) => void
  /** Mark all in-progress and pending todos as stopped (on error) */
  stopDeepResearchTodos: () => void
  /** Stop all spinners (todos, LLM steps, agents, tool calls). Pass true for success, false/undefined for error. */
  stopAllDeepResearchSpinners: (isSuccessfulCompletion?: boolean) => void
  /** Clear all deep research state (for new research) */
  clearDeepResearch: () => void
  /** Set job ID for loaded (non-streaming) report data - enables cache lookup */
  setLoadedJobId: (jobId: string) => void
  /** Mark that full stream data has been loaded for current job */
  setStreamLoaded: (loaded: boolean) => void

  // Deep research ThinkingTab actions (LLM steps, agents, tool calls, files)

  /** Add a new LLM step (on llm.start) */
  addDeepResearchLLMStep: (
    step: Omit<DeepResearchLLMStep, 'id' | 'timestamp' | 'isComplete'>
  ) => string
  /** Append content to an LLM step (on llm.chunk) */
  appendToDeepResearchLLMStep: (stepId: string, content: string) => void
  /** Complete an LLM step with thinking and usage (on llm.end) */
  completeDeepResearchLLMStep: (
    stepId: string,
    thinking?: string,
    usage?: { input_tokens: number; output_tokens: number }
  ) => void
  /** Add a new agent (on workflow.start) */
  addDeepResearchAgent: (agent: Omit<DeepResearchAgent, 'id' | 'startedAt' | 'status'>) => string
  /** Add a new agent with a specific ID (for linking with tool calls) */
  addDeepResearchAgentWithId: (
    id: string,
    agent: Omit<DeepResearchAgent, 'id' | 'startedAt' | 'status'>
  ) => string
  /** Complete an agent (on workflow.end) */
  completeDeepResearchAgent: (agentId: string, output?: string) => void
  /** Add a new tool call (on tool.start) */
  addDeepResearchToolCall: (
    toolCall: Omit<DeepResearchToolCall, 'id' | 'timestamp' | 'status'>
  ) => string
  /** Complete a tool call (on tool.end) */
  completeDeepResearchToolCall: (toolCallId: string, output?: string) => void
  /** Get tool calls for a specific agent */
  getAgentToolCalls: (agentId: string) => DeepResearchToolCall[]
  /** Add a file artifact (on artifact.update type: "file") */
  addDeepResearchFile: (file: Omit<DeepResearchFile, 'id' | 'timestamp'>) => string

  // Plan actions (for chat/HITL restore flows)

  /** Add a plan message (clarification, plan preview, etc.) */
  addPlanMessage: (message: Omit<PlanMessage, 'id' | 'timestamp'>) => string
  /** Update a plan message with user response */
  updatePlanMessageResponse: (messageId: string, response: string) => void
  /** Clear all plan messages (for new request) */
  clearPlanMessages: () => void
  /** Persist current planMessages to the conversation for HITL recovery */
  persistPlanMessages: () => void

  // Session restoration

  /** Restore ephemeral state (thinkingSteps, reportContent, citations) from a conversation's messages */
  restoreSessionState: (conversation: Conversation) => void

  // Session busy checks (for disabling UI controls)

  /**
   * Check if a specific session has active operations.
   * Scans message history for background jobs - the only way to detect jobs in non-current sessions.
   * @param conversationId - The conversation ID to check
   * @returns true if the session has active operations (shallow or deep research)
   */
  isSessionBusy: (conversationId: string) => boolean
  /**
   * Check if ANY session has active operations.
   * Used to disable "Delete All Sessions" button.
   * @returns true if any session has active operations
   */
  hasAnyBusySession: () => boolean
}

/** Combined chat store type */
export type ChatStore = ChatState & ChatActions
