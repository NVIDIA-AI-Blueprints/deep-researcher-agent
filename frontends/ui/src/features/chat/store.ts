// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Chat Store
 *
 * Zustand store for managing chat state including messages,
 * conversations, and streaming status.
 */

import { create } from 'zustand'
import {
  devtools,
  persist,
  createJSONStorage,
  type StorageValue,
  type PersistStorage,
} from 'zustand/middleware'
import { v4 as uuidv4 } from 'uuid'
import type {
  ChatStore,
  ChatState,
  Conversation,
  ChatMessage,
  StatusType,
  PromptType,
  PendingInteraction,
  FileCardData,
  ErrorCode,
  ThinkingStep,
  FileUploadStatusType,
  DeepResearchJobStatus,
  CitationSource,
  PlanMessage,
  DeepResearchLLMStep,
  DeepResearchAgent,
  DeepResearchToolCall,
  DeepResearchFile,
  DeepResearchBannerType,
  DeepResearchTodo,
} from './types'
import { getErrorMeta } from './lib/error-registry'
import {
  saveDeepResearchToSession,
  loadDeepResearchFromSession as _loadDeepResearchFromSession,
  clearDeepResearchSession,
  clearAllDeepResearchSessions,
} from './lib/deep-research-session-storage'
import { isUnavailableDeepResearchJobError } from './lib/deep-research-errors'
import { hasActiveDeepResearchJob, hasNoUserChatMessages } from './lib/session-activity'
import { discardSessionDocumentsResources } from '@/features/documents/discard-session-resources'
import { useDocumentsStore } from '@/features/documents/store'
import {
  logStorageWrite,
  logQuotaExceededPruning,
  logCriticalSessionsClear,
  logStorageAvailability,
  logExternalStorageEvent,
  logStoreHydration,
} from './lib/storage-logger'
import { pruneMessageForStorage } from './lib/prune-message-for-storage'
import { normalizeDeepResearchTodos } from './lib/deep-research-todos'
import { ensureStorageCapacity, checkStorageHealth } from './lib/storage-manager'
import { useLayoutStore } from '@/features/layout/store'

const isQuotaExceededError = (error: unknown): boolean => {
  if (!(error instanceof Error)) return false
  if (error.name === 'QuotaExceededError') return true
  return /quota|exceeded|storage/i.test(error.message)
}

type PersistedChatState = {
  currentUserId: ChatState['currentUserId']
  conversations: ChatState['conversations']
  currentConversation: ChatState['currentConversation']
  pendingInteraction: ChatState['pendingInteraction']
}

type PersistedChatStorageValue = StorageValue<PersistedChatState>

const DEEP_RESEARCH_TODO_PERSIST_DEBOUNCE_MS = 1000
let deepResearchTodoPersistTimer: ReturnType<typeof setTimeout> | null = null

const areDeepResearchTodosEqual = (
  left: DeepResearchTodo[] | undefined,
  right: DeepResearchTodo[] | undefined
): boolean => {
  const leftTodos = left ?? []
  const rightTodos = right ?? []
  if (leftTodos.length !== rightTodos.length) return false

  return leftTodos.every((todo, index) => {
    const other = rightTodos[index]
    return todo.id === other.id && todo.content === other.content && todo.status === other.status
  })
}

const prunePersistedChatState = (value: PersistedChatStorageValue): PersistedChatStorageValue => {
  const state = value.state

  const conversations: Conversation[] = (state.conversations ?? []).map((conv) => ({
    ...conv,
    messages: (conv.messages ?? []).map(pruneMessageForStorage),
  }))

  // Store only the ID reference — the full object already lives in conversations[].
  // On read, getItem reconstructs currentConversation from conversations by ID.
  // This avoids serializing the active session's messages twice in JSON.
  const currentConversationId = state.currentConversation?.id ?? null

  return {
    ...value,
    state: {
      currentUserId: state.currentUserId ?? null,
      conversations,
      currentConversation: currentConversationId as unknown as Conversation | null,
      pendingInteraction: state.pendingInteraction ?? null,
    },
  }
}

const createResilientStorage = (): PersistStorage<PersistedChatState> | undefined => {
  const base = createJSONStorage<PersistedChatState>(() => localStorage)
  if (!base) {
    logStorageAvailability(false)
    return undefined
  }

  return {
    getItem: async (name: string): Promise<PersistedChatStorageValue | null> => {
      const raw = await base.getItem(name)
      if (!raw) return null

      // Strip connection error messages — they are transient and should not
      // survive page reloads. Persisting them causes the UI to show a permanent
      // "failed to connect" state that only clears when the user wipes cache.
      const stripConnectionErrors = (conversations: Conversation[]) =>
        conversations.map((c) => ({
          ...c,
          messages: c.messages.filter(
            (m) => !(m.messageType === 'error' && m.errorData?.errorCode?.startsWith('connection.'))
          ),
        }))

      if (raw.state.conversations) {
        raw.state.conversations = stripConnectionErrors(raw.state.conversations)
      }

      // Reconstruct currentConversation from the ID stored by prunePersistedChatState.
      const storedId = raw.state.currentConversation as unknown as string | null
      if (storedId) {
        const conversations = raw.state.conversations ?? []
        raw.state.currentConversation = conversations.find((c) => c.id === storedId) ?? null
      }

      return raw
    },
    removeItem: base.removeItem,
    setItem: (name: string, value: PersistedChatStorageValue) => {
      const prunedValue = prunePersistedChatState(value)
      const serializedValue = JSON.stringify(prunedValue)

      try {
        if (localStorage.getItem(name) === serializedValue) return

        localStorage.setItem(name, serializedValue)
        logStorageWrite(
          prunedValue.state.conversations ?? [],
          prunedValue.state.currentUserId ?? null
        )
      } catch (error) {
        if (!isQuotaExceededError(error)) {
          throw error
        }

        const beforeConversations = prunedValue.state.conversations ?? []
        const beforeCount = beforeConversations.length
        const beforeSizeKB = Math.round((JSON.stringify(beforeConversations).length * 2) / 1024)

        logQuotaExceededPruning(beforeCount, beforeCount, beforeSizeKB, beforeSizeKB)

        // Last resort: clear all conversations
        try {
          const lostSessionIds = beforeConversations.map((c) => c.id)

          base.removeItem(name)
          base.setItem(name, {
            ...value,
            state: {
              currentUserId: value.state.currentUserId ?? null,
              conversations: [],
              currentConversation: null,
              pendingInteraction: null,
            },
          })

          logCriticalSessionsClear(value.state.currentUserId ?? null, lostSessionIds, error)
        } catch (finalError) {
          console.error('[SessionsStore] ❌ CATASTROPHIC: Failed to clear sessions', {
            error: finalError instanceof Error ? finalError.message : String(finalError),
          })
        }
      }
    },
  }
}

const initialState: ChatState = {
  currentUserId: null,
  currentConversation: null,
  conversations: [],
  isStreaming: false,
  isLoading: false,
  // State for thinking steps association
  currentUserMessageId: null,
  // State for Details Panel
  thinkingSteps: [],
  activeThinkingStepId: null,
  reportContent: '',
  reportContentCategory: null,
  currentStatus: null,
  // State for HITL (human-in-the-loop)
  pendingInteraction: null,
  respondToInteractionFn: null,
  // State for deep research SSE streaming
  deepResearchJobId: null,
  deepResearchLastEventId: null,
  isDeepResearchStreaming: false,
  deepResearchStatus: null,
  deepResearchOwnerConversationId: null,
  activeDeepResearchMessageId: null,
  deepResearchCitations: [],
  deepResearchTodos: [],
  // State for ThinkingTab sub-tabs (LLM steps, agents, tool calls, files)
  deepResearchLLMSteps: [],
  deepResearchAgents: [],
  deepResearchToolCalls: [],
  deepResearchFiles: [],
  deepResearchStreamLoaded: false,
  // State for HITL plan/chat messages
  planMessages: [],
}

/**
 * Create a new conversation with default values
 * @param userId - The user ID who owns this conversation
 */
const createNewConversation = (userId: string): Conversation => ({
  id: `s_${uuidv4().replace(/-/g, '_')}`, // Milvus: letters, numbers, underscores only (no hyphens)
  userId,
  title: '',
  messages: [],
  createdAt: new Date(),
  updatedAt: new Date(),
})

/**
 * Generate a title from the first user message
 */
const generateTitle = (content: string): string => {
  const maxLength = 50
  const trimmed = content.trim()
  if (trimmed.length <= maxLength) {
    return trimmed
  }
  return trimmed.substring(0, maxLength) + '...'
}

/**
 * Helper to update conversation in list
 */
const updateConversationInList = (
  conversations: Conversation[],
  updatedConversation: Conversation
): Conversation[] => {
  return conversations.map((c) => (c.id === updatedConversation.id ? updatedConversation : c))
}

const getLatestDeepResearchMessage = (conversation: Conversation): ChatMessage | null => {
  for (let i = conversation.messages.length - 1; i >= 0; i--) {
    const message = conversation.messages[i]
    if (message.messageType === 'agent_response' && message.deepResearchJobId) {
      return message
    }
  }
  return null
}

const isCompletedDeepResearchReportMessage = (message: ChatMessage): boolean =>
  Boolean(
    !message.deepResearchReportExpired &&
    message.deepResearchJobId &&
    message.deepResearchJobStatus === 'success' &&
    (message.showViewReport || message.reportContent?.trim())
  )

const patchLatestDeepResearchJobMessage = (
  conversation: Conversation,
  jobId: string,
  patch: Partial<ChatMessage>
): Conversation => {
  const messageIndex = [...conversation.messages]
    .reverse()
    .findIndex(
      (message) => message.messageType === 'agent_response' && message.deepResearchJobId === jobId
    )

  if (messageIndex < 0) return conversation

  const actualIndex = conversation.messages.length - 1 - messageIndex
  const messages = conversation.messages.map((message, index) =>
    index === actualIndex ? { ...message, ...patch } : message
  )

  return { ...conversation, messages }
}

const createDeepResearchBannerMessage = (
  bannerType: DeepResearchBannerType,
  jobId: string,
  stats?: { totalTokens?: number; toolCallCount?: number }
): ChatMessage => ({
  id: uuidv4(),
  role: 'assistant',
  content: '',
  timestamp: new Date(),
  messageType: 'deep_research_banner',
  deepResearchBannerData: {
    bannerType,
    jobId,
    totalTokens: stats?.totalTokens,
    toolCallCount: stats?.toolCallCount,
  },
  // Starting banners carry job metadata so restored sessions can reconnect or cancel correctly.
  ...(bannerType === 'starting' && {
    deepResearchJobId: jobId,
    deepResearchJobStatus: 'submitted' as const,
    isDeepResearchActive: true,
  }),
})

const withDeepResearchBanner = (
  conversation: Conversation,
  bannerType: DeepResearchBannerType,
  jobId: string,
  stats?: { totalTokens?: number; toolCallCount?: number }
): Conversation => {
  const isTerminalBanner = bannerType !== 'starting'
  const filteredMessages = isTerminalBanner
    ? conversation.messages.filter(
        (message) =>
          !(
            message.messageType === 'deep_research_banner' &&
            message.deepResearchBannerData?.jobId === jobId
          )
      )
    : conversation.messages

  return {
    ...conversation,
    messages: [...filteredMessages, createDeepResearchBannerMessage(bannerType, jobId, stats)],
    updatedAt: new Date(),
  }
}

const patchConversationMessageById = (
  conversation: Conversation,
  messageId: string,
  patch: Partial<ChatMessage>
): Conversation => {
  let didPatch = false
  const messages = conversation.messages.map((message) => {
    if (message.id !== messageId) return message
    didPatch = true
    return { ...message, ...patch }
  })

  return didPatch ? { ...conversation, messages, updatedAt: new Date() } : conversation
}

/**
 * A protected per-user source (e.g. Google Drive) must be connected before it
 * can be part of the active selection — otherwise it appears in "Selected Data
 * Sources" and is submitted while unusable. Mirrors the card toggle, "Enable
 * All", and the initial-fetch gate in the layout store.
 */
const isSelectableDataSource = (source: {
  per_user_auth?: { required?: boolean; status?: string | null } | null
}): boolean => !(source.per_user_auth?.required && source.per_user_auth.status !== 'connected')

const getDefaultEnabledDataSourceIds = (): string[] => {
  const layoutStore = useLayoutStore.getState()
  return (layoutStore.availableDataSources ?? []).filter(isSelectableDataSource).map((source) => source.id)
}

const restoreConversationDataSources = (conversation: Conversation): void => {
  const layoutStore = useLayoutStore.getState()

  if (conversation.enabledDataSourceIds) {
    // Only restore sources that are still available AND currently selectable —
    // a protected source saved as enabled must not come back while it's not
    // connected (e.g. an old session that had Google Drive on).
    const selectableIds = new Set(
      (layoutStore.availableDataSources ?? []).filter(isSelectableDataSource).map((source) => source.id)
    )
    const validIds = conversation.enabledDataSourceIds.filter((id) => selectableIds.has(id))
    layoutStore.setEnabledDataSources(validIds)
    return
  }

  const defaultIds = getDefaultEnabledDataSourceIds()
  layoutStore.setEnabledDataSources(defaultIds)
}

/**
 * When leaving an upload-only session (no user chat), remove it and tear down
 * its document collection. Skips if uploads/ingestion are still in flight or
 * the session is waiting on HITL / active deep research.
 */
const maybeDiscardAbandonedUploadOnlySession = (
  get: () => ChatStore,
  sessionId: string | null | undefined
): void => {
  if (!sessionId) return

  const { conversations, currentUserId, pendingInteraction, currentConversation } = get()
  if (pendingInteraction && currentConversation?.id === sessionId) return

  const conv = conversations.find((c) => c.id === sessionId && c.userId === currentUserId)
  if (!conv) return
  if (!hasNoUserChatMessages(conv.messages)) return
  if (hasActiveDeepResearchJob(conv.messages)) return

  const docsInFlight = useDocumentsStore
    .getState()
    .trackedFiles.some(
      (f) =>
        f.collectionName === sessionId && (f.status === 'uploading' || f.status === 'ingesting')
    )
  if (docsInFlight) return

  discardSessionDocumentsResources(sessionId)
  get().deleteConversation(sessionId)
}

export const useChatStore = create<ChatStore>()(
  devtools(
    persist(
      (set, get) => ({
        ...initialState,

        setCurrentUser: (userId: string | null) => {
          const { conversations, currentConversation, currentUserId: previousUserId } = get()

          // Clear current conversation if:
          // 1. User is logging out (userId is null), OR
          // 2. User changed to a different user whose conversations don't include current one
          const shouldClearCurrent =
            currentConversation && (userId === null || currentConversation.userId !== userId)

          if (userId !== previousUserId) {
            useLayoutStore.getState().resetComposerState()
          }

          // Find first conversation for new user to auto-select
          const userConversations = userId ? conversations.filter((c) => c.userId === userId) : []
          const newCurrentConversation = shouldClearCurrent
            ? userConversations[0] || null
            : currentConversation

          set(
            {
              currentUserId: userId,
              currentConversation: newCurrentConversation,
            },
            false,
            'setCurrentUser'
          )

          // Restore session state from auto-selected conversation, or clear if none
          if (newCurrentConversation) {
            get().restoreSessionState(newCurrentConversation)
            restoreConversationDataSources(newCurrentConversation)
          } else {
            // No conversation selected - clear all ephemeral state
            set(
              {
                thinkingSteps: [],
                activeThinkingStepId: null,
                reportContent: '',
                reportContentCategory: null,
                currentStatus: null,
                planMessages: [],
                deepResearchCitations: [],
                deepResearchTodos: [],
                deepResearchLLMSteps: [],
                deepResearchAgents: [],
                deepResearchToolCalls: [],
                deepResearchFiles: [],
                deepResearchStreamLoaded: false,
                // Clear deep research job state
                deepResearchJobId: null,
                deepResearchLastEventId: null,
                isDeepResearchStreaming: false,
                deepResearchStatus: null,
                deepResearchOwnerConversationId: null,
                activeDeepResearchMessageId: null,
                // Clear HITL pending interaction
                pendingInteraction: null,
              },
              false,
              'setCurrentUser:clearState'
            )
          }
        },

        getUserConversations: () => {
          const { conversations, currentUserId } = get()
          if (!currentUserId) return []
          return conversations.filter((c) => c.userId === currentUserId)
        },

        createConversation: () => {
          const { currentUserId } = get()
          if (!currentUserId) {
            throw new Error('Cannot create conversation without authenticated user')
          }
          const layoutState = useLayoutStore.getState()
          const defaultEnabledDataSourceIds = getDefaultEnabledDataSourceIds()
          layoutState.setEnabledDataSources(defaultEnabledDataSourceIds)
          const newConversation: Conversation = {
            ...createNewConversation(currentUserId),
            enabledDataSourceIds: defaultEnabledDataSourceIds,
          }
          set(
            (state) => ({
              conversations: [newConversation, ...state.conversations],
              currentConversation: newConversation,
              // Clear all ResearchPanel content for new conversation
              thinkingSteps: [],
              activeThinkingStepId: null,
              reportContent: '',
              reportContentCategory: null,
              currentStatus: null,
              planMessages: [],
              deepResearchCitations: [],
              deepResearchTodos: [],
              deepResearchLLMSteps: [],
              deepResearchAgents: [],
              deepResearchToolCalls: [],
              deepResearchFiles: [],
              deepResearchStreamLoaded: false,
              // Clear deep research job state for new conversation
              deepResearchJobId: null,
              deepResearchLastEventId: null,
              isDeepResearchStreaming: false,
              deepResearchStatus: null,
              deepResearchOwnerConversationId: null,
              activeDeepResearchMessageId: null,
              // Clear HITL pending interaction
              pendingInteraction: null,
            }),
            false,
            'createConversation'
          )
          return newConversation
        },

        startNewSessionDraft: () => {
          const { currentUserId, currentConversation } = get()
          if (!currentUserId) {
            throw new Error('Cannot start session draft without authenticated user')
          }

          maybeDiscardAbandonedUploadOnlySession(get, currentConversation?.id)

          const layoutState = useLayoutStore.getState()
          const defaultEnabledDataSourceIds = getDefaultEnabledDataSourceIds()
          layoutState.setEnabledDataSources(defaultEnabledDataSourceIds)

          set(
            {
              currentConversation: null,
              // Clear shallow WebSocket state for draft sessions. A dev
              // reload or abandoned socket can otherwise leave the global
              // streaming flag true and make a fresh session look busy.
              isStreaming: false,
              isLoading: false,
              currentUserMessageId: null,
              // Clear all ResearchPanel content for draft session
              thinkingSteps: [],
              activeThinkingStepId: null,
              reportContent: '',
              reportContentCategory: null,
              currentStatus: null,
              planMessages: [],
              deepResearchCitations: [],
              deepResearchTodos: [],
              deepResearchLLMSteps: [],
              deepResearchAgents: [],
              deepResearchToolCalls: [],
              deepResearchFiles: [],
              deepResearchStreamLoaded: false,
              // Clear deep research job state for draft session
              deepResearchJobId: null,
              deepResearchLastEventId: null,
              isDeepResearchStreaming: false,
              deepResearchStatus: null,
              deepResearchOwnerConversationId: null,
              activeDeepResearchMessageId: null,
              // Clear HITL pending interaction
              pendingInteraction: null,
            },
            false,
            'startNewSessionDraft'
          )
        },

        ensureSession: () => {
          const { currentConversation, currentUserId } = get()

          if (currentConversation?.id) {
            return currentConversation.id
          }
          if (!currentUserId) {
            return undefined
          }

          ensureStorageCapacity(currentConversation?.id ?? null, currentUserId)

          const layoutState = useLayoutStore.getState()
          const defaultEnabledDataSourceIds = getDefaultEnabledDataSourceIds()
          layoutState.setEnabledDataSources(defaultEnabledDataSourceIds)
          const newConversation: Conversation = {
            ...createNewConversation(currentUserId),
            enabledDataSourceIds: defaultEnabledDataSourceIds,
          }
          set(
            (state) => ({
              conversations: [newConversation, ...state.conversations],
              currentConversation: newConversation,
              // Clear all ResearchPanel content for new session
              thinkingSteps: [],
              activeThinkingStepId: null,
              reportContent: '',
              reportContentCategory: null,
              currentStatus: null,
              planMessages: [],
              deepResearchCitations: [],
              deepResearchTodos: [],
              deepResearchLLMSteps: [],
              deepResearchAgents: [],
              deepResearchToolCalls: [],
              deepResearchFiles: [],
              deepResearchStreamLoaded: false,
              // Clear deep research job state for new session
              deepResearchJobId: null,
              deepResearchLastEventId: null,
              isDeepResearchStreaming: false,
              deepResearchStatus: null,
              deepResearchOwnerConversationId: null,
              activeDeepResearchMessageId: null,
              // Clear HITL pending interaction
              pendingInteraction: null,
            }),
            false,
            'ensureSession'
          )
          return newConversation.id
        },

        selectConversation: (conversationId: string) => {
          const beforeLeave = get()
          const leavingId =
            beforeLeave.currentConversation?.id &&
            beforeLeave.currentConversation.id !== conversationId
              ? beforeLeave.currentConversation.id
              : undefined

          if (leavingId) {
            maybeDiscardAbandonedUploadOnlySession(get, leavingId)
          }

          const {
            conversations,
            currentUserId,
            currentConversation,
            isDeepResearchStreaming,
            deepResearchOwnerConversationId,
            activeDeepResearchMessageId,
            deepResearchLastEventId,
          } = get()

          if (currentConversation?.id !== conversationId) {
            ensureStorageCapacity(conversationId, currentUserId)
          }

          const conversation = conversations.find((c) => c.id === conversationId)

          if (conversation && conversation.userId === currentUserId) {
            // Save lastEventId if actively streaming before clearing
            if (
              currentConversation &&
              currentConversation.id !== conversationId &&
              isDeepResearchStreaming &&
              deepResearchOwnerConversationId === currentConversation.id &&
              activeDeepResearchMessageId
            ) {
              get().patchConversationMessage(
                deepResearchOwnerConversationId,
                activeDeepResearchMessageId,
                { deepResearchLastEventId: deepResearchLastEventId || undefined }
              )
              // Persist full deep research state to sessionStorage before clearing
              // so switching back can restore todos, citations, agents, etc.
              get().persistDeepResearchToSession()
            }

            // Close research panel when switching conversations
            // This ensures fresh data loads when panel is reopened
            useLayoutStore.getState().closeRightPanel()

            // Always clear deep research ephemeral state when switching conversations
            // SSE will be disconnected by hook cleanup when deepResearchJobId becomes null
            set(
              {
                currentConversation: conversation,
                deepResearchJobId: null,
                deepResearchLastEventId: null,
                isDeepResearchStreaming: false,
                deepResearchStatus: null,
                deepResearchOwnerConversationId: null,
                activeDeepResearchMessageId: null,
                deepResearchCitations: [],
                deepResearchTodos: [],
                deepResearchLLMSteps: [],
                deepResearchAgents: [],
                deepResearchToolCalls: [],
                deepResearchFiles: [],
                deepResearchStreamLoaded: false,
                reportContent: '',
                reportContentCategory: null,
              },
              false,
              'selectConversation'
            )

            // Restore basic session state (thinkingSteps) from messages
            get().restoreSessionState(conversation)
            restoreConversationDataSources(conversation)
          }
        },

        addUserMessage: (
          content: string,
          metadata?: {
            enabledDataSources?: string[]
            messageFiles?: Array<{ id: string; fileName: string }>
            selectedModel?: string
          }
        ) => {
          const { currentConversation, conversations, currentUserId } = get()

          // Create conversation if none exists
          let conversation = currentConversation
          if (!conversation) {
            if (!currentUserId) {
              throw new Error('Cannot create conversation without authenticated user')
            }
            const layoutState = useLayoutStore.getState()
            conversation = {
              ...createNewConversation(currentUserId),
              enabledDataSourceIds: [...layoutState.enabledDataSourceIds],
            }
          }

          const newMessage: ChatMessage = {
            id: uuidv4(),
            role: 'user',
            content,
            timestamp: new Date(),
            messageType: 'user',
            enabledDataSources: metadata?.enabledDataSources,
            messageFiles: metadata?.messageFiles,
            selectedModel: metadata?.selectedModel,
          }
          // Update title on first user message (ignore file_upload_status and other system messages)
          const hasUserMessage = conversation.messages.some((m) => m.messageType === 'user')
          const shouldUpdateTitle = !hasUserMessage

          const updatedConversation: Conversation = {
            ...conversation,
            title: shouldUpdateTitle ? generateTitle(content) : conversation.title,
            messages: [...conversation.messages, newMessage],
            updatedAt: new Date(),
          }

          // Update conversations list
          const existingIndex = conversations.findIndex((c) => c.id === updatedConversation.id)
          let updatedConversations: Conversation[]

          if (existingIndex >= 0) {
            updatedConversations = updateConversationInList(conversations, updatedConversation)
          } else {
            updatedConversations = [updatedConversation, ...conversations]
          }

          set(
            {
              currentConversation: updatedConversation,
              conversations: updatedConversations,
              isLoading: true,
              // Set current user message ID for associating thinking steps
              currentUserMessageId: newMessage.id,
              // Clear active thinking step for new request (but keep historical steps)
              activeThinkingStepId: null,
              // NOTE: planMessages and reportContent are NOT cleared here
              // They persist until a NEW deep research job starts (see startDeepResearch)
            },
            false,
            'addUserMessage'
          )

          return newMessage
        },

        startAssistantMessage: () => {
          const { currentConversation, conversations } = get()
          if (!currentConversation) {
            throw new Error('No active conversation')
          }

          const newMessage: ChatMessage = {
            id: uuidv4(),
            role: 'assistant',
            content: '',
            timestamp: new Date(),
            messageType: 'assistant',
            isStreaming: true,
          }

          const updatedConversation: Conversation = {
            ...currentConversation,
            messages: [...currentConversation.messages, newMessage],
            updatedAt: new Date(),
          }

          const updatedConversations = updateConversationInList(conversations, updatedConversation)

          set(
            {
              currentConversation: updatedConversation,
              conversations: updatedConversations,
              isStreaming: true,
              isLoading: false,
            },
            false,
            'startAssistantMessage'
          )

          return newMessage
        },

        appendToAssistantMessage: (content: string) => {
          const { currentConversation, conversations } = get()
          if (!currentConversation) return

          const messages = currentConversation.messages
          const lastMessage = messages[messages.length - 1]

          if (!lastMessage || lastMessage.role !== 'assistant' || !lastMessage.isStreaming) {
            return
          }

          const updatedMessage: ChatMessage = {
            ...lastMessage,
            content: lastMessage.content + content,
          }

          const updatedMessages = [...messages.slice(0, -1), updatedMessage]

          const updatedConversation: Conversation = {
            ...currentConversation,
            messages: updatedMessages,
            updatedAt: new Date(),
          }

          const updatedConversations = updateConversationInList(conversations, updatedConversation)

          set(
            {
              currentConversation: updatedConversation,
              conversations: updatedConversations,
            },
            false,
            'appendToAssistantMessage'
          )
        },

        completeAssistantMessage: () => {
          const { currentConversation, conversations } = get()
          if (!currentConversation) return

          const messages = currentConversation.messages
          const lastMessage = messages[messages.length - 1]

          if (!lastMessage || lastMessage.role !== 'assistant') {
            set({ isStreaming: false }, false, 'completeAssistantMessage')
            return
          }

          const updatedMessage: ChatMessage = {
            ...lastMessage,
            isStreaming: false,
          }

          const updatedMessages = [...messages.slice(0, -1), updatedMessage]

          const updatedConversation: Conversation = {
            ...currentConversation,
            messages: updatedMessages,
            updatedAt: new Date(),
          }

          const updatedConversations = updateConversationInList(conversations, updatedConversation)

          set(
            {
              currentConversation: updatedConversation,
              conversations: updatedConversations,
              isStreaming: false,
            },
            false,
            'completeAssistantMessage'
          )
        },

        setLoading: (isLoading: boolean) => {
          set({ isLoading }, false, 'setLoading')
        },

        setStreaming: (isStreaming: boolean) => {
          set({ isStreaming }, false, 'setStreaming')
        },

        deleteConversation: (conversationId: string) => {
          const { currentConversation, conversations, deepResearchJobId, isDeepResearchStreaming } =
            get()

          // Find the conversation being deleted
          const conversationToDelete = conversations.find((c) => c.id === conversationId)

          // Check if this conversation has an active deep research job
          // Either from current ephemeral state (if deleting current conversation)
          // or from persisted message data
          let jobIdToCancel: string | null = null

          if (
            currentConversation?.id === conversationId &&
            isDeepResearchStreaming &&
            deepResearchJobId
          ) {
            // Deleting current conversation with active streaming
            jobIdToCancel = deepResearchJobId
          } else if (conversationToDelete) {
            // Check if conversation has a job ID in its messages
            const lastAgentResponse = [...conversationToDelete.messages]
              .reverse()
              .find((m) => m.messageType === 'agent_response' && m.deepResearchJobId)

            if (
              lastAgentResponse?.deepResearchJobId &&
              lastAgentResponse.deepResearchJobStatus !== 'success' &&
              lastAgentResponse.deepResearchJobStatus !== 'failure' &&
              lastAgentResponse.deepResearchJobStatus !== 'interrupted'
            ) {
              // Job might still be running
              jobIdToCancel = lastAgentResponse.deepResearchJobId
            }
          }

          // Cancel the job asynchronously (fire and forget)
          if (jobIdToCancel) {
            import('@/adapters/api/deep-research-client').then(({ cancelJob }) => {
              cancelJob(jobIdToCancel!).catch((err) => {
                console.warn('Failed to cancel deep research job on session delete:', err)
              })
            })
          }

          const updatedConversations = conversations.filter((c) => c.id !== conversationId)

          // If deleting the current conversation with active streaming, clear deep research state
          const isCurrentWithActiveResearch =
            currentConversation?.id === conversationId && isDeepResearchStreaming

          set(
            {
              conversations: updatedConversations,
              currentConversation:
                currentConversation?.id === conversationId ? null : currentConversation,
              // Clear deep research state if deleting current conversation with active job
              ...(isCurrentWithActiveResearch && {
                deepResearchJobId: null,
                deepResearchLastEventId: null,
                isDeepResearchStreaming: false,
                deepResearchStatus: null,
                deepResearchOwnerConversationId: null,
                activeDeepResearchMessageId: null,
                deepResearchCitations: [],
                deepResearchTodos: [],
                deepResearchLLMSteps: [],
                deepResearchAgents: [],
                deepResearchToolCalls: [],
                deepResearchFiles: [],
                deepResearchStreamLoaded: false,
                reportContent: '',
                reportContentCategory: null,
              }),
            },
            false,
            'deleteConversation'
          )
        },

        deleteAllConversations: () => {
          const {
            conversations,
            currentUserId,
            currentConversation,
            isDeepResearchStreaming,
            deepResearchJobId,
          } = get()

          if (!currentUserId) return

          // Get all conversations for the current user
          const userConversations = conversations.filter((c) => c.userId === currentUserId)

          // Collect job IDs from conversations with potentially active deep research
          const jobIdsToCancel: string[] = []

          // Add current streaming job if active
          if (isDeepResearchStreaming && deepResearchJobId) {
            jobIdsToCancel.push(deepResearchJobId)
          }

          // Check all user conversations for potentially active jobs
          for (const conv of userConversations) {
            const lastAgentResponse = [...conv.messages]
              .reverse()
              .find((m) => m.messageType === 'agent_response' && m.deepResearchJobId)

            if (
              lastAgentResponse?.deepResearchJobId &&
              lastAgentResponse.deepResearchJobStatus !== 'success' &&
              lastAgentResponse.deepResearchJobStatus !== 'failure' &&
              lastAgentResponse.deepResearchJobStatus !== 'interrupted' &&
              !jobIdsToCancel.includes(lastAgentResponse.deepResearchJobId)
            ) {
              jobIdsToCancel.push(lastAgentResponse.deepResearchJobId)
            }
          }

          // Cancel all jobs asynchronously (fire and forget)
          if (jobIdsToCancel.length > 0) {
            import('@/adapters/api/deep-research-client').then(async ({ cancelJob }) => {
              const results = await Promise.allSettled(
                jobIdsToCancel.map((jobId) => cancelJob(jobId))
              )

              results.forEach((result, index) => {
                if (result.status === 'fulfilled') return
                console.warn(
                  'Failed to cancel deep research job on delete all sessions:',
                  jobIdsToCancel[index],
                  result.reason
                )
              })
            })
          }

          // Clear all deep research session storage
          clearAllDeepResearchSessions()

          // Filter out all conversations belonging to current user
          const remainingConversations = conversations.filter((c) => c.userId !== currentUserId)

          // Check if current conversation belongs to user being cleared
          const shouldClearCurrent =
            currentConversation && currentConversation.userId === currentUserId

          set(
            {
              conversations: remainingConversations,
              currentConversation: shouldClearCurrent ? null : currentConversation,
              // Clear all deep research state
              deepResearchJobId: null,
              deepResearchLastEventId: null,
              isDeepResearchStreaming: false,
              deepResearchStatus: null,
              deepResearchOwnerConversationId: null,
              activeDeepResearchMessageId: null,
              deepResearchCitations: [],
              deepResearchTodos: [],
              deepResearchLLMSteps: [],
              deepResearchAgents: [],
              deepResearchToolCalls: [],
              deepResearchFiles: [],
              deepResearchStreamLoaded: false,
              // Clear other ephemeral state
              thinkingSteps: [],
              activeThinkingStepId: null,
              reportContent: '',
              reportContentCategory: null,
              currentStatus: null,
              planMessages: [],
              pendingInteraction: null,
            },
            false,
            'deleteAllConversations'
          )
        },

        updateConversationTitle: (conversationId: string, title: string) => {
          const { currentConversation, conversations } = get()

          const updatedConversations = conversations.map((c) =>
            c.id === conversationId ? { ...c, title, updatedAt: new Date() } : c
          )

          const updatedCurrentConversation =
            currentConversation?.id === conversationId
              ? { ...currentConversation, title, updatedAt: new Date() }
              : currentConversation

          set(
            {
              conversations: updatedConversations,
              currentConversation: updatedCurrentConversation,
            },
            false,
            'updateConversationTitle'
          )
        },

        saveDataSourcesToConversation: (ids: string[]) => {
          let { currentConversation, conversations } = get()

          if (!currentConversation) {
            const sessionId = get().ensureSession()
            if (!sessionId) return
            currentConversation = get().currentConversation
            conversations = get().conversations
            if (!currentConversation) return
          }

          const updatedConversation: Conversation = {
            ...currentConversation,
            enabledDataSourceIds: ids,
          }

          set(
            {
              currentConversation: updatedConversation,
              conversations: updateConversationInList(conversations, updatedConversation),
            },
            false,
            'saveDataSourcesToConversation'
          )
        },

        // ============================================================
        // New actions for thinking/report content and status/prompts
        // ============================================================

        addThinkingStep: (step: Omit<ThinkingStep, 'id' | 'timestamp' | 'userMessageId'>) => {
          const { currentUserMessageId, currentConversation, conversations } = get()
          if (!currentUserMessageId) {
            console.warn('addThinkingStep called without currentUserMessageId')
            return ''
          }

          const stepId = uuidv4()
          const newStep: ThinkingStep = {
            ...step,
            id: stepId,
            userMessageId: currentUserMessageId,
            timestamp: new Date(),
          }

          // Update ephemeral store
          let updatedConversation = currentConversation
          let updatedConversations = conversations

          // Also persist to the user message in conversation for session persistence
          if (currentConversation) {
            const updatedMessages = currentConversation.messages.map((msg) => {
              if (msg.id === currentUserMessageId) {
                return {
                  ...msg,
                  thinkingSteps: [...(msg.thinkingSteps || []), newStep],
                }
              }
              return msg
            })

            updatedConversation = {
              ...currentConversation,
              messages: updatedMessages,
              updatedAt: new Date(),
            }

            updatedConversations = updateConversationInList(conversations, updatedConversation)
          }

          set(
            {
              thinkingSteps: [...get().thinkingSteps, newStep],
              activeThinkingStepId: stepId,
              currentConversation: updatedConversation,
              conversations: updatedConversations,
            },
            false,
            'addThinkingStep'
          )

          return stepId
        },

        getThinkingStepsForMessage: (userMessageId: string) => {
          const { thinkingSteps } = get()
          return thinkingSteps.filter((step) => step.userMessageId === userMessageId)
        },

        appendToThinkingStep: (stepId: string, content: string) => {
          const { currentConversation, conversations, thinkingSteps } = get()

          // Update ephemeral store
          const updatedThinkingSteps = thinkingSteps.map((step) =>
            step.id === stepId ? { ...step, content: step.content + content } : step
          )

          // Find the userMessageId for this step to update persisted message
          const step = thinkingSteps.find((s) => s.id === stepId)
          let updatedConversation = currentConversation
          let updatedConversations = conversations

          if (step && currentConversation) {
            const updatedMessages = currentConversation.messages.map((msg) => {
              if (msg.id === step.userMessageId && msg.thinkingSteps) {
                return {
                  ...msg,
                  thinkingSteps: msg.thinkingSteps.map((s) =>
                    s.id === stepId ? { ...s, content: s.content + content } : s
                  ),
                }
              }
              return msg
            })

            updatedConversation = {
              ...currentConversation,
              messages: updatedMessages,
              updatedAt: new Date(),
            }

            updatedConversations = updateConversationInList(conversations, updatedConversation)
          }

          set(
            {
              thinkingSteps: updatedThinkingSteps,
              currentConversation: updatedConversation,
              conversations: updatedConversations,
            },
            false,
            'appendToThinkingStep'
          )
        },

        completeThinkingStep: (stepId: string) => {
          const { currentConversation, conversations, thinkingSteps, activeThinkingStepId } = get()

          // Update ephemeral store
          const updatedThinkingSteps = thinkingSteps.map((step) =>
            step.id === stepId
              ? {
                  ...step,
                  isComplete: true,
                  completedAt: step.completedAt ?? new Date(),
                  status: step.status === 'error' ? step.status : ('success' as const),
                }
              : step
          )

          // Find the userMessageId for this step to update persisted message
          const step = thinkingSteps.find((s) => s.id === stepId)
          let updatedConversation = currentConversation
          let updatedConversations = conversations

          if (step && currentConversation) {
            const updatedMessages = currentConversation.messages.map((msg) => {
              if (msg.id === step.userMessageId && msg.thinkingSteps) {
                return {
                  ...msg,
                  thinkingSteps: msg.thinkingSteps.map((s) =>
                    s.id === stepId
                      ? {
                          ...s,
                          isComplete: true,
                          completedAt: s.completedAt ?? new Date(),
                          status: s.status === 'error' ? s.status : ('success' as const),
                        }
                      : s
                  ),
                }
              }
              return msg
            })

            updatedConversation = {
              ...currentConversation,
              messages: updatedMessages,
              updatedAt: new Date(),
            }

            updatedConversations = updateConversationInList(conversations, updatedConversation)
          }

          set(
            {
              thinkingSteps: updatedThinkingSteps,
              activeThinkingStepId: activeThinkingStepId === stepId ? null : activeThinkingStepId,
              currentConversation: updatedConversation,
              conversations: updatedConversations,
            },
            false,
            'completeThinkingStep'
          )
        },

        failThinkingStep: (stepId: string) => {
          const { currentConversation, conversations, thinkingSteps, activeThinkingStepId } = get()

          const updatedThinkingSteps = thinkingSteps.map((step) =>
            step.id === stepId
              ? {
                  ...step,
                  isComplete: true,
                  completedAt: step.completedAt ?? new Date(),
                  status: 'error' as const,
                }
              : step
          )

          const step = thinkingSteps.find((s) => s.id === stepId)
          let updatedConversation = currentConversation
          let updatedConversations = conversations

          if (step && currentConversation) {
            const updatedMessages = currentConversation.messages.map((msg) => {
              if (msg.id === step.userMessageId && msg.thinkingSteps) {
                return {
                  ...msg,
                  thinkingSteps: msg.thinkingSteps.map((s) =>
                    s.id === stepId
                      ? {
                          ...s,
                          isComplete: true,
                          completedAt: s.completedAt ?? new Date(),
                          status: 'error' as const,
                        }
                      : s
                  ),
                }
              }
              return msg
            })

            updatedConversation = {
              ...currentConversation,
              messages: updatedMessages,
              updatedAt: new Date(),
            }

            updatedConversations = updateConversationInList(conversations, updatedConversation)
          }

          set(
            {
              thinkingSteps: updatedThinkingSteps,
              activeThinkingStepId: activeThinkingStepId === stepId ? null : activeThinkingStepId,
              currentConversation: updatedConversation,
              conversations: updatedConversations,
            },
            false,
            'failThinkingStep'
          )
        },

        updateThinkingStepByFunctionName: (
          functionName: string,
          content: string,
          isComplete: boolean
        ) => {
          const { currentConversation, conversations, thinkingSteps, currentUserMessageId } = get()

          // Update ephemeral store
          const updatedThinkingSteps = thinkingSteps.map((step) =>
            step.functionName === functionName && step.userMessageId === currentUserMessageId
              ? { ...step, content, isComplete }
              : step
          )

          // Find the step to get its userMessageId for persistence
          const step = thinkingSteps.find(
            (s) => s.functionName === functionName && s.userMessageId === currentUserMessageId
          )
          let updatedConversation = currentConversation
          let updatedConversations = conversations

          if (step && currentConversation) {
            const updatedMessages = currentConversation.messages.map((msg) => {
              if (msg.id === step.userMessageId && msg.thinkingSteps) {
                return {
                  ...msg,
                  thinkingSteps: msg.thinkingSteps.map((s) =>
                    s.functionName === functionName ? { ...s, content, isComplete } : s
                  ),
                }
              }
              return msg
            })

            updatedConversation = {
              ...currentConversation,
              messages: updatedMessages,
              updatedAt: new Date(),
            }

            updatedConversations = updateConversationInList(conversations, updatedConversation)
          }

          set(
            {
              thinkingSteps: updatedThinkingSteps,
              currentConversation: updatedConversation,
              conversations: updatedConversations,
            },
            false,
            'updateThinkingStepByFunctionName'
          )
        },

        findThinkingStepByFunctionName: (functionName: string) => {
          const { thinkingSteps, currentUserMessageId } = get()
          if (!currentUserMessageId) return undefined
          return thinkingSteps.find(
            (step) =>
              step.functionName === functionName && step.userMessageId === currentUserMessageId
          )
        },

        setReportContent: (content: string, category?: 'research_notes' | 'final_report') => {
          set(
            { reportContent: content, reportContentCategory: category ?? null },
            false,
            'setReportContent'
          )
        },

        clearThinkingSteps: () => {
          set({ thinkingSteps: [], activeThinkingStepId: null }, false, 'clearThinkingSteps')
        },

        clearReportContent: () => {
          set({ reportContent: '', reportContentCategory: null }, false, 'clearReportContent')
        },

        setCurrentStatus: (status: StatusType | null) => {
          set({ currentStatus: status }, false, 'setCurrentStatus')
        },

        addStatusCard: (type: StatusType, message?: string) => {
          const { currentConversation, conversations } = get()
          if (!currentConversation) return

          const statusMessage: ChatMessage = {
            id: uuidv4(),
            role: 'assistant',
            content: message || '',
            timestamp: new Date(),
            messageType: 'status',
            statusType: type,
          }

          const updatedConversation: Conversation = {
            ...currentConversation,
            messages: [...currentConversation.messages, statusMessage],
            updatedAt: new Date(),
          }

          const updatedConversations = updateConversationInList(conversations, updatedConversation)

          set(
            {
              currentConversation: updatedConversation,
              conversations: updatedConversations,
              currentStatus: type,
            },
            false,
            'addStatusCard'
          )
        },

        addAgentPrompt: (
          type: PromptType,
          content: string,
          options?: string[],
          placeholder?: string,
          promptId?: string,
          parentId?: string,
          inputType?: 'text' | 'multiple_choice' | 'binary_choice' | 'approval' | 'notification'
        ) => {
          const { currentConversation, conversations, planMessages } = get()
          if (!currentConversation) return

          const promptMessage: ChatMessage = {
            id: uuidv4(),
            role: 'assistant',
            content,
            timestamp: new Date(),
            messageType: 'prompt',
            promptType: type,
            promptId,
            promptParentId: parentId,
            promptInputType: inputType,
            promptOptions: options,
            promptPlaceholder: placeholder,
            isPromptResponded: false,
            // Persist current planMessages for session restoration during HITL wait
            planMessages: planMessages.length > 0 ? [...planMessages] : undefined,
          }

          const updatedConversation: Conversation = {
            ...currentConversation,
            messages: [...currentConversation.messages, promptMessage],
            updatedAt: new Date(),
          }

          const updatedConversations = updateConversationInList(conversations, updatedConversation)

          // Pause streaming/loading while waiting for user response
          set(
            {
              currentConversation: updatedConversation,
              conversations: updatedConversations,
              isLoading: false,
              isStreaming: false,
            },
            false,
            'addAgentPrompt'
          )
        },

        respondToPrompt: (messageId: string, response: string) => {
          const { currentConversation, conversations } = get()
          if (!currentConversation) return

          const updatedMessages = currentConversation.messages.map((msg) =>
            msg.id === messageId
              ? { ...msg, promptResponse: response, isPromptResponded: true }
              : msg
          )

          const updatedConversation: Conversation = {
            ...currentConversation,
            messages: updatedMessages,
            updatedAt: new Date(),
          }

          const updatedConversations = updateConversationInList(conversations, updatedConversation)

          set(
            {
              currentConversation: updatedConversation,
              conversations: updatedConversations,
              isLoading: true, // Resume loading after user responds
              pendingInteraction: null, // Clear pending interaction
            },
            false,
            'respondToPrompt'
          )

          // NOTE: This only updates local state. To send the response to the backend,
          // the UI component or hook should call sendPromptResponse() from chat-client.ts
          // after this action completes. The backend integration depends on the
          // /generate/respond endpoint being implemented.
        },

        // ============================================================
        // Actions for agent responses and HITL
        // ============================================================

        addAgentResponse: (content: string, showViewReport?: boolean) => {
          const {
            currentConversation,
            conversations,
            reportContent,
            deepResearchCitations,
            planMessages,
            deepResearchTodos,
            deepResearchLLMSteps,
            deepResearchAgents,
            deepResearchToolCalls,
            deepResearchFiles,
            // Job persistence fields
            deepResearchJobId,
            deepResearchLastEventId,
            deepResearchStatus,
          } = get()
          if (!currentConversation) return

          // Include all ResearchPanel content for session persistence
          const responseMessage: ChatMessage = {
            id: uuidv4(),
            role: 'assistant',
            content,
            timestamp: new Date(),
            messageType: 'agent_response',
            showViewReport,
            // Persist ResearchPanel content with this response
            reportContent: reportContent || undefined,
            citations: deepResearchCitations.length > 0 ? [...deepResearchCitations] : undefined,
            // Persist additional ResearchPanel tabs
            planMessages: planMessages.length > 0 ? [...planMessages] : undefined,
            deepResearchTodos: deepResearchTodos.length > 0 ? [...deepResearchTodos] : undefined,
            deepResearchLLMSteps:
              deepResearchLLMSteps.length > 0 ? [...deepResearchLLMSteps] : undefined,
            deepResearchAgents: deepResearchAgents.length > 0 ? [...deepResearchAgents] : undefined,
            deepResearchToolCalls:
              deepResearchToolCalls.length > 0 ? [...deepResearchToolCalls] : undefined,
            deepResearchFiles: deepResearchFiles.length > 0 ? [...deepResearchFiles] : undefined,
            // Persist deep research job metadata for session restoration
            deepResearchJobId: deepResearchJobId || undefined,
            deepResearchLastEventId: deepResearchLastEventId || undefined,
            deepResearchJobStatus: deepResearchStatus || undefined,
          }

          const updatedConversation: Conversation = {
            ...currentConversation,
            messages: [...currentConversation.messages, responseMessage],
            updatedAt: new Date(),
          }

          const updatedConversations = updateConversationInList(conversations, updatedConversation)

          set(
            {
              currentConversation: updatedConversation,
              conversations: updatedConversations,
            },
            false,
            'addAgentResponse'
          )

          // Proactive storage check after response — this is when storage
          // meaningfully grows, not just on session create/switch.
          if (!checkStorageHealth().isHealthy) {
            const { currentUserId } = get()
            ensureStorageCapacity(currentConversation.id, currentUserId)
          }
        },

        addAgentResponseWithMeta: (
          content: string,
          showViewReport: boolean,
          meta: Partial<ChatMessage>
        ): string => {
          const { currentConversation, conversations } = get()
          if (!currentConversation) return ''

          const messageId = uuidv4()
          const responseMessage: ChatMessage = {
            id: messageId,
            role: 'assistant',
            content,
            timestamp: new Date(),
            messageType: 'agent_response',
            showViewReport,
            ...meta,
          }

          const updatedConversation: Conversation = {
            ...currentConversation,
            messages: [...currentConversation.messages, responseMessage],
            updatedAt: new Date(),
          }

          const updatedConversations = updateConversationInList(conversations, updatedConversation)

          set(
            {
              currentConversation: updatedConversation,
              conversations: updatedConversations,
            },
            false,
            'addAgentResponseWithMeta'
          )

          return messageId
        },

        patchConversationMessage: (
          conversationId: string,
          messageId: string,
          patch: Partial<ChatMessage>
        ) => {
          const { currentConversation, conversations } = get()

          const targetConversation = conversations.find((c) => c.id === conversationId)
          if (!targetConversation) return

          const updatedMessages = targetConversation.messages.map((msg) =>
            msg.id === messageId ? { ...msg, ...patch } : msg
          )

          const updatedConversation: Conversation = {
            ...targetConversation,
            messages: updatedMessages,
            updatedAt: new Date(),
          }

          const updatedConversations = updateConversationInList(conversations, updatedConversation)

          const updatedCurrent =
            currentConversation?.id === conversationId ? updatedConversation : currentConversation

          set(
            {
              currentConversation: updatedCurrent,
              conversations: updatedConversations,
            },
            false,
            'patchConversationMessage'
          )
        },

        setPendingInteraction: (interaction: PendingInteraction | null) => {
          set({ pendingInteraction: interaction }, false, 'setPendingInteraction')
        },

        clearPendingInteraction: () => {
          set({ pendingInteraction: null }, false, 'clearPendingInteraction')
        },

        setRespondToInteractionFn: (fn) => {
          set({ respondToInteractionFn: fn }, false, 'setRespondToInteractionFn')
        },

        // ============================================================
        // Actions for file and error cards
        // ============================================================

        addFileCard: (data: FileCardData) => {
          const { currentConversation, conversations } = get()
          if (!currentConversation) return

          const fileMessage: ChatMessage = {
            id: uuidv4(),
            role: 'assistant',
            content: data.fileName,
            timestamp: new Date(),
            messageType: 'file',
            fileData: data,
          }

          const updatedConversation: Conversation = {
            ...currentConversation,
            messages: [...currentConversation.messages, fileMessage],
            updatedAt: new Date(),
          }

          const updatedConversations = updateConversationInList(conversations, updatedConversation)

          set(
            {
              currentConversation: updatedConversation,
              conversations: updatedConversations,
            },
            false,
            'addFileCard'
          )
        },

        updateFileCard: (messageId: string, data: Partial<FileCardData>) => {
          const { currentConversation, conversations } = get()
          if (!currentConversation) return

          const updatedMessages = currentConversation.messages.map((msg) =>
            msg.id === messageId && msg.fileData
              ? {
                  ...msg,
                  fileData: { ...msg.fileData, ...data },
                  content: data.fileName || msg.content,
                }
              : msg
          )

          const updatedConversation: Conversation = {
            ...currentConversation,
            messages: updatedMessages,
            updatedAt: new Date(),
          }

          const updatedConversations = updateConversationInList(conversations, updatedConversation)

          set(
            {
              currentConversation: updatedConversation,
              conversations: updatedConversations,
            },
            false,
            'updateFileCard'
          )
        },

        addErrorCard: (code: ErrorCode, message?: string, details?: string) => {
          const { currentConversation, conversations } = get()
          if (!currentConversation) return

          const errorMeta = getErrorMeta(code)

          const errorMessage: ChatMessage = {
            id: uuidv4(),
            role: 'assistant',
            content: message || errorMeta.defaultMessage,
            timestamp: new Date(),
            messageType: 'error',
            errorData: {
              errorCode: code,
              errorMessage: message,
              errorDetails: details,
            },
          }

          const updatedConversation: Conversation = {
            ...currentConversation,
            messages: [...currentConversation.messages, errorMessage],
            updatedAt: new Date(),
          }

          const updatedConversations = updateConversationInList(conversations, updatedConversation)

          set(
            {
              currentConversation: updatedConversation,
              conversations: updatedConversations,
            },
            false,
            'addErrorCard'
          )
        },

        dismissErrorCard: (messageId: string) => {
          const { currentConversation, conversations } = get()
          if (!currentConversation) return

          // Remove the error message from the conversation
          const updatedMessages = currentConversation.messages.filter((msg) => msg.id !== messageId)

          const updatedConversation: Conversation = {
            ...currentConversation,
            messages: updatedMessages,
            updatedAt: new Date(),
          }

          const updatedConversations = updateConversationInList(conversations, updatedConversation)

          set(
            {
              currentConversation: updatedConversation,
              conversations: updatedConversations,
            },
            false,
            'dismissErrorCard'
          )
        },

        dismissConnectionErrors: () => {
          const { currentConversation, conversations } = get()
          if (!currentConversation) return

          const updatedMessages = currentConversation.messages.filter(
            (msg) =>
              !(msg.messageType === 'error' && msg.errorData?.errorCode?.startsWith('connection.'))
          )

          if (updatedMessages.length === currentConversation.messages.length) return

          const updatedConversation: Conversation = {
            ...currentConversation,
            messages: updatedMessages,
            updatedAt: new Date(),
          }

          const updatedConversations = updateConversationInList(conversations, updatedConversation)

          set(
            {
              currentConversation: updatedConversation,
              conversations: updatedConversations,
            },
            false,
            'dismissConnectionErrors'
          )
        },

        // ============================================================
        // Actions for file upload status banners
        // ============================================================

        addFileUploadStatusCard: (
          type: FileUploadStatusType,
          fileCount: number,
          jobId: string,
          sessionId?: string
        ) => {
          const { currentConversation, conversations } = get()

          // Find target conversation: use sessionId if provided, otherwise currentConversation
          const targetConversation = sessionId
            ? conversations.find((c) => c.id === sessionId)
            : currentConversation

          if (!targetConversation) return

          const statusMessage: ChatMessage = {
            id: uuidv4(),
            role: 'assistant',
            content: '',
            timestamp: new Date(),
            messageType: 'file_upload_status',
            fileUploadStatusData: {
              type,
              fileCount,
              jobId,
            },
          }

          const updatedConversation: Conversation = {
            ...targetConversation,
            messages: [...targetConversation.messages, statusMessage],
            updatedAt: new Date(),
          }

          const updatedConversations = updateConversationInList(conversations, updatedConversation)

          // Only update currentConversation if we modified it
          const updatedCurrent =
            currentConversation?.id === targetConversation.id
              ? updatedConversation
              : currentConversation

          set(
            {
              currentConversation: updatedCurrent,
              conversations: updatedConversations,
            },
            false,
            'addFileUploadStatusCard'
          )
        },

        removeFileUploadWarning: () => {
          const { currentConversation, conversations } = get()
          if (!currentConversation) return

          // Remove all pending_warning file upload status messages from current conversation
          const updatedMessages = currentConversation.messages.filter(
            (msg) =>
              !(
                msg.messageType === 'file_upload_status' &&
                msg.fileUploadStatusData?.type === 'pending_warning'
              )
          )

          // Skip if nothing was removed
          if (updatedMessages.length === currentConversation.messages.length) return

          const updatedConversation: Conversation = {
            ...currentConversation,
            messages: updatedMessages,
            updatedAt: new Date(),
          }

          const updatedConversations = updateConversationInList(conversations, updatedConversation)

          set(
            {
              currentConversation: updatedConversation,
              conversations: updatedConversations,
            },
            false,
            'removeFileUploadWarning'
          )
        },

        // ============================================================
        // Actions for deep research banners
        // ============================================================

        addDeepResearchBanner: (
          bannerType: DeepResearchBannerType,
          jobId: string,
          conversationId?: string,
          stats?: { totalTokens?: number; toolCallCount?: number }
        ) => {
          const { currentConversation, conversations } = get()

          // Find target conversation: use conversationId if provided, otherwise currentConversation
          const targetConversation = conversationId
            ? conversations.find((c) => c.id === conversationId)
            : currentConversation

          if (!targetConversation) return

          const updatedConversation = withDeepResearchBanner(
            targetConversation,
            bannerType,
            jobId,
            stats
          )

          const updatedConversations = updateConversationInList(conversations, updatedConversation)

          // Only update currentConversation if we modified it
          const updatedCurrent =
            currentConversation?.id === targetConversation.id
              ? updatedConversation
              : currentConversation

          set(
            {
              currentConversation: updatedCurrent,
              conversations: updatedConversations,
            },
            false,
            'addDeepResearchBanner'
          )
        },

        // ============================================================
        // Actions for deep research SSE streaming
        // ============================================================

        startDeepResearch: (jobId: string, messageId?: string) => {
          const { currentConversation } = get()
          set(
            {
              deepResearchJobId: jobId,
              deepResearchLastEventId: null,
              isDeepResearchStreaming: true,
              deepResearchStatus: 'submitted',
              deepResearchOwnerConversationId: currentConversation?.id || null,
              activeDeepResearchMessageId: messageId || null,
              // Clear deep research execution content (but keep planMessages from planning phase)
              // planMessages are preserved to show the plan created during clarification
              reportContent: '',
              reportContentCategory: null,
              deepResearchCitations: [],
              deepResearchTodos: [],
              deepResearchLLMSteps: [],
              deepResearchAgents: [],
              deepResearchToolCalls: [],
              deepResearchFiles: [],
              deepResearchStreamLoaded: false,
            },
            false,
            'startDeepResearch'
          )
        },

        updateDeepResearchStatus: (status: DeepResearchJobStatus) => {
          set({ deepResearchStatus: status }, false, 'updateDeepResearchStatus')
        },

        completeDeepResearch: () => {
          const { deepResearchJobId } = get()
          // Clear sessionStorage since job is complete
          if (deepResearchJobId) {
            clearDeepResearchSession(deepResearchJobId)
          }
          set(
            {
              isDeepResearchStreaming: false,
              // Keep jobId, status, and citations for reference
            },
            false,
            'completeDeepResearch'
          )
        },

        addDeepResearchCitation: (url: string, content: string, isCited?: boolean) => {
          const { deepResearchCitations } = get()

          // Check if citation with same URL already exists
          const existingIndex = deepResearchCitations.findIndex((c) => c.url === url)

          if (existingIndex >= 0) {
            // Update existing citation - if it's being marked as cited, update that
            const updatedCitations = deepResearchCitations.map((c, i) => {
              if (i === existingIndex) {
                return {
                  ...c,
                  content: content || c.content,
                  // Once cited, always cited (citation_use trumps citation_source)
                  isCited: isCited || c.isCited,
                }
              }
              return c
            })

            set(
              { deepResearchCitations: updatedCitations },
              false,
              'addDeepResearchCitation:update'
            )
          } else {
            // Add new citation
            const newCitation: CitationSource = {
              id: uuidv4(),
              url,
              content,
              timestamp: new Date(),
              isCited,
            }

            set(
              {
                deepResearchCitations: [...deepResearchCitations, newCitation],
              },
              false,
              'addDeepResearchCitation'
            )
          }
        },

        setDeepResearchTodos: (todos: Array<{ content: string; status: string }>) => {
          const typedTodos = normalizeDeepResearchTodos(todos)
          const {
            conversations,
            deepResearchOwnerConversationId,
            activeDeepResearchMessageId,
          } = get()

          if (!deepResearchOwnerConversationId || !activeDeepResearchMessageId) {
            set({ deepResearchTodos: typedTodos }, false, 'setDeepResearchTodos')
            return
          }

          const targetConversation = conversations.find(
            (conversation) => conversation.id === deepResearchOwnerConversationId
          )

          if (!targetConversation) {
            set({ deepResearchTodos: typedTodos }, false, 'setDeepResearchTodos')
            return
          }

          set({ deepResearchTodos: typedTodos }, false, 'setDeepResearchTodos')

          if (deepResearchTodoPersistTimer) {
            clearTimeout(deepResearchTodoPersistTimer)
          }

          const scheduledOwnerConversationId = deepResearchOwnerConversationId
          const scheduledMessageId = activeDeepResearchMessageId

          // Keep the live UI responsive while coalescing persisted task snapshots.
          // SSE can emit several todo events in quick succession; only the latest
          // snapshot needs to survive a page refresh.
          deepResearchTodoPersistTimer = setTimeout(() => {
            deepResearchTodoPersistTimer = null

            const latestState = get()
            if (
              latestState.deepResearchOwnerConversationId !== scheduledOwnerConversationId ||
              latestState.activeDeepResearchMessageId !== scheduledMessageId
            ) {
              return
            }

            const latestConversation = latestState.conversations.find(
              (conversation) => conversation.id === scheduledOwnerConversationId
            )
            if (!latestConversation) return

            const latestMessage = latestConversation.messages.find(
              (message) => message.id === scheduledMessageId
            )
            if (areDeepResearchTodosEqual(latestMessage?.deepResearchTodos, typedTodos)) return

            const updatedConversation = patchConversationMessageById(
              latestConversation,
              scheduledMessageId,
              { deepResearchTodos: typedTodos }
            )
            const updatedConversations = updateConversationInList(
              latestState.conversations,
              updatedConversation
            )
            const updatedCurrent =
              latestState.currentConversation?.id === updatedConversation.id
                ? updatedConversation
                : latestState.currentConversation

            set(
              {
                conversations: updatedConversations,
                currentConversation: updatedCurrent,
              },
              false,
              'setDeepResearchTodos:persist'
            )
          }, DEEP_RESEARCH_TODO_PERSIST_DEBOUNCE_MS)
        },

        stopDeepResearchTodos: () => {
          if (deepResearchTodoPersistTimer) {
            clearTimeout(deepResearchTodoPersistTimer)
            deepResearchTodoPersistTimer = null
          }

          const {
            currentConversation,
            conversations,
            deepResearchTodos,
            deepResearchOwnerConversationId,
            activeDeepResearchMessageId,
          } = get()
          const stoppedTodos = deepResearchTodos.map((todo) => ({
            ...todo,
            status:
              todo.status === 'in_progress' || todo.status === 'pending'
                ? ('stopped' as const)
                : todo.status,
          }))

          if (!deepResearchOwnerConversationId || !activeDeepResearchMessageId) {
            set({ deepResearchTodos: stoppedTodos }, false, 'stopDeepResearchTodos')
            return
          }

          const targetConversation = conversations.find(
            (conversation) => conversation.id === deepResearchOwnerConversationId
          )

          if (!targetConversation) {
            set({ deepResearchTodos: stoppedTodos }, false, 'stopDeepResearchTodos')
            return
          }

          const updatedConversation = patchConversationMessageById(
            targetConversation,
            activeDeepResearchMessageId,
            { deepResearchTodos: stoppedTodos }
          )
          const updatedConversations = updateConversationInList(conversations, updatedConversation)
          const updatedCurrent =
            currentConversation?.id === updatedConversation.id ? updatedConversation : currentConversation

          set(
            {
              deepResearchTodos: stoppedTodos,
              conversations: updatedConversations,
              currentConversation: updatedCurrent,
            },
            false,
            'stopDeepResearchTodos'
          )
        },

        stopAllDeepResearchSpinners: (isSuccessfulCompletion = false) => {
          const {
            currentConversation,
            conversations,
            deepResearchTodos,
            deepResearchLLMSteps,
            deepResearchAgents,
            deepResearchToolCalls,
            deepResearchOwnerConversationId,
            activeDeepResearchMessageId,
          } = get()

          // Stop todos (pending/in_progress → stopped or completed)
          const stoppedTodos = deepResearchTodos.map((todo) => ({
            ...todo,
            status:
              todo.status === 'in_progress' || todo.status === 'pending'
                ? isSuccessfulCompletion
                  ? ('completed' as const)
                  : ('stopped' as const)
                : todo.status,
          }))

          // Complete LLM steps (mark incomplete ones as complete)
          const stoppedLLMSteps = deepResearchLLMSteps.map((step) => ({
            ...step,
            isComplete: true,
          }))

          // A terminal job status does not prove that an individual workflow
          // finished. Keep workflow.end authoritative for observed progress.
          const stoppedAgents = deepResearchAgents.map((agent) => ({
            ...agent,
            status: agent.status === 'running' ? ('error' as const) : agent.status,
          }))

          // Stop tool calls (running → complete or error based on job success)
          const stoppedToolCalls = deepResearchToolCalls.map((toolCall) => ({
            ...toolCall,
            status:
              toolCall.status === 'running'
                ? isSuccessfulCompletion
                  ? ('complete' as const)
                  : ('error' as const)
                : toolCall.status,
          }))

          const update = {
            deepResearchTodos: stoppedTodos,
            deepResearchLLMSteps: stoppedLLMSteps,
            deepResearchAgents: stoppedAgents,
            deepResearchToolCalls: stoppedToolCalls,
          }

          if (!deepResearchOwnerConversationId || !activeDeepResearchMessageId) {
            set(update, false, 'stopAllDeepResearchSpinners')
            return
          }

          const targetConversation = conversations.find(
            (conversation) => conversation.id === deepResearchOwnerConversationId
          )

          if (!targetConversation) {
            set(update, false, 'stopAllDeepResearchSpinners')
            return
          }

          const updatedConversation = patchConversationMessageById(
            targetConversation,
            activeDeepResearchMessageId,
            { deepResearchTodos: stoppedTodos }
          )
          const updatedConversations = updateConversationInList(conversations, updatedConversation)
          const updatedCurrent =
            currentConversation?.id === updatedConversation.id ? updatedConversation : currentConversation

          set(
            {
              ...update,
              conversations: updatedConversations,
              currentConversation: updatedCurrent,
            },
            false,
            'stopAllDeepResearchSpinners'
          )
        },

        clearDeepResearch: () => {
          set(
            {
              deepResearchJobId: null,
              deepResearchLastEventId: null,
              isDeepResearchStreaming: false,
              deepResearchStatus: null,
              deepResearchOwnerConversationId: null,
              activeDeepResearchMessageId: null,
              deepResearchCitations: [],
              deepResearchTodos: [],
              deepResearchLLMSteps: [],
              deepResearchAgents: [],
              deepResearchToolCalls: [],
              deepResearchFiles: [],
              deepResearchStreamLoaded: false,
            },
            false,
            'clearDeepResearch'
          )
        },

        setLoadedJobId: (jobId: string) => {
          set({ deepResearchJobId: jobId }, false, 'setLoadedJobId')
        },

        setStreamLoaded: (loaded: boolean) => {
          set({ deepResearchStreamLoaded: loaded }, false, 'setStreamLoaded')
        },

        setDeepResearchLastEventId: (eventId: string | null) => {
          set({ deepResearchLastEventId: eventId }, false, 'setDeepResearchLastEventId')
        },

        persistDeepResearchToSession: () => {
          const {
            deepResearchJobId,
            deepResearchLastEventId,
            deepResearchOwnerConversationId,
            activeDeepResearchMessageId,
            deepResearchStatus,
            isDeepResearchStreaming,
          } = get()

          // Only persist if there's an active streaming job
          if (!deepResearchJobId || !isDeepResearchStreaming) {
            return
          }

          saveDeepResearchToSession({
            jobId: deepResearchJobId,
            lastEventId: deepResearchLastEventId,
            ownerConversationId: deepResearchOwnerConversationId,
            activeMessageId: activeDeepResearchMessageId,
            status: deepResearchStatus,
          })
        },

        saveDeepResearchProgress: () => {
          const { currentConversation, isDeepResearchStreaming, deepResearchJobId, reportContent } =
            get()

          // Only save if there's an active deep research session
          if (!currentConversation || !isDeepResearchStreaming || !deepResearchJobId) {
            return
          }

          // Save current progress by creating an agent response with all accumulated data
          // This persists: reportContent, citations, todos, LLM steps, agents, tool calls, files, and job metadata
          const statusMessage = reportContent
            ? 'Research in progress...'
            : 'Deep research started. Progress will be restored when you return.'
          get().addAgentResponse(statusMessage, !!reportContent)
        },

        reconnectToActiveJob: async () => {
          const { currentConversation, isDeepResearchStreaming } = get()
          if (!currentConversation || isDeepResearchStreaming) return

          const conversationId = currentConversation.id

          // Find messages with in-progress jobs (running or submitted)
          const activeJobMessage = [...currentConversation.messages]
            .reverse()
            .find(
              (m) =>
                m.messageType === 'agent_response' &&
                m.deepResearchJobId &&
                m.isDeepResearchActive &&
                (m.deepResearchJobStatus === 'running' || m.deepResearchJobStatus === 'submitted')
            )

          if (!activeJobMessage?.deepResearchJobId) {
            return // No active job to reconnect
          }

          const jobId = activeJobMessage.deepResearchJobId
          const messageId = activeJobMessage.id

          try {
            const { getJobStatus } = await import('@/adapters/api/deep-research-client')

            // Verify conversation hasn't changed
            if (get().currentConversation?.id !== conversationId) return
            if (get().isDeepResearchStreaming) return

            const statusResponse = await getJobStatus(jobId)
            const currentStatus = statusResponse.status

            // Final check before setting state
            if (get().currentConversation?.id !== conversationId) return
            if (get().isDeepResearchStreaming) return

            if (currentStatus === 'running' || currentStatus === 'submitted') {
              // Start with empty arrays and null lastEventId to force a full SSE
              // replay from the beginning. The catch-up buffer collects all events
              // and flushes them to the store in a single setState call.
              set(
                {
                  deepResearchJobId: jobId,
                  deepResearchLastEventId: null,
                  isDeepResearchStreaming: true,
                  deepResearchStatus: currentStatus,
                  deepResearchOwnerConversationId: conversationId,
                  activeDeepResearchMessageId: messageId,
                  deepResearchCitations: [],
                  deepResearchTodos: [],
                  deepResearchLLMSteps: [],
                  deepResearchAgents: [],
                  deepResearchToolCalls: [],
                  deepResearchFiles: [],
                  deepResearchStreamLoaded: false,
                  reportContent: '',
                  reportContentCategory: null,
                  currentStatus: 'researching',
                },
                false,
                'reconnectToActiveJob'
              )
            } else {
              // Job completed while we were away - clear session storage and update message status
              clearDeepResearchSession(jobId)
              // Defensive cleanup: if sessionStorage restored items in 'running' state, fix them
              get().stopAllDeepResearchSpinners(currentStatus === 'success')
              get().patchConversationMessage(conversationId, messageId, {
                deepResearchJobStatus: currentStatus,
                isDeepResearchActive: false,
                showViewReport: currentStatus === 'success',
              })
              // Add terminal banner (also removes orphaned 'starting' banner for this job)
              const terminalBannerType: DeepResearchBannerType =
                currentStatus === 'success' ? 'success' : 'failure'
              get().addDeepResearchBanner(terminalBannerType, jobId, conversationId)
            }
          } catch (error) {
            console.warn('Failed to reconnect to active job:', error)
            if (isUnavailableDeepResearchJobError(error)) {
              clearDeepResearchSession(jobId)
              get().patchConversationMessage(conversationId, activeJobMessage.id, {
                deepResearchJobStatus: 'failure',
                isDeepResearchActive: false,
                showViewReport: Boolean(activeJobMessage.reportContent?.trim()),
              })
              get().addDeepResearchBanner('failure', jobId, conversationId)
            } else {
              // Mark as inactive to prevent retry loops
              get().patchConversationMessage(conversationId, activeJobMessage.id, {
                isDeepResearchActive: false,
              })
            }
          }
        },

        cleanupOrphanedStartingBanners: async () => {
          const { currentConversation } = get()
          if (!currentConversation) return

          const conversationId = currentConversation.id
          const syncTrackingMessageToTerminalState = (
            jobId: string,
            terminalStatus: DeepResearchJobStatus
          ): void => {
            const conversation = get().conversations.find((c) => c.id === conversationId)
            if (!conversation) return

            const trackingMessage = [...conversation.messages]
              .reverse()
              .find((m) => m.messageType === 'agent_response' && m.deepResearchJobId === jobId)

            if (!trackingMessage?.id) return

            const hasPartialReport = Boolean(trackingMessage.reportContent?.trim())
            get().patchConversationMessage(conversationId, trackingMessage.id, {
              deepResearchJobStatus: terminalStatus,
              isDeepResearchActive: false,
              showViewReport: terminalStatus === 'success' || hasPartialReport,
            })
          }
          const bannerTypeToTerminalStatus = (
            bannerType: DeepResearchBannerType | undefined
          ): DeepResearchJobStatus => {
            // Preserve the distinction between explicit user cancellation and
            // terminal failures such as expiry/deletion. Cancelled jobs map to
            // the interrupted job status; backend lookup failures map to failure.
            if (bannerType === 'success') return 'success'
            if (bannerType === 'cancelled') return 'interrupted'
            return 'failure'
          }

          const startingBanners = currentConversation.messages.filter(
            (m) =>
              m.messageType === 'deep_research_banner' &&
              m.deepResearchBannerData?.bannerType === 'starting'
          )

          if (startingBanners.length === 0) return

          // Separate into banners with an existing terminal banner vs those needing a REST check
          const orphanedIds: string[] = []
          const needsCheck: Array<{ bannerId: string; jobId: string }> = []

          for (const banner of startingBanners) {
            const bannerJobId = banner.deepResearchBannerData!.jobId

            const matchingTerminalBanner = currentConversation.messages.find(
              (m) =>
                m.messageType === 'deep_research_banner' &&
                m.deepResearchBannerData?.jobId === bannerJobId &&
                m.id !== banner.id &&
                ['success', 'failure', 'cancelled', 'expired'].includes(
                  m.deepResearchBannerData?.bannerType || ''
                )
            )

            if (matchingTerminalBanner) {
              const terminalStatus = bannerTypeToTerminalStatus(
                matchingTerminalBanner.deepResearchBannerData?.bannerType
              )
              syncTrackingMessageToTerminalState(bannerJobId, terminalStatus)
              orphanedIds.push(banner.id)
            } else {
              needsCheck.push({ bannerId: banner.id, jobId: bannerJobId })
            }
          }

          // Remove starting banners that already have a matching terminal banner
          if (orphanedIds.length > 0) {
            const conv = get().currentConversation
            if (conv && conv.id === conversationId) {
              const filtered = conv.messages.filter((m) => !orphanedIds.includes(m.id))
              const updatedConversation: Conversation = {
                ...conv,
                messages: filtered,
                updatedAt: new Date(),
              }
              const updatedConversations = updateConversationInList(
                get().conversations,
                updatedConversation
              )
              set(
                {
                  currentConversation: updatedConversation,
                  conversations: updatedConversations,
                },
                false,
                'cleanupOrphanedStartingBanners/removeOrphans'
              )
            }
          }

          // Poll REST API for remaining starting banners without a terminal counterpart
          if (needsCheck.length > 0) {
            try {
              const { getJobStatus } = await import('@/adapters/api/deep-research-client')
              for (const { jobId } of needsCheck) {
                // Bail out if conversation changed during async work
                if (get().currentConversation?.id !== conversationId) return
                try {
                  const statusResponse = await getJobStatus(jobId)
                  const terminalStatuses = ['success', 'failure', 'interrupted']
                  if (terminalStatuses.includes(statusResponse.status)) {
                    syncTrackingMessageToTerminalState(jobId, statusResponse.status)
                    const terminalType: DeepResearchBannerType =
                      statusResponse.status === 'success' ? 'success' : 'failure'
                    // addDeepResearchBanner removes the starting banner and adds the terminal one
                    get().addDeepResearchBanner(terminalType, jobId, conversationId)
                  }
                } catch (error) {
                  if (isUnavailableDeepResearchJobError(error)) {
                    clearDeepResearchSession(jobId)
                    syncTrackingMessageToTerminalState(jobId, 'failure')
                    get().addDeepResearchBanner('failure', jobId, conversationId)
                  }
                  // Other job check failures are likely transient — leave banner as-is
                }
              }
            } catch {
              // Module import failed — skip REST checks
            }
          }
        },

        refreshDeepResearchSessionStatuses: async () => {
          const { currentUserId, conversations } = get()

          if (!currentUserId) return

          const candidates = conversations
            .filter((conversation) => conversation.userId === currentUserId)
            .map((conversation) => ({
              conversation,
              message: getLatestDeepResearchMessage(conversation),
            }))
            .filter(
              (candidate): candidate is { conversation: Conversation; message: ChatMessage } =>
                Boolean(candidate.message?.deepResearchJobId)
            )

          if (candidates.length === 0) return

          let getJobStatus: typeof import('@/adapters/api/deep-research-client').getJobStatus
          try {
            const deepResearchClient = await import('@/adapters/api/deep-research-client')
            getJobStatus = deepResearchClient.getJobStatus
          } catch {
            return
          }

          type JobRefreshResult =
            | { kind: 'reachable'; status: DeepResearchJobStatus }
            | { kind: 'unavailable' }
            | { kind: 'transient_error' }

          const checkedJobs = new Map<string, JobRefreshResult>()

          for (const { message } of candidates) {
            const jobId = message.deepResearchJobId
            if (!jobId) continue
            let result = checkedJobs.get(jobId)

            if (!result) {
              try {
                const response = await getJobStatus(jobId)
                result = { kind: 'reachable', status: response.status }
              } catch (error) {
                if (!isUnavailableDeepResearchJobError(error)) {
                  result = { kind: 'transient_error' }
                  checkedJobs.set(jobId, result)
                  continue
                }
                result = { kind: 'unavailable' }
              }

              checkedJobs.set(jobId, result)
            }
          }

          if ([...checkedJobs.values()].every((result) => result.kind === 'transient_error')) {
            return
          }

          const latestState = get()
          const inactiveJobIds = new Set<string>()

          const updatedConversations = latestState.conversations.map((conversation) => {
            if (conversation.userId !== currentUserId) return conversation

            const message = getLatestDeepResearchMessage(conversation)
            const jobId = message?.deepResearchJobId
            if (!message || !jobId) return conversation

            const result = checkedJobs.get(jobId)
            if (!result || result.kind === 'transient_error') return conversation

            if (result.kind === 'unavailable') {
              clearDeepResearchSession(jobId)
              inactiveJobIds.add(jobId)

              const hadCompletedReport =
                isCompletedDeepResearchReportMessage(message) ||
                Boolean(message.deepResearchReportExpired)

              const patchedConversation = patchLatestDeepResearchJobMessage(conversation, jobId, {
                deepResearchJobStatus: 'failure',
                isDeepResearchActive: false,
                showViewReport: false,
                deepResearchReportExpired: hadCompletedReport,
              })

              return hadCompletedReport
                ? withDeepResearchBanner(patchedConversation, 'expired', jobId)
                : patchedConversation
            }

            if (result.status === 'submitted' || result.status === 'running') {
              return patchLatestDeepResearchJobMessage(conversation, jobId, {
                deepResearchJobStatus: result.status,
                isDeepResearchActive: true,
                deepResearchReportExpired: false,
              })
            }

            inactiveJobIds.add(jobId)
            clearDeepResearchSession(jobId)

            return patchLatestDeepResearchJobMessage(conversation, jobId, {
              deepResearchJobStatus: result.status,
              isDeepResearchActive: false,
              showViewReport: result.status === 'success' || Boolean(message.reportContent?.trim()),
              deepResearchReportExpired: false,
            })
          })

          const updatedCurrentConversation =
            latestState.currentConversation === null
              ? null
              : (updatedConversations.find(
                  (conversation) => conversation.id === latestState.currentConversation?.id
                ) ?? latestState.currentConversation)

          const shouldClearActiveJob =
            latestState.deepResearchJobId !== null &&
            inactiveJobIds.has(latestState.deepResearchJobId)

          set(
            {
              conversations: updatedConversations,
              currentConversation: updatedCurrentConversation,
              ...(shouldClearActiveJob && {
                deepResearchJobId: null,
                deepResearchLastEventId: null,
                isDeepResearchStreaming: false,
                deepResearchStatus: null,
                deepResearchOwnerConversationId: null,
                activeDeepResearchMessageId: null,
                deepResearchCitations: [],
                deepResearchTodos: [],
                deepResearchLLMSteps: [],
                deepResearchAgents: [],
                deepResearchToolCalls: [],
                deepResearchFiles: [],
                deepResearchStreamLoaded: false,
                reportContent: '',
                reportContentCategory: null,
                currentStatus: null,
                pendingInteraction: null,
              }),
            },
            false,
            'refreshDeepResearchSessionStatuses'
          )
        },

        // ============================================================
        // Actions for deep research ThinkingTab sub-tabs
        // ============================================================

        addDeepResearchLLMStep: (
          step: Omit<DeepResearchLLMStep, 'id' | 'timestamp' | 'isComplete'>
        ) => {
          const stepId = uuidv4()
          const newStep: DeepResearchLLMStep = {
            ...step,
            id: stepId,
            timestamp: new Date(),
            isComplete: false,
          }

          set(
            (state) => ({
              deepResearchLLMSteps: [...state.deepResearchLLMSteps, newStep],
            }),
            false,
            'addDeepResearchLLMStep'
          )

          return stepId
        },

        appendToDeepResearchLLMStep: (stepId: string, content: string) => {
          set(
            (state) => ({
              deepResearchLLMSteps: state.deepResearchLLMSteps.map((step) =>
                step.id === stepId ? { ...step, content: step.content + content } : step
              ),
            }),
            false,
            'appendToDeepResearchLLMStep'
          )
        },

        completeDeepResearchLLMStep: (
          stepId: string,
          thinking?: string,
          usage?: { input_tokens: number; output_tokens: number }
        ) => {
          set(
            (state) => ({
              deepResearchLLMSteps: state.deepResearchLLMSteps.map((step) =>
                step.id === stepId ? { ...step, isComplete: true, thinking, usage } : step
              ),
            }),
            false,
            'completeDeepResearchLLMStep'
          )
        },

        addDeepResearchAgent: (agent: Omit<DeepResearchAgent, 'id' | 'startedAt' | 'status'>) => {
          const agentId = uuidv4()
          const newAgent: DeepResearchAgent = {
            ...agent,
            id: agentId,
            startedAt: new Date(),
            status: 'running',
          }

          set(
            (state) => ({
              deepResearchAgents: [...state.deepResearchAgents, newAgent],
            }),
            false,
            'addDeepResearchAgent'
          )

          return agentId
        },

        addDeepResearchAgentWithId: (
          id: string,
          agent: Omit<DeepResearchAgent, 'id' | 'startedAt' | 'status'>
        ) => {
          const { deepResearchAgents } = get()

          if (deepResearchAgents.some((a) => a.id === id)) {
            return id
          }

          const newAgent: DeepResearchAgent = {
            ...agent,
            id,
            startedAt: new Date(),
            status: 'running',
          }

          set(
            (state) => ({
              deepResearchAgents: [...state.deepResearchAgents, newAgent],
            }),
            false,
            'addDeepResearchAgentWithId'
          )

          return id
        },

        completeDeepResearchAgent: (agentId: string, output?: string) => {
          set(
            (state) => ({
              deepResearchAgents: state.deepResearchAgents.map((agent) =>
                agent.id === agentId
                  ? { ...agent, status: 'complete' as const, output, completedAt: new Date() }
                  : agent
              ),
            }),
            false,
            'completeDeepResearchAgent'
          )
        },

        addDeepResearchToolCall: (
          toolCall: Omit<DeepResearchToolCall, 'id' | 'timestamp' | 'status'>
        ) => {
          const toolCallId = uuidv4()
          const newToolCall: DeepResearchToolCall = {
            ...toolCall,
            id: toolCallId,
            timestamp: new Date(),
            status: 'running',
          }

          set(
            (state) => ({
              deepResearchToolCalls: [...state.deepResearchToolCalls, newToolCall],
            }),
            false,
            'addDeepResearchToolCall'
          )

          return toolCallId
        },

        getAgentToolCalls: (agentId: string) => {
          const { deepResearchToolCalls } = get()
          return deepResearchToolCalls.filter((tc) => tc.agentId === agentId)
        },

        completeDeepResearchToolCall: (toolCallId: string, output?: string) => {
          set(
            (state) => ({
              deepResearchToolCalls: state.deepResearchToolCalls.map((toolCall) =>
                toolCall.id === toolCallId
                  ? { ...toolCall, status: 'complete' as const, output }
                  : toolCall
              ),
            }),
            false,
            'completeDeepResearchToolCall'
          )
        },

        addDeepResearchFile: (file: Omit<DeepResearchFile, 'id' | 'timestamp'>) => {
          const { deepResearchFiles } = get()
          const existingIndex = deepResearchFiles.findIndex((f) => f.filename === file.filename)

          if (existingIndex >= 0) {
            // Update existing text or durable generated-artifact metadata.
            const updatedFiles = deepResearchFiles.map((f, i) =>
              i === existingIndex ? { ...f, ...file, timestamp: new Date() } : f
            )
            set({ deepResearchFiles: updatedFiles }, false, 'addDeepResearchFile:update')
            return deepResearchFiles[existingIndex].id
          }

          const fileId = uuidv4()
          const newFile: DeepResearchFile = {
            ...file,
            id: fileId,
            timestamp: new Date(),
          }

          set(
            (state) => ({
              deepResearchFiles: [...state.deepResearchFiles, newFile],
            }),
            false,
            'addDeepResearchFile'
          )

          return fileId
        },

        // ============================================================
        // Actions for plan/HITL messages
        // ============================================================

        addPlanMessage: (message: Omit<PlanMessage, 'id' | 'timestamp'>) => {
          const messageId = uuidv4()
          const newMessage: PlanMessage = {
            ...message,
            id: messageId,
            timestamp: new Date(),
          }

          set(
            (state) => ({
              planMessages: [...state.planMessages, newMessage],
            }),
            false,
            'addPlanMessage'
          )

          // Persist planMessages for HITL recovery on page refresh
          get().persistPlanMessages()

          return messageId
        },

        updatePlanMessageResponse: (messageId: string, response: string) => {
          set(
            (state) => ({
              planMessages: state.planMessages.map((msg) =>
                msg.id === messageId ? { ...msg, userResponse: response } : msg
              ),
            }),
            false,
            'updatePlanMessageResponse'
          )

          // Persist planMessages for HITL recovery on page refresh
          get().persistPlanMessages()
        },

        clearPlanMessages: () => {
          set({ planMessages: [] }, false, 'clearPlanMessages')
        },

        persistPlanMessages: () => {
          const { currentConversation, conversations, planMessages } = get()
          if (!currentConversation || planMessages.length === 0) return

          // Find the most recent unresponded prompt message to attach planMessages to
          // This ensures planMessages survive page refresh during HITL flows
          const messages = currentConversation.messages
          const lastPromptIndex = [...messages]
            .reverse()
            .findIndex((m) => m.messageType === 'prompt' && !m.isPromptResponded)

          if (lastPromptIndex >= 0) {
            const actualIndex = messages.length - 1 - lastPromptIndex
            const updatedMessages = messages.map((msg, idx) =>
              idx === actualIndex ? { ...msg, planMessages: [...planMessages] } : msg
            )

            const updatedConversation: Conversation = {
              ...currentConversation,
              messages: updatedMessages,
              updatedAt: new Date(),
            }

            const updatedConversations = updateConversationInList(
              conversations,
              updatedConversation
            )

            set(
              {
                currentConversation: updatedConversation,
                conversations: updatedConversations,
              },
              false,
              'persistPlanMessages'
            )
          }
        },

        // ============================================================
        // Session restoration
        // ============================================================

        restoreSessionState: (conversation: Conversation) => {
          // Aggregate thinkingSteps from all user messages in the conversation
          const allSteps = conversation.messages
            .filter((m) => m.thinkingSteps && m.thinkingSteps.length > 0)
            .flatMap((m) => m.thinkingSteps!)

          // Get latest ResearchPanel content from last agent_response message
          const lastAgentResponse = [...conversation.messages]
            .reverse()
            .find((m) => m.messageType === 'agent_response')

          // Check for unresponded HITL prompt to restore pendingInteraction
          const unrespondedPrompt = [...conversation.messages]
            .reverse()
            .find((m) => m.messageType === 'prompt' && !m.isPromptResponded)

          let restoredPendingInteraction: PendingInteraction | null = null
          if (
            unrespondedPrompt?.promptId &&
            unrespondedPrompt?.promptParentId &&
            unrespondedPrompt?.promptInputType
          ) {
            restoredPendingInteraction = {
              id: unrespondedPrompt.promptId,
              parentId: unrespondedPrompt.promptParentId,
              inputType: unrespondedPrompt.promptInputType,
              text: unrespondedPrompt.content,
              options: unrespondedPrompt.promptOptions,
            }
          }

          // Restore planMessages from unresponded prompt (during HITL wait) or last agent response
          const restoredPlanMessages =
            unrespondedPrompt?.planMessages || lastAgentResponse?.planMessages || []

          // Restore only lightweight local research state. Heavy stream details
          // are removed by pruneMessageForStorage and fetched on demand via
          // importStreamOnly() when the user opens ResearchPanel tabs.
          const restoredDeepResearchTodos = lastAgentResponse?.deepResearchTodos || []

          set(
            {
              thinkingSteps: allSteps,
              activeThinkingStepId: null,
              // DO NOT restore heavy research data - will be fetched from backend on demand
              reportContent: '',
              reportContentCategory: null,
              deepResearchCitations: [],
              deepResearchTodos: restoredDeepResearchTodos,
              deepResearchLLMSteps: [],
              deepResearchAgents: [],
              deepResearchToolCalls: [],
              deepResearchFiles: [],
              // ONLY restore planMessages - cannot be fetched from backend (WebSocket only)
              planMessages: restoredPlanMessages,
              // Clear streaming/loading state for restored sessions
              // In-progress jobs will reconnect via reconnectToActiveJob
              isStreaming: false,
              isLoading: false,
              currentStatus: null,
              // Restore pending HITL interaction from unresponded prompt message
              pendingInteraction: restoredPendingInteraction,
              // Restore job ID so research data can be fetched on demand
              deepResearchJobId: lastAgentResponse?.deepResearchJobId || null,
              deepResearchLastEventId: null,
              isDeepResearchStreaming: false,
              deepResearchStatus: null,
              activeDeepResearchMessageId: lastAgentResponse?.id || null,
              deepResearchOwnerConversationId: conversation.id,
              // Set to false to trigger lazy loading when tabs are opened
              deepResearchStreamLoaded: false,
            },
            false,
            'restoreSessionState'
          )

          // Detect interrupted responses: if the last meaningful message is a user
          // message with thinking steps but no following response, the response was
          // interrupted by a page refresh or browser close mid-stream.
          // Skip if there's a pending HITL interaction (user is expected to respond).
          if (!restoredPendingInteraction) {
            const meaningfulTypes = new Set([
              'user',
              'assistant',
              'agent_response',
              'error',
              'prompt',
            ])
            const lastMeaningful = [...conversation.messages]
              .reverse()
              .find((m) => meaningfulTypes.has(m.messageType ?? ''))

            if (lastMeaningful?.messageType === 'user' && lastMeaningful.thinkingSteps?.length) {
              get().addErrorCard(
                'agent.response_interrupted',
                'Your previous request was not completed. Please resend your message.'
              )
            }
          }
        },

        // ============================================================
        // Session busy checks (for disabling UI controls)
        // ============================================================

        /**
         * Check if a specific session has active operations.
         * MUST scan message history because ephemeral state is cleared on session switch.
         *
         * @param conversationId - The conversation ID to check
         * @returns true if the session has active operations (shallow or deep research)
         */
        isSessionBusy: (conversationId: string) => {
          const state = get()

          // Check if this is the current session with active shallow thinking (WebSocket)
          if (state.currentConversation?.id === conversationId && state.isStreaming) {
            return true
          }

          // Check if this is the currently streaming deep research session (ephemeral check)
          if (
            state.deepResearchOwnerConversationId === conversationId &&
            state.isDeepResearchStreaming
          ) {
            return true
          }

          // CRITICAL: Check message history for background deep research jobs
          // This is the ONLY way to detect jobs in non-current sessions
          // Uses shared utility that scans from end for O(1) typical performance
          const conversation = state.conversations.find((c) => c.id === conversationId)
          if (conversation && hasActiveDeepResearchJob(conversation.messages)) {
            return true
          }

          return false
        },

        /**
         * Check if ANY session has active operations.
         * Used to disable "Delete All Sessions" button.
         *
         * @returns true if any session has active operations
         */
        hasAnyBusySession: () => {
          const state = get()
          // Check global pending interaction (persisted, survives refresh)
          if (state.pendingInteraction !== null) return true
          return state.conversations.some((conv) => state.isSessionBusy(conv.id))
        },
      }),
      {
        name: 'deep-researcher-chat-store',
        storage: typeof window === 'undefined' ? undefined : createResilientStorage(),
        partialize: (state) => ({
          // Persist conversations and user context, not streaming state or panel content
          currentUserId: state.currentUserId,
          conversations: state.conversations,
          currentConversation: state.currentConversation,
          // Persist pending HITL interaction for page refresh recovery
          pendingInteraction: state.pendingInteraction,
        }),
        // After rehydration, refresh persisted job metadata. Missing backend
        // reports are represented in-session instead of deleting the chat.
        onRehydrateStorage: () => (state) => {
          if (!state || typeof window === 'undefined') return
          queueMicrotask(() => {
            void useChatStore.getState().refreshDeepResearchSessionStatuses()
          })
        },
      }
    ),
    { name: 'ChatStore' }
  )
)

// ============================================================
// Selectors
// ============================================================

export const selectHasConnectionError = (state: ChatStore): boolean =>
  state.currentConversation?.messages.some(
    (m) => m.messageType === 'error' && m.errorData?.errorCode?.startsWith('connection.')
  ) ?? false

/**
 * Resolve the deep-research job id for artifact resolution (report images, PDF/markdown
 * export). Prefers the active streaming job, then falls back to the latest deep-research
 * message in the conversation — the active id is cleared once a job stops streaming, but a
 * finished report still needs the id to fetch its artifact content.
 */
export const selectResolvedDeepResearchJobId = (state: ChatStore): string | undefined => {
  if (state.deepResearchJobId) return state.deepResearchJobId
  const conversation = state.currentConversation
  if (!conversation) return undefined
  return getLatestDeepResearchMessage(conversation)?.deepResearchJobId ?? undefined
}

// ============================================================
// Storage Event Monitoring (for debugging session clearing)
// ============================================================

if (typeof window !== 'undefined') {
  // Log initial hydration state (dev-only)
  const initialState = useChatStore.getState()
  logStoreHydration(true, initialState.conversations?.length ?? 0, initialState.currentUserId)

  // Monitor storage events from other tabs or browser extensions
  window.addEventListener('storage', (event) => {
    // Only log events related to our chat store
    if (event.key === 'deep-researcher-chat-store') {
      logExternalStorageEvent(event.key, event.oldValue, event.newValue)

      // If the store was cleared externally, this is critical
      if (event.oldValue !== null && event.newValue === null) {
        console.error(
          '[SessionsStore] ❌ CRITICAL: Storage cleared by external source (browser extension, dev tools, or another tab)'
        )
      }
    }
  })
}
