// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, test, expect, beforeEach, afterEach, vi } from 'vitest'
import { useChatStore } from './store'
import type { Conversation, PendingInteraction, FileCardData } from './types'

const STORAGE_KEY = 'deep-researcher-chat-store'
const mockLayoutState = vi.hoisted(() => ({
  closeRightPanel: vi.fn(),
  enabledDataSourceIds: ['web_search'],
  availableDataSources: [{ id: 'web_search' }, { id: 'knowledge_base', requires_auth: true }],
  setEnabledDataSources: vi.fn(),
  resetComposerState: vi.fn(),
}))
const mockDeepResearchApi = vi.hoisted(() => ({
  getJobStatus: vi.fn(),
  cancelJob: vi.fn(),
}))

// Mock the layout store
vi.mock('@/features/layout/store', () => ({
  useLayoutStore: {
    getState: () => mockLayoutState,
  },
}))

vi.mock('@/adapters/api/deep-research-client', () => ({
  getJobStatus: mockDeepResearchApi.getJobStatus,
  cancelJob: mockDeepResearchApi.cancelJob,
}))

const mockDiscardSessionResources = vi.hoisted(() => vi.fn())

vi.mock('@/features/documents/discard-session-resources', () => ({
  discardSessionDocumentsResources: mockDiscardSessionResources,
}))

describe('useChatStore', () => {
  beforeEach(() => {
    // Clear localStorage before each test
    localStorage.removeItem(STORAGE_KEY)
    mockLayoutState.closeRightPanel.mockClear()
    mockLayoutState.setEnabledDataSources.mockClear()
    mockLayoutState.resetComposerState.mockClear()
    mockLayoutState.enabledDataSourceIds = ['web_search']
    mockLayoutState.availableDataSources = [
      { id: 'web_search' },
      { id: 'knowledge_base', requires_auth: true },
    ]
    mockDiscardSessionResources.mockClear()
    mockDeepResearchApi.getJobStatus.mockReset()
    mockDeepResearchApi.cancelJob.mockReset()
    // Reset store to initial state before each test
    useChatStore.setState({
      currentUserId: null,
      currentConversation: null,
      conversations: [],
      isStreaming: false,
      isLoading: false,
      currentUserMessageId: null,
      thinkingSteps: [],
      activeThinkingStepId: null,
      reportContent: '',
      currentStatus: null,
      pendingInteraction: null,
    })
  })

  afterEach(() => {
    vi.useRealTimers()
    // Clean up localStorage after each test
    localStorage.removeItem(STORAGE_KEY)
  })

  describe('initial state', () => {
    test('has correct default values', () => {
      const state = useChatStore.getState()

      expect(state.currentUserId).toBeNull()
      expect(state.currentConversation).toBeNull()
      expect(state.conversations).toEqual([])
      expect(state.isStreaming).toBe(false)
      expect(state.isLoading).toBe(false)
      expect(state.currentUserMessageId).toBeNull()
      expect(state.thinkingSteps).toEqual([])
      expect(state.activeThinkingStepId).toBeNull()
      expect(state.reportContent).toBe('')
      expect(state.currentStatus).toBeNull()
      expect(state.pendingInteraction).toBeNull()
    })
  })

  describe('setCurrentUser', () => {
    test('sets user ID', () => {
      useChatStore.getState().setCurrentUser('user-1')

      expect(useChatStore.getState().currentUserId).toBe('user-1')
    })

    test('clears thinking state when user changes', () => {
      useChatStore.setState({
        currentUserId: 'user-1',
        currentUserMessageId: 'msg-1',
        thinkingSteps: [
          {
            id: '1',
            userMessageId: 'msg-1',
            category: 'agents',
            functionName: 'test',
            displayName: 'Test',
            content: '',
            timestamp: new Date(),
            isComplete: false,
          },
        ],
        activeThinkingStepId: '1',
        reportContent: 'Some report',
        currentStatus: 'thinking',
      })

      useChatStore.getState().setCurrentUser('user-2')

      const state = useChatStore.getState()
      expect(state.thinkingSteps).toEqual([])
      expect(state.activeThinkingStepId).toBeNull()
      expect(state.reportContent).toBe('')
      expect(state.currentStatus).toBeNull()
    })

    test('auto-selects first conversation for new user', () => {
      const conv1: Conversation = {
        id: 'conv-1',
        userId: 'user-1',
        title: 'Conv 1',
        messages: [],
        createdAt: new Date(),
        updatedAt: new Date(),
      }
      const conv2: Conversation = {
        id: 'conv-2',
        userId: 'user-2',
        title: 'Conv 2',
        messages: [],
        createdAt: new Date(),
        updatedAt: new Date(),
      }

      useChatStore.setState({
        currentUserId: 'user-1',
        currentConversation: conv1,
        conversations: [conv1, conv2],
      })

      useChatStore.getState().setCurrentUser('user-2')

      expect(useChatStore.getState().currentConversation).toEqual(conv2)
      expect(mockLayoutState.setEnabledDataSources).toHaveBeenCalledWith([
        'web_search',
        'knowledge_base',
      ])
    })

    test('clears current conversation when logging out', () => {
      const conv: Conversation = {
        id: 'conv-1',
        userId: 'user-1',
        title: 'Conv 1',
        messages: [],
        createdAt: new Date(),
        updatedAt: new Date(),
      }

      useChatStore.setState({
        currentUserId: 'user-1',
        currentConversation: conv,
        conversations: [conv],
      })

      useChatStore.getState().setCurrentUser(null)

      expect(useChatStore.getState().currentConversation).toBeNull()
    })

    test('clears layout composer state on account switch', () => {
      useChatStore.setState({ currentUserId: 'user-1' })

      useChatStore.getState().setCurrentUser('user-2')

      expect(mockLayoutState.resetComposerState).toHaveBeenCalledTimes(1)
    })

    test('clears layout composer state on logout', () => {
      useChatStore.setState({ currentUserId: 'user-1' })

      useChatStore.getState().setCurrentUser(null)

      expect(mockLayoutState.resetComposerState).toHaveBeenCalledTimes(1)
    })

    test('does not clear composer state when the user is unchanged', () => {
      useChatStore.setState({ currentUserId: 'user-1' })

      useChatStore.getState().setCurrentUser('user-1')

      expect(mockLayoutState.resetComposerState).not.toHaveBeenCalled()
    })
  })

  describe('getUserConversations', () => {
    test('returns empty array when no user', () => {
      useChatStore.setState({
        currentUserId: null,
        conversations: [
          {
            id: 'conv-1',
            userId: 'user-1',
            title: 'Conv',
            messages: [],
            createdAt: new Date(),
            updatedAt: new Date(),
          },
        ],
      })

      expect(useChatStore.getState().getUserConversations()).toEqual([])
    })

    test('returns only conversations for current user', () => {
      const conv1: Conversation = {
        id: 'conv-1',
        userId: 'user-1',
        title: 'User 1 Conv',
        messages: [],
        createdAt: new Date(),
        updatedAt: new Date(),
      }
      const conv2: Conversation = {
        id: 'conv-2',
        userId: 'user-2',
        title: 'User 2 Conv',
        messages: [],
        createdAt: new Date(),
        updatedAt: new Date(),
      }

      useChatStore.setState({
        currentUserId: 'user-1',
        conversations: [conv1, conv2],
      })

      const result = useChatStore.getState().getUserConversations()

      expect(result).toHaveLength(1)
      expect(result[0].id).toBe('conv-1')
    })
  })

  describe('createConversation', () => {
    test('creates new conversation for current user', () => {
      useChatStore.setState({ currentUserId: 'user-1' })

      const conv = useChatStore.getState().createConversation()

      expect(conv.userId).toBe('user-1')
      expect(conv.title).toBe('')
      expect(conv.messages).toEqual([])
      expect(useChatStore.getState().currentConversation).toEqual(conv)
      expect(useChatStore.getState().conversations).toContainEqual(conv)
    })

    test('enables all available data sources for new conversations by default', () => {
      useChatStore.setState({ currentUserId: 'user-1' })

      const conv = useChatStore.getState().createConversation()

      expect(conv.enabledDataSourceIds).toEqual(['web_search', 'knowledge_base'])
      expect(mockLayoutState.setEnabledDataSources).toHaveBeenCalledWith([
        'web_search',
        'knowledge_base',
      ])
    })

    test('throws when no user is authenticated', () => {
      expect(() => useChatStore.getState().createConversation()).toThrow(
        'Cannot create conversation without authenticated user'
      )
    })

    test('clears thinking state on new conversation', () => {
      useChatStore.setState({
        currentUserId: 'user-1',
        thinkingSteps: [
          {
            id: '1',
            userMessageId: 'msg-1',
            category: 'agents',
            functionName: 'test',
            displayName: 'Test',
            content: '',
            timestamp: new Date(),
            isComplete: false,
          },
        ],
        reportContent: 'Old report',
      })

      useChatStore.getState().createConversation()

      const state = useChatStore.getState()
      expect(state.thinkingSteps).toEqual([])
      expect(state.reportContent).toBe('')
    })
  })

  describe('ensureSession', () => {
    test('returns existing conversation ID', () => {
      const conv: Conversation = {
        id: 'existing-conv',
        userId: 'user-1',
        title: 'Existing',
        messages: [],
        createdAt: new Date(),
        updatedAt: new Date(),
      }
      useChatStore.setState({ currentUserId: 'user-1', currentConversation: conv })

      const result = useChatStore.getState().ensureSession()

      expect(result).toBe('existing-conv')
    })

    test('creates new conversation if none exists', () => {
      useChatStore.setState({ currentUserId: 'user-1', currentConversation: null })

      const result = useChatStore.getState().ensureSession()

      expect(result).toBeDefined()
      expect(useChatStore.getState().currentConversation).not.toBeNull()
    })

    test('returns undefined when no user', () => {
      useChatStore.setState({ currentUserId: null, currentConversation: null })

      const result = useChatStore.getState().ensureSession()

      expect(result).toBeUndefined()
    })
  })

  describe('upload-only session cleanup', () => {
    const uploadOnlyConv = (id: string): Conversation => ({
      id,
      userId: 'user-1',
      title: 'New chat',
      messages: [
        {
          id: 'banner-1',
          role: 'assistant',
          content: '',
          timestamp: new Date(),
          messageType: 'file_upload_status',
          fileUploadStatusData: { type: 'uploaded', fileCount: 2, jobId: 'job-1' },
        },
      ],
      createdAt: new Date(),
      updatedAt: new Date(),
    })

    test('startNewSessionDraft removes upload-only session and discards documents', () => {
      const conv = uploadOnlyConv('upload-only-1')
      useChatStore.setState({
        currentUserId: 'user-1',
        currentConversation: conv,
        conversations: [conv],
      })

      useChatStore.getState().startNewSessionDraft()

      expect(mockDiscardSessionResources).toHaveBeenCalledWith('upload-only-1')
      expect(useChatStore.getState().conversations.some((c) => c.id === 'upload-only-1')).toBe(
        false
      )
      expect(useChatStore.getState().currentConversation).toBeNull()
    })

    test('startNewSessionDraft keeps session after user has chatted', () => {
      const conv: Conversation = {
        ...uploadOnlyConv('with-user'),
        messages: [
          {
            id: 'u1',
            role: 'user',
            content: 'hello',
            timestamp: new Date(),
            messageType: 'user',
          },
        ],
      }
      useChatStore.setState({
        currentUserId: 'user-1',
        currentConversation: conv,
        conversations: [conv],
      })

      useChatStore.getState().startNewSessionDraft()

      expect(mockDiscardSessionResources).not.toHaveBeenCalled()
      expect(useChatStore.getState().conversations.some((c) => c.id === 'with-user')).toBe(true)
    })

    test('startNewSessionDraft clears stale shallow streaming state', () => {
      const conv: Conversation = {
        ...uploadOnlyConv('stale-thinking'),
        messages: [
          {
            id: 'u1',
            role: 'user',
            content: 'hello',
            timestamp: new Date(),
            messageType: 'user',
          },
        ],
      }
      useChatStore.setState({
        currentUserId: 'user-1',
        currentConversation: conv,
        conversations: [conv],
        isStreaming: true,
        isLoading: true,
        currentUserMessageId: 'u1',
        currentStatus: 'thinking',
      })

      useChatStore.getState().startNewSessionDraft()

      expect(useChatStore.getState().isStreaming).toBe(false)
      expect(useChatStore.getState().isLoading).toBe(false)
      expect(useChatStore.getState().currentUserMessageId).toBeNull()
      expect(useChatStore.getState().currentStatus).toBeNull()
    })

    test('selectConversation removes prior upload-only session when switching away', () => {
      const uploadOnly = uploadOnlyConv('u-only')
      const other: Conversation = {
        id: 'other',
        userId: 'user-1',
        title: 'Other',
        messages: [
          {
            id: 'm1',
            role: 'user',
            content: 'hi',
            timestamp: new Date(),
            messageType: 'user',
          },
        ],
        createdAt: new Date(),
        updatedAt: new Date(),
      }
      useChatStore.setState({
        currentUserId: 'user-1',
        currentConversation: uploadOnly,
        conversations: [uploadOnly, other],
      })

      useChatStore.getState().selectConversation('other')

      expect(mockDiscardSessionResources).toHaveBeenCalledWith('u-only')
      expect(useChatStore.getState().conversations.some((c) => c.id === 'u-only')).toBe(false)
      expect(useChatStore.getState().currentConversation?.id).toBe('other')
    })

    test('selectConversation does not remove upload-only session while files are uploading', async () => {
      const { useDocumentsStore } = await import('@/features/documents/store')
      const uploadOnly = uploadOnlyConv('u-busy')
      const other: Conversation = {
        id: 'other-2',
        userId: 'user-1',
        title: 'Other',
        messages: [],
        createdAt: new Date(),
        updatedAt: new Date(),
      }
      useDocumentsStore.setState({
        trackedFiles: [
          {
            id: 'tf-1',
            file: new File(['x'], 'x.txt'),
            fileName: 'x.txt',
            fileSize: 1,
            status: 'uploading',
            progress: 0,
            collectionName: 'u-busy',
            uploadedAt: new Date().toISOString(),
          },
        ],
      })
      useChatStore.setState({
        currentUserId: 'user-1',
        currentConversation: uploadOnly,
        conversations: [uploadOnly, other],
      })

      useChatStore.getState().selectConversation('other-2')

      expect(mockDiscardSessionResources).not.toHaveBeenCalled()
      expect(useChatStore.getState().conversations.some((c) => c.id === 'u-busy')).toBe(true)

      useDocumentsStore.setState({ trackedFiles: [] })
    })
  })

  describe('selectConversation', () => {
    test('selects conversation owned by current user', () => {
      const conv: Conversation = {
        id: 'conv-1',
        userId: 'user-1',
        title: 'Conv 1',
        messages: [],
        createdAt: new Date(),
        updatedAt: new Date(),
      }
      useChatStore.setState({
        currentUserId: 'user-1',
        conversations: [conv],
        currentConversation: null,
      })

      useChatStore.getState().selectConversation('conv-1')

      expect(useChatStore.getState().currentConversation).toEqual(conv)
    })

    test('does not select conversation owned by different user', () => {
      const conv: Conversation = {
        id: 'conv-1',
        userId: 'user-2',
        title: 'Conv 1',
        messages: [],
        createdAt: new Date(),
        updatedAt: new Date(),
      }
      useChatStore.setState({
        currentUserId: 'user-1',
        conversations: [conv],
        currentConversation: null,
      })

      useChatStore.getState().selectConversation('conv-1')

      expect(useChatStore.getState().currentConversation).toBeNull()
    })

    test('clears thinking state on selection', () => {
      const conv: Conversation = {
        id: 'conv-1',
        userId: 'user-1',
        title: 'Conv',
        messages: [],
        createdAt: new Date(),
        updatedAt: new Date(),
      }
      useChatStore.setState({
        currentUserId: 'user-1',
        conversations: [conv],
        thinkingSteps: [
          {
            id: '1',
            userMessageId: 'msg-1',
            category: 'agents',
            functionName: 'test',
            displayName: 'Test',
            content: '',
            timestamp: new Date(),
            isComplete: false,
          },
        ],
        reportContent: 'Old',
      })

      useChatStore.getState().selectConversation('conv-1')

      expect(useChatStore.getState().thinkingSteps).toEqual([])
      expect(useChatStore.getState().reportContent).toBe('')
    })
  })

  describe('addUserMessage', () => {
    test('adds user message to current conversation', () => {
      const conv: Conversation = {
        id: 'conv-1',
        userId: 'user-1',
        title: '',
        messages: [],
        createdAt: new Date(),
        updatedAt: new Date(),
      }
      useChatStore.setState({
        currentUserId: 'user-1',
        currentConversation: conv,
        conversations: [conv],
      })

      const msg = useChatStore.getState().addUserMessage('Hello')

      expect(msg.role).toBe('user')
      expect(msg.content).toBe('Hello')
      expect(useChatStore.getState().currentConversation?.messages).toHaveLength(1)
    })

    test('captures selectedModel from metadata onto the message', () => {
      const conv: Conversation = {
        id: 'conv-1',
        userId: 'user-1',
        title: '',
        messages: [],
        createdAt: new Date(),
        updatedAt: new Date(),
      }
      useChatStore.setState({
        currentUserId: 'user-1',
        currentConversation: conv,
        conversations: [conv],
      })

      const msg = useChatStore.getState().addUserMessage('Hello', { selectedModel: 'gpt-5.4' })

      expect(msg.selectedModel).toBe('gpt-5.4')
    })

    test('updates title on first message', () => {
      const conv: Conversation = {
        id: 'conv-1',
        userId: 'user-1',
        title: '',
        messages: [],
        createdAt: new Date(),
        updatedAt: new Date(),
      }
      useChatStore.setState({
        currentUserId: 'user-1',
        currentConversation: conv,
        conversations: [conv],
      })

      useChatStore.getState().addUserMessage('What is the capital of France?')

      expect(useChatStore.getState().currentConversation?.title).toBe(
        'What is the capital of France?'
      )
    })

    test('updates title on first user message when file upload status messages exist', () => {
      const conv: Conversation = {
        id: 'conv-1',
        userId: 'user-1',
        title: '',
        messages: [
          {
            id: 'status-1',
            role: 'assistant',
            content: '',
            timestamp: new Date(),
            messageType: 'file_upload_status',
            fileUploadStatusData: { type: 'uploaded', fileCount: 1, jobId: 'job-1' },
          },
        ],
        createdAt: new Date(),
        updatedAt: new Date(),
      }
      useChatStore.setState({
        currentUserId: 'user-1',
        currentConversation: conv,
        conversations: [conv],
      })

      useChatStore.getState().addUserMessage('Summarize my document')

      expect(useChatStore.getState().currentConversation?.title).toBe('Summarize my document')
    })

    test('truncates long titles to 50 characters', () => {
      const conv: Conversation = {
        id: 'conv-1',
        userId: 'user-1',
        title: '',
        messages: [],
        createdAt: new Date(),
        updatedAt: new Date(),
      }
      useChatStore.setState({
        currentUserId: 'user-1',
        currentConversation: conv,
        conversations: [conv],
      })

      const longMessage = 'A'.repeat(100)
      useChatStore.getState().addUserMessage(longMessage)

      expect(useChatStore.getState().currentConversation?.title).toBe('A'.repeat(50) + '...')
    })

    test('creates conversation if none exists', () => {
      mockLayoutState.enabledDataSourceIds = ['web_search', 'knowledge_base']
      useChatStore.setState({
        currentUserId: 'user-1',
        currentConversation: null,
        conversations: [],
      })

      useChatStore.getState().addUserMessage('Hello')

      expect(useChatStore.getState().currentConversation).not.toBeNull()
      expect(useChatStore.getState().conversations).toHaveLength(1)
      expect(useChatStore.getState().currentConversation?.enabledDataSourceIds).toEqual([
        'web_search',
        'knowledge_base',
      ])
    })

    test('throws when no user authenticated', () => {
      useChatStore.setState({ currentUserId: null, currentConversation: null })

      expect(() => useChatStore.getState().addUserMessage('Hello')).toThrow(
        'Cannot create conversation without authenticated user'
      )
    })

    test('sets loading state and updates currentUserMessageId', () => {
      const conv: Conversation = {
        id: 'conv-1',
        userId: 'user-1',
        title: '',
        messages: [],
        createdAt: new Date(),
        updatedAt: new Date(),
      }
      useChatStore.setState({
        currentUserId: 'user-1',
        currentConversation: conv,
        conversations: [conv],
        currentUserMessageId: 'old-msg-id',
        thinkingSteps: [
          {
            id: '1',
            userMessageId: 'old-msg-id',
            category: 'agents',
            functionName: 'test',
            displayName: 'Test',
            content: '',
            timestamp: new Date(),
            isComplete: false,
          },
        ],
      })

      const message = useChatStore.getState().addUserMessage('Hello')

      expect(useChatStore.getState().isLoading).toBe(true)
      // New behavior: thinking steps are preserved (associated with previous message)
      expect(useChatStore.getState().thinkingSteps).toHaveLength(1)
      // currentUserMessageId is updated to the new message
      expect(useChatStore.getState().currentUserMessageId).toBe(message.id)
    })
  })

  describe('assistant message streaming', () => {
    const setupConversation = () => {
      const conv: Conversation = {
        id: 'conv-1',
        userId: 'user-1',
        title: 'Test',
        messages: [],
        createdAt: new Date(),
        updatedAt: new Date(),
      }
      useChatStore.setState({
        currentUserId: 'user-1',
        currentConversation: conv,
        conversations: [conv],
      })
      return conv
    }

    test('startAssistantMessage creates streaming message', () => {
      setupConversation()

      const msg = useChatStore.getState().startAssistantMessage()

      expect(msg.role).toBe('assistant')
      expect(msg.content).toBe('')
      expect(msg.isStreaming).toBe(true)
      expect(useChatStore.getState().isStreaming).toBe(true)
      expect(useChatStore.getState().isLoading).toBe(false)
    })

    test('startAssistantMessage throws when no conversation', () => {
      useChatStore.setState({ currentConversation: null })

      expect(() => useChatStore.getState().startAssistantMessage()).toThrow(
        'No active conversation'
      )
    })

    test('appendToAssistantMessage appends to streaming message', () => {
      setupConversation()
      useChatStore.getState().startAssistantMessage()

      useChatStore.getState().appendToAssistantMessage('Hello ')
      useChatStore.getState().appendToAssistantMessage('world!')

      const messages = useChatStore.getState().currentConversation?.messages
      expect(messages?.[0].content).toBe('Hello world!')
    })

    test('appendToAssistantMessage does nothing if no streaming message', () => {
      setupConversation()

      useChatStore.getState().appendToAssistantMessage('Hello')

      expect(useChatStore.getState().currentConversation?.messages).toHaveLength(0)
    })

    test('completeAssistantMessage marks message as complete', () => {
      setupConversation()
      useChatStore.getState().startAssistantMessage()
      useChatStore.getState().appendToAssistantMessage('Response')

      useChatStore.getState().completeAssistantMessage()

      const messages = useChatStore.getState().currentConversation?.messages
      expect(messages?.[0].isStreaming).toBe(false)
      expect(useChatStore.getState().isStreaming).toBe(false)
    })
  })

  describe('loading state', () => {
    test('setLoading sets loading state', () => {
      useChatStore.getState().setLoading(true)
      expect(useChatStore.getState().isLoading).toBe(true)

      useChatStore.getState().setLoading(false)
      expect(useChatStore.getState().isLoading).toBe(false)
    })

    test('setStreaming sets streaming state', () => {
      useChatStore.getState().setStreaming(true)
      expect(useChatStore.getState().isStreaming).toBe(true)

      useChatStore.getState().setStreaming(false)
      expect(useChatStore.getState().isStreaming).toBe(false)
    })
  })

  describe('conversation management', () => {
    test('deleteConversation removes conversation', () => {
      const conv: Conversation = {
        id: 'conv-1',
        userId: 'user-1',
        title: 'Test',
        messages: [],
        createdAt: new Date(),
        updatedAt: new Date(),
      }
      useChatStore.setState({ currentConversation: conv, conversations: [conv] })

      useChatStore.getState().deleteConversation('conv-1')

      expect(useChatStore.getState().conversations).toHaveLength(0)
      expect(useChatStore.getState().currentConversation).toBeNull()
    })

    test('deleteConversation keeps current if different', () => {
      const conv1: Conversation = {
        id: 'conv-1',
        userId: 'user-1',
        title: 'Test 1',
        messages: [],
        createdAt: new Date(),
        updatedAt: new Date(),
      }
      const conv2: Conversation = {
        id: 'conv-2',
        userId: 'user-1',
        title: 'Test 2',
        messages: [],
        createdAt: new Date(),
        updatedAt: new Date(),
      }
      useChatStore.setState({ currentConversation: conv1, conversations: [conv1, conv2] })

      useChatStore.getState().deleteConversation('conv-2')

      expect(useChatStore.getState().currentConversation).toEqual(conv1)
    })

    test('deleteConversation removes session from localStorage', async () => {
      // Create conversations and wait for persist
      const conv1: Conversation = {
        id: 'conv-persist-1',
        userId: 'user-1',
        title: 'Session to Delete',
        messages: [],
        createdAt: new Date('2024-01-01'),
        updatedAt: new Date('2024-01-01'),
      }
      const conv2: Conversation = {
        id: 'conv-persist-2',
        userId: 'user-1',
        title: 'Session to Keep',
        messages: [],
        createdAt: new Date('2024-01-02'),
        updatedAt: new Date('2024-01-02'),
      }
      useChatStore.setState({
        currentUserId: 'user-1',
        currentConversation: conv1,
        conversations: [conv1, conv2],
      })

      // Wait for Zustand persist to sync to localStorage
      await vi.waitFor(() => {
        const stored = localStorage.getItem(STORAGE_KEY)
        expect(stored).not.toBeNull()
        const parsed = JSON.parse(stored!)
        expect(parsed.state.conversations).toHaveLength(2)
      })

      // Verify initial localStorage state
      const beforeDelete = JSON.parse(localStorage.getItem(STORAGE_KEY)!)
      expect(beforeDelete.state.conversations.map((c: Conversation) => c.id)).toContain(
        'conv-persist-1'
      )
      expect(beforeDelete.state.conversations.map((c: Conversation) => c.id)).toContain(
        'conv-persist-2'
      )

      // Delete the first conversation
      useChatStore.getState().deleteConversation('conv-persist-1')

      // Wait for Zustand persist to sync the deletion to localStorage
      await vi.waitFor(() => {
        const stored = localStorage.getItem(STORAGE_KEY)
        const parsed = JSON.parse(stored!)
        expect(parsed.state.conversations).toHaveLength(1)
      })

      // Verify localStorage was updated correctly
      const afterDelete = JSON.parse(localStorage.getItem(STORAGE_KEY)!)

      // The deleted session should NOT be in localStorage
      expect(afterDelete.state.conversations.map((c: Conversation) => c.id)).not.toContain(
        'conv-persist-1'
      )

      // The other session should still be in localStorage
      expect(afterDelete.state.conversations.map((c: Conversation) => c.id)).toContain(
        'conv-persist-2'
      )

      // currentConversation should be cleared since we deleted the current one
      expect(afterDelete.state.currentConversation).toBeNull()
    })

    test('deleteConversation updates currentConversation in localStorage when deleting current', async () => {
      const conv: Conversation = {
        id: 'conv-current',
        userId: 'user-1',
        title: 'Current Session',
        messages: [],
        createdAt: new Date(),
        updatedAt: new Date(),
      }
      useChatStore.setState({
        currentUserId: 'user-1',
        currentConversation: conv,
        conversations: [conv],
      })

      // Wait for initial persist (currentConversation stored as ID string)
      await vi.waitFor(() => {
        const stored = localStorage.getItem(STORAGE_KEY)
        expect(stored).not.toBeNull()
        const parsed = JSON.parse(stored!)
        expect(parsed.state.currentConversation).toBe('conv-current')
      })

      // Delete the current conversation
      useChatStore.getState().deleteConversation('conv-current')

      // Wait for persist to sync
      await vi.waitFor(() => {
        const stored = localStorage.getItem(STORAGE_KEY)
        const parsed = JSON.parse(stored!)
        expect(parsed.state.conversations).toHaveLength(0)
      })

      // Verify currentConversation is cleared in localStorage
      const stored = JSON.parse(localStorage.getItem(STORAGE_KEY)!)
      expect(stored.state.currentConversation).toBeNull()
      expect(stored.state.conversations).toHaveLength(0)
    })

    test('refreshDeepResearchSessionStatuses marks unavailable completed reports expired without deleting sessions', async () => {
      const expiredConversation: Conversation = {
        id: 'conv-expired',
        userId: 'user-1',
        title: 'Expired Report',
        messages: [
          {
            id: 'msg-expired',
            role: 'assistant',
            content: '',
            timestamp: new Date(),
            messageType: 'agent_response',
            deepResearchJobId: 'job-expired',
            deepResearchJobStatus: 'success',
            showViewReport: true,
          },
          {
            id: 'success-banner',
            role: 'assistant',
            content: '',
            timestamp: new Date(),
            messageType: 'deep_research_banner',
            deepResearchBannerData: { bannerType: 'success', jobId: 'job-expired' },
          },
        ],
        createdAt: new Date(),
        updatedAt: new Date(),
      }
      const reachableConversation: Conversation = {
        id: 'conv-reachable',
        userId: 'user-1',
        title: 'Reachable Report',
        messages: [
          {
            id: 'msg-reachable',
            role: 'assistant',
            content: '',
            timestamp: new Date(),
            messageType: 'agent_response',
            deepResearchJobId: 'job-reachable',
            deepResearchJobStatus: 'success',
          },
        ],
        createdAt: new Date(),
        updatedAt: new Date(),
      }
      const otherUserConversation: Conversation = {
        id: 'conv-other-user',
        userId: 'user-2',
        title: 'Other User Report',
        messages: [
          {
            id: 'msg-other-user',
            role: 'assistant',
            content: '',
            timestamp: new Date(),
            messageType: 'agent_response',
            deepResearchJobId: 'job-other-user',
            deepResearchJobStatus: 'success',
          },
        ],
        createdAt: new Date(),
        updatedAt: new Date(),
      }

      mockDeepResearchApi.getJobStatus.mockImplementation(async (jobId: string) => {
        if (jobId === 'job-expired') {
          throw new Error('Failed to get job status: 404')
        }
        return { job_id: jobId, status: 'success', error: null }
      })

      useChatStore.setState({
        currentUserId: 'user-1',
        currentConversation: expiredConversation,
        conversations: [expiredConversation, reachableConversation, otherUserConversation],
        deepResearchJobId: 'job-expired',
        deepResearchOwnerConversationId: 'conv-expired',
        activeDeepResearchMessageId: 'msg-expired',
        reportContent: 'stale report',
      })

      await useChatStore.getState().refreshDeepResearchSessionStatuses()

      expect(mockDeepResearchApi.getJobStatus).toHaveBeenCalledTimes(2)
      expect(mockDeepResearchApi.getJobStatus).toHaveBeenCalledWith('job-expired')
      expect(mockDeepResearchApi.getJobStatus).toHaveBeenCalledWith('job-reachable')
      expect(mockDeepResearchApi.getJobStatus).not.toHaveBeenCalledWith('job-other-user')

      const state = useChatStore.getState()
      expect(state.conversations.map((c) => c.id)).toEqual([
        'conv-expired',
        'conv-reachable',
        'conv-other-user',
      ])
      expect(state.currentConversation?.id).toBe('conv-expired')
      expect(state.deepResearchJobId).toBeNull()
      expect(state.deepResearchOwnerConversationId).toBeNull()
      expect(state.activeDeepResearchMessageId).toBeNull()
      expect(state.reportContent).toBe('')

      const expiredMessage = state.conversations
        .find((c) => c.id === 'conv-expired')
        ?.messages.find((m) => m.id === 'msg-expired')
      expect(expiredMessage?.deepResearchJobStatus).toBe('failure')
      expect(expiredMessage?.isDeepResearchActive).toBe(false)
      expect(expiredMessage?.showViewReport).toBe(false)
      expect(expiredMessage?.deepResearchReportExpired).toBe(true)

      const expiredConversationAfter = state.conversations.find((c) => c.id === 'conv-expired')
      const reportBanners =
        expiredConversationAfter?.messages.filter(
          (m) =>
            m.messageType === 'deep_research_banner' &&
            m.deepResearchBannerData?.jobId === 'job-expired'
        ) ?? []
      expect(reportBanners).toHaveLength(1)
      expect(reportBanners[0].deepResearchBannerData?.bannerType).toBe('expired')
    })

    test('refreshDeepResearchSessionStatuses keeps old chat-only sessions without backend checks', async () => {
      const oldChatConversation: Conversation = {
        id: 'conv-chat',
        userId: 'user-1',
        title: 'Old Chat Session',
        messages: [
          {
            id: 'msg-user',
            role: 'user',
            content: 'hello',
            timestamp: new Date(),
            messageType: 'user',
          },
          {
            id: 'msg-assistant',
            role: 'assistant',
            content: 'hi',
            timestamp: new Date(),
            messageType: 'agent_response',
          },
        ],
        createdAt: new Date('2026-01-01T00:00:00Z'),
        updatedAt: new Date('2026-01-01T00:00:00Z'),
      }

      useChatStore.setState({
        currentUserId: 'user-1',
        currentConversation: oldChatConversation,
        conversations: [oldChatConversation],
      })

      await useChatStore.getState().refreshDeepResearchSessionStatuses()

      expect(mockDeepResearchApi.getJobStatus).not.toHaveBeenCalled()
      expect(useChatStore.getState().conversations.map((c) => c.id)).toEqual(['conv-chat'])
      expect(useChatStore.getState().currentConversation?.id).toBe('conv-chat')
    })

    test('refreshDeepResearchSessionStatuses unlocks missing active jobs without marking them as expired reports', async () => {
      const runningConversation: Conversation = {
        id: 'conv-running',
        userId: 'user-1',
        title: 'Running Report',
        messages: [
          {
            id: 'msg-running',
            role: 'assistant',
            content: '',
            timestamp: new Date(),
            messageType: 'agent_response',
            deepResearchJobId: 'job-running',
            deepResearchJobStatus: 'running',
            isDeepResearchActive: true,
          },
        ],
        createdAt: new Date(),
        updatedAt: new Date(),
      }

      mockDeepResearchApi.getJobStatus.mockRejectedValue(new Error('Failed to get job status: 404'))

      useChatStore.setState({
        currentUserId: 'user-1',
        currentConversation: runningConversation,
        conversations: [runningConversation],
        deepResearchJobId: 'job-running',
        deepResearchOwnerConversationId: 'conv-running',
        activeDeepResearchMessageId: 'msg-running',
        isDeepResearchStreaming: true,
      })

      await useChatStore.getState().refreshDeepResearchSessionStatuses()

      const state = useChatStore.getState()
      const runningMessage = state.conversations[0].messages[0]
      expect(state.conversations.map((c) => c.id)).toEqual(['conv-running'])
      expect(runningMessage.deepResearchJobStatus).toBe('failure')
      expect(runningMessage.isDeepResearchActive).toBe(false)
      expect(runningMessage.deepResearchReportExpired).toBeFalsy()
      expect(state.deepResearchJobId).toBeNull()
      expect(state.isDeepResearchStreaming).toBe(false)
      expect(state.isSessionBusy('conv-running')).toBe(false)
    })

    test('updateConversationTitle updates title', () => {
      const conv: Conversation = {
        id: 'conv-1',
        userId: 'user-1',
        title: 'Old Title',
        messages: [],
        createdAt: new Date(),
        updatedAt: new Date(),
      }
      useChatStore.setState({ currentConversation: conv, conversations: [conv] })

      useChatStore.getState().updateConversationTitle('conv-1', 'New Title')

      expect(useChatStore.getState().currentConversation?.title).toBe('New Title')
      expect(useChatStore.getState().conversations[0].title).toBe('New Title')
    })
  })

  describe('thinking steps', () => {
    // Helper to set up a user message context for thinking steps tests
    const setupUserMessageContext = () => {
      useChatStore.getState().setCurrentUser('test-user')
      const message = useChatStore.getState().addUserMessage('Test message')
      return message.id
    }

    test('addThinkingStep adds step and returns ID', () => {
      const userMessageId = setupUserMessageContext()

      const stepId = useChatStore.getState().addThinkingStep({
        category: 'agents',
        functionName: 'intent_classifier',
        displayName: 'Intent Classifier',
        content: 'Initial thought',
        isComplete: false,
      })

      expect(stepId).toBeDefined()
      const steps = useChatStore.getState().thinkingSteps
      expect(steps).toHaveLength(1)
      expect(steps[0].category).toBe('agents')
      expect(steps[0].functionName).toBe('intent_classifier')
      expect(steps[0].displayName).toBe('Intent Classifier')
      expect(steps[0].content).toBe('Initial thought')
      expect(steps[0].isComplete).toBe(false)
      expect(steps[0].userMessageId).toBe(userMessageId)
      expect(useChatStore.getState().activeThinkingStepId).toBe(stepId)
    })

    test('addThinkingStep returns empty string without currentUserMessageId', () => {
      // Don't set up user message context
      const stepId = useChatStore.getState().addThinkingStep({
        category: 'agents',
        functionName: 'test',
        displayName: 'Test',
        content: '',
        isComplete: false,
      })

      expect(stepId).toBe('')
      expect(useChatStore.getState().thinkingSteps).toHaveLength(0)
    })

    test('appendToThinkingStep appends content', () => {
      setupUserMessageContext()

      const stepId = useChatStore.getState().addThinkingStep({
        category: 'agents',
        functionName: 'test_agent',
        displayName: 'Test Agent',
        content: 'Hello ',
        isComplete: false,
      })

      useChatStore.getState().appendToThinkingStep(stepId, 'world!')

      expect(useChatStore.getState().thinkingSteps[0].content).toBe('Hello world!')
    })

    test('completeThinkingStep marks step complete', () => {
      setupUserMessageContext()

      const stepId = useChatStore.getState().addThinkingStep({
        category: 'tasks',
        functionName: '<workflow>',
        displayName: 'Workflow',
        content: '',
        isComplete: false,
      })

      useChatStore.getState().completeThinkingStep(stepId)

      expect(useChatStore.getState().thinkingSteps[0].isComplete).toBe(true)
      expect(useChatStore.getState().activeThinkingStepId).toBeNull()
    })

    test('completeThinkingStep sets status success and completedAt', () => {
      setupUserMessageContext()

      const stepId = useChatStore.getState().addThinkingStep({
        category: 'tools',
        functionName: 'web_search_tool',
        displayName: 'Searching the web',
        content: '',
        isComplete: false,
        status: 'running',
      })

      useChatStore.getState().completeThinkingStep(stepId)

      const step = useChatStore.getState().thinkingSteps[0]
      expect(step.status).toBe('success')
      expect(step.completedAt).toBeInstanceOf(Date)
    })

    test('completeThinkingStep preserves an existing error status', () => {
      setupUserMessageContext()

      const stepId = useChatStore.getState().addThinkingStep({
        category: 'tools',
        functionName: 'web_search_tool',
        displayName: 'Searching the web',
        content: '',
        isComplete: false,
        status: 'error',
      })

      useChatStore.getState().completeThinkingStep(stepId)

      expect(useChatStore.getState().thinkingSteps[0].status).toBe('error')
    })

    test('failThinkingStep marks step failed with error status and completedAt', () => {
      setupUserMessageContext()

      const stepId = useChatStore.getState().addThinkingStep({
        category: 'tools',
        functionName: 'web_search_tool',
        displayName: 'Searching the web',
        content: '',
        isComplete: false,
        status: 'running',
      })

      useChatStore.getState().failThinkingStep(stepId)

      const step = useChatStore.getState().thinkingSteps[0]
      expect(step.isComplete).toBe(true)
      expect(step.status).toBe('error')
      expect(step.completedAt).toBeInstanceOf(Date)
      expect(useChatStore.getState().activeThinkingStepId).toBeNull()
    })

    test('failThinkingStep persists error status to the owning message', () => {
      const userMessageId = setupUserMessageContext()

      const stepId = useChatStore.getState().addThinkingStep({
        category: 'tools',
        functionName: 'web_search_tool',
        displayName: 'Searching the web',
        content: '',
        isComplete: false,
        status: 'running',
      })

      useChatStore.getState().failThinkingStep(stepId)

      const message = useChatStore
        .getState()
        .currentConversation?.messages.find((m) => m.id === userMessageId)
      expect(message?.thinkingSteps?.[0].status).toBe('error')
    })

    test('clearThinkingSteps clears all steps', () => {
      setupUserMessageContext()

      useChatStore.getState().addThinkingStep({
        category: 'agents',
        functionName: 'agent1',
        displayName: 'Agent 1',
        content: '',
        isComplete: false,
      })
      useChatStore.getState().addThinkingStep({
        category: 'tools',
        functionName: 'web_search_tool',
        displayName: 'Web Search Tool',
        content: '',
        isComplete: false,
      })

      useChatStore.getState().clearThinkingSteps()

      expect(useChatStore.getState().thinkingSteps).toEqual([])
      expect(useChatStore.getState().activeThinkingStepId).toBeNull()
    })

    test('updateThinkingStepByFunctionName updates step', () => {
      setupUserMessageContext()

      useChatStore.getState().addThinkingStep({
        category: 'tools',
        functionName: 'web_search_tool',
        displayName: 'Web Search Tool',
        content: 'Searching...',
        isComplete: false,
      })

      useChatStore
        .getState()
        .updateThinkingStepByFunctionName(
          'web_search_tool',
          'Search complete: found 5 results',
          true
        )

      const step = useChatStore.getState().thinkingSteps[0]
      expect(step.content).toBe('Search complete: found 5 results')
      expect(step.isComplete).toBe(true)
    })

    test('findThinkingStepByFunctionName finds existing step', () => {
      setupUserMessageContext()

      useChatStore.getState().addThinkingStep({
        category: 'agents',
        functionName: 'intent_classifier',
        displayName: 'Intent Classifier',
        content: 'Classifying...',
        isComplete: false,
      })

      const found = useChatStore.getState().findThinkingStepByFunctionName('intent_classifier')

      expect(found).toBeDefined()
      expect(found?.functionName).toBe('intent_classifier')
    })

    test('findThinkingStepByFunctionName returns undefined for non-existent step', () => {
      const found = useChatStore.getState().findThinkingStepByFunctionName('non_existent')

      expect(found).toBeUndefined()
    })

    test('getThinkingStepsForMessage filters by userMessageId', () => {
      useChatStore.getState().setCurrentUser('test-user')

      // Add first user message and its thinking step
      const message1 = useChatStore.getState().addUserMessage('Message 1')
      useChatStore.getState().addThinkingStep({
        category: 'agents',
        functionName: 'agent1',
        displayName: 'Agent 1',
        content: 'Step for message 1',
        isComplete: false,
      })

      // Add second user message and its thinking step
      const message2 = useChatStore.getState().addUserMessage('Message 2')
      useChatStore.getState().addThinkingStep({
        category: 'tools',
        functionName: 'tool1',
        displayName: 'Tool 1',
        content: 'Step for message 2',
        isComplete: false,
      })

      // Get steps for each message
      const stepsForMessage1 = useChatStore.getState().getThinkingStepsForMessage(message1.id)
      const stepsForMessage2 = useChatStore.getState().getThinkingStepsForMessage(message2.id)

      expect(stepsForMessage1).toHaveLength(1)
      expect(stepsForMessage1[0].functionName).toBe('agent1')
      expect(stepsForMessage2).toHaveLength(1)
      expect(stepsForMessage2[0].functionName).toBe('tool1')
    })

    test('getThinkingStepsForMessage includes deep research steps so the inline trace mirrors the live run', () => {
      useChatStore.getState().setCurrentUser('test-user')

      const message = useChatStore.getState().addUserMessage('Test message')

      useChatStore.getState().addThinkingStep({
        category: 'agents',
        functionName: 'websocket_agent',
        displayName: 'WebSocket Agent',
        content: 'WebSocket step',
        isComplete: false,
        isDeepResearch: false,
      })

      useChatStore.getState().addThinkingStep({
        category: 'agents',
        functionName: 'deep_research_agent',
        displayName: 'Deep Research Agent',
        content: 'Deep research step',
        isComplete: false,
        isDeepResearch: true,
      })

      const ephemeral = useChatStore.getState().getThinkingStepsForMessage(message.id)
      const persisted =
        useChatStore
          .getState()
          .currentConversation?.messages.find((m) => m.id === message.id)?.thinkingSteps ?? []

      expect(ephemeral.map((s) => s.functionName)).toEqual(['websocket_agent', 'deep_research_agent'])
      expect(persisted.map((s) => s.functionName)).toEqual(['websocket_agent', 'deep_research_agent'])
    })
  })

  describe('report content', () => {
    test('setReportContent sets content', () => {
      useChatStore.getState().setReportContent('# Report\n\nContent here')

      expect(useChatStore.getState().reportContent).toBe('# Report\n\nContent here')
    })

    test('clearReportContent clears content', () => {
      useChatStore.setState({ reportContent: 'Some content' })

      useChatStore.getState().clearReportContent()

      expect(useChatStore.getState().reportContent).toBe('')
    })
  })

  describe('status and prompts', () => {
    const setupConversation = () => {
      const conv: Conversation = {
        id: 'conv-1',
        userId: 'user-1',
        title: 'Test',
        messages: [],
        createdAt: new Date(),
        updatedAt: new Date(),
      }
      useChatStore.setState({
        currentUserId: 'user-1',
        currentConversation: conv,
        conversations: [conv],
      })
      return conv
    }

    test('setCurrentStatus sets status', () => {
      useChatStore.getState().setCurrentStatus('searching')

      expect(useChatStore.getState().currentStatus).toBe('searching')
    })

    test('addStatusCard adds status message', () => {
      setupConversation()

      useChatStore.getState().addStatusCard('searching', 'Searching documents...')

      const messages = useChatStore.getState().currentConversation?.messages
      expect(messages).toHaveLength(1)
      expect(messages?.[0].messageType).toBe('status')
      expect(messages?.[0].statusType).toBe('searching')
      expect(messages?.[0].content).toBe('Searching documents...')
    })

    test('addAgentPrompt adds prompt message', () => {
      setupConversation()

      useChatStore
        .getState()
        .addAgentPrompt('choice', 'Select an option', ['A', 'B', 'C'], 'Choose one')

      const messages = useChatStore.getState().currentConversation?.messages
      expect(messages).toHaveLength(1)
      expect(messages?.[0].messageType).toBe('prompt')
      expect(messages?.[0].promptType).toBe('choice')
      expect(messages?.[0].promptOptions).toEqual(['A', 'B', 'C'])
      expect(messages?.[0].promptPlaceholder).toBe('Choose one')
      expect(messages?.[0].isPromptResponded).toBe(false)
      expect(useChatStore.getState().isStreaming).toBe(false)
    })

    test('respondToPrompt updates prompt message', () => {
      setupConversation()
      useChatStore.getState().addAgentPrompt('choice', 'Pick one', ['A', 'B'])
      const promptId = useChatStore.getState().currentConversation!.messages[0].id!

      useChatStore.getState().respondToPrompt(promptId, 'A')

      const msg = useChatStore.getState().currentConversation?.messages[0]
      expect(msg?.promptResponse).toBe('A')
      expect(msg?.isPromptResponded).toBe(true)
      expect(useChatStore.getState().isLoading).toBe(true)
    })
  })

  describe('agent responses and HITL', () => {
    const setupConversation = () => {
      const conv: Conversation = {
        id: 'conv-1',
        userId: 'user-1',
        title: 'Test',
        messages: [],
        createdAt: new Date(),
        updatedAt: new Date(),
      }
      useChatStore.setState({
        currentUserId: 'user-1',
        currentConversation: conv,
        conversations: [conv],
      })
      return conv
    }

    test('addAgentResponse adds response message', () => {
      setupConversation()

      useChatStore.getState().addAgentResponse('Here is your answer', true)

      const messages = useChatStore.getState().currentConversation?.messages
      expect(messages).toHaveLength(1)
      expect(messages?.[0].messageType).toBe('agent_response')
      expect(messages?.[0].content).toBe('Here is your answer')
      expect(messages?.[0].showViewReport).toBe(true)
    })

    test('setPendingInteraction sets interaction', () => {
      const interaction: PendingInteraction = {
        id: 'int-1',
        parentId: 'parent-1',
        inputType: 'text',
        text: 'Enter your name',
      }

      useChatStore.getState().setPendingInteraction(interaction)

      expect(useChatStore.getState().pendingInteraction).toEqual(interaction)
    })

    test('clearPendingInteraction clears interaction', () => {
      useChatStore.setState({
        pendingInteraction: { id: 'int-1', parentId: 'p1', inputType: 'text', text: 'Test' },
      })

      useChatStore.getState().clearPendingInteraction()

      expect(useChatStore.getState().pendingInteraction).toBeNull()
    })
  })

  describe('file cards', () => {
    const setupConversation = () => {
      const conv: Conversation = {
        id: 'conv-1',
        userId: 'user-1',
        title: 'Test',
        messages: [],
        createdAt: new Date(),
        updatedAt: new Date(),
      }
      useChatStore.setState({
        currentUserId: 'user-1',
        currentConversation: conv,
        conversations: [conv],
      })
      return conv
    }

    test('addFileCard adds file message', () => {
      setupConversation()

      const fileData: FileCardData = {
        fileName: 'document.pdf',
        fileSize: 1024,
        fileStatus: 'uploading',
        progress: 50,
      }

      useChatStore.getState().addFileCard(fileData)

      const messages = useChatStore.getState().currentConversation?.messages
      expect(messages).toHaveLength(1)
      expect(messages?.[0].messageType).toBe('file')
      expect(messages?.[0].fileData).toEqual(fileData)
    })

    test('updateFileCard updates file data', () => {
      setupConversation()
      useChatStore.getState().addFileCard({
        fileName: 'doc.pdf',
        fileSize: 1024,
        fileStatus: 'uploading',
        progress: 0,
      })
      const msgId = useChatStore.getState().currentConversation!.messages[0].id!

      useChatStore.getState().updateFileCard(msgId, { fileStatus: 'success', progress: 100 })

      const msg = useChatStore.getState().currentConversation?.messages[0]
      expect(msg?.fileData?.fileStatus).toBe('success')
      expect(msg?.fileData?.progress).toBe(100)
    })
  })

  describe('error cards', () => {
    const setupConversation = () => {
      const conv: Conversation = {
        id: 'conv-1',
        userId: 'user-1',
        title: 'Test',
        messages: [],
        createdAt: new Date(),
        updatedAt: new Date(),
      }
      useChatStore.setState({
        currentUserId: 'user-1',
        currentConversation: conv,
        conversations: [conv],
      })
      return conv
    }

    test('addErrorCard adds error message with defaults from registry', () => {
      setupConversation()

      useChatStore.getState().addErrorCard('connection.lost')

      const messages = useChatStore.getState().currentConversation?.messages
      expect(messages).toHaveLength(1)
      expect(messages?.[0].messageType).toBe('error')
      expect(messages?.[0].errorData?.errorCode).toBe('connection.lost')
    })

    test('addErrorCard uses custom message', () => {
      setupConversation()

      useChatStore
        .getState()
        .addErrorCard('connection.failed', 'Custom error message', 'Details here')

      const msg = useChatStore.getState().currentConversation?.messages[0]
      expect(msg?.content).toBe('Custom error message')
      expect(msg?.errorData?.errorDetails).toBe('Details here')
    })

    test('dismissErrorCard removes error message', () => {
      setupConversation()
      useChatStore.getState().addErrorCard('system.unknown')
      const msgId = useChatStore.getState().currentConversation!.messages[0].id!

      useChatStore.getState().dismissErrorCard(msgId)

      expect(useChatStore.getState().currentConversation?.messages).toHaveLength(0)
    })
  })

  describe('file upload status cards', () => {
    test('addFileUploadStatusCard adds to current conversation', () => {
      const conv: Conversation = {
        id: 'conv-1',
        userId: 'user-1',
        title: 'Test',
        messages: [],
        createdAt: new Date(),
        updatedAt: new Date(),
      }
      useChatStore.setState({
        currentUserId: 'user-1',
        currentConversation: conv,
        conversations: [conv],
      })

      useChatStore.getState().addFileUploadStatusCard('uploaded', 3, 'job-123')

      const messages = useChatStore.getState().currentConversation?.messages
      expect(messages).toHaveLength(1)
      expect(messages?.[0].messageType).toBe('file_upload_status')
      expect(messages?.[0].fileUploadStatusData?.type).toBe('uploaded')
      expect(messages?.[0].fileUploadStatusData?.fileCount).toBe(3)
      expect(messages?.[0].fileUploadStatusData?.jobId).toBe('job-123')
    })

    test('addFileUploadStatusCard adds to specific session', () => {
      const conv1: Conversation = {
        id: 'conv-1',
        userId: 'user-1',
        title: 'Current',
        messages: [],
        createdAt: new Date(),
        updatedAt: new Date(),
      }
      const conv2: Conversation = {
        id: 'conv-2',
        userId: 'user-1',
        title: 'Target',
        messages: [],
        createdAt: new Date(),
        updatedAt: new Date(),
      }
      useChatStore.setState({
        currentUserId: 'user-1',
        currentConversation: conv1,
        conversations: [conv1, conv2],
      })

      useChatStore.getState().addFileUploadStatusCard('uploaded', 2, 'job-456', 'conv-2')

      // Current conversation should be unchanged
      expect(useChatStore.getState().currentConversation?.messages).toHaveLength(0)

      // Target conversation should have the message
      const targetConv = useChatStore.getState().conversations.find((c) => c.id === 'conv-2')
      expect(targetConv?.messages).toHaveLength(1)
      expect(targetConv?.messages[0].fileUploadStatusData?.type).toBe('uploaded')
    })
  })

  describe('restoreSessionState — interrupted response detection', () => {
    const createConversation = (
      messages: Partial<Conversation['messages'][0]>[]
    ): Conversation => ({
      id: 'conv-restore',
      userId: 'user-1',
      title: 'Restore Test',
      messages: messages.map((m, i) => ({
        id: `msg-${i}`,
        role: (m.role ?? 'user') as 'user' | 'assistant' | 'system',
        content: m.content ?? '',
        timestamp: new Date(),
        ...m,
      })),
      createdAt: new Date(),
      updatedAt: new Date(),
    })

    test('adds error card when last meaningful message is user with thinking steps', () => {
      const conv = createConversation([
        {
          role: 'user',
          messageType: 'user',
          content: 'Tell me about AI',
          thinkingSteps: [
            {
              id: 's1',
              userMessageId: 'msg-0',
              category: 'tasks',
              functionName: 'fn',
              displayName: 'Searching',
              content: '',
              isComplete: true,
              timestamp: new Date(),
            },
          ],
        },
      ])

      // Set currentConversation before calling restoreSessionState
      useChatStore.setState({ currentConversation: conv, conversations: [conv] })
      useChatStore.getState().restoreSessionState(conv)

      // Should have added an error card
      const messages = useChatStore.getState().currentConversation?.messages ?? []
      expect(messages).toHaveLength(2)
      expect(messages[1].messageType).toBe('error')
      expect(messages[1].errorData?.errorCode).toBe('agent.response_interrupted')
    })

    test('does NOT add error card when last message is an assistant response', () => {
      const conv = createConversation([
        {
          role: 'user',
          messageType: 'user',
          content: 'Hello',
          thinkingSteps: [
            {
              id: 's1',
              userMessageId: 'msg-0',
              category: 'tasks',
              functionName: 'fn',
              displayName: 'Thinking',
              content: '',
              isComplete: true,
              timestamp: new Date(),
            },
          ],
        },
        { role: 'assistant', messageType: 'agent_response', content: 'Hi there!' },
      ])

      useChatStore.setState({ currentConversation: conv, conversations: [conv] })
      useChatStore.getState().restoreSessionState(conv)

      // No error card added — response was completed
      const messages = useChatStore.getState().currentConversation?.messages ?? []
      expect(messages).toHaveLength(2)
      expect(messages.every((m) => m.messageType !== 'error')).toBe(true)
    })

    test('restores last known deep research todos from the stored agent response', () => {
      const storedTodos = [
        { id: 'todo-1', content: 'Search current sources', status: 'completed' as const },
        { id: 'todo-2', content: 'Draft report', status: 'in_progress' as const },
      ]
      const conv = createConversation([
        {
          role: 'assistant',
          messageType: 'agent_response',
          content: 'Report is still loading',
          deepResearchJobId: 'job-123',
          deepResearchTodos: storedTodos,
          deepResearchLLMSteps: [
            {
              id: 'llm-1',
              name: 'model',
              content: 'heavy content',
              timestamp: new Date(),
              isComplete: true,
            },
          ],
        },
      ])

      useChatStore.setState({
        currentConversation: conv,
        conversations: [conv],
        deepResearchTodos: [],
        deepResearchLLMSteps: [],
      })

      useChatStore.getState().restoreSessionState(conv)

      const state = useChatStore.getState()
      expect(state.deepResearchTodos).toEqual(storedTodos)
      expect(state.deepResearchLLMSteps).toEqual([])
      expect(state.deepResearchJobId).toBe('job-123')
    })

    test('persists latest deep research todos onto the active tracking message', async () => {
      vi.useFakeTimers()
      const conv = createConversation([
        {
          id: 'tracking-msg',
          role: 'assistant',
          messageType: 'agent_response',
          content: '',
          deepResearchJobId: 'job-123',
          deepResearchJobStatus: 'running',
          isDeepResearchActive: true,
        },
      ])

      useChatStore.setState({
        currentConversation: conv,
        conversations: [conv],
        deepResearchOwnerConversationId: conv.id,
        activeDeepResearchMessageId: 'tracking-msg',
      })

      useChatStore.getState().setDeepResearchTodos([
        { content: 'Search current sources', status: 'in_progress' },
      ])

      await vi.advanceTimersByTimeAsync(1000)

      const trackingMessage = useChatStore
        .getState()
        .currentConversation?.messages.find((m) => m.id === 'tracking-msg')

      expect(trackingMessage?.deepResearchTodos).toEqual([
        {
          id: 'todo-0-search-current-sourc',
          content: 'Search current sources',
          status: 'in_progress',
        },
      ])
    })

    test('debounces persisted deep research todo snapshots during active streams', async () => {
      vi.useFakeTimers()
      const conv = createConversation([
        {
          id: 'tracking-msg',
          role: 'assistant',
          messageType: 'agent_response',
          content: '',
          deepResearchJobId: 'job-123',
          deepResearchJobStatus: 'running',
          isDeepResearchActive: true,
        },
      ])

      useChatStore.setState({
        currentConversation: conv,
        conversations: [conv],
        deepResearchOwnerConversationId: conv.id,
        activeDeepResearchMessageId: 'tracking-msg',
      })

      useChatStore.getState().setDeepResearchTodos([
        { content: 'Search current sources', status: 'pending' },
      ])
      useChatStore.getState().setDeepResearchTodos([
        { content: 'Search current sources', status: 'in_progress' },
      ])

      expect(
        useChatStore.getState().currentConversation?.messages[0].deepResearchTodos
      ).toBeUndefined()

      await vi.advanceTimersByTimeAsync(999)

      expect(
        useChatStore.getState().currentConversation?.messages[0].deepResearchTodos
      ).toBeUndefined()

      await vi.advanceTimersByTimeAsync(1)

      expect(useChatStore.getState().currentConversation?.messages[0].deepResearchTodos).toEqual([
        {
          id: 'todo-0-search-current-sourc',
          content: 'Search current sources',
          status: 'in_progress',
        },
      ])
    })

    test('persists stopped deep research todos onto the active tracking message', () => {
      const conv = createConversation([
        {
          id: 'tracking-msg',
          role: 'assistant',
          messageType: 'agent_response',
          content: '',
          deepResearchJobId: 'job-123',
          deepResearchJobStatus: 'running',
          isDeepResearchActive: true,
          deepResearchTodos: [
            { id: 'todo-1', content: 'Running task', status: 'in_progress' },
          ],
        },
      ])

      useChatStore.setState({
        currentConversation: conv,
        conversations: [conv],
        deepResearchOwnerConversationId: conv.id,
        activeDeepResearchMessageId: 'tracking-msg',
        deepResearchTodos: [
          { id: 'todo-1', content: 'Running task', status: 'in_progress' },
        ],
      })

      useChatStore.getState().stopAllDeepResearchSpinners(false)

      const trackingMessage = useChatStore
        .getState()
        .currentConversation?.messages.find((m) => m.id === 'tracking-msg')

      expect(trackingMessage?.deepResearchTodos).toEqual([
        { id: 'todo-1', content: 'Running task', status: 'stopped' },
      ])
    })

    test('does not fabricate workflow completion from a successful job status', () => {
      useChatStore.setState({
        deepResearchTodos: [
          { id: 'todo-1', content: 'Legacy task', status: 'in_progress' },
        ],
        deepResearchAgents: [
          {
            id: 'agent-start-only',
            name: 'researcher-agent',
            status: 'running',
            startedAt: new Date(),
          },
        ],
      })

      useChatStore.getState().stopAllDeepResearchSpinners(true)

      expect(useChatStore.getState().deepResearchTodos[0]?.status).toBe('completed')
      expect(useChatStore.getState().deepResearchAgents[0]?.status).toBe('error')
    })

    test('does NOT add error card when user message has no thinking steps', () => {
      const conv = createConversation([{ role: 'user', messageType: 'user', content: 'Hello' }])

      useChatStore.setState({ currentConversation: conv, conversations: [conv] })
      useChatStore.getState().restoreSessionState(conv)

      // No error card — no thinking steps means processing never started
      const messages = useChatStore.getState().currentConversation?.messages ?? []
      expect(messages).toHaveLength(1)
    })

    test('does NOT add error card when pending HITL interaction exists', () => {
      const conv = createConversation([
        {
          role: 'user',
          messageType: 'user',
          content: 'Research AI',
          thinkingSteps: [
            {
              id: 's1',
              userMessageId: 'msg-0',
              category: 'tasks',
              functionName: 'fn',
              displayName: 'Planning',
              content: '',
              isComplete: true,
              timestamp: new Date(),
            },
          ],
        },
        {
          role: 'assistant',
          messageType: 'prompt',
          content: 'Approve this plan?',
          promptId: 'p-1',
          promptParentId: 'msg-0',
          promptInputType: 'approval',
          isPromptResponded: false,
        },
      ])

      useChatStore.setState({ currentConversation: conv, conversations: [conv] })
      useChatStore.getState().restoreSessionState(conv)

      // No error card — unresponded prompt restores pendingInteraction, not an interruption
      const messages = useChatStore.getState().currentConversation?.messages ?? []
      expect(messages.every((m) => m.errorData?.errorCode !== 'agent.response_interrupted')).toBe(
        true
      )
    })

    test('does NOT double-add error card on repeated restore calls', () => {
      const conv = createConversation([
        {
          role: 'user',
          messageType: 'user',
          content: 'Tell me about AI',
          thinkingSteps: [
            {
              id: 's1',
              userMessageId: 'msg-0',
              category: 'tasks',
              functionName: 'fn',
              displayName: 'Searching',
              content: '',
              isComplete: true,
              timestamp: new Date(),
            },
          ],
        },
      ])

      useChatStore.setState({ currentConversation: conv, conversations: [conv] })

      // First restore — adds error card
      useChatStore.getState().restoreSessionState(conv)
      const afterFirst = useChatStore.getState().currentConversation?.messages ?? []
      expect(
        afterFirst.filter((m) => m.errorData?.errorCode === 'agent.response_interrupted')
      ).toHaveLength(1)

      // Second restore with updated conversation (now includes error card)
      const updatedConv = useChatStore.getState().currentConversation!
      useChatStore.getState().restoreSessionState(updatedConv)
      const afterSecond = useChatStore.getState().currentConversation?.messages ?? []
      expect(
        afterSecond.filter((m) => m.errorData?.errorCode === 'agent.response_interrupted')
      ).toHaveLength(1)
    })
  })

  describe('cleanupOrphanedStartingBanners', () => {
    const createConversation = (
      messages: Partial<Conversation['messages'][0]>[]
    ): Conversation => ({
      id: 'conv-orphaned',
      userId: 'user-1',
      title: 'Orphaned Banner Test',
      messages: messages.map((m, i) => ({
        id: m.id ?? `msg-${i}`,
        role: (m.role ?? 'assistant') as 'user' | 'assistant' | 'system',
        content: m.content ?? '',
        timestamp: new Date(),
        ...m,
      })),
      createdAt: new Date(),
      updatedAt: new Date(),
    })

    test('syncs stale tracking message when terminal banner already exists', async () => {
      const conv = createConversation([
        {
          id: 'tracking-msg',
          messageType: 'agent_response',
          deepResearchJobId: 'job-123',
          deepResearchJobStatus: 'running',
          isDeepResearchActive: true,
        },
        {
          id: 'starting-banner',
          messageType: 'deep_research_banner',
          deepResearchBannerData: { bannerType: 'starting', jobId: 'job-123' },
        },
        {
          id: 'failure-banner',
          messageType: 'deep_research_banner',
          deepResearchBannerData: { bannerType: 'failure', jobId: 'job-123' },
        },
      ])

      useChatStore.setState({ currentConversation: conv, conversations: [conv] })

      await useChatStore.getState().cleanupOrphanedStartingBanners()

      const updatedMessages = useChatStore.getState().currentConversation?.messages ?? []
      const trackingMessage = updatedMessages.find((m) => m.id === 'tracking-msg')

      expect(updatedMessages.some((m) => m.id === 'starting-banner')).toBe(false)
      expect(trackingMessage?.deepResearchJobStatus).toBe('failure')
      expect(trackingMessage?.isDeepResearchActive).toBe(false)
    })

    test('syncs stale tracking message after REST status resolves terminal state', async () => {
      mockDeepResearchApi.getJobStatus.mockResolvedValue({
        job_id: 'job-456',
        status: 'failure',
        error: 'expired',
      })

      const conv = createConversation([
        {
          id: 'tracking-msg',
          messageType: 'agent_response',
          deepResearchJobId: 'job-456',
          deepResearchJobStatus: 'running',
          isDeepResearchActive: true,
        },
        {
          id: 'starting-banner',
          messageType: 'deep_research_banner',
          deepResearchBannerData: { bannerType: 'starting', jobId: 'job-456' },
        },
      ])

      useChatStore.setState({ currentConversation: conv, conversations: [conv] })

      await useChatStore.getState().cleanupOrphanedStartingBanners()

      const updatedMessages = useChatStore.getState().currentConversation?.messages ?? []
      const trackingMessage = updatedMessages.find((m) => m.id === 'tracking-msg')
      const terminalBanner = updatedMessages.find(
        (m) =>
          m.messageType === 'deep_research_banner' &&
          m.deepResearchBannerData?.jobId === 'job-456' &&
          m.deepResearchBannerData?.bannerType === 'failure'
      )

      expect(trackingMessage?.deepResearchJobStatus).toBe('failure')
      expect(trackingMessage?.isDeepResearchActive).toBe(false)
      expect(updatedMessages.some((m) => m.id === 'starting-banner')).toBe(false)
      expect(terminalBanner).toBeTruthy()
    })
  })

  describe('reconnectToActiveJob', () => {
    const createConversation = (
      messages: Partial<Conversation['messages'][0]>[]
    ): Conversation => ({
      id: 'conv-reconnect',
      userId: 'user-1',
      title: 'Reconnect Test',
      messages: messages.map((m, i) => ({
        id: m.id ?? `msg-${i}`,
        role: (m.role ?? 'assistant') as 'user' | 'assistant' | 'system',
        content: m.content ?? '',
        timestamp: new Date(),
        ...m,
      })),
      createdAt: new Date(),
      updatedAt: new Date(),
    })

    test('marks missing active job as failed when status lookup returns 404', async () => {
      mockDeepResearchApi.getJobStatus.mockRejectedValue(new Error('Failed to get job status: 404'))

      const conv = createConversation([
        {
          id: 'tracking-msg',
          messageType: 'agent_response',
          deepResearchJobId: 'job-missing',
          deepResearchJobStatus: 'running',
          isDeepResearchActive: true,
        },
        {
          id: 'starting-banner',
          messageType: 'deep_research_banner',
          deepResearchBannerData: { bannerType: 'starting', jobId: 'job-missing' },
        },
      ])

      useChatStore.setState({ currentConversation: conv, conversations: [conv] })

      await useChatStore.getState().reconnectToActiveJob()

      const updatedMessages = useChatStore.getState().currentConversation?.messages ?? []
      const trackingMessage = updatedMessages.find((m) => m.id === 'tracking-msg')
      const failureBanner = updatedMessages.find(
        (m) =>
          m.messageType === 'deep_research_banner' &&
          m.deepResearchBannerData?.jobId === 'job-missing' &&
          m.deepResearchBannerData?.bannerType === 'failure'
      )

      expect(trackingMessage?.deepResearchJobStatus).toBe('failure')
      expect(trackingMessage?.isDeepResearchActive).toBe(false)
      expect(updatedMessages.some((m) => m.id === 'starting-banner')).toBe(false)
      expect(failureBanner).toBeTruthy()
    })
  })

})
