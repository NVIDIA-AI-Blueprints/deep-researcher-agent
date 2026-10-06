// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { render, screen } from '@/test-utils'
import userEvent from '@testing-library/user-event'
import { vi, describe, test, expect, beforeEach } from 'vitest'
import { ChatArea } from './ChatArea'

const mockRespondToPrompt = vi.fn()
const mockDismissErrorCard = vi.fn()
const mockGetThinkingStepsForMessage = vi.fn((_messageId: string) => [] as { id: string; displayName: string }[])
const mockChatThinking = vi.fn((_props: unknown) => <div data-testid="chat-thinking">Thinking...</div>)

vi.mock('@/features/chat', () => ({
  useChatStore: vi.fn((selector?: (s: any) => any) => {
    const state = {
      currentConversation: { messages: [] },
      isLoading: false,
      isStreaming: false,
      thinkingSteps: [],
      respondToPrompt: mockRespondToPrompt,
      dismissErrorCard: mockDismissErrorCard,
      getThinkingStepsForMessage: mockGetThinkingStepsForMessage,
    }
    return selector ? selector(state) : state
  }),
  AgentPrompt: ({ content }: { content: string }) => (
    <div data-testid="agent-prompt">{content}</div>
  ),
  AgentResponse: ({ content }: { content: string }) => (
    <div data-testid="agent-response">{content}</div>
  ),
  ErrorBanner: ({ message }: { message: string }) => <div data-testid="error-card">{message}</div>,
  FileUploadBanner: ({ type }: { type: string }) => <div data-testid="file-banner">{type}</div>,
  UserMessage: ({ content }: { content: string }) => (
    <div data-testid="user-message">{content}</div>
  ),
  ChatThinking: (props: unknown) => mockChatThinking(props),
}))

import { useChatStore } from '@/features/chat'

describe('ChatArea', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  test('renders welcome state when not authenticated', () => {
    render(<ChatArea isAuthenticated={false} />)

    expect(screen.getByText('Welcome to Deep Researcher Agent')).toBeInTheDocument()
    expect(screen.getByText(/sign in with your account/i)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /sign in with.*sso/i })).toBeInTheDocument()
  })

  test('renders welcome state when authenticated with no messages', () => {
    render(<ChatArea isAuthenticated={true} />)

    expect(screen.getByText('What do you want to know?')).toBeInTheDocument()
    expect(screen.getByText(/connected data sources/i)).toBeInTheDocument()
  })

  test('does not render example prompt chips in the welcome state', () => {
    render(<ChatArea isAuthenticated={true} />)

    expect(
      screen.queryByRole('button', { name: /which customers are most likely to churn/i })
    ).not.toBeInTheDocument()
  })

  test('calls onSignIn when sign in button clicked', async () => {
    const user = userEvent.setup()
    const onSignIn = vi.fn()

    render(<ChatArea isAuthenticated={false} onSignIn={onSignIn} />)

    await user.click(screen.getByRole('button', { name: /sign in with.*sso/i }))

    expect(onSignIn).toHaveBeenCalled()
  })

  test('renders user messages', () => {
    vi.mocked(useChatStore).mockImplementation((selector?: (s: any) => any) => {
      const state = {
        currentConversation: {
          messages: [{ id: 'msg-1', role: 'user', content: 'Hello world', messageType: 'user' }],
        },
        isLoading: false,
        isStreaming: false,
        thinkingSteps: [],
        respondToPrompt: mockRespondToPrompt,
        dismissErrorCard: mockDismissErrorCard,
        getThinkingStepsForMessage: mockGetThinkingStepsForMessage,
      }
      return selector ? selector(state) : state
    })

    render(<ChatArea isAuthenticated={true} />)

    expect(screen.getByTestId('user-message')).toHaveTextContent('Hello world')
  })

  test('renders status messages', () => {
    vi.mocked(useChatStore).mockImplementation((selector?: (s: any) => any) => {
      const state = {
        currentConversation: {
          messages: [
            {
              id: 'msg-1',
              role: 'assistant',
              content: 'Processing...',
              messageType: 'status',
              statusType: 'thinking',
            },
          ],
        },
        isLoading: false,
        isStreaming: false,
        thinkingSteps: [],
        respondToPrompt: mockRespondToPrompt,
        dismissErrorCard: mockDismissErrorCard,
        getThinkingStepsForMessage: mockGetThinkingStepsForMessage,
      }
      return selector ? selector(state) : state
    })

    render(<ChatArea isAuthenticated={true} />)

    expect(screen.getByRole('status')).toBeInTheDocument()
  })

  test('renders agent prompts', () => {
    vi.mocked(useChatStore).mockImplementation((selector?: (s: any) => any) => {
      const state = {
        currentConversation: {
          messages: [
            {
              id: 'msg-1',
              role: 'assistant',
              content: 'Please provide more details',
              messageType: 'prompt',
              promptType: 'input',
            },
          ],
        },
        isLoading: false,
        isStreaming: false,
        respondToPrompt: mockRespondToPrompt,
        dismissErrorCard: mockDismissErrorCard,
      }
      return selector ? selector(state) : state
    })

    render(<ChatArea isAuthenticated={true} />)

    expect(screen.getByTestId('agent-prompt')).toBeInTheDocument()
  })

  test('renders agent responses', () => {
    vi.mocked(useChatStore).mockImplementation((selector?: (s: any) => any) => {
      const state = {
        currentConversation: {
          messages: [
            {
              id: 'msg-1',
              role: 'assistant',
              content: 'Here is your answer',
              messageType: 'agent_response',
            },
          ],
        },
        isLoading: false,
        isStreaming: false,
        respondToPrompt: mockRespondToPrompt,
        dismissErrorCard: mockDismissErrorCard,
      }
      return selector ? selector(state) : state
    })

    render(<ChatArea isAuthenticated={true} />)

    expect(screen.getByTestId('agent-response')).toHaveTextContent('Here is your answer')
  })

  test('renders file messages', () => {
    vi.mocked(useChatStore).mockImplementation((selector?: (s: any) => any) => {
      const state = {
        currentConversation: {
          messages: [
            {
              id: 'msg-1',
              role: 'assistant',
              content: '',
              messageType: 'file',
              fileData: {
                fileName: 'document.pdf',
                fileSize: 1024,
                fileStatus: 'success',
              },
            },
          ],
        },
        isLoading: false,
        isStreaming: false,
        thinkingSteps: [],
        respondToPrompt: mockRespondToPrompt,
        dismissErrorCard: mockDismissErrorCard,
        getThinkingStepsForMessage: mockGetThinkingStepsForMessage,
      }
      return selector ? selector(state) : state
    })

    render(<ChatArea isAuthenticated={true} />)

    expect(screen.getByText(/document\.pdf/)).toBeInTheDocument()
  })

  test('renders error banners', () => {
    vi.mocked(useChatStore).mockImplementation((selector?: (s: any) => any) => {
      const state = {
        currentConversation: {
          messages: [
            {
              id: 'msg-1',
              role: 'assistant',
              content: '',
              messageType: 'error',
              errorData: {
                errorCode: 'E001',
                errorMessage: 'Something went wrong',
              },
            },
          ],
        },
        isLoading: false,
        isStreaming: false,
        thinkingSteps: [],
        respondToPrompt: mockRespondToPrompt,
        dismissErrorCard: mockDismissErrorCard,
        getThinkingStepsForMessage: mockGetThinkingStepsForMessage,
      }
      return selector ? selector(state) : state
    })

    render(<ChatArea isAuthenticated={true} />)

    expect(screen.getByTestId('error-card')).toBeInTheDocument()
  })

  test('does not render assistant messages (full reports)', () => {
    vi.mocked(useChatStore).mockImplementation((selector?: (s: any) => any) => {
      const state = {
        currentConversation: {
          messages: [
            {
              id: 'msg-1',
              role: 'assistant',
              content: 'Full report content',
              messageType: 'assistant',
            },
          ],
        },
        isLoading: false,
        isStreaming: false,
        respondToPrompt: mockRespondToPrompt,
        dismissErrorCard: mockDismissErrorCard,
      }
      return selector ? selector(state) : state
    })

    render(<ChatArea isAuthenticated={true} />)

    expect(screen.getByText('What do you want to know?')).toBeInTheDocument()
  })

  test('renders chat messages area with aria-label', () => {
    render(<ChatArea isAuthenticated={true} />)

    expect(screen.getByLabelText(/chat messages/i)).toBeInTheDocument()
  })

  test('handles null currentConversation', () => {
    vi.mocked(useChatStore).mockImplementation((selector?: (s: any) => any) => {
      const state = {
        currentConversation: null,
        isLoading: false,
        isStreaming: false,
        respondToPrompt: mockRespondToPrompt,
        dismissErrorCard: mockDismissErrorCard,
      }
      return selector ? selector(state) : state
    })

    render(<ChatArea isAuthenticated={true} />)

    expect(screen.getByText('What do you want to know?')).toBeInTheDocument()
  })

  test('renders file upload banners', () => {
    vi.mocked(useChatStore).mockImplementation((selector?: (s: any) => any) => {
      const state = {
        currentConversation: {
          messages: [
            {
              id: 'msg-1',
              role: 'assistant',
              content: '',
              messageType: 'file_upload_status',
              fileUploadStatusData: {
                type: 'uploaded',
                fileCount: 2,
              },
            },
          ],
        },
        isLoading: false,
        isStreaming: false,
        respondToPrompt: mockRespondToPrompt,
        dismissErrorCard: mockDismissErrorCard,
      }
      return selector ? selector(state) : state
    })

    render(<ChatArea isAuthenticated={true} />)

    expect(screen.getByTestId('file-banner')).toBeInTheDocument()
  })

  test('keeps earlier interrupted thinking state after a later completed turn', () => {
    mockGetThinkingStepsForMessage.mockImplementation((messageId: string) => {
      if (messageId === 'user-1') return [{ id: 'step-1', displayName: 'Step 1' }]
      if (messageId === 'user-2') return [{ id: 'step-2', displayName: 'Step 2' }]
      return []
    })

    vi.mocked(useChatStore).mockImplementation((selector?: (s: any) => any) => {
      const state = {
        currentConversation: {
          messages: [
            { id: 'user-1', role: 'user', content: 'First question', messageType: 'user' },
            {
              id: 'user-2',
              role: 'user',
              content: 'Second question',
              messageType: 'user',
              selectedModel: 'test-model',
              timestamp: new Date('2026-06-29T16:00:00.000Z'),
            },
            {
              id: 'answer-2',
              role: 'assistant',
              content: 'Second answer',
              messageType: 'agent_response',
              timestamp: new Date('2026-06-29T16:01:07.000Z'),
            },
          ],
        },
        isLoading: false,
        isStreaming: false,
        thinkingSteps: [],
        respondToPrompt: mockRespondToPrompt,
        dismissErrorCard: mockDismissErrorCard,
        getThinkingStepsForMessage: mockGetThinkingStepsForMessage,
      }
      return selector ? selector(state) : state
    })

    render(<ChatArea isAuthenticated={true} />)

    expect(mockChatThinking).toHaveBeenCalledTimes(2)

    const firstCallProps = mockChatThinking.mock.calls[0][0] as {
      isInterrupted?: boolean
      isThinking?: boolean
    }
    const secondCallProps = mockChatThinking.mock.calls[1][0] as {
      isInterrupted?: boolean
      isThinking?: boolean
      responseStartedAt?: Date
      responseCompletedAt?: Date
    }

    expect(firstCallProps.isInterrupted).toBe(true)
    expect(firstCallProps.isThinking).toBe(false)

    expect(secondCallProps.isInterrupted).toBe(false)
    expect(secondCallProps.isThinking).toBe(false)
    expect(secondCallProps.responseStartedAt).toEqual(new Date('2026-06-29T16:00:00.000Z'))
    expect(secondCallProps.responseCompletedAt).toEqual(new Date('2026-06-29T16:01:07.000Z'))
  })

  test('keeps earlier interrupted thinking state while a new message is actively streaming', () => {
    mockGetThinkingStepsForMessage.mockImplementation((messageId: string) => {
      if (messageId === 'user-1') return [{ id: 'step-1', displayName: 'Step 1' }]
      if (messageId === 'user-2') return [{ id: 'step-2', displayName: 'Step 2' }]
      return []
    })

    vi.mocked(useChatStore).mockImplementation((selector?: (s: any) => any) => {
      const state = {
        currentConversation: {
          messages: [
            { id: 'user-1', role: 'user', content: 'First question', messageType: 'user' },
            {
              id: 'user-2',
              role: 'user',
              content: 'Second question',
              messageType: 'user',
              selectedModel: 'test-model',
            },
          ],
        },
        isLoading: true,
        isStreaming: true,
        currentUserMessageId: 'user-2',
        thinkingSteps: [],
        respondToPrompt: mockRespondToPrompt,
        dismissErrorCard: mockDismissErrorCard,
        getThinkingStepsForMessage: mockGetThinkingStepsForMessage,
      }
      return selector ? selector(state) : state
    })

    render(<ChatArea isAuthenticated={true} />)

    expect(mockChatThinking).toHaveBeenCalledTimes(2)

    const firstCallProps = mockChatThinking.mock.calls[0][0] as {
      isInterrupted?: boolean
      isThinking?: boolean
    }
    const secondCallProps = mockChatThinking.mock.calls[1][0] as {
      isInterrupted?: boolean
      isThinking?: boolean
    }

    expect(firstCallProps.isInterrupted).toBe(true)
    expect(firstCallProps.isThinking).toBe(false)

    expect(secondCallProps.isThinking).toBe(true)
    expect(secondCallProps.isInterrupted).toBe(false)
    expect((mockChatThinking.mock.calls[1][0] as { model?: string }).model).toBe('test-model')
  })
})
