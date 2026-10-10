// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { render, screen } from '@/test-utils'
import userEvent from '@testing-library/user-event'
import { vi, describe, test, expect, beforeEach } from 'vitest'
import { MainLayout } from './MainLayout'

const mockUpdateSessionUrl = vi.fn()
const mockClearSessionUrl = vi.fn()
const mockSelectConversation = vi.fn()
const mockStartNewSessionDraft = vi.fn()
const mockDeleteConversation = vi.fn()
const mockDeleteAllConversations = vi.fn()
const mockUpdateConversationTitle = vi.fn()
const mockOpenRightPanel = vi.fn()

vi.mock('@/hooks/use-session-url', () => ({
  useSessionUrl: vi.fn(() => ({
    updateSessionUrl: mockUpdateSessionUrl,
    clearSessionUrl: mockClearSessionUrl,
  })),
}))

vi.mock('@/features/chat', () => ({
  useChatStore: vi.fn((selector?: (s: any) => any) => {
    const state = {
      currentConversation: { id: 'session-1', title: 'Test Session' },
      getUserConversations: vi.fn(() => []),
      selectConversation: mockSelectConversation,
      startNewSessionDraft: mockStartNewSessionDraft,
      deleteConversation: mockDeleteConversation,
      deleteAllConversations: mockDeleteAllConversations,
      updateConversationTitle: mockUpdateConversationTitle,
      isStreaming: false,
      pendingInteraction: null,
      isDeepResearchStreaming: false,
      deepResearchOwnerConversationId: null,
    }
    return selector ? selector(state) : state
  }),
  useDeepResearch: vi.fn(() => ({
    isResearching: false,
    connect: vi.fn(),
    disconnect: vi.fn(),
    cancel: vi.fn(),
  })),
  NoSourcesBanner: () => <div data-testid="no-sources-banner">No Sources Banner</div>,
}))

vi.mock('../store', () => ({
  useLayoutStore: vi.fn((selector?: (s: any) => any) => {
    const state = {
      rightPanel: null,
      isSessionsPanelOpen: false,
      setSessionsPanelOpen: vi.fn(),
      enabledDataSourceIds: ['source-1', 'source-2'],
      openRightPanel: mockOpenRightPanel,
    }
    return selector ? selector(state) : state
  }),
}))

vi.mock('./AppBar', () => ({
  AppBar: ({
    sessionTitle,
    onNewSession,
    isNewSessionDisabled,
  }: {
    sessionTitle: string
    onNewSession?: () => void
    isNewSessionDisabled?: boolean
  }) => (
    <>
      <div data-testid="app-bar">{sessionTitle}</div>
      <button type="button" onClick={onNewSession} disabled={isNewSessionDisabled}>
        Header New Session
      </button>
    </>
  ),
}))

vi.mock('./SessionsPanel', () => ({
  SessionsPanel: () => <div data-testid="sessions-panel">Sessions Panel</div>,
}))

vi.mock('./ChatArea', () => ({
  ChatArea: () => <div data-testid="chat-area">Chat Area</div>,
}))

vi.mock('./InputArea', () => ({
  InputArea: () => <div data-testid="input-area">Input Area</div>,
}))

vi.mock('./ResearchPanel', () => ({
  ResearchPanel: () => <div data-testid="research-panel">Research Panel</div>,
}))

vi.mock('./DataSourcesPanel', () => ({
  DataSourcesPanel: () => (
    <div data-testid="data-sources-panel">
      <button type="button" aria-label="Close data sources panel">
        Close
      </button>
    </div>
  ),
}))

vi.mock('./DeepResearchRail', () => ({
  DeepResearchRail: () => <div data-testid="deep-research-rail">Deep Research Rail</div>,
}))

import { useChatStore } from '@/features/chat'

describe('MainLayout', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  test('renders authenticated main sections', () => {
    render(<MainLayout isAuthenticated={true} />)

    expect(screen.getByTestId('app-bar')).toBeInTheDocument()
    expect(screen.getByTestId('sessions-panel')).toBeInTheDocument()
    expect(screen.getByTestId('chat-area')).toBeInTheDocument()
    expect(screen.getByTestId('input-area')).toBeInTheDocument()
    expect(screen.getByTestId('research-panel')).toBeInTheDocument()
    expect(screen.getByTestId('data-sources-panel')).toBeInTheDocument()
    expect(screen.getByTestId('deep-research-rail')).toBeInTheDocument()
  })

  test('hides the sessions sidebar and data sources panel when unauthenticated', () => {
    render(<MainLayout />)

    expect(screen.getByTestId('app-bar')).toBeInTheDocument()
    expect(screen.queryByTestId('sessions-panel')).not.toBeInTheDocument()
    expect(screen.getByTestId('chat-area')).toBeInTheDocument()
    expect(screen.getByTestId('input-area')).toBeInTheDocument()
    expect(screen.getByTestId('research-panel')).toBeInTheDocument()
    expect(screen.queryByTestId('data-sources-panel')).not.toBeInTheDocument()
    expect(screen.getByTestId('deep-research-rail')).toBeInTheDocument()
  })

  test('passes session title to AppBar', () => {
    render(<MainLayout />)

    expect(screen.getByTestId('app-bar')).toHaveTextContent('Test Session')
  })

  test('shows no session title when no current conversation', () => {
    vi.mocked(useChatStore).mockImplementationOnce((selector?: (s: any) => any) => {
      const state = {
        currentConversation: null,
        getUserConversations: vi.fn(() => []),
        selectConversation: vi.fn(),
        startNewSessionDraft: vi.fn(),
        deleteConversation: vi.fn(),
        deleteAllConversations: vi.fn(),
        updateConversationTitle: vi.fn(),
        isStreaming: false,
        pendingInteraction: null,
        isDeepResearchStreaming: false,
        deepResearchOwnerConversationId: null,
      }
      return selector ? selector(state) : state
    })

    render(<MainLayout />)

    expect(screen.getByTestId('app-bar')).toHaveTextContent('')
  })

  test('passes auth state to components', () => {
    const onSignIn = vi.fn()
    const onSignOut = vi.fn()
    const user = { name: 'Test User', email: 'test@nvidia.com' }

    render(
      <MainLayout isAuthenticated={true} user={user} onSignIn={onSignIn} onSignOut={onSignOut} />
    )

    expect(screen.getByTestId('app-bar')).toBeInTheDocument()
    expect(screen.getByTestId('chat-area')).toBeInTheDocument()
    expect(screen.getByTestId('input-area')).toBeInTheDocument()
  })

  test('wires the AppBar new session action to draft session flow', async () => {
    const user = userEvent.setup()

    render(<MainLayout isAuthenticated={true} />)

    await user.click(screen.getByRole('button', { name: /header new session/i }))

    expect(mockStartNewSessionDraft).toHaveBeenCalledOnce()
    expect(mockClearSessionUrl).toHaveBeenCalledOnce()
    expect(mockOpenRightPanel).toHaveBeenCalledWith('data-sources')
  })

  test('does not open data sources from new session while unauthenticated', async () => {
    const user = userEvent.setup()

    render(<MainLayout />)

    await user.click(screen.getByRole('button', { name: /header new session/i }))

    expect(mockStartNewSessionDraft).toHaveBeenCalledOnce()
    expect(mockClearSessionUrl).toHaveBeenCalledOnce()
    expect(mockOpenRightPanel).not.toHaveBeenCalled()
  })

  test('disables new session action while shallow streaming is active', () => {
    vi.mocked(useChatStore).mockImplementation((selector?: (s: any) => any) => {
      const state = {
        currentConversation: { id: 'session-1', title: 'Test Session' },
        getUserConversations: vi.fn(() => []),
        selectConversation: vi.fn(),
        startNewSessionDraft: vi.fn(),
        deleteConversation: vi.fn(),
        deleteAllConversations: vi.fn(),
        updateConversationTitle: vi.fn(),
        isStreaming: true,
        pendingInteraction: null,
        isDeepResearchStreaming: false,
        deepResearchOwnerConversationId: null,
      }
      return selector ? selector(state) : state
    })

    render(<MainLayout />)

    expect(screen.getByRole('button', { name: /header new session/i })).toBeDisabled()
  })

  test('chat region flexes to fill the space between the side panels', () => {
    render(<MainLayout isAuthenticated={true} />)

    const centerColumn = screen.getByTestId('chat-area').parentElement
    expect(centerColumn).toHaveClass('flex-1')
  })

  test('keeps the Deep Research rail and Data Sources close control reachable at 1024px instead of clipping them off-screen', () => {
    const originalInnerWidth = window.innerWidth
    Object.defineProperty(window, 'innerWidth', { configurable: true, value: 1024 })
    try {
      render(<MainLayout isAuthenticated={true} />)

      const contentRow = screen.getByTestId('main-content-row')
      const rail = screen.getByTestId('deep-research-rail')
      const closeControl = screen.getByRole('button', { name: 'Close data sources panel' })

      expect(contentRow).toContainElement(rail)
      expect(contentRow).toContainElement(closeControl)

      expect(contentRow.className).toMatch(/overflow-x-auto/)
      expect(contentRow.className).not.toMatch(/overflow-hidden/)

      Object.defineProperty(contentRow, 'clientWidth', { configurable: true, value: 1024 })
      Object.defineProperty(contentRow, 'scrollWidth', { configurable: true, value: 1240 })
      expect(contentRow.scrollWidth).toBeGreaterThan(contentRow.clientWidth)

      contentRow.scrollLeft = contentRow.scrollWidth - contentRow.clientWidth
      expect(contentRow.scrollLeft).toBeGreaterThan(0)
    } finally {
      Object.defineProperty(window, 'innerWidth', {
        configurable: true,
        value: originalInnerWidth,
      })
    }
  })
})
