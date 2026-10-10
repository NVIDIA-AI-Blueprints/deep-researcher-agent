// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { render, screen, waitFor } from '@/test-utils'
import userEvent from '@testing-library/user-event'
import { vi, describe, test, expect, beforeEach } from 'vitest'
import { SessionsPanel } from './SessionsPanel'

const mockToggleSessionsSidebar = vi.fn()
const mockSetSessionsCollapsed = vi.fn()

vi.mock('../store', () => ({
  useLayoutStore: vi.fn((selector?: (s: any) => any) => {
    const state = {
      sessionsCollapsed: false,
      sessionsAutoCollapsed: false,
      toggleSessionsSidebar: mockToggleSessionsSidebar,
      setSessionsCollapsed: mockSetSessionsCollapsed,
    }
    return selector ? selector(state) : state
  }),
}))

vi.mock('@/features/chat', () => ({
  useChatStore: vi.fn(),
}))

vi.mock('./DeleteSessionConfirmationModal', () => ({
  DeleteSessionConfirmationModal: ({
    open,
    onConfirm,
    onOpenChange,
  }: {
    open: boolean
    onConfirm: () => void
    onOpenChange: (open: boolean) => void
  }) =>
    open ? (
      <div data-testid="delete-modal">
        <button onClick={onConfirm}>Confirm Delete</button>
        <button onClick={() => onOpenChange(false)}>Cancel</button>
      </div>
    ) : null,
}))

import { useLayoutStore } from '../store'
import { useChatStore } from '@/features/chat'

/**
 * Helper to create a mock chat store state.
 * Components select individual fields via useChatStore((state) => state.X).
 */
const createMockChatState = (
  overrides: {
    isSessionBusy?: (sessionId: string) => boolean
    hasAnyBusySession?: () => boolean
    isStreaming?: boolean
    pendingInteraction?: { id: string; type: string; content: string } | null
    refreshDeepResearchSessionStatuses?: () => Promise<void>
  } = {}
) => ({
  isSessionBusy: overrides.isSessionBusy ?? (() => false),
  hasAnyBusySession: overrides.hasAnyBusySession ?? (() => false),
  isStreaming: overrides.isStreaming ?? false,
  pendingInteraction: overrides.pendingInteraction ?? null,
  refreshDeepResearchSessionStatuses: overrides.refreshDeepResearchSessionStatuses ?? vi.fn(),
})

const setupChatStoreMock = (overrides: Parameters<typeof createMockChatState>[0] = {}) => {
  const state = createMockChatState(overrides)
  vi.mocked(useChatStore).mockImplementation((selector: (s: any) => any) => {
    if (typeof selector === 'function') {
      return selector(state)
    }
    return undefined
  })
}

const setupLayoutStoreMock = (collapsed = false) => {
  vi.mocked(useLayoutStore).mockImplementation((selector?: (s: any) => any) => {
    const state = {
      sessionsCollapsed: collapsed,
      sessionsAutoCollapsed: false,
      toggleSessionsSidebar: mockToggleSessionsSidebar,
      setSessionsCollapsed: mockSetSessionsCollapsed,
    }
    return selector ? selector(state) : state
  })
}

describe('SessionsPanel', () => {
  const today = new Date()
  const yesterday = new Date(today)
  yesterday.setDate(yesterday.getDate() - 1)

  const mockSessions = [
    { id: 'session-1', title: 'First Session', date: today },
    { id: 'session-2', title: 'Second Session', date: yesterday },
  ]

  beforeEach(() => {
    vi.clearAllMocks()
    setupChatStoreMock()

    setupLayoutStoreMock()
  })

  test('renders panel with heading', () => {
    render(<SessionsPanel sessions={mockSessions} />)

    expect(screen.getByText('Sessions')).toBeInTheDocument()
  })

  test('renders new session button', () => {
    render(<SessionsPanel sessions={mockSessions} />)

    expect(screen.getByText('New Session')).toBeInTheDocument()
  })

  test('renders session list grouped by date', () => {
    render(<SessionsPanel sessions={mockSessions} />)

    expect(screen.getByText('Today')).toBeInTheDocument()
    expect(screen.getByText('Yesterday')).toBeInTheDocument()
    expect(screen.getByText('First Session')).toBeInTheDocument()
    expect(screen.getByText('Second Session')).toBeInTheDocument()
  })

  test('buckets older sessions into relative ranges', () => {
    const now = Date.now()
    const day = 86_400_000
    render(
      <SessionsPanel
        sessions={[
          { id: 'a', title: 'Today chat', date: new Date(now) },
          { id: 'b', title: 'This week chat', date: new Date(now - 3 * day) },
          { id: 'c', title: 'This month chat', date: new Date(now - 10 * day) },
        ]}
      />
    )

    expect(screen.getByText('Today')).toBeInTheDocument()
    expect(screen.getByText('Previous 7 Days')).toBeInTheDocument()
    expect(screen.getByText('Previous 30 Days')).toBeInTheDocument()
  })

  test('preserves the caller session order within a bucket instead of re-sorting by date', () => {
    const now = Date.now()
    render(
      <SessionsPanel
        sessions={[
          { id: 'older', title: 'Older today', date: new Date(now - 2 * 60_000) },
          { id: 'newer', title: 'Newer today', date: new Date(now - 60_000) },
        ]}
      />
    )

    const older = screen.getByText('Older today')
    const newer = screen.getByText('Newer today')
    expect(older.compareDocumentPosition(newer) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
  })

  test('shows empty state when no sessions', () => {
    render(<SessionsPanel sessions={[]} />)

    expect(screen.getByText('No sessions yet')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /start a new session/i })).toBeInTheDocument()
  })

  test('calls onNewSession when new session button clicked', async () => {
    const user = userEvent.setup()
    const onNewSession = vi.fn()

    render(<SessionsPanel sessions={mockSessions} onNewSession={onNewSession} />)

    await user.click(screen.getByText('New Session'))

    expect(onNewSession).toHaveBeenCalled()
  })

  test('calls onSelectSession when session clicked', async () => {
    const user = userEvent.setup()
    const onSelectSession = vi.fn()

    render(<SessionsPanel sessions={mockSessions} onSelectSession={onSelectSession} />)

    await user.click(screen.getByRole('button', { name: /session: first session/i }))

    expect(onSelectSession).toHaveBeenCalledWith('session-1')
  })

  test('highlights selected session', () => {
    render(<SessionsPanel sessions={mockSessions} selectedSessionId="session-1" />)

    const firstSession = screen.getByRole('button', { name: /session: first session/i })
    expect(firstSession).toHaveClass('bg-surface-raised')
  })

  test('shows edit and delete icons on hover', async () => {
    const user = userEvent.setup()
    render(<SessionsPanel sessions={mockSessions} />)

    const sessionItem = screen.getByRole('button', { name: /session: first session/i })
    await user.hover(sessionItem)

    expect(screen.getByRole('button', { name: /rename session/i })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /delete session/i })).toBeInTheDocument()
  })

  test('renders footer text', () => {
    render(<SessionsPanel sessions={mockSessions} />)

    expect(screen.getByText(/Chat sessions are saved in this browser/i)).toBeInTheDocument()
  })

  test('returns focus to the search trigger when search closes on Escape', async () => {
    const user = userEvent.setup()
    render(<SessionsPanel sessions={mockSessions} />)

    await user.click(screen.getByRole('button', { name: /search sessions/i }))
    const input = screen.getByRole('textbox', { name: /search sessions/i })
    await user.type(input, '{Escape}')

    await waitFor(() => {
      expect(screen.getByRole('button', { name: /search sessions/i })).toHaveFocus()
    })
  })

  test('checks persisted deep research jobs when the sidebar is expanded', () => {
    const refreshDeepResearchSessionStatuses = vi.fn().mockResolvedValue(undefined)
    setupChatStoreMock({ refreshDeepResearchSessionStatuses })

    render(<SessionsPanel sessions={mockSessions} />)

    expect(refreshDeepResearchSessionStatuses).toHaveBeenCalledTimes(1)
  })

  test('does not start overlapping deep research status refreshes', async () => {
    let collapsed = false
    let resolveRefresh: () => void = () => {}
    const refreshDeepResearchSessionStatuses = vi.fn(
      () =>
        new Promise<void>((resolve) => {
          resolveRefresh = resolve
        })
    )
    setupChatStoreMock({ refreshDeepResearchSessionStatuses })
    vi.mocked(useLayoutStore).mockImplementation((selector?: (s: any) => any) => {
      const state = {
        sessionsCollapsed: collapsed,
        sessionsAutoCollapsed: false,
        toggleSessionsSidebar: mockToggleSessionsSidebar,
        setSessionsCollapsed: mockSetSessionsCollapsed,
      }
      return selector ? selector(state) : state
    })

    const { rerender } = render(<SessionsPanel sessions={mockSessions} />)

    expect(refreshDeepResearchSessionStatuses).toHaveBeenCalledTimes(1)

    collapsed = true
    rerender(<SessionsPanel sessions={[...mockSessions]} />)
    collapsed = false
    rerender(<SessionsPanel sessions={[...mockSessions]} />)

    expect(refreshDeepResearchSessionStatuses).toHaveBeenCalledTimes(1)

    resolveRefresh()
    await Promise.resolve()
    await Promise.resolve()
    await Promise.resolve()

    collapsed = true
    rerender(<SessionsPanel sessions={[...mockSessions]} />)
    await Promise.resolve()
    collapsed = false
    rerender(<SessionsPanel sessions={[...mockSessions]} />)

    await vi.waitFor(() => {
      expect(refreshDeepResearchSessionStatuses).toHaveBeenCalledTimes(2)
    })
  })

  test('renders the icon rail when collapsed', () => {
    setupLayoutStoreMock(true)

    render(<SessionsPanel sessions={mockSessions} />)

    expect(screen.getByRole('button', { name: /expand sessions sidebar/i })).toBeInTheDocument()
    expect(screen.getByText('Sessions')).toHaveClass('opacity-0')
    expect(screen.queryByText('First Session')).not.toBeInTheDocument()
  })

  test('does not refresh deep research job state when collapsed', () => {
    const refreshDeepResearchSessionStatuses = vi.fn().mockResolvedValue(undefined)
    setupChatStoreMock({ refreshDeepResearchSessionStatuses })
    setupLayoutStoreMock(true)

    render(<SessionsPanel sessions={mockSessions} />)

    expect(refreshDeepResearchSessionStatuses).not.toHaveBeenCalled()
  })

  test('shows chat icon for sessions with no report', async () => {
    render(<SessionsPanel sessions={mockSessions} />)

    await vi.waitFor(() => {
      expect(document.querySelector('svg[data-src$="/line/chat-single.svg"]')).toBeInTheDocument()
    })
  })

  test('shows document-checkmark icon for completed report sessions', async () => {
    render(
      <SessionsPanel
        sessions={[
          {
            id: 'session-report',
            title: 'Completed Report',
            date: new Date(),
            hasCompletedReport: true,
          },
        ]}
      />
    )

    await vi.waitFor(() => {
      expect(
        document.querySelector('svg[data-src$="/line/document-checkmark.svg"]')
      ).toBeInTheDocument()
    })
  })

  test('shows select-ellipse icon for expired report sessions', async () => {
    render(
      <SessionsPanel
        sessions={[
          {
            id: 'session-expired',
            title: 'Expired Report',
            date: new Date(),
            hasExpiredReport: true,
          },
        ]}
      />
    )

    await vi.waitFor(() => {
      expect(
        document.querySelector('svg[data-src$="/line/select-ellipse.svg"]')
      ).toBeInTheDocument()
    })
  })

  test('shows spinner for active shallow sessions', () => {
    setupChatStoreMock({
      isSessionBusy: (sessionId: string) => sessionId === 'session-1',
      hasAnyBusySession: () => true,
    })

    render(<SessionsPanel sessions={mockSessions} />)

    expect(screen.getByRole('status', { name: /session active/i })).toBeInTheDocument()
  })
})

describe('SessionsPanel - Session Switching', () => {
  const mockSessions = [
    { id: 'session-1', title: 'Deep Research Session', date: new Date() },
    { id: 'session-2', title: 'Idle Session', date: new Date() },
  ]

  beforeEach(() => {
    vi.clearAllMocks()
    setupChatStoreMock()
    setupLayoutStoreMock()
  })

  test('allows switching sessions during active deep research (server-side SSE)', async () => {
    setupChatStoreMock({
      isSessionBusy: (sessionId: string) => sessionId === 'session-1',
      hasAnyBusySession: () => true,
      isStreaming: false,
      pendingInteraction: null,
    })

    const user = userEvent.setup()
    const onSelectSession = vi.fn()
    render(
      <SessionsPanel
        sessions={mockSessions}
        selectedSessionId="session-2"
        onSelectSession={onSelectSession}
      />
    )

    const deepResearchSession = screen.getByRole('button', {
      name: /session: deep research session/i,
    })
    expect(deepResearchSession).not.toHaveClass('cursor-not-allowed')
    expect(deepResearchSession).toHaveAttribute('aria-disabled', 'false')

    await user.click(deepResearchSession)
    expect(onSelectSession).toHaveBeenCalledWith('session-1')
  })

  test('blocks switching when shallow thinking (WebSocket) is active', async () => {
    setupChatStoreMock({
      isStreaming: true,
      pendingInteraction: null,
    })

    const user = userEvent.setup()
    const onSelectSession = vi.fn()
    render(
      <SessionsPanel
        sessions={mockSessions}
        selectedSessionId="session-1"
        onSelectSession={onSelectSession}
      />
    )

    const session2 = screen.getByRole('button', {
      name: /session: idle session \(processing in progress\)/i,
    })
    expect(session2).toHaveClass('cursor-not-allowed')
    expect(session2).toHaveAttribute('aria-disabled', 'true')

    await user.click(session2)
    expect(onSelectSession).not.toHaveBeenCalled()
  })

  test('blocks switching while a HITL interaction is pending (navigation guard)', async () => {
    setupChatStoreMock({
      isStreaming: false,
      pendingInteraction: { id: 'p1', type: 'approval', content: 'Approve plan?' },
    })

    const user = userEvent.setup()
    const onSelectSession = vi.fn()
    render(
      <SessionsPanel
        sessions={mockSessions}
        selectedSessionId="session-1"
        onSelectSession={onSelectSession}
      />
    )

    const session2 = screen.getByRole('button', {
      name: /session: idle session/i,
    })
    expect(session2).toHaveClass('cursor-not-allowed')

    await user.click(session2)
    expect(onSelectSession).not.toHaveBeenCalled()
  })

  test('allows switching between sessions when nothing is active', async () => {
    setupChatStoreMock({
      isStreaming: false,
      pendingInteraction: null,
    })

    const user = userEvent.setup()
    const onSelectSession = vi.fn()
    render(
      <SessionsPanel
        sessions={mockSessions}
        selectedSessionId="session-1"
        onSelectSession={onSelectSession}
      />
    )

    const session2 = screen.getByRole('button', { name: /session: idle session/i })
    expect(session2).not.toHaveClass('cursor-not-allowed')

    await user.click(session2)
    expect(onSelectSession).toHaveBeenCalledWith('session-2')
  })
})

describe('SessionsPanel - New Session Button', () => {
  const mockSessions = [{ id: 'session-1', title: 'First Session', date: new Date() }]

  beforeEach(() => {
    vi.clearAllMocks()
    setupChatStoreMock()
    setupLayoutStoreMock()
  })

  test('disables new session button when shallow streaming is active', () => {
    setupChatStoreMock({ isStreaming: true })

    render(<SessionsPanel sessions={mockSessions} />)

    const newSessionBtn = screen.getByRole('button', {
      name: /start new session \(disabled during active operations\)/i,
    })
    expect(newSessionBtn).toBeDisabled()
  })

  test('disables new session button while a HITL interaction is pending (navigation guard)', () => {
    setupChatStoreMock({
      pendingInteraction: { id: 'p1', type: 'approval', content: 'Approve?' },
    })

    render(<SessionsPanel sessions={mockSessions} />)

    const newSessionBtn = screen.getByRole('button', {
      name: /start new session \(disabled during active operations\)/i,
    })
    expect(newSessionBtn).toBeDisabled()
  })

  test('enables new session button during active deep research (server-side)', () => {
    setupChatStoreMock({
      isSessionBusy: (sessionId: string) => sessionId === 'session-1',
      hasAnyBusySession: () => true,
      isStreaming: false,
      pendingInteraction: null,
    })

    render(<SessionsPanel sessions={mockSessions} />)

    const newSessionBtn = screen.getByRole('button', { name: /^start new session$/i })
    expect(newSessionBtn).not.toBeDisabled()
  })

  test('enables new session button when no active operations', () => {
    setupChatStoreMock({ isStreaming: false, pendingInteraction: null })

    render(<SessionsPanel sessions={mockSessions} />)

    const newSessionBtn = screen.getByRole('button', { name: /^start new session$/i })
    expect(newSessionBtn).not.toBeDisabled()
  })
})

describe('SessionsPanel - Delete Button States', () => {
  const mockSessions = [
    { id: 'session-1', title: 'First Session', date: new Date() },
    { id: 'session-2', title: 'Second Session', date: new Date() },
  ]

  beforeEach(() => {
    vi.clearAllMocks()
    setupChatStoreMock()
    setupLayoutStoreMock()
  })

  test('disables individual delete button when session has active deep research', async () => {
    setupChatStoreMock({
      isSessionBusy: (sessionId: string) => sessionId === 'session-1',
      hasAnyBusySession: () => true,
      isStreaming: false,
      pendingInteraction: null,
    })

    const user = userEvent.setup()
    render(<SessionsPanel sessions={mockSessions} />)

    const firstSession = screen.getByRole('button', { name: /session: first session/i })
    await user.hover(firstSession)

    const deleteButton = screen.getByRole('button', { name: /delete session \(disabled\)/i })
    expect(deleteButton).toBeDisabled()
  })

  test('disables individual delete button when shallow streaming is active (global block)', async () => {
    setupChatStoreMock({ isStreaming: true })

    const user = userEvent.setup()
    render(<SessionsPanel sessions={mockSessions} />)

    const firstSession = screen.getByRole('button', {
      name: /session: first session \(processing in progress\)/i,
    })
    await user.hover(firstSession)

    const deleteButton = screen.getByRole('button', { name: /delete session \(disabled\)/i })
    expect(deleteButton).toBeDisabled()
  })

  test('enables delete button when session is idle and no global block', async () => {
    setupChatStoreMock({
      isSessionBusy: () => false,
      hasAnyBusySession: () => false,
      isStreaming: false,
      pendingInteraction: null,
    })

    const user = userEvent.setup()
    render(<SessionsPanel sessions={mockSessions} />)

    const firstSession = screen.getByRole('button', { name: /session: first session/i })
    await user.hover(firstSession)

    const deleteButton = screen.getByRole('button', { name: /^delete session$/i })
    expect(deleteButton).not.toBeDisabled()
  })

  test('disables "Delete All" button when any session is busy', () => {
    setupChatStoreMock({
      isSessionBusy: () => false,
      hasAnyBusySession: () => true,
      isStreaming: false,
      pendingInteraction: null,
    })

    render(<SessionsPanel sessions={mockSessions} />)

    const deleteAllButton = screen.getByRole('button', {
      name: /delete all sessions \(disabled\)/i,
    })
    expect(deleteAllButton).toBeDisabled()
  })

  test('enables "Delete All" button when no sessions are busy', () => {
    setupChatStoreMock({
      isSessionBusy: () => false,
      hasAnyBusySession: () => false,
      isStreaming: false,
      pendingInteraction: null,
    })

    render(<SessionsPanel sessions={mockSessions} />)

    const deleteAllButton = screen.getByRole('button', { name: /^delete all sessions$/i })
    expect(deleteAllButton).not.toBeDisabled()
  })

  test('has appropriate title attribute on disabled delete button for active session', async () => {
    setupChatStoreMock({
      isSessionBusy: (sessionId: string) => sessionId === 'session-1',
      hasAnyBusySession: () => true,
      isStreaming: false,
      pendingInteraction: null,
    })

    const user = userEvent.setup()
    render(<SessionsPanel sessions={mockSessions} />)

    const firstSession = screen.getByRole('button', { name: /session: first session/i })
    await user.hover(firstSession)

    const deleteButton = screen.getByRole('button', { name: /delete session \(disabled\)/i })
    expect(deleteButton).toHaveAttribute('title', 'Cannot delete while operations are in progress')
  })
})
