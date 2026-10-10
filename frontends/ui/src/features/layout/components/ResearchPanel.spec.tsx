// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { render, screen, fireEvent, waitFor, act } from '@/test-utils'
import userEvent from '@testing-library/user-event'
import { vi, describe, test, expect, beforeEach } from 'vitest'
import { ResearchPanel, FALLBACK_WIDE_WIDTH } from './ResearchPanel'

const mockCloseRightPanel = vi.fn()
let mockRightPanel: string | null = 'research'

vi.mock('../store', () => ({
  useLayoutStore: vi.fn((selector?: (s: Record<string, unknown>) => unknown) => {
    const state = {
      rightPanel: mockRightPanel,
      closeRightPanel: mockCloseRightPanel,
    }
    return selector ? selector(state) : state
  }),
}))

vi.mock('@/adapters/auth', () => ({
  useAuth: vi.fn(() => ({ idToken: 'mock-token' })),
}))

vi.mock('@/adapters/api', () => ({
  cancelJob: vi.fn().mockResolvedValue(undefined),
}))

interface MockChatState {
  isDeepResearchStreaming: boolean
  deepResearchJobId: string | null
  reportContent: string
  deepResearchCitations: Array<{
    id: string
    url: string
    content: string
    timestamp: Date
    isCited?: boolean
  }>
  deepResearchAgents: unknown[]
  deepResearchTodos: unknown[]
}

const defaultChatState: MockChatState = {
  isDeepResearchStreaming: false,
  deepResearchJobId: null,
  reportContent: '',
  deepResearchCitations: [],
  deepResearchAgents: [],
  deepResearchTodos: [],
}

let mockChatState: MockChatState = { ...defaultChatState }
let mockIsLoadJobDataLoading = false
const mockLoadResearchPanelTab = vi.fn()
const mockImportStreamOnly = vi.fn()

vi.mock('@/features/chat', () => ({
  useChatStore: Object.assign(
    (selector: (s: MockChatState) => unknown) => selector(mockChatState),
    { getState: () => mockChatState }
  ),
  selectResolvedDeepResearchJobId: (s: MockChatState) => s.deepResearchJobId,
  useLoadJobData: () => ({
    loadResearchPanelTab: mockLoadResearchPanelTab,
    importStreamOnly: mockImportStreamOnly,
    isLoading: mockIsLoadJobDataLoading,
  }),
}))

vi.mock('./TasksTab', () => ({
  TasksTab: () => <div data-testid="tasks-tab">Tasks Tab Content</div>,
}))

vi.mock('./ThinkingTab', () => ({
  ThinkingTab: () => <div data-testid="thinking-tab">Thinking Tab Content</div>,
}))

vi.mock('./ReportTab', () => ({
  ReportTab: ({ children }: { children?: React.ReactNode }) => (
    <div data-testid="report-tab">Report Tab Content {children}</div>
  ),
}))

describe('ResearchPanel', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockRightPanel = 'research'
    mockChatState = { ...defaultChatState }
    mockIsLoadJobDataLoading = false
    mockLoadResearchPanelTab.mockResolvedValue(undefined)
    mockImportStreamOnly.mockResolvedValue(undefined)
  })

  describe('panel visibility', () => {
    test.each(['research', 'thinking', 'citations'] as const)(
      'is open for the %s rail panel',
      (panel) => {
        mockRightPanel = panel
        render(<ResearchPanel isAuthenticated={true} />)

        expect(screen.getByTestId('research-panel-close')).toBeInTheDocument()
        expect(screen.getByTestId('research-panel')).toHaveAttribute('aria-hidden', 'false')
      }
    )

    test.each(['data-sources', 'settings', null] as const)(
      'is hidden when rightPanel is %s',
      (panel) => {
        mockRightPanel = panel
        render(<ResearchPanel isAuthenticated={true} />)

        expect(screen.getByTestId('research-panel')).toHaveAttribute('aria-hidden', 'true')
      }
    )
  })

  describe('panel header', () => {
    test.each([
      ['research', 'Research'],
      ['thinking', 'Thinking'],
      ['citations', 'Citations'],
    ] as const)('shows the %s label in the header', (panel, label) => {
      mockRightPanel = panel
      render(<ResearchPanel isAuthenticated={true} />)

      expect(screen.getByText(label)).toBeInTheDocument()
    })

    test('does not render the former Tasks/Thinking/Report segmented control', () => {
      render(<ResearchPanel isAuthenticated={true} />)

      expect(screen.queryByRole('radio')).not.toBeInTheDocument()
    })

    test('does not render the former "Show Research" toggle tab', () => {
      render(<ResearchPanel isAuthenticated={true} />)

      expect(screen.queryByText('Show Research')).not.toBeInTheDocument()
      expect(screen.queryByTestId('research-panel-toggle')).not.toBeInTheDocument()
    })
  })

  describe('content wiring', () => {
    test('research shows the ReportTab (with export footer)', () => {
      mockRightPanel = 'research'
      render(<ResearchPanel isAuthenticated={true} />)

      expect(screen.getByTestId('report-tab')).toBeInTheDocument()
    })

    test('thinking shows the ThinkingTab', () => {
      mockRightPanel = 'thinking'
      render(<ResearchPanel isAuthenticated={true} />)

      expect(screen.getByTestId('thinking-tab')).toBeInTheDocument()
    })

    test('thinking folds workflow Task progress in when there is progress', async () => {
      mockRightPanel = 'thinking'
      mockChatState = { ...defaultChatState, deepResearchTodos: [{ id: 't1' }] }
      const user = userEvent.setup()
      render(<ResearchPanel isAuthenticated={true} />)

      const toggle = screen.getByText('Task progress')
      expect(toggle).toBeInTheDocument()
      expect(screen.queryByTestId('tasks-tab')).not.toBeInTheDocument()

      await user.click(toggle)
      expect(screen.getByTestId('tasks-tab')).toBeInTheDocument()
    })

    test('thinking hides Task progress when there is none', () => {
      mockRightPanel = 'thinking'
      render(<ResearchPanel isAuthenticated={true} />)

      expect(screen.queryByText('Task progress')).not.toBeInTheDocument()
    })

    test('citations shows an empty state when there are no sources', () => {
      mockRightPanel = 'citations'
      render(<ResearchPanel isAuthenticated={true} />)

      expect(screen.getByText('Sources the agent reads will appear here.')).toBeInTheDocument()
    })

    test('citations lists deep-research citations when present', () => {
      mockRightPanel = 'citations'
      mockChatState = {
        ...defaultChatState,
        deepResearchCitations: [
          {
            id: 'c1',
            url: 'https://example.com/report',
            content: 'A cited finding',
            timestamp: new Date('2026-05-01T00:00:00Z'),
          },
        ],
      }
      render(<ResearchPanel isAuthenticated={true} />)

      expect(screen.getByText('example.com')).toBeInTheDocument()
      expect(screen.queryByText('Sources the agent reads will appear here.')).not.toBeInTheDocument()
    })

    test('citations consolidates cited and uncited sources behind the All/Cited filter', () => {
      mockRightPanel = 'citations'
      mockChatState = {
        ...defaultChatState,
        deepResearchCitations: [
          {
            id: 'read',
            url: 'https://read.example',
            content: 'Read but not cited',
            timestamp: new Date('2026-05-01T00:00:00Z'),
            isCited: false,
          },
          {
            id: 'cited',
            url: 'https://cited.example',
            content: 'Cited in report',
            timestamp: new Date('2026-05-02T00:00:00Z'),
            isCited: true,
          },
        ],
      }
      render(<ResearchPanel isAuthenticated={true} />)

      expect(screen.getAllByRole('radio').map((radio) => radio.textContent)).toEqual([
        'All',
        'Cited (1)',
      ])
      expect(screen.getByText('Cited in report')).toBeInTheDocument()
      expect(screen.getByText('Other sources found')).toBeInTheDocument()
      expect(screen.getByText('cited.example')).toBeInTheDocument()
      expect(screen.getByText('read.example')).toBeInTheDocument()
    })

    test('does not render an artifacts panel', () => {
      mockRightPanel = 'artifacts'
      render(<ResearchPanel isAuthenticated={true} />)

      expect(screen.getByTestId('research-panel')).toHaveAttribute('aria-hidden', 'true')
    })
  })

  describe('close button', () => {
    test('calls closeRightPanel when clicked', async () => {
      const user = userEvent.setup()
      render(<ResearchPanel isAuthenticated={true} />)

      await user.click(screen.getByTestId('research-panel-close'))

      expect(mockCloseRightPanel).toHaveBeenCalled()
    })
  })

  describe('stop researching button', () => {
    test('is hidden when not streaming', () => {
      mockChatState = { ...defaultChatState, isDeepResearchStreaming: false }
      render(<ResearchPanel isAuthenticated={true} />)

      expect(screen.queryByTestId('research-panel-stop')).not.toBeInTheDocument()
    })

    test('is shown and enabled when streaming', () => {
      mockChatState = { ...defaultChatState, isDeepResearchStreaming: true }
      render(<ResearchPanel isAuthenticated={true} />)

      expect(screen.getByTestId('research-panel-stop')).toBeInTheDocument()
      expect(screen.getByTestId('research-panel-stop')).not.toBeDisabled()
    })

    test('renders a divider between stop and close only while streaming', () => {
      mockChatState = { ...defaultChatState, isDeepResearchStreaming: false }
      const { unmount } = render(<ResearchPanel isAuthenticated={true} />)
      expect(screen.queryByTestId('research-panel-header-divider')).not.toBeInTheDocument()
      unmount()

      mockChatState = { ...defaultChatState, isDeepResearchStreaming: true }
      render(<ResearchPanel isAuthenticated={true} />)
      expect(screen.getByTestId('research-panel-header-divider')).toBeInTheDocument()
    })
  })

  describe('resize grip', () => {
    const getPanelWidth = () =>
      (screen.getByTestId('research-panel') as HTMLElement).style.width

    test('exposes a resize separator only while the panel is open', () => {
      mockRightPanel = 'research'
      const { unmount } = render(<ResearchPanel isAuthenticated={true} />)
      expect(screen.getByRole('separator', { name: 'Resize panel' })).toBeInTheDocument()
      unmount()

      mockRightPanel = null
      render(<ResearchPanel isAuthenticated={true} />)
      expect(screen.queryByTestId('research-panel-resize')).not.toBeInTheDocument()
    })

    test('nudges the width with the arrow keys', () => {
      mockRightPanel = 'research'
      render(<ResearchPanel isAuthenticated={true} />)
      const grip = screen.getByRole('separator', { name: 'Resize panel' })

      const expectedDefault = document.createElement('div')
      expectedDefault.style.width = 'min(60%, 900px, 70vw)'
      expect(getPanelWidth()).toBe(expectedDefault.style.width)

      fireEvent.keyDown(grip, { key: 'ArrowLeft' })
      const widened = getPanelWidth()
      expect(widened).toMatch(/px$/)
      expect(parseInt(widened, 10)).toBeGreaterThan(480)
    })

    test('clamps to the minimum width on repeated narrowing', () => {
      mockRightPanel = 'research'
      render(<ResearchPanel isAuthenticated={true} />)
      const grip = screen.getByRole('separator', { name: 'Resize panel' })

      for (let i = 0; i < 20; i++) {
        fireEvent.keyDown(grip, { key: 'ArrowRight' })
      }

      expect(getPanelWidth()).toBe('480px')
    })

    test('clamps to the viewport-bounded maximum on repeated widening', () => {
      mockRightPanel = 'research'
      render(<ResearchPanel isAuthenticated={true} />)
      const grip = screen.getByRole('separator', { name: 'Resize panel' })

      for (let i = 0; i < 40; i++) {
        fireEvent.keyDown(grip, { key: 'ArrowLeft' })
      }

      const max = Math.min(900, Math.round(window.innerWidth * 0.7))
      expect(getPanelWidth()).toBe(`${max}px`)
    })

    test('tracks the pointer while dragging the grip', () => {
      mockRightPanel = 'research'
      render(<ResearchPanel isAuthenticated={true} />)
      const grip = screen.getByRole('separator', { name: 'Resize panel' })

      fireEvent.pointerDown(grip, { button: 0, clientX: 1000, pointerId: 1 })
      fireEvent.pointerMove(grip, { clientX: 1100, pointerId: 1 })
      fireEvent.pointerUp(grip, { clientX: 1100, pointerId: 1 })

      expect(getPanelWidth()).toBe(`${FALLBACK_WIDE_WIDTH - 100}px`)
    })

    test('pointercancel clears the resizing state so resize does not stay stuck', () => {
      mockRightPanel = 'research'
      render(<ResearchPanel isAuthenticated={true} />)
      const grip = screen.getByRole('separator', { name: 'Resize panel' })
      const panel = screen.getByTestId('research-panel')

      fireEvent.pointerDown(grip, { button: 0, clientX: 1000, pointerId: 1 })
      expect(panel).toHaveClass('select-none')

      fireEvent.pointerCancel(grip, { pointerId: 1 })
      expect(panel).not.toHaveClass('select-none')

      const widthAfterCancel = getPanelWidth()
      fireEvent.pointerMove(grip, { clientX: 1200, pointerId: 1 })
      expect(getPanelWidth()).toBe(widthAfterCancel)
    })

    test('exposes the resize range on the separator for assistive tech', () => {
      mockRightPanel = 'research'
      render(<ResearchPanel isAuthenticated={true} />)
      const grip = screen.getByRole('separator', { name: 'Resize panel' })

      expect(grip).toHaveAttribute('aria-valuemin')
      expect(grip).toHaveAttribute('aria-valuemax')
      expect(grip).toHaveAttribute('aria-valuenow')

      const min = Number(grip.getAttribute('aria-valuemin'))
      const max = Number(grip.getAttribute('aria-valuemax'))
      const now = Number(grip.getAttribute('aria-valuenow'))
      expect(min).toBeLessThan(max)
      expect(now).toBeGreaterThanOrEqual(min)
      expect(now).toBeLessThanOrEqual(max)
    })

    test('re-clamps a fixed panel width down to the new maximum when the viewport shrinks after a resize', () => {
      mockRightPanel = 'research'
      const originalInnerWidth = window.innerWidth
      Object.defineProperty(window, 'innerWidth', { configurable: true, value: 1600 })
      try {
        render(<ResearchPanel isAuthenticated={true} />)
        const grip = screen.getByRole('separator', { name: 'Resize panel' })

        fireEvent.pointerDown(grip, { button: 0, clientX: 1500, pointerId: 1 })
        fireEvent.pointerMove(grip, { clientX: 400, pointerId: 1 })
        fireEvent.pointerUp(grip, { clientX: 400, pointerId: 1 })

        const wideMax = Math.min(900, Math.round(1600 * 0.7))
        expect(getPanelWidth()).toBe(`${wideMax}px`)

        Object.defineProperty(window, 'innerWidth', { configurable: true, value: 800 })
        act(() => {
          window.dispatchEvent(new Event('resize'))
        })

        const narrowMax = Math.min(900, Math.round(800 * 0.7))
        expect(narrowMax).toBeLessThan(wideMax)
        expect(getPanelWidth()).toBe(`${narrowMax}px`)
      } finally {
        Object.defineProperty(window, 'innerWidth', {
          configurable: true,
          value: originalInnerWidth,
        })
      }
    })

    test('keeps the resize range valid when the viewport is narrower than the panel min', () => {
      mockRightPanel = 'research'
      const originalInnerWidth = window.innerWidth
      Object.defineProperty(window, 'innerWidth', { configurable: true, value: 600 })
      try {
        render(<ResearchPanel isAuthenticated={true} />)
        const grip = screen.getByRole('separator', { name: 'Resize panel' })

        const min = Number(grip.getAttribute('aria-valuemin'))
        const max = Number(grip.getAttribute('aria-valuemax'))
        const now = Number(grip.getAttribute('aria-valuenow'))
        expect(min).toBeLessThanOrEqual(max)
        expect(now).toBeGreaterThanOrEqual(min)
        expect(now).toBeLessThanOrEqual(max)
      } finally {
        Object.defineProperty(window, 'innerWidth', {
          configurable: true,
          value: originalInnerWidth,
        })
      }
    })
  })

  describe('lazy data loading', () => {
    test('loads the report when the research panel opens for a job', () => {
      mockRightPanel = 'research'
      mockChatState = { ...defaultChatState, deepResearchJobId: 'job-123' }
      render(<ResearchPanel isAuthenticated={true} />)

      expect(mockLoadResearchPanelTab).toHaveBeenCalledWith('job-123', 'report')
    })

    test('replays the stream for stream-backed panels', () => {
      mockRightPanel = 'thinking'
      mockChatState = { ...defaultChatState, deepResearchJobId: 'job-123' }
      render(<ResearchPanel isAuthenticated={true} />)

      expect(mockImportStreamOnly).toHaveBeenCalledWith('job-123')
    })

    test('loads once and does not re-attempt while the panel stays open, even across a loading toggle', async () => {
      mockRightPanel = 'research'
      mockChatState = { ...defaultChatState, deepResearchJobId: 'job-123' }

      const { rerender } = render(<ResearchPanel isAuthenticated={true}>a</ResearchPanel>)
      expect(mockLoadResearchPanelTab).toHaveBeenCalledTimes(1)

      await act(async () => {
        await new Promise((resolve) => setTimeout(resolve, 0))
      })

      mockIsLoadJobDataLoading = true
      rerender(<ResearchPanel isAuthenticated={true}>b</ResearchPanel>)
      mockIsLoadJobDataLoading = false
      rerender(<ResearchPanel isAuthenticated={true}>c</ResearchPanel>)

      await act(async () => {
        await new Promise((resolve) => setTimeout(resolve, 0))
      })

      expect(mockLoadResearchPanelTab).toHaveBeenCalledTimes(1)
    })

    test('reopening the panel for the same job re-loads (retry on fresh open)', async () => {
      mockRightPanel = 'research'
      mockChatState = { ...defaultChatState, deepResearchJobId: 'job-123' }

      const { rerender } = render(<ResearchPanel isAuthenticated={true}>a</ResearchPanel>)
      expect(mockLoadResearchPanelTab).toHaveBeenCalledTimes(1)

      mockRightPanel = null
      rerender(<ResearchPanel isAuthenticated={true}>b</ResearchPanel>)
      mockRightPanel = 'research'
      rerender(<ResearchPanel isAuthenticated={true}>c</ResearchPanel>)

      await waitFor(() => expect(mockLoadResearchPanelTab).toHaveBeenCalledTimes(2))
    })
  })
})
