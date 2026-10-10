// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { render, screen } from '@/test-utils'
import userEvent from '@testing-library/user-event'
import { vi, describe, test, expect, beforeEach } from 'vitest'
import { DeepResearchRail } from './DeepResearchRail'

const mockOpenRightPanel = vi.fn()
const mockCloseRightPanel = vi.fn()
let mockRightPanel: string | null = null

const railState = () => ({
  rightPanel: mockRightPanel,
  openRightPanel: mockOpenRightPanel,
  closeRightPanel: mockCloseRightPanel,
})

vi.mock('../store', () => ({
  useLayoutStore: Object.assign(
    vi.fn((selector?: (s: ReturnType<typeof railState>) => unknown) => {
      const state = railState()
      return selector ? selector(state) : state
    }),
    { getState: () => railState() }
  ),
}))

let mockIsDeepResearchStreaming = false

vi.mock('@/features/chat', () => ({
  useChatStore: (selector: (s: { isDeepResearchStreaming: boolean }) => unknown) =>
    selector({ isDeepResearchStreaming: mockIsDeepResearchStreaming }),
}))

const ITEM_LABELS = ['Data Sources', 'Citations', 'Research', 'Thinking']

describe('DeepResearchRail', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockRightPanel = null
    mockIsDeepResearchStreaming = false
  })

  test('renders the header and all four nav items', () => {
    render(<DeepResearchRail isAuthenticated={true} />)

    expect(screen.getByText('Deep Research')).toBeInTheDocument()
    for (const label of ITEM_LABELS) {
      expect(screen.getByRole('button', { name: label })).toBeInTheDocument()
    }
  })

  test('does not render an Artifacts nav item', () => {
    render(<DeepResearchRail isAuthenticated={true} />)

    expect(screen.queryByRole('button', { name: 'Artifacts' })).not.toBeInTheDocument()
  })

  test('renders items in order with Thinking pinned at the bottom', () => {
    render(<DeepResearchRail isAuthenticated={true} />)

    const buttons = screen.getAllByRole('button').map((b) => b.getAttribute('aria-label'))
    expect(buttons).toEqual(ITEM_LABELS)
    expect(buttons[buttons.length - 1]).toBe('Thinking')
  })

  test('opens the matching panel when an item is clicked', async () => {
    const user = userEvent.setup()
    render(<DeepResearchRail isAuthenticated={true} />)

    await user.click(screen.getByRole('button', { name: 'Citations' }))
    expect(mockOpenRightPanel).toHaveBeenCalledWith('citations')

    await user.click(screen.getByRole('button', { name: 'Thinking' }))
    expect(mockOpenRightPanel).toHaveBeenCalledWith('thinking')
  })

  test('highlights the active item', () => {
    mockRightPanel = 'research'
    render(<DeepResearchRail isAuthenticated={true} />)

    expect(screen.getByRole('button', { name: 'Research' })).toHaveAttribute('aria-pressed', 'true')
    expect(screen.getByRole('button', { name: 'Citations' })).toHaveAttribute(
      'aria-pressed',
      'false'
    )
  })

  test('clicking the active item closes its panel (toggle)', async () => {
    mockRightPanel = 'research'
    const user = userEvent.setup()
    render(<DeepResearchRail isAuthenticated={true} />)

    await user.click(screen.getByRole('button', { name: 'Research' }))

    expect(mockCloseRightPanel).toHaveBeenCalled()
    expect(mockOpenRightPanel).not.toHaveBeenCalled()
  })

  test('disables items and does nothing when unauthenticated', async () => {
    const user = userEvent.setup()
    render(<DeepResearchRail isAuthenticated={false} />)

    const item = screen.getByRole('button', { name: 'Data Sources' })
    expect(item).toBeDisabled()

    await user.click(item)
    expect(mockOpenRightPanel).not.toHaveBeenCalled()
    expect(mockCloseRightPanel).not.toHaveBeenCalled()
  })

  test('enabled nav buttons use a pointer cursor', () => {
    render(<DeepResearchRail isAuthenticated={true} />)

    expect(screen.getByRole('button', { name: 'Research' })).toHaveClass('cursor-pointer')
  })

  test('disabled nav buttons keep the not-allowed cursor and are not pointer', () => {
    render(<DeepResearchRail isAuthenticated={false} />)

    const item = screen.getByRole('button', { name: 'Research' })
    expect(item).toHaveClass('cursor-not-allowed')
    expect(item).not.toHaveClass('cursor-pointer')
  })

  describe('deep research activity indicator', () => {
    test('shows a live activity indicator while deep research is streaming', () => {
      mockIsDeepResearchStreaming = true
      render(<DeepResearchRail isAuthenticated={true} />)

      expect(screen.getByTestId('deep-research-activity')).toBeInTheDocument()
    })

    test('hides the activity indicator when not streaming', () => {
      mockIsDeepResearchStreaming = false
      render(<DeepResearchRail isAuthenticated={true} />)

      expect(screen.queryByTestId('deep-research-activity')).not.toBeInTheDocument()
    })
  })
})
