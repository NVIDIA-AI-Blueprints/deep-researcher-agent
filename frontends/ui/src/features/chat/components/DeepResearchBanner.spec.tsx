// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { render, screen } from '@/test-utils'
import userEvent from '@testing-library/user-event'
import { describe, expect, test, vi } from 'vitest'
import { DeepResearchBanner } from './DeepResearchBanner'

const mockOpenRightPanel = vi.fn()
const mockSetResearchPanelTab = vi.fn()
const mockLoadResearchPanelTab = vi.fn()

vi.mock('@/features/layout/store', () => ({
  useLayoutStore: vi.fn(
    (
      selector?: (s: {
        openRightPanel: typeof mockOpenRightPanel
        setResearchPanelTab: typeof mockSetResearchPanelTab
      }) => unknown
    ) => {
      const state = {
        openRightPanel: mockOpenRightPanel,
        setResearchPanelTab: mockSetResearchPanelTab,
      }
      return selector ? selector(state) : state
    }
  ),
}))

vi.mock('../hooks/use-load-job-data', () => ({
  useLoadJobData: () => ({
    loadResearchPanelTab: mockLoadResearchPanelTab,
  }),
}))

describe('DeepResearchBanner', () => {
  test('shows only the View Report banner action', () => {
    render(<DeepResearchBanner bannerType="success" jobId="job-1" />)

    expect(screen.getByRole('button', { name: 'View Report' })).toBeInTheDocument()
  })

  test('clicking the completed banner action loads the research report and does not open Thinking', async () => {
    const user = userEvent.setup()
    render(<DeepResearchBanner bannerType="success" jobId="job-1" />)

    await user.click(screen.getByRole('button', { name: 'View Report' }))

    expect(mockLoadResearchPanelTab).toHaveBeenCalledWith('job-1', 'report')
    expect(mockOpenRightPanel).not.toHaveBeenCalledWith('thinking')
  })

  test.each([
    ['starting', 'View Progress'],
    ['failure', 'View Thinking'],
    ['cancelled', 'View Progress'],
  ] as const)('hides the %s banner action', (bannerType, actionLabel) => {
    render(<DeepResearchBanner bannerType={bannerType} jobId="job-1" />)

    expect(screen.queryByRole('button', { name: actionLabel })).not.toBeInTheDocument()
  })

  test('renders an expired report warning without an action', () => {
    render(<DeepResearchBanner bannerType="expired" jobId="job-1" />)

    expect(screen.getByText('Report Expired')).toBeInTheDocument()
    expect(
      screen.getByText(/The report has expired and is no longer available/i)
    ).toBeInTheDocument()
    expect(screen.queryByRole('button')).not.toBeInTheDocument()
  })
})
