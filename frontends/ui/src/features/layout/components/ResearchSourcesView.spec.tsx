// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { render, screen, within } from '@/test-utils'
import userEvent from '@testing-library/user-event'
import { vi, describe, test, expect, beforeEach } from 'vitest'
import { ResearchSourcesView } from './ResearchSourcesView'

interface MockCitation {
  id: string
  url: string
  content: string
  timestamp: Date
  isCited?: boolean
}

interface MockState {
  reportContent?: string
  deepResearchCitations: MockCitation[]
}

const defaultState: MockState = {
  reportContent: '',
  deepResearchCitations: [],
}

let mockState: MockState = { ...defaultState }

vi.mock('@/features/chat', () => ({
  useChatStore: vi.fn((selector?: (state: MockState) => unknown) => {
    return selector ? selector(mockState) : mockState
  }),
}))

const citedAndRead: MockCitation[] = [
  {
    id: 'read-source',
    url: 'https://read.example',
    content: 'Source read during research',
    timestamp: new Date('2026-05-01T00:00:00Z'),
    isCited: false,
  },
  {
    id: 'cited-source',
    url: 'https://cited.example',
    content: 'Source cited in final report',
    timestamp: new Date('2026-05-02T00:00:00Z'),
    isCited: true,
  },
]

describe('ResearchSourcesView', () => {
  beforeEach(() => {
    mockState = { ...defaultState }
  })

  test('shows an empty state when there are no sources', () => {
    render(<ResearchSourcesView />)

    expect(screen.getByText('Sources the agent reads will appear here.')).toBeInTheDocument()
    expect(screen.queryByRole('radio')).not.toBeInTheDocument()
  })

  test('renders the All / Cited filter over the merged list', () => {
    mockState = { ...defaultState, deepResearchCitations: citedAndRead }
    render(<ResearchSourcesView />)

    expect(screen.getAllByRole('radio').map((radio) => radio.textContent)).toEqual([
      'All',
      'Cited (1)',
    ])
  })

  test('shows both cited and uncited sources grouped in All mode', () => {
    mockState = { ...defaultState, deepResearchCitations: citedAndRead }
    render(<ResearchSourcesView />)

    expect(screen.getByText('Cited in report')).toBeInTheDocument()
    expect(screen.getByText('Other sources found')).toBeInTheDocument()
    expect(screen.getByText('cited.example')).toBeInTheDocument()
    expect(screen.getByText('read.example')).toBeInTheDocument()
  })

  test('filters to only cited sources on demand', async () => {
    const user = userEvent.setup()
    mockState = { ...defaultState, deepResearchCitations: citedAndRead }
    render(<ResearchSourcesView />)

    await user.click(screen.getByRole('radio', { name: /Cited \(1\)/i }))

    expect(screen.getByText('cited.example')).toBeInTheDocument()
    expect(screen.queryByText('read.example')).not.toBeInTheDocument()
    expect(screen.queryByText('Other sources found')).not.toBeInTheDocument()
  })

  test('shows a cited-specific empty state when nothing was cited', async () => {
    const user = userEvent.setup()
    mockState = {
      ...defaultState,
      deepResearchCitations: [citedAndRead[0]],
    }
    render(<ResearchSourcesView />)

    await user.click(screen.getByRole('radio', { name: /Cited \(0\)/i }))

    expect(screen.getByText('No sources were cited in the report.')).toBeInTheDocument()
    expect(
      screen.queryByText('Sources the agent reads will appear here.')
    ).not.toBeInTheDocument()
  })

  test('cited sources and their numbers come from the report [N], not discovery order', () => {
    mockState = {
      reportContent:
        'Body [1] and [2].\n\n## Sources\n[1] Report first: https://cited-b.example\n[2] Report second: https://cited-a.example',
      deepResearchCitations: [
        {
          id: 'a',
          url: 'https://cited-a.example',
          content: '',
          timestamp: new Date('2026-05-01T00:00:00Z'),
          isCited: true,
        },
        {
          id: 'b',
          url: 'https://cited-b.example',
          content: '',
          timestamp: new Date('2026-05-02T00:00:00Z'),
          isCited: true,
        },
        {
          id: 'read',
          url: 'https://read.example',
          content: '',
          timestamp: new Date('2026-05-03T00:00:00Z'),
          isCited: false,
        },
      ],
    }
    render(<ResearchSourcesView />)

    const citedRegion = screen.getByRole('region', { name: 'Cited in report' })
    const rows = within(citedRegion).getAllByRole('listitem')
    expect(rows).toHaveLength(2)
    expect(within(rows[0]).getByText('1')).toBeInTheDocument()
    expect(within(rows[0]).getByText('cited-b.example')).toBeInTheDocument()
    expect(within(rows[1]).getByText('2')).toBeInTheDocument()
    expect(within(rows[1]).getByText('cited-a.example')).toBeInTheDocument()

    const otherRegion = screen.getByRole('region', { name: 'Other sources found' })
    expect(within(otherRegion).getByText('read.example')).toBeInTheDocument()
    expect(within(otherRegion).queryByText('cited-a.example')).not.toBeInTheDocument()
    expect(within(otherRegion).queryByText('cited-b.example')).not.toBeInTheDocument()
  })

  test('numbers uncited sources after the highest report [N], even with gaps', () => {
    mockState = {
      reportContent:
        'Body [1] and [5].\n\n## Sources\n[1] Report first: https://cited-a.example\n[5] Report fifth: https://cited-b.example',
      deepResearchCitations: [
        {
          id: 'read',
          url: 'https://read.example',
          content: '',
          timestamp: new Date('2026-05-03T00:00:00Z'),
          isCited: false,
        },
      ],
    }
    render(<ResearchSourcesView />)

    const citedRegion = screen.getByRole('region', { name: 'Cited in report' })
    expect(within(citedRegion).getByText('1')).toBeInTheDocument()
    expect(within(citedRegion).getByText('5')).toBeInTheDocument()

    const otherRegion = screen.getByRole('region', { name: 'Other sources found' })
    const otherRows = within(otherRegion).getAllByRole('listitem')
    expect(otherRows).toHaveLength(1)
    expect(within(otherRows[0]).getByText('6')).toBeInTheDocument()
    expect(within(otherRows[0]).getByText('read.example')).toBeInTheDocument()
    expect(within(otherRegion).queryByText('1')).not.toBeInTheDocument()
    expect(within(otherRegion).queryByText('5')).not.toBeInTheDocument()
  })

  test('treats a citation that differs only by a trailing slash as cited', () => {
    mockState = {
      reportContent: 'Body [1].\n\n## Sources\n[1] Report page: https://example.com/page',
      deepResearchCitations: [
        {
          id: 'dup',
          url: 'https://example.com/page/',
          content: '',
          timestamp: new Date('2026-05-01T00:00:00Z'),
          isCited: false,
        },
      ],
    }
    render(<ResearchSourcesView />)

    const citedRegion = screen.getByRole('region', { name: 'Cited in report' })
    expect(within(citedRegion).getByText('example.com')).toBeInTheDocument()

    expect(screen.queryByText('Other sources found')).not.toBeInTheDocument()
    expect(screen.getByRole('radio', { name: /Cited \(1\)/i })).toBeInTheDocument()
    expect(screen.getAllByRole('listitem')).toHaveLength(1)
  })

  test('does not re-list a url-less cited document under Other sources found', () => {
    mockState = {
      reportContent: 'Body [1].\n\n## Sources\n[1] q3_report.pdf, p.4',
      deepResearchCitations: [
        {
          id: 'file-cite',
          url: '',
          content: 'q3_report.pdf, p.4',
          timestamp: new Date('2026-05-01T00:00:00Z'),
          isCited: true,
        },
      ],
    }
    render(<ResearchSourcesView />)

    const citedRegion = screen.getByRole('region', { name: 'Cited in report' })
    expect(within(citedRegion).getByText('q3_report.pdf, p.4')).toBeInTheDocument()

    expect(screen.queryByText('Other sources found')).not.toBeInTheDocument()
    expect(screen.getByRole('radio', { name: /Cited \(1\)/i })).toBeInTheDocument()
    expect(screen.getAllByRole('listitem')).toHaveLength(1)
  })
})
