// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { render, screen } from '@/test-utils'
import { vi, describe, test, expect } from 'vitest'
import { ReportTab } from './ReportTab'

// Mock the chat store
vi.mock('@/features/chat', () => ({
  useChatStore: vi.fn((selector?: (s: any) => any) => {
    const state = {
      reportContent: '',
      isStreaming: false,
      currentStatus: null,
    }
    return selector ? selector(state) : state
  }),
  selectResolvedDeepResearchJobId: (state: any) => state?.deepResearchJobId,
}))

// Mock MarkdownRenderer
vi.mock('@/shared/components/MarkdownRenderer', () => ({
  MarkdownRenderer: ({
    content,
    isStreaming,
    sources,
  }: {
    content: string
    isStreaming?: boolean
    sources?: Array<{ index: number; url?: string }>
  }) => (
    <div
      data-testid="markdown"
      data-streaming={isStreaming}
      data-sources={JSON.stringify((sources ?? []).map((s) => ({ index: s.index, url: s.url ?? null })))}
    >
      {content}
      {isStreaming && <span data-testid="streaming-indicator">Generating report...</span>}
    </div>
  ),
}))

// Mock ExportFooter
vi.mock('./ExportFooter', () => ({
  ExportFooter: () => <div data-testid="export-footer">Export Footer</div>,
}))

// Mock SourceStrip so the structured-citations path renders hermetically
vi.mock('@/shared/components/Sources/SourceStrip', () => ({
  SourceStrip: ({ sources }: { sources: unknown[] }) => (
    <div data-testid="source-strip">{sources.length} sources</div>
  ),
}))

vi.mock('./FileCard', () => ({
  FileCard: ({ file }: { file: { name: string } }) => (
    <div data-testid="file-card">{file.name}</div>
  ),
}))

import userEvent from '@testing-library/user-event'
import { useChatStore } from '@/features/chat'

describe('ReportTab', () => {
  test('displays empty state when no report content', () => {
    render(<ReportTab />)

    expect(screen.getByText(/report content will appear here/i)).toBeInTheDocument()
    // Icon is rendered as SVG, verify by checking the document icon is present
    expect(document.querySelector('svg')).toBeInTheDocument()
  })

  test('shows an in-progress state when empty while deep research is streaming', () => {
    vi.mocked(useChatStore).mockImplementation((selector?: (s: any) => any) => {
      const state = {
        reportContent: '',
        isStreaming: true,
        currentStatus: 'researching',
        isDeepResearchStreaming: true,
      }
      return selector ? selector(state) : state
    })

    render(<ReportTab />)

    expect(screen.getByText(/deep research in progress/i)).toBeInTheDocument()
    expect(screen.queryByText(/report content will appear here/i)).not.toBeInTheDocument()
  })

  test('shows the static empty state when empty and not streaming', () => {
    vi.mocked(useChatStore).mockImplementation((selector?: (s: any) => any) => {
      const state = {
        reportContent: '',
        isStreaming: false,
        currentStatus: null,
        isDeepResearchStreaming: false,
      }
      return selector ? selector(state) : state
    })

    render(<ReportTab />)

    expect(screen.getByText(/report content will appear here/i)).toBeInTheDocument()
    expect(screen.queryByText(/deep research in progress/i)).not.toBeInTheDocument()
  })

  test('renders report content via MarkdownRenderer', () => {
    vi.mocked(useChatStore).mockImplementation((selector?: (s: any) => any) => {
      const state = {
        reportContent: '# Report Title\n\nReport content here',
        isStreaming: false,
        currentStatus: null,
      }
      return selector ? selector(state) : state
    })

    render(<ReportTab />)

    expect(screen.getByTestId('markdown')).toHaveTextContent('# Report Title')
  })

  test('renders title when provided', () => {
    vi.mocked(useChatStore).mockImplementation((selector?: (s: any) => any) => {
      const state = {
        reportContent: 'Some content',
        isStreaming: false,
        currentStatus: null,
      }
      return selector ? selector(state) : state
    })

    render(<ReportTab />)

    expect(screen.getByText('Some content')).toBeInTheDocument()
  })

  test('shows generating indicator when streaming and writing', () => {
    vi.mocked(useChatStore).mockImplementation((selector?: (s: any) => any) => {
      const state = {
        reportContent: 'Partial content...',
        isStreaming: true,
        currentStatus: 'writing',
      }
      return selector ? selector(state) : state
    })

    render(<ReportTab />)

    // Check that MarkdownRenderer receives isStreaming prop and shows indicator
    expect(screen.getByTestId('streaming-indicator')).toBeInTheDocument()
    expect(screen.getByText('Generating report...')).toBeInTheDocument()
  })

  test('renders children when provided', () => {
    render(
      <ReportTab>
        <div>Custom content</div>
      </ReportTab>
    )

    expect(screen.getByText('Custom content')).toBeInTheDocument()
    expect(screen.queryByTestId('markdown')).not.toBeInTheDocument()
  })

  test('always renders export footer', () => {
    render(<ReportTab />)

    expect(screen.getByTestId('export-footer')).toBeInTheDocument()
  })

  test('uses the report own parsed references and strips the block even without discovery citations', () => {
    vi.mocked(useChatStore).mockImplementation((selector?: (s: any) => any) => {
      const state = {
        reportContent:
          'The sky is blue [1].\n\n## Sources\n- [1] Sky facts - https://example.com/sky',
        isStreaming: false,
        currentStatus: null,
        deepResearchCitations: [],
      }
      return selector ? selector(state) : state
    })

    render(<ReportTab />)

    const markdown = screen.getByTestId('markdown')
    expect(markdown).not.toHaveTextContent('example.com')
    expect(screen.getByTestId('source-strip')).toBeInTheDocument()
    const sources = JSON.parse(markdown.getAttribute('data-sources') ?? '[]')
    expect(sources).toEqual([{ index: 1, url: 'https://example.com/sky' }])
  })

  test('keeps the body intact when the trailing block is unparseable and there are no discovery citations', () => {
    vi.mocked(useChatStore).mockImplementation((selector?: (s: any) => any) => {
      const state = {
        reportContent: 'The sky is blue.\n\n## Sources\nInternal notes without markers',
        isStreaming: false,
        currentStatus: null,
        deepResearchCitations: [],
      }
      return selector ? selector(state) : state
    })

    render(<ReportTab />)

    const markdown = screen.getByTestId('markdown')
    expect(markdown).toHaveTextContent('Internal notes without markers')
    expect(screen.queryByTestId('source-strip')).not.toBeInTheDocument()
  })

  test('resolves each [N] marker to the source the report assigned, not discovery order', () => {
    vi.mocked(useChatStore).mockImplementation((selector?: (s: any) => any) => {
      const state = {
        reportContent:
          'Alpha [1] and Beta [2].\n\n## Sources\n- [1] Beta source - https://beta.example.com\n- [2] Alpha source - https://alpha.example.com',
        isStreaming: false,
        currentStatus: null,
        deepResearchCitations: [
          { id: 'd1', url: 'https://alpha.example.com', content: 'Alpha source', isCited: true },
          { id: 'd2', url: 'https://beta.example.com', content: 'Beta source', isCited: true },
        ],
      }
      return selector ? selector(state) : state
    })

    render(<ReportTab />)

    const sources = JSON.parse(screen.getByTestId('markdown').getAttribute('data-sources') ?? '[]')
    expect(sources).toEqual([
      { index: 1, url: 'https://beta.example.com' },
      { index: 2, url: 'https://alpha.example.com' },
    ])
  })

  test('renders gracefully across a gap in the report [N] numbering', () => {
    vi.mocked(useChatStore).mockImplementation((selector?: (s: any) => any) => {
      const state = {
        reportContent:
          'First [1] and third [3].\n\n## Sources\n- [1] First - https://one.example.com\n- [3] Third - https://three.example.com',
        isStreaming: false,
        currentStatus: null,
        deepResearchCitations: [],
      }
      return selector ? selector(state) : state
    })

    render(<ReportTab />)

    const sources = JSON.parse(screen.getByTestId('markdown').getAttribute('data-sources') ?? '[]')
    expect(sources).toEqual([
      { index: 1, url: 'https://one.example.com' },
      { index: 3, url: 'https://three.example.com' },
    ])
  })

  test('resolves the dense sequential case where report order matches discovery', () => {
    vi.mocked(useChatStore).mockImplementation((selector?: (s: any) => any) => {
      const state = {
        reportContent:
          'A [1], B [2], C [3].\n\n## Sources\n- [1] First - https://one.example.com\n- [2] Second - https://two.example.com\n- [3] Third - https://three.example.com',
        isStreaming: false,
        currentStatus: null,
        deepResearchCitations: [],
      }
      return selector ? selector(state) : state
    })

    render(<ReportTab />)

    const sources = JSON.parse(screen.getByTestId('markdown').getAttribute('data-sources') ?? '[]')
    expect(sources).toEqual([
      { index: 1, url: 'https://one.example.com' },
      { index: 2, url: 'https://two.example.com' },
      { index: 3, url: 'https://three.example.com' },
    ])
  })

  test('falls back to discovery citations keyed by marker number when the report has no structured block', () => {
    vi.mocked(useChatStore).mockImplementation((selector?: (s: any) => any) => {
      const state = {
        reportContent: 'Fact one [1]. Fact two [2].',
        isStreaming: false,
        currentStatus: null,
        deepResearchCitations: [
          { id: 'd1', url: 'https://one.example.com', content: 'One', isCited: false },
          { id: 'd2', url: 'https://two.example.com', content: 'Two', isCited: true },
        ],
      }
      return selector ? selector(state) : state
    })

    render(<ReportTab />)

    const sources = JSON.parse(screen.getByTestId('markdown').getAttribute('data-sources') ?? '[]')
    expect(sources).toEqual([
      { index: 1, url: 'https://one.example.com' },
      { index: 2, url: 'https://two.example.com' },
    ])
  })

  test('strips the redundant markdown references when structured citations exist', () => {
    vi.mocked(useChatStore).mockImplementation((selector?: (s: any) => any) => {
      const state = {
        reportContent:
          'The sky is blue [1].\n\n## Sources\n- [1] Sky facts - https://example.com/sky',
        isStreaming: false,
        currentStatus: null,
        deepResearchCitations: [
          { id: 'c1', url: 'https://example.com/sky', content: 'Sky facts', isCited: true },
        ],
      }
      return selector ? selector(state) : state
    })

    render(<ReportTab />)

    const markdown = screen.getByTestId('markdown')
    expect(markdown).not.toHaveTextContent('example.com')
    expect(screen.getByTestId('source-strip')).toBeInTheDocument()
  })

  test('bounds the expanded generated-files list with a scroll area', async () => {
    const user = userEvent.setup()
    vi.mocked(useChatStore).mockImplementation((selector?: (s: any) => any) => {
      const state = {
        reportContent: 'Report body',
        isStreaming: false,
        currentStatus: null,
        deepResearchFiles: [
          { id: 'f1', name: 'a.md' },
          { id: 'f2', name: 'b.md' },
        ],
      }
      return selector ? selector(state) : state
    })

    render(<ReportTab />)

    await user.click(screen.getByRole('button', { name: /generated files/i }))

    const list = screen.getAllByTestId('file-card')[0].parentElement?.parentElement
    expect(list?.className).toMatch(/overflow-y-auto/)
    expect(list?.className).toMatch(/max-h-/)
  })
})
