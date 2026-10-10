// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { render, screen } from '@/test-utils'
import userEvent from '@testing-library/user-event'
import { vi, describe, test, expect, beforeEach } from 'vitest'
import { AgentResponse } from './AgentResponse'

const mockOpenRightPanel = vi.fn()
const mockSetResearchPanelTab = vi.fn()

vi.mock('@/features/layout/store', () => ({
  useLayoutStore: vi.fn((selector?: (s: any) => any) => {
    const state = {
      openRightPanel: mockOpenRightPanel,
      setResearchPanelTab: mockSetResearchPanelTab,
    }
    return selector ? selector(state) : state
  }),
}))

const chatState = vi.hoisted(() => ({
  reportContent: '',
  deepResearchJobId: null as string | null,
  isDeepResearchStreaming: false,
  deepResearchStreamLoaded: false,
  currentConversation: null,
  patchConversationMessage: vi.fn(),
  reconnectToActiveJob: vi.fn(),
}))

vi.mock('../store', () => ({
  useChatStore: vi.fn((selector?: (s: any) => any) =>
    selector ? selector(chatState) : chatState
  ),
}))

vi.mock('@/adapters/api', () => ({
  cancelJob: vi.fn(),
}))

vi.mock('@/adapters/auth', () => ({
  useAuth: () => ({
    accessToken: null,
  }),
}))

const mockImportJobStream = vi.fn()
const mockLoadResearchPanelTab = vi.fn()
const hookState = vi.hoisted(() => ({ error: null as string | null, isLoading: false }))

vi.mock('../hooks', () => ({
  useLoadJobData: () => ({
    loadReport: vi.fn(),
    importJobStream: mockImportJobStream,
    loadResearchPanelTab: mockLoadResearchPanelTab,
    isLoading: hookState.isLoading,
    error: hookState.error,
    clearError: vi.fn(),
  }),
}))

vi.mock('@/shared/components/MarkdownRenderer', () => ({
  MarkdownRenderer: ({ content, className }: { content: string; className?: string }) => (
    <span
      data-testid="markdown-body"
      className={['markdown-content', className].filter(Boolean).join(' ')}
    >
      {content}
    </span>
  ),
}))

describe('AgentResponse', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockImportJobStream.mockClear()
    mockLoadResearchPanelTab.mockClear()
    hookState.error = null
    hookState.isLoading = false
    chatState.reportContent = ''
    chatState.deepResearchJobId = null
    chatState.isDeepResearchStreaming = false
    chatState.deepResearchStreamLoaded = false
  })

  test('renders response content', () => {
    render(<AgentResponse content="Here is your answer" />)

    expect(screen.getByText('Here is your answer')).toBeInTheDocument()
  })

  test('returns null for empty content', () => {
    render(<AgentResponse content="" />)

    expect(screen.queryByTestId('markdown-body')).not.toBeInTheDocument()
  })

  test('returns null for whitespace-only content', () => {
    render(<AgentResponse content="   " />)

    expect(screen.queryByTestId('markdown-body')).not.toBeInTheDocument()
  })

  test('displays timestamp when provided', () => {
    const timestamp = new Date('2024-01-15T14:30:00')

    render(<AgentResponse content="Response" timestamp={timestamp} />)

    expect(screen.getByText(/\d{1,2}:\d{2}/)).toBeInTheDocument()
  })

  test('handles ISO string timestamp', () => {
    render(<AgentResponse content="Response" timestamp="2024-01-15T14:30:00Z" />)

    expect(screen.getByText(/\d{1,2}:\d{2}/)).toBeInTheDocument()
  })

  test('shows "View Report" button when showViewReport is true', () => {
    render(<AgentResponse content="Response" showViewReport={true} />)

    expect(screen.getByRole('button', { name: 'View Report' })).toBeInTheDocument()
  })

  test('hides "View Report" button when showViewReport is false', () => {
    render(<AgentResponse content="Response" showViewReport={false} />)

    expect(screen.queryByRole('button', { name: 'View Report' })).not.toBeInTheDocument()
  })

  test('clicking "View Report" opens research panel with report tab', async () => {
    const user = userEvent.setup()

    render(<AgentResponse content="Response" showViewReport={true} />)

    await user.click(screen.getByRole('button', { name: 'View Report' }))

    expect(mockSetResearchPanelTab).toHaveBeenCalledWith('report')
    expect(mockOpenRightPanel).toHaveBeenCalledWith('research')
  })

  test('renders without timestamp', () => {
    render(<AgentResponse content="Response without timestamp" />)

    expect(screen.getByText('Response without timestamp')).toBeInTheDocument()
    expect(screen.queryByText(/\d{1,2}:\d{2}/)).not.toBeInTheDocument()
  })

  test('renders long content', () => {
    const longContent = 'This is a very long response. '.repeat(50)

    const { container } = render(<AgentResponse content={longContent} />)

    expect(container.textContent).toContain('This is a very long response.')
  })

  test('uses shared research panel loader when clicking "View Report" with jobId', async () => {
    const user = userEvent.setup()

    render(<AgentResponse content="Response" showViewReport={true} jobId="test-job-123" />)

    await user.click(screen.getByRole('button', { name: 'View Report' }))

    expect(mockLoadResearchPanelTab).toHaveBeenCalledWith('test-job-123', 'report')
    expect(mockImportJobStream).not.toHaveBeenCalled()
  })

  test('surfaces a report-load error and retries loading the report', async () => {
    const user = userEvent.setup()
    hookState.error = 'network timeout'

    render(<AgentResponse content="Response" showViewReport={true} jobId="job-1" />)

    const alert = screen.getByRole('alert')
    expect(alert).toHaveTextContent('network timeout')
    await user.click(screen.getByRole('button', { name: /retry/i }))
    expect(mockLoadResearchPanelTab).toHaveBeenCalledWith('job-1', 'report')
  })

  test('disables a completed job View Report while another job is streaming, instead of routing to it', async () => {
    const user = userEvent.setup()
    chatState.isDeepResearchStreaming = true
    chatState.deepResearchJobId = 'other-job'

    render(<AgentResponse content="Old answer" jobId="job-A" deepResearchJobStatus="success" />)

    const button = screen.getByRole('button', {
      name: /available once the running research job finishes/i,
    })
    expect(button).toBeDisabled()

    await user.click(button)

    expect(mockSetResearchPanelTab).not.toHaveBeenCalledWith('tasks')
    expect(mockOpenRightPanel).not.toHaveBeenCalled()
  })

  test('clicking "View Progress" for an active job opens the Thinking panel, not the report placeholder', async () => {
    const user = userEvent.setup()

    render(<AgentResponse content="Working on it" jobId="job-active" isDeepResearchActive={true} />)

    await user.click(screen.getByRole('button', { name: 'View Progress' }))

    expect(mockOpenRightPanel).toHaveBeenCalledWith('thinking')
    expect(mockOpenRightPanel).not.toHaveBeenCalledWith('research')
    expect(mockSetResearchPanelTab).not.toHaveBeenCalled()
  })

  test('clicking "View Report" for a completed job loads the research report, not Thinking', async () => {
    const user = userEvent.setup()

    render(<AgentResponse content="Done" jobId="job-done" deepResearchJobStatus="success" />)

    await user.click(screen.getByRole('button', { name: 'View Report' }))

    expect(mockLoadResearchPanelTab).toHaveBeenCalledWith('job-done', 'report')
    expect(mockOpenRightPanel).not.toHaveBeenCalledWith('thinking')
  })

  test('strips a baked references block from the rendered body and lists sources', () => {
    const content =
      'NVIDIA shipped record volume [1].\n\n**References:**\n- [1] NVIDIA Q4 results - https://www.nvidia.com/news'

    render(<AgentResponse content={content} />)

    const body = screen.getByTestId('markdown-body')
    expect(body).toHaveTextContent('NVIDIA shipped record volume [1].')
    expect(body).not.toHaveTextContent('References:')
    expect(screen.getByRole('region', { name: 'Sources' })).toBeInTheDocument()
    expect(screen.getByText('nvidia.com')).toBeInTheDocument()
  })

  test('renders no Sources list when there is no references block', () => {
    render(<AgentResponse content="Just a plain answer." />)

    expect(screen.queryByRole('region', { name: 'Sources' })).not.toBeInTheDocument()
  })

  test('keeps a blank-line-separated bullet as a separate top-level list, not a child of the numbered item', () => {
    const content = '1. First\n2. Second\n\n- Separate topic'

    render(<AgentResponse content={content} />)

    const body = screen.getByTestId('markdown-body')
    expect(body.textContent).toContain('\n- Separate topic')
    expect(body.textContent).not.toContain('   - Separate topic')
  })

  test('leaves two blank-line-separated ordered lists with their own numbering', () => {
    const content = '1. First\n2. Second\n\n1. Fresh start\n2. Next'

    render(<AgentResponse content={content} />)

    const body = screen.getByTestId('markdown-body')
    expect(body.textContent).toContain('\n1. Fresh start')
    expect(body.textContent).not.toContain('3. Fresh start')
  })

  test('Copy action copies the original full content including references', async () => {
    const user = userEvent.setup()
    const content =
      'Answer body [1].\n\n**References:**\n- [1] NVIDIA Q4 results - https://www.nvidia.com/news'

    render(<AgentResponse content={content} />)

    await user.click(screen.getByRole('button', { name: 'Copy answer' }))

    expect(await navigator.clipboard.readText()).toBe(content)
  })
})

describe('AgentResponse answer reveal animation', () => {
  test('answer markdown wrapper opts into the staggered reveal (default variant)', () => {
    const { container } = render(<AgentResponse content="Here is your answer" />)
    expect(container.querySelector('.markdown-content.answer-reveal')).toBeTruthy()
  })

  test('answer markdown wrapper opts into the staggered reveal (inline variant)', () => {
    const { container } = render(<AgentResponse content="Here is your answer" variant="inline" />)
    expect(container.querySelector('.markdown-content.answer-reveal')).toBeTruthy()
  })
})
