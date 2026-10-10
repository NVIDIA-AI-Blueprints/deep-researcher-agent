// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { render, screen } from '@/test-utils'
import userEvent from '@testing-library/user-event'
import { vi, describe, test, expect, beforeEach } from 'vitest'
import { ThinkingTab } from './ThinkingTab'

interface MockFile {
  id: string
  filename: string
  content: string
}

interface MockState {
  deepResearchFiles: MockFile[]
}

const defaultState: MockState = {
  deepResearchFiles: [],
}

let mockState: MockState = { ...defaultState }

vi.mock('@/features/chat', () => ({
  useChatStore: vi.fn((selector?: (state: MockState) => unknown) => {
    return selector ? selector(mockState) : mockState
  }),
}))

vi.mock('./AgentsTab', () => ({
  AgentsTab: () => <div data-testid="agents-tab">Steps Content</div>,
}))

vi.mock('./FileCard', () => ({
  FileCard: ({ file }: { file: MockFile }) => (
    <div data-testid="file-card">{file.filename}</div>
  ),
}))

describe('ThinkingTab', () => {
  beforeEach(() => {
    mockState = { ...defaultState }
  })

  test('renders the steps trace directly', () => {
    render(<ThinkingTab />)

    expect(screen.getByTestId('agents-tab')).toBeInTheDocument()
  })

  test('no longer renders a Steps/Sources sub-tab bar', () => {
    render(<ThinkingTab />)

    expect(screen.queryByRole('radio')).not.toBeInTheDocument()
    expect(screen.queryByRole('radio', { name: /Sources/i })).not.toBeInTheDocument()
  })

  test('does not render any source list (sources live in the Citations panel)', () => {
    render(<ThinkingTab />)

    expect(screen.queryByText('Cited in report')).not.toBeInTheDocument()
    expect(screen.queryByText('Other sources found')).not.toBeInTheDocument()
  })

  test('folds generated files into a thin disclosure only when files exist', async () => {
    const user = userEvent.setup()
    mockState = {
      ...defaultState,
      deepResearchFiles: [{ id: 'f1', filename: 'report.md', content: 'body' }],
    }

    render(<ThinkingTab />)

    expect(screen.getByText('Generated files (1)')).toBeInTheDocument()
    expect(screen.queryByTestId('file-card')).not.toBeInTheDocument()

    await user.click(screen.getByText('Generated files (1)'))
    expect(screen.getByTestId('file-card')).toBeInTheDocument()
  })

  test('hides the generated-files section when there are no files', () => {
    render(<ThinkingTab />)

    expect(screen.queryByText(/Generated files/i)).not.toBeInTheDocument()
  })
})
