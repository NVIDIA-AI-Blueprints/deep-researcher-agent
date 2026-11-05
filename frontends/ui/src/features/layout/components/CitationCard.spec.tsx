// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { render, screen } from '@/test-utils'
import { vi, describe, test, expect, beforeEach } from 'vitest'
import { CitationCard, getConfidenceBadge } from './CitationCard'
import type { CitationSource } from '@/features/chat/types'

describe('CitationCard', () => {
  const createCitation = (overrides: Partial<CitationSource> = {}): CitationSource => ({
    id: 'citation-1',
    url: 'https://example.com/article',
    content: 'Citation content',
    timestamp: new Date('2024-01-15T14:30:00'),
    isCited: false,
    ...overrides,
  })

  beforeEach(() => {
    vi.clearAllMocks()
  })

  describe('basic rendering', () => {
    test('renders as a link', () => {
      render(<CitationCard citation={createCitation()} />)

      const link = screen.getByRole('link')
      expect(link).toHaveAttribute('href', 'https://example.com/article')
      expect(link).toHaveAttribute('target', '_blank')
      expect(link).toHaveAttribute('rel', 'noopener noreferrer')
    })

    test('displays domain name', () => {
      render(<CitationCard citation={createCitation({ url: 'https://www.example.com/page' })} />)

      expect(screen.getByText('example.com')).toBeInTheDocument()
    })

    test('displays full URL', () => {
      render(
        <CitationCard
          citation={createCitation({ url: 'https://example.com/full/path/here' })}
        />
      )

      expect(screen.getByText('https://example.com/full/path/here')).toBeInTheDocument()
    })

    test('displays timestamp', () => {
      render(
        <CitationCard
          citation={createCitation({ timestamp: new Date('2024-01-15T14:30:00') })}
        />
      )

      expect(screen.getByText(/\d{1,2}:\d{2}/)).toBeInTheDocument()
    })
  })

  describe('cited vs referenced state', () => {
    test('shows check icon when cited', () => {
      render(<CitationCard citation={createCitation({ isCited: true })} />)

      const domain = screen.getByText('example.com')
      expect(domain).toBeInTheDocument()
    })

    test('shows link icon when referenced (not cited)', () => {
      render(<CitationCard citation={createCitation({ isCited: false })} />)

      const domain = screen.getByText('example.com')
      expect(domain).toBeInTheDocument()
    })
  })

  describe('confidence badge', () => {
    test('shows no badge for exact verified citations', () => {
      render(
        <CitationCard
          citation={createCitation({
            isCited: true,
            verification: { verified: true, confidence: 1, matchKind: 'exact' },
          })}
        />
      )

      expect(screen.queryByText(/Unverified|Partial match/)).not.toBeInTheDocument()
    })

    test('shows no badge when the backend sent no verification data', () => {
      render(<CitationCard citation={createCitation({ isCited: true })} />)

      expect(screen.queryByText(/Unverified|Partial match/)).not.toBeInTheDocument()
    })

    test('shows an unverified badge for low-confidence citations', () => {
      render(
        <CitationCard
          citation={createCitation({
            isCited: true,
            verification: { verified: false, confidence: 0.3, matchKind: 'host_only' },
          })}
        />
      )

      const badge = screen.getByText('Unverified · 30%')
      expect(badge).toBeInTheDocument()
      expect(
        screen.getByLabelText(
          'Citation could not be verified against research sources (confidence 30%, only the site was found in research results)'
        )
      ).toBeInTheDocument()
    })

    test('shows a partial-match badge for repaired verified citations', () => {
      render(
        <CitationCard
          citation={createCitation({
            isCited: true,
            verification: { verified: true, confidence: 0.7, matchKind: 'repaired' },
          })}
        />
      )

      expect(screen.getByText('Partial match · 70%')).toBeInTheDocument()
    })

    test('does not badge referenced (uncited) sources', () => {
      render(
        <CitationCard
          citation={createCitation({
            isCited: false,
            verification: { verified: false, confidence: 0.3, matchKind: 'host_only' },
          })}
        />
      )

      expect(screen.queryByText(/Unverified/)).not.toBeInTheDocument()
    })

    test('getConfidenceBadge treats normalized matches as confident', () => {
      expect(getConfidenceBadge({ verified: true, confidence: 0.95, matchKind: 'normalized' })).toBeNull()
      expect(getConfidenceBadge({ verified: false })).toEqual(
        expect.objectContaining({ label: 'Unverified', color: 'yellow' })
      )
    })
  })

  describe('domain extraction', () => {
    test('removes www prefix from domain', () => {
      render(
        <CitationCard
          citation={createCitation({ url: 'https://www.wikipedia.org/wiki/Test' })}
        />
      )

      expect(screen.getByText('wikipedia.org')).toBeInTheDocument()
    })

    test('handles complex URLs', () => {
      render(
        <CitationCard
          citation={createCitation({ url: 'https://sub.domain.example.com/path?query=1' })}
        />
      )

      expect(screen.getByText('sub.domain.example.com')).toBeInTheDocument()
    })

    test('handles invalid URLs gracefully', () => {
      render(<CitationCard citation={createCitation({ url: 'not-a-valid-url' })} />)

      const urlTexts = screen.getAllByText('not-a-valid-url')
      expect(urlTexts.length).toBeGreaterThan(0)
    })
  })

  describe('timestamp handling', () => {
    test('handles Date object timestamp', () => {
      render(
        <CitationCard
          citation={createCitation({ timestamp: new Date('2024-01-15T09:15:00') })}
        />
      )

      expect(screen.getByText(/\d{1,2}:\d{2}/)).toBeInTheDocument()
    })

    test('handles ISO string timestamp', () => {
      render(
        <CitationCard
          citation={createCitation({
            timestamp: '2024-01-15T14:30:00Z' as unknown as Date,
          })}
        />
      )

      expect(screen.getByText(/\d{1,2}:\d{2}/)).toBeInTheDocument()
    })
  })
})
