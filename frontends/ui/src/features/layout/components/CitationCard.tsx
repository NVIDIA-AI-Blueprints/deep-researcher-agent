// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * CitationCard Component
 *
 * Non-collapsible card displaying a single citation/source as a clickable link.
 * Shows title/domain and full URL.
 *
 * SSE Events:
 * - artifact.update type: "citation_source" - Referenced (discovered during search)
 * - artifact.update type: "citation_use" - Cited (actually used in report). Final-report
 *   events may carry citation-verifier fields (verified, confidence, match_kind); cited
 *   sources that were repaired or kept unverified show a confidence badge.
 */

'use client'

import { type FC } from 'react'
import { Badge, Flex, Text } from '@/adapters/ui'
import { Link, Check } from '@/adapters/ui/icons'
import type { CitationSource, CitationVerification } from '@/features/chat/types'

interface CitationCardProps {
  /** Citation information */
  citation: CitationSource
}

/**
 * Format timestamp for display
 */
const formatTime = (date: Date | string): string => {
  const dateObj = typeof date === 'string' ? new Date(date) : date
  return dateObj.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}

/**
 * Extract domain from URL for display
 */
const getDomain = (url: string): string => {
  try {
    const urlObj = new URL(url)
    return urlObj.hostname.replace('www.', '')
  } catch {
    return url.substring(0, 30)
  }
}

/** Verified matches at or above this confidence are shown without a badge */
const HIGH_CONFIDENCE_THRESHOLD = 0.9

const MATCH_KIND_LABELS: Record<string, string> = {
  exact: 'exact match',
  normalized: 'normalized URL match',
  repaired: 'repaired URL match',
  citation_key: 'document match',
  backfilled: 'matched by title',
  host_only: 'only the site was found in research results',
  unmatched: 'not found in research results',
}

interface ConfidenceBadgeInfo {
  label: string
  description: string
  color: 'yellow' | 'gray'
}

/**
 * Decide whether a cited source needs a confidence badge.
 * Returns null for confident, verified citations and for sources without verifier data.
 */
export const getConfidenceBadge = (
  verification: CitationVerification | undefined
): ConfidenceBadgeInfo | null => {
  if (!verification) return null
  const { verified, confidence, matchKind } = verification
  const percent = typeof confidence === 'number' ? `${Math.round(confidence * 100)}%` : undefined
  const kindLabel = matchKind ? MATCH_KIND_LABELS[matchKind] ?? matchKind : undefined
  const details = [percent && `confidence ${percent}`, kindLabel].filter(Boolean).join(', ')

  if (verified === false) {
    return {
      label: percent ? `Unverified · ${percent}` : 'Unverified',
      description: `Citation could not be verified against research sources${details ? ` (${details})` : ''}`,
      color: 'yellow',
    }
  }
  if (typeof confidence === 'number' && confidence < HIGH_CONFIDENCE_THRESHOLD) {
    return {
      label: `Partial match · ${percent}`,
      description: `Citation verified with a partial match${details ? ` (${details})` : ''}`,
      color: 'gray',
    }
  }
  return null
}

/**
 * Non-collapsible card showing a citation source as a clickable link.
 */
export const CitationCard: FC<CitationCardProps> = ({ citation }) => {
  const badge = citation.isCited ? getConfidenceBadge(citation.verification) : null
  const isUnverified = badge?.color === 'yellow'
  const accentColor = isUnverified
    ? 'var(--text-color-feedback-warning)'
    : citation.isCited
      ? 'var(--text-color-feedback-success)'
      : 'var(--text-color-subtle)'

  return (
    <a
      href={citation.url}
      target="_blank"
      rel="noopener noreferrer"
      className="block"
    >
      <Flex
        direction="col"
        className="rounded-lg border overflow-hidden bg-surface-sunken border-base hover:bg-surface-raised-50 transition-colors"
      >
        {}
        <Flex align="center" gap="2" className="w-full px-3 py-2">
          {}
          <span
            className="shrink-0"
            style={{ color: accentColor }}
            aria-hidden="true"
          >
            {citation.isCited && !isUnverified ? (
              <Check className="h-4 w-4" />
            ) : (
              <Link className="h-4 w-4" />
            )}
          </span>

          {}
          <Text
            kind="label/semibold/sm"
            className="flex-1 min-w-0 truncate"
            style={{ color: accentColor }}
          >
            {getDomain(citation.url)}
          </Text>

          {badge && (
            <Badge
              kind="outline"
              color={badge.color}
              className="shrink-0"
              title={badge.description}
              aria-label={badge.description}
            >
              {badge.label}
            </Badge>
          )}

          {}
          <Text kind="body/regular/xs" className="text-subtle shrink-0">
            {formatTime(citation.timestamp)}
          </Text>
        </Flex>

        {}
        <Flex className="px-3 pb-2 border-t border-base">
          <Text kind="body/regular/sm" className="text-subtle truncate mt-1 break-all">
            {citation.url}
          </Text>
        </Flex>
      </Flex>
    </a>
  )
}
