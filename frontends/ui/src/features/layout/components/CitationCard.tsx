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
 * - artifact.update type: "citation_use" - Cited (actually used in report)
 */

'use client'

import { type FC } from 'react'
import { Flex, Text } from '@/adapters/ui'
import { Link, Check } from '@/adapters/ui/icons'
import type { CitationSource } from '@/features/chat/types'

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

/**
 * Non-collapsible card showing a citation source as a clickable link.
 */
export const CitationCard: FC<CitationCardProps> = ({ citation }) => {
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
            style={{
              color: citation.isCited
                ? 'var(--text-color-feedback-success)'
                : 'var(--text-color-subtle)',
            }}
            aria-hidden="true"
          >
            {citation.isCited ? (
              <Check className="h-4 w-4" />
            ) : (
              <Link className="h-4 w-4" />
            )}
          </span>

          {}
          <Text
            kind="label/semibold/sm"
            className="flex-1 min-w-0 truncate"
            style={{
              color: citation.isCited
                ? 'var(--text-color-feedback-success)'
                : 'var(--text-color-subtle)',
            }}
          >
            {getDomain(citation.url)} {citation.confidence < 0.9 ? "⚠️" : ""}
          </Text>

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
