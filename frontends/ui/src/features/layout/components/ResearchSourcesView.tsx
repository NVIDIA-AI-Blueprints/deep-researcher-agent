// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * ResearchSourcesView Component
 *
 * The single, consolidated home for every source the deep-research agent
 * touched, rendered by the Citations rail panel. Cited sources come from the
 * finished report's own reference block via the authoritative
 * {@link splitReferences} metadata that ReportTab uses, so their numbers match
 * the report's inline [N]. Sources the agent read but the report did not cite
 * are listed under "Other sources found". When the report has no parseable
 * references (for example an interrupted run), it falls back to the stream
 * citations split by their cited flag. An All / Cited filter toggles between the
 * groups; rows use the compact single-line SourceList styling.
 */

'use client'

import { type FC, useCallback, useMemo, useState } from 'react'
import { Flex, SegmentedControl, Text } from '@/adapters/ui'
import { Book } from '@/adapters/ui/icons'
import { useChatStore } from '@/features/chat'
import { SourceList } from '@/shared/components/Sources/SourceList'
import { mapCitationSource } from '@/shared/components/Sources/source-utils'
import { splitReferences } from '@/shared/components/Sources/parse-references'
import type { SourceRef } from '@/shared/components/Sources/types'
import { EMPTY_RESEARCH_DETAILS_HELP_TEXT } from './research-empty-state-copy'

/** Source-list filter: every source, or only those cited in the report. */
type SourceFilter = 'all' | 'cited'

/**
 * Normalize a URL for cited/uncited comparison: lowercase host, drop a leading
 * `www.`, strip the fragment, and remove a trailing slash. Falls back to a
 * lightweight string normalize when the value is not a parseable URL.
 */
function normalizeUrl(url: string): string {
  try {
    const parsed = new URL(url)
    const host = parsed.host.toLowerCase().replace(/^www\./, '')
    const path = parsed.pathname.replace(/\/$/, '')
    return `${parsed.protocol}${host}${path}${parsed.search}`
  } catch {
    return url.trim().toLowerCase().replace(/#.*$/, '').replace(/\/$/, '')
  }
}

export const ResearchSourcesView: FC = () => {
  const reportContent = useChatStore((s) => s.reportContent)
  const deepResearchCitations = useChatStore((s) => s.deepResearchCitations)
  const [filter, setFilter] = useState<SourceFilter>('all')

  const { cited, other, total } = useMemo(() => {
    const reportStr = typeof reportContent === 'string' ? reportContent : ''
    const reportSources = splitReferences(reportStr).sources
    const citations = deepResearchCitations ?? []

    if (reportSources.length > 0) {
      const citedUrls = new Set(
        reportSources
          .map((source) => source.url)
          .filter((url): url is string => Boolean(url))
          .map(normalizeUrl)
      )
      const lastReportIndex = reportSources.reduce((max, s) => Math.max(max, s.index), 0)
      const otherSources: SourceRef[] = citations
        .filter((citation) => citation.url && !citedUrls.has(normalizeUrl(citation.url)))
        .map((citation, index) => mapCitationSource(citation, lastReportIndex + index))
      return {
        cited: reportSources,
        other: otherSources,
        total: reportSources.length + otherSources.length,
      }
    }

    const mapped = citations.map((citation, index) => ({
      source: mapCitationSource(citation, index),
      isCited: Boolean(citation.isCited),
    }))
    return {
      cited: mapped.filter((entry) => entry.isCited).map((entry) => entry.source),
      other: mapped.filter((entry) => !entry.isCited).map((entry) => entry.source),
      total: mapped.length,
    }
  }, [reportContent, deepResearchCitations])

  const handleFilterChange = useCallback((value: string) => {
    setFilter(value as SourceFilter)
  }, [])

  return (
    <Flex direction="col" gap="4" className="h-full min-h-0">
      <Flex direction="col" gap="2" className="shrink-0">
        <Flex align="center" gap="2">
          <Text kind="label/semibold/md" className="text-primary">
            Sources
          </Text>
          {total > 0 && (
            <Text kind="body/regular/xs" className="text-secondary tabular-nums">
              {total}
            </Text>
          )}
        </Flex>
        {total > 0 && (
          <div>
            <SegmentedControl
              value={filter}
              onValueChange={handleFilterChange}
              size="small"
              items={[
                { value: 'all', children: 'All' },
                { value: 'cited', children: `Cited (${cited.length})` },
              ]}
            />
          </div>
        )}
      </Flex>

      {total === 0 ? (
        <Flex direction="col" align="center" justify="center" className="flex-1 py-8 text-center">
          <Book className="text-secondary mb-3 h-8 w-8" />
          <Text kind="body/regular/md" className="text-secondary">
            Sources the agent reads will appear here.
          </Text>
          <Text kind="body/regular/sm" className="text-secondary mt-2">
            {EMPTY_RESEARCH_DETAILS_HELP_TEXT}
          </Text>
        </Flex>
      ) : filter === 'cited' && cited.length === 0 ? (
        <Flex direction="col" align="center" justify="center" className="flex-1 py-8 text-center">
          <Book className="text-secondary mb-3 h-8 w-8" />
          <Text kind="body/regular/md" className="text-secondary">
            No sources were cited in the report.
          </Text>
          <Text kind="body/regular/sm" className="text-secondary mt-2">
            Switch to All to see every source the agent found.
          </Text>
        </Flex>
      ) : (
        <div className="min-h-0 flex-1 overflow-y-auto">
          {filter === 'cited' ? (
            <SourceList sources={cited} title="Cited in report" />
          ) : (
            <>
              <SourceList sources={cited} title="Cited in report" />
              <SourceList sources={other} title="Other sources found" />
            </>
          )}
        </div>
      )}
    </Flex>
  )
}
