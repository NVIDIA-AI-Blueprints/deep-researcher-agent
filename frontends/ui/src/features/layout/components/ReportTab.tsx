// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * ReportTab Component
 *
 * Displays research output in two visual modes:
 *   1. Research Notes (intermediate) -- preview styling with a header badge
 *   2. Final Report -- full-width rendered markdown with export footer
 *
 * Shows streaming indicator when report is being generated.
 * Includes export footer for Markdown and PDF export (final report only).
 */

'use client'

import { type FC, type ReactNode, useMemo, useState } from 'react'
import { Flex, Text } from '@/adapters/ui'
import { useShallow } from 'zustand/react/shallow'
import { Document, ChevronDown } from '@/adapters/ui/icons'
import { MarkdownRenderer } from '@/shared/components/MarkdownRenderer'
import { SourceStrip } from '@/shared/components/Sources/SourceStrip'
import { mapCitationSource } from '@/shared/components/Sources/source-utils'
import { splitReferences, stripTrailingReferences } from '@/shared/components/Sources/parse-references'
import { useChatStore, selectResolvedDeepResearchJobId } from '@/features/chat'
import { ExportFooter } from './ExportFooter'
import { FileCard } from './FileCard'

interface ReportTabProps {
  /** Optional custom content to display instead of store content */
  children?: ReactNode
}

/**
 * Report tab content - displays research output.
 * Subscribes to chat store for report content, category, and streaming state.
 * Renders research notes with a subtle preview treatment and the final report at full prominence.
 */
export const ReportTab: FC<ReportTabProps> = ({ children }) => {
  const {
    reportContent,
    reportContentCategory,
    isStreaming,
    currentStatus,
    isDeepResearchStreaming,
    deepResearchCitations,
    deepResearchFiles,
  } = useChatStore(
    useShallow((s) => ({
      reportContent: s.reportContent,
      reportContentCategory: s.reportContentCategory,
      isStreaming: s.isStreaming,
      currentStatus: s.currentStatus,
      isDeepResearchStreaming: s.isDeepResearchStreaming,
      deepResearchCitations: s.deepResearchCitations,
      deepResearchFiles: s.deepResearchFiles,
    }))
  )
  // Resolve the owning job id (active or latest finished) so artifact:// images render.
  const deepResearchJobId = useChatStore(selectResolvedDeepResearchJobId)

  const [showFiles, setShowFiles] = useState(false)

  const reportContentStr = typeof reportContent === 'string' ? reportContent : ''

  const { reportBody, sources } = useMemo(() => {
    const split = splitReferences(reportContentStr)
    if (split.sources.length > 0) {
      return { reportBody: split.body, sources: split.sources }
    }
    const fallback = (deepResearchCitations ?? []).map(mapCitationSource)
    return {
      reportBody: fallback.length > 0 ? stripTrailingReferences(reportContentStr) : reportContentStr,
      sources: fallback,
    }
  }, [reportContentStr, deepResearchCitations])

  const isEmpty = !reportContentStr.trim()
  const isGeneratingReport = isStreaming && currentStatus === 'writing'
  const isResearchNotes = reportContentCategory === 'research_notes'

  return (
    <Flex direction="col" className="h-full">
      {/* Scrollable content area */}
      <Flex direction="col" gap="4" className="flex-1 overflow-y-auto">
        {children ? (
          children
        ) : isEmpty ? (
          isDeepResearchStreaming ? (
            <Flex
              direction="col"
              align="center"
              justify="center"
              gap="3"
              className="flex-1 py-8 text-center"
            >
              <span
                role="status"
                aria-label="Deep research in progress"
                className="h-2.5 w-2.5 animate-pulse rounded-full bg-[color:var(--color-brand)] motion-reduce:animate-none"
              />
              <Text kind="body/regular/md" className="text-secondary">
                Deep research in progress...
              </Text>
            </Flex>
          ) : (
            <Flex direction="col" align="center" justify="center" className="flex-1 py-8 text-center">
              <Document className="text-subtle mb-3 h-8 w-8" />
              <Text kind="body/regular/md" className="text-subtle">
                Report content will appear here when available.
              </Text>
            </Flex>
          )
        ) : isResearchNotes ? (
          /* Research notes: preview treatment */
          <Flex direction="col" gap="3" className="flex-1">
            <Flex
              align="center"
              gap="2"
              className="shrink-0 rounded-md border border-yellow-200 bg-yellow-50 px-3 py-2 dark:border-yellow-800 dark:bg-yellow-950"
            >
              <div className="h-2 w-2 animate-pulse rounded-full bg-yellow-500" />
              <Text kind="body/regular/sm" className="text-yellow-700 dark:text-yellow-300">
                Research notes from agents. Final report is still being generated.
              </Text>
            </Flex>
            <div className="flex-1 opacity-80">
              <MarkdownRenderer
                content={reportContentStr}
                isStreaming={false}
                className="max-w-none"
                artifactJobId={deepResearchJobId ?? undefined}
              />
            </div>
          </Flex>
        ) : (
          /* Final report: full prominence */
          <Flex direction="col" gap="4" className="flex-1">
            <MarkdownRenderer
              content={reportBody}
              isStreaming={isGeneratingReport}
              className="max-w-none"
              variant="answer"
              sources={sources}
              artifactJobId={deepResearchJobId ?? undefined}
            />
            {sources.length > 0 && (
              <div className="border-base border-t pt-4">
                <SourceStrip sources={sources} />
              </div>
            )}
          </Flex>
        )}
      </Flex>

      {/* Generated files */}
      {!children && (deepResearchFiles?.length ?? 0) > 0 && (
        <Flex direction="col" gap="2" className="border-base shrink-0 border-t pt-3">
          <button
            type="button"
            onClick={() => setShowFiles((v) => !v)}
            aria-expanded={showFiles}
            className="text-secondary hover:text-primary flex cursor-pointer items-center gap-1.5 self-start transition-colors"
          >
            <Document className="h-4 w-4" aria-hidden="true" />
            <Text kind="body/regular/sm">Generated files ({deepResearchFiles.length})</Text>
            <ChevronDown
              className={`h-4 w-4 transition-transform duration-200 ${showFiles ? 'rotate-180' : ''}`}
              aria-hidden="true"
            />
          </button>
          {showFiles && (
            <Flex direction="col" gap="2" className="max-h-64 overflow-y-auto">
              {deepResearchFiles.map((file) => (
                <div key={file.id} className="shrink-0">
                  <FileCard file={file} />
                </div>
              ))}
            </Flex>
          )}
        </Flex>
      )}

      {/* Export footer - only meaningful for the final report */}
      <ExportFooter />
    </Flex>
  )
}
