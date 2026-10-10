// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * AgentResponse Component
 *
 * Displays a completed agent response in the chat area.
 * Used for short answers that don't need the full report panel.
 * Left-aligned with distinct styling from user messages.
 */

'use client'

import { type FC, useCallback, useMemo } from 'react'
import { Flex, Text, Button } from '@/adapters/ui'
import { useShallow } from 'zustand/react/shallow'
import { ChevronRight, LoadingSpinner } from '@/adapters/ui/icons'
import { MarkdownRenderer } from '@/shared/components/MarkdownRenderer'
import { SourceList } from '@/shared/components/Sources/SourceList'
import { splitReferences, tabularizeEntityLines } from '@/shared/components/Sources/parse-references'
import { CopyButton } from '@/shared/components/Actions/CopyButton'
import { formatTime } from '@/shared/utils/format-time'
import { useLayoutStore } from '@/features/layout/store'
import { useChatStore } from '../store'
import { useLoadJobData } from '../hooks'

export interface AgentResponseProps {
  /** Response content from the agent */
  content: string
  /** Timestamp of the response (Date or ISO string from persisted state) */
  timestamp?: Date | string
  /** Whether to show a button to view the full report */
  showViewReport?: boolean
  /** Display variant - 'default' has box styling, 'inline' has no box (for use inside containers) */
  variant?: 'default' | 'inline'
  /** Deep research job ID for loading report data on-demand */
  jobId?: string
  /** Whether this message has active (streaming) deep research */
  isDeepResearchActive?: boolean
  /** Job status for determining button behavior */
  deepResearchJobStatus?: 'submitted' | 'running' | 'success' | 'failure' | 'interrupted'
}

/**
 * Agent response bubble component for completed responses
 */
export const AgentResponse: FC<AgentResponseProps> = ({
  content,
  timestamp,
  showViewReport = false,
  variant = 'default',
  jobId,
  isDeepResearchActive = false,
  deepResearchJobStatus,
}) => {
  const openRightPanel = useLayoutStore((s) => s.openRightPanel)
  const setResearchPanelTab = useLayoutStore((s) => s.setResearchPanelTab)

  const { reportContent, deepResearchJobId, isDeepResearchStreaming, deepResearchStreamLoaded } =
    useChatStore(
      useShallow((s) => ({
        reportContent: s.reportContent,
        deepResearchJobId: s.deepResearchJobId,
        isDeepResearchStreaming: s.isDeepResearchStreaming,
        deepResearchStreamLoaded: s.deepResearchStreamLoaded,
      }))
    )
  const reconnectToActiveJob = useChatStore((s) => s.reconnectToActiveJob)
  const { loadResearchPanelTab, isLoading, error } = useLoadJobData()

  const { body, sources } = useMemo(() => {
    const split = splitReferences(content ?? '')
    return { body: tabularizeEntityLines(split.body), sources: split.sources }
  }, [content])

  const isJobActive =
    isDeepResearchActive ||
    deepResearchJobStatus === 'submitted' ||
    deepResearchJobStatus === 'running'
  const isJobComplete =
    deepResearchJobStatus === 'success' ||
    deepResearchJobStatus === 'failure' ||
    deepResearchJobStatus === 'interrupted'
  const shouldShowButton = showViewReport || (jobId && (isJobActive || isJobComplete))
  const buttonText = isJobActive ? 'View Progress' : 'View Report'

  const isAnotherJobStreaming = Boolean(
    isDeepResearchStreaming && deepResearchJobId && deepResearchJobId !== jobId
  )
  const blockedByActiveJob = !isJobActive && isAnotherJobStreaming

  const handleViewReport = useCallback(async () => {
    if (blockedByActiveJob) return

    if (isJobActive) {
      if (!isDeepResearchStreaming || deepResearchJobId !== jobId) {
        await reconnectToActiveJob()
      }
      openRightPanel('thinking')
      return
    }

    const hasExistingDataForThisJob =
      jobId &&
      deepResearchJobId === jobId &&
      deepResearchStreamLoaded &&
      reportContent &&
      reportContent.trim().length > 0

    if (hasExistingDataForThisJob) {
      setResearchPanelTab('report')
      openRightPanel('research')
      return
    }

    if (jobId) {
      await loadResearchPanelTab(jobId, 'report')
    } else {
      setResearchPanelTab('report')
      openRightPanel('research')
    }
  }, [
    jobId,
    deepResearchJobId,
    reportContent,
    deepResearchStreamLoaded,
    isJobActive,
    blockedByActiveJob,
    isDeepResearchStreaming,
    loadResearchPanelTab,
    reconnectToActiveJob,
    setResearchPanelTab,
    openRightPanel,
  ])

  if (!content || !content.trim() || content === 'null') {
    return null
  }

  const reportButton = shouldShowButton && (
    <Flex direction="col" align="end" gap="1" className="mt-1">
      {error && (
        <Text
          role="alert"
          kind="label/regular/xs"
          className="text-error max-w-full text-right"
        >
          Could not load the report: {error}
        </Text>
      )}
      <Button
        kind="tertiary"
        size="tiny"
        onClick={handleViewReport}
        disabled={isLoading || blockedByActiveJob}
        aria-label={
          isLoading
            ? 'Loading...'
            : blockedByActiveJob
              ? `${buttonText} (available once the running research job finishes)`
              : error
                ? `Retry: ${buttonText}`
                : buttonText
        }
        title={
          blockedByActiveJob
            ? 'Available once the running research job finishes'
            : error
              ? `Error: ${error}`
              : isLoading
                ? 'Loading...'
                : buttonText
        }
      >
        <Flex align="center" gap="1">
          {isLoading ? (
            <>
              <LoadingSpinner size="small" aria-label="Loading" className="h-3 w-3" />
              <Text kind="label/regular/xs">Loading...</Text>
            </>
          ) : (
            <>
              <Text kind="label/regular/xs">{error ? 'Retry' : buttonText}</Text>
              <ChevronRight className="h-3 w-3" aria-hidden="true" />
            </>
          )}
        </Flex>
      </Button>
    </Flex>
  )

  const meta = (
    <Flex align="center" gap="2" className="agent-final-meta mt-1.5">
      <CopyButton text={content} label="Copy answer" />
      {timestamp && (
        <Text kind="body/regular/xs" className="mono-meta text-subtle">
          {formatTime(timestamp)}
        </Text>
      )}
    </Flex>
  )

  if (variant === 'inline') {
    return (
      <Flex
        direction="col"
        gap="2"
        className="agent-final-response w-full overflow-hidden break-words pl-2"
      >
        <MarkdownRenderer
          content={body}
          variant="answer"
          sources={sources}
          className="answer-reveal"
        />
        {sources.length > 0 && <SourceList sources={sources} />}
        {reportButton}
        {meta}
      </Flex>
    )
  }

  return (
    <Flex justify="start" className="w-full">
      <Flex direction="col" className="agent-final-response w-full max-w-[85%]">
        <Flex direction="col" gap="2" className="overflow-hidden break-words pl-2">
          <MarkdownRenderer
            content={body}
            variant="answer"
            sources={sources}
            className="answer-reveal"
          />
          {sources.length > 0 && <SourceList sources={sources} />}
          {reportButton}
        </Flex>
        {meta}
      </Flex>
    </Flex>
  )
}
