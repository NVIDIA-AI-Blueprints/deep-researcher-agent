// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * ResearchPanel Component
 *
 * The Deep Research content panel. It opens immediately to the left of the
 * DeepResearchRail and swaps its content based on the active rail item:
 *   - research  -> ReportTab (report + Markdown/PDF export footer)
 *   - thinking  -> ThinkingTab (reasoning/steps trace) with the workflow Task
 *                  progress folded in as a disclosure above it
 *   - citations -> every source the agent touched (ResearchSourcesView)
 *
 * Data Sources is handled by its own DataSourcesPanel; the three items above and
 * Data Sources are mutually exclusive via the shared rightPanel slot, so only
 * one panel is ever open. This panel PUSHES the chat area rather than overlaying
 * and exposes a drag/keyboard resize grip on its left (chat-facing) edge.
 */

'use client'

import {
  type FC,
  type KeyboardEvent as ReactKeyboardEvent,
  type PointerEvent as ReactPointerEvent,
  type ReactNode,
  memo,
  useCallback,
  useEffect,
  useRef,
  useState,
} from 'react'
import { Flex, Button, Spinner, Text } from '@/adapters/ui'
import { CheckCircle, ChevronDown, Close, StopCircle } from '@/adapters/ui/icons'
import { cancelJob } from '@/adapters/api'
import { useShallow } from 'zustand/react/shallow'
import { cn } from '@/shared/lib/cn'
import { useChatStore, useLoadJobData, selectResolvedDeepResearchJobId } from '@/features/chat'
import { useAuth } from '@/adapters/auth'
import { useReducedMotion } from '@/hooks/use-reduced-motion'
import { useLayoutStore } from '../store'
import { TasksTab } from './TasksTab'
import { ThinkingTab } from './ThinkingTab'
import { ReportTab } from './ReportTab'
import { ResearchSourcesView } from './ResearchSourcesView'
import { getRailPanelLabel } from './DeepResearchRail'
import type { RightPanelType } from '../types'

/** Rail panels rendered by this component (Data Sources lives in its own panel). */
const PANEL_TABS = new Set<RightPanelType>(['research', 'thinking', 'citations'])

/** Panels that read replayed stream data (citations, steps). */
const STREAM_BACKED_TABS = new Set<RightPanelType>(['thinking', 'citations'])

/** Wide panels get more room for their rich content. */
const WIDE_TABS = new Set<RightPanelType>(['research', 'thinking'])

/** Minimum open width per panel class (px). */
const WIDE_MIN_WIDTH = 480
const NARROW_MIN_WIDTH = 420

/** Upper bounds so a resized panel never covers the whole viewport. */
const MAX_WIDTH_PX = 900
const MAX_WIDTH_VW_RATIO = 0.7

/** Nudge step for keyboard resize (px). */
const KEYBOARD_STEP_PX = 24

/** Width used as the drag/keyboard start when the live panel size is unknown. */
export const FALLBACK_WIDE_WIDTH = 640

/** Fallback timeout: if the SSE stream doesn't deliver the interrupted
 *  status within this window after cancel, clean up the UI optimistically. */
const CANCEL_FALLBACK_TIMEOUT_MS = 5000

/** Largest allowed width for the current viewport. */
const getMaxWidth = (): number => {
  if (typeof window === 'undefined') return MAX_WIDTH_PX
  return Math.min(MAX_WIDTH_PX, Math.round(window.innerWidth * MAX_WIDTH_VW_RATIO))
}

/** Clamp a proposed width to the active panel's [min, max] range. */
const clampPanelWidth = (px: number, wide: boolean): number => {
  const min = wide ? WIDE_MIN_WIDTH : NARROW_MIN_WIDTH
  const max = getMaxWidth()
  const lo = Math.min(min, max)
  return Math.max(lo, Math.min(Math.round(px), max))
}

interface ResearchPanelProps {
  /** Content to display in the report view */
  children?: ReactNode
  /** Whether the user is authenticated */
  isAuthenticated?: boolean
}

/**
 * Thinking view: the reasoning/steps trace, with the observed workflow Task
 * progress folded in as a collapsible disclosure above it (shown only when
 * there is progress to display). This is where the former Tasks tab now lives.
 */
const ThinkingView: FC = () => {
  const { deepResearchAgents, deepResearchTodos } = useChatStore(
    useShallow((s) => ({
      deepResearchAgents: s.deepResearchAgents,
      deepResearchTodos: s.deepResearchTodos,
    }))
  )
  const [showTasks, setShowTasks] = useState(false)
  const hasTaskProgress = deepResearchAgents.length > 0 || deepResearchTodos.length > 0

  return (
    <Flex direction="col" gap="3" className="h-full min-h-0">
      {hasTaskProgress && (
        <Flex direction="col" gap="2" className="border-base shrink-0 border-b pb-3">
          <button
            type="button"
            onClick={() => setShowTasks((v) => !v)}
            aria-expanded={showTasks}
            className="text-secondary hover:text-primary flex cursor-pointer items-center gap-1.5 self-start transition-colors"
          >
            <CheckCircle className="h-4 w-4" aria-hidden="true" />
            <Text kind="body/regular/sm">Task progress</Text>
            <ChevronDown
              className={`h-4 w-4 transition-transform duration-200 ${showTasks ? 'rotate-180' : ''}`}
              aria-hidden="true"
            />
          </button>
          {showTasks && (
            <div className="max-h-64 overflow-y-auto">
              <TasksTab />
            </div>
          )}
        </Flex>
      )}
      <div className="min-h-0 flex-1 overflow-hidden">
        <ThinkingTab />
      </div>
    </Flex>
  )
}

/**
 * The Deep Research content panel, driven by the rail selection.
 */
export const ResearchPanel: FC<ResearchPanelProps> = memo(function ResearchPanel({
  children,
  isAuthenticated = false,
}) {
  const rightPanel = useLayoutStore((s) => s.rightPanel)
  const closeRightPanel = useLayoutStore((s) => s.closeRightPanel)
  const isOpen = PANEL_TABS.has(rightPanel)

  const isDeepResearchStreaming = useChatStore((s) => s.isDeepResearchStreaming)
  const deepResearchJobId = useChatStore(selectResolvedDeepResearchJobId)
  const { loadResearchPanelTab, importStreamOnly, isLoading: isStreamLoading } = useLoadJobData()
  const { idToken } = useAuth()

  const prefersReducedMotion = useReducedMotion()
  const cancelFallbackRef = useRef<NodeJS.Timeout | null>(null)
  const loadKeyRef = useRef<string | null>(null)

  const panelRef = useRef<HTMLDivElement>(null)
  const resizeStartRef = useRef<{ startX: number; startWidth: number } | null>(null)
  const [panelWidth, setPanelWidth] = useState<number | null>(null)
  const [isResizing, setIsResizing] = useState(false)

  useEffect(() => {
    return () => {
      if (cancelFallbackRef.current) {
        clearTimeout(cancelFallbackRef.current)
        cancelFallbackRef.current = null
      }
    }
  }, [])

  useEffect(() => {
    if (!isOpen) {
      loadKeyRef.current = null
      return
    }
    if (!isAuthenticated || !deepResearchJobId || isStreamLoading) return
    const key = `${rightPanel}:${deepResearchJobId}`
    if (loadKeyRef.current === key) return
    loadKeyRef.current = key
    if (rightPanel === 'research') {
      void loadResearchPanelTab(deepResearchJobId, 'report')
    } else if (STREAM_BACKED_TABS.has(rightPanel)) {
      void importStreamOnly(deepResearchJobId)
    }
  }, [
    isAuthenticated,
    isOpen,
    rightPanel,
    deepResearchJobId,
    isStreamLoading,
    loadResearchPanelTab,
    importStreamOnly,
  ])

  const handleClose = useCallback(() => {
    closeRightPanel()
  }, [closeRightPanel])

  const handleStopResearch = useCallback(async () => {
    if (!deepResearchJobId) return
    const cancelledJobId = deepResearchJobId
    try {
      await cancelJob(cancelledJobId, idToken || undefined)

      if (cancelFallbackRef.current) clearTimeout(cancelFallbackRef.current)
      cancelFallbackRef.current = setTimeout(() => {
        cancelFallbackRef.current = null
        const state = useChatStore.getState()
        if (!state.isDeepResearchStreaming || state.deepResearchJobId !== cancelledJobId) {
          return
        }
        console.warn(
          '[ResearchPanel] Cancel fallback: SSE did not deliver interrupted status. Cleaning up locally.'
        )
        state.stopAllDeepResearchSpinners()
        const ownerConvId = state.deepResearchOwnerConversationId
        const messageId = state.activeDeepResearchMessageId
        const hasReport = Boolean(state.reportContent?.trim())
        if (ownerConvId && messageId) {
          state.patchConversationMessage(ownerConvId, messageId, {
            content: '',
            deepResearchJobStatus: 'interrupted',
            isDeepResearchActive: false,
            showViewReport: hasReport,
          })
        }
        state.addDeepResearchBanner('cancelled', cancelledJobId, ownerConvId || undefined)
        state.completeDeepResearch()
        state.setStreaming(false)
      }, CANCEL_FALLBACK_TIMEOUT_MS)
    } catch (error) {
      console.error('Failed to cancel job:', error)
    }
  }, [deepResearchJobId, idToken])

  const isWide = WIDE_TABS.has(rightPanel)
  const panelLabel = getRailPanelLabel(rightPanel)
  const minWidth = isWide ? WIDE_MIN_WIDTH : NARROW_MIN_WIDTH

  useEffect(() => {
    setPanelWidth((current) => (current == null ? current : clampPanelWidth(current, isWide)))
  }, [isWide])

  useEffect(() => {
    const reclampToViewport = () => {
      setPanelWidth((current) => (current == null ? current : clampPanelWidth(current, isWide)))
    }
    window.addEventListener('resize', reclampToViewport)
    return () => window.removeEventListener('resize', reclampToViewport)
  }, [isWide])

  const measuredWidth = useCallback((): number => {
    if (panelWidth != null) return panelWidth
    const rect = panelRef.current?.getBoundingClientRect()
    if (rect && rect.width > 0) return Math.round(rect.width)
    return isWide ? FALLBACK_WIDE_WIDTH : NARROW_MIN_WIDTH
  }, [panelWidth, isWide])

  const applyWidth = useCallback(
    (px: number) => {
      setPanelWidth(clampPanelWidth(px, isWide))
    },
    [isWide]
  )

  const handleResizePointerDown = useCallback(
    (e: ReactPointerEvent<HTMLDivElement>) => {
      if (e.button !== 0) return
      resizeStartRef.current = { startX: e.clientX, startWidth: measuredWidth() }
      e.currentTarget.setPointerCapture?.(e.pointerId)
      setIsResizing(true)
    },
    [measuredWidth]
  )

  const handleResizePointerMove = useCallback(
    (e: ReactPointerEvent<HTMLDivElement>) => {
      const start = resizeStartRef.current
      if (!start) return
      applyWidth(start.startWidth + (start.startX - e.clientX))
    },
    [applyWidth]
  )

  const handleResizePointerUp = useCallback((e: ReactPointerEvent<HTMLDivElement>) => {
    if (!resizeStartRef.current) return
    resizeStartRef.current = null
    e.currentTarget.releasePointerCapture?.(e.pointerId)
    setIsResizing(false)
  }, [])

  const handleResizeKeyDown = useCallback(
    (e: ReactKeyboardEvent<HTMLDivElement>) => {
      if (e.key === 'ArrowLeft') {
        e.preventDefault()
        applyWidth(measuredWidth() + KEYBOARD_STEP_PX)
      } else if (e.key === 'ArrowRight') {
        e.preventDefault()
        applyWidth(measuredWidth() - KEYBOARD_STEP_PX)
      }
    },
    [applyWidth, measuredWidth]
  )

  const maxWidth = getMaxWidth()
  const currentPanelWidth = clampPanelWidth(panelWidth ?? measuredWidth(), isWide)

  const openWidth =
    panelWidth != null
      ? `${clampPanelWidth(panelWidth, isWide)}px`
      : isWide
        ? `min(60%, ${MAX_WIDTH_PX}px, ${Math.round(MAX_WIDTH_VW_RATIO * 100)}vw)`
        : `${NARROW_MIN_WIDTH}px`

  return (
    <div
      ref={panelRef}
      data-testid="research-panel"
      className={cn(
        'border-base bg-surface-base relative h-full shrink-0 overflow-hidden',
        isOpen && 'border-l',
        isResizing && 'select-none'
      )}
      style={{
        width: isOpen ? openWidth : '0px',
        minWidth: isOpen ? `${minWidth}px` : '0px',
        transition:
          prefersReducedMotion || isResizing
            ? 'none'
            : 'width 400ms ease-in-out, min-width 400ms ease-in-out',
      }}
      aria-hidden={!isOpen}
    >
      {isOpen && (
        <div
          role="separator"
          aria-orientation="vertical"
          aria-label="Resize panel"
          aria-valuenow={currentPanelWidth}
          aria-valuemin={Math.min(minWidth, maxWidth)}
          aria-valuemax={maxWidth}
          tabIndex={0}
          onPointerDown={handleResizePointerDown}
          onPointerMove={handleResizePointerMove}
          onPointerUp={handleResizePointerUp}
          onPointerCancel={handleResizePointerUp}
          onKeyDown={handleResizeKeyDown}
          data-testid="research-panel-resize"
          className="group focus-visible:ring-brand absolute left-0 top-0 z-10 flex h-full w-3 cursor-col-resize touch-none select-none items-center justify-center outline-none focus-visible:ring-2 focus-visible:ring-inset"
        >
          <span
            aria-hidden="true"
            className="text-subtle group-hover:text-secondary flex flex-col items-center justify-center gap-1 transition-colors"
          >
            <span className="h-1 w-1 rounded-full bg-current" />
            <span className="h-1 w-1 rounded-full bg-current" />
            <span className="h-1 w-1 rounded-full bg-current" />
            <span className="h-1 w-1 rounded-full bg-current" />
          </span>
        </div>
      )}
      <Flex
        direction="col"
        className="h-full w-full"
        style={{
          visibility: isOpen ? 'visible' : 'hidden',
          opacity: isOpen ? 1 : 0,
          transition: prefersReducedMotion
            ? 'none'
            : isOpen
              ? 'opacity 100ms ease-in-out, visibility 0ms'
              : 'opacity 100ms ease-in-out 300ms, visibility 0ms 400ms',
        }}
      >
        {/* Header: active item name + optional stop + close */}
        <Flex
          align="center"
          justify="between"
          className="border-base h-[var(--header-height)] shrink-0 border-b px-6"
        >
          <Text kind="label/semibold/md" className="text-primary truncate">
            {panelLabel}
          </Text>
          <Flex align="center" gap="2">
            {isDeepResearchStreaming && (
              <>
                <Button
                  kind="tertiary"
                  size="small"
                  onClick={handleStopResearch}
                  aria-label="Stop researching"
                  title="Stop researching"
                  data-testid="research-panel-stop"
                >
                  <StopCircle className="mr-2 h-4 w-4" aria-hidden="true" />
                  Stop Researching
                </Button>
                <span
                  aria-hidden="true"
                  data-testid="research-panel-header-divider"
                  className="border-base ml-1 h-5 border-l"
                />
              </>
            )}
            <Button
              kind="tertiary"
              size="small"
              onClick={handleClose}
              aria-label={`Close ${panelLabel || 'panel'}`}
              title="Close panel"
              data-testid="research-panel-close"
            >
              <Close className="h-4 w-4" aria-hidden="true" />
            </Button>
          </Flex>
        </Flex>

        {/* Body */}
        <Flex direction="col" className="flex-1 overflow-hidden px-6 py-5">
          {isStreamLoading ? (
            <Flex direction="col" align="center" justify="center" className="h-full gap-4">
              <Spinner size="medium" aria-label="Loading research data" />
              <Text kind="body/regular/md" className="text-tertiary">
                {rightPanel === 'research' ? 'Loading report...' : 'Loading research data...'}
              </Text>
            </Flex>
          ) : rightPanel === 'research' ? (
            <ReportTab>{children}</ReportTab>
          ) : rightPanel === 'thinking' ? (
            <ThinkingView />
          ) : rightPanel === 'citations' ? (
            <ResearchSourcesView />
          ) : null}
        </Flex>
      </Flex>
    </div>
  )
})
