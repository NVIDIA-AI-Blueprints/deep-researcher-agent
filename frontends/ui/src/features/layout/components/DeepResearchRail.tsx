// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * DeepResearchRail Component
 *
 * Persistent far-right vertical navigation for the Deep Research workspace.
 * Each item toggles a content panel that opens immediately to its left:
 *   - Data Sources, Citations, Research (primary group, top)
 *   - Thinking (pinned to the bottom, visually separated)
 *
 * The active item gets a dark rounded highlight. Clicking the active item again
 * closes its panel. The rail replaces the former "Show Research" vertical tab.
 */

'use client'

import { type FC, type ComponentType, memo, useCallback } from 'react'
import { Flex, Text } from '@/adapters/ui'
import { ChartFlow, DocumentCheckmark, Document, Globe } from '@/adapters/ui/icons'
import { cn } from '@/shared/lib/cn'
import { useChatStore } from '@/features/chat'
import { useLayoutStore } from '../store'
import type { RightPanelType } from '../types'

/** A single Deep Research rail entry. */
export interface DeepResearchRailItem {
  id: Exclude<RightPanelType, null | 'settings'>
  label: string
  Icon: ComponentType<{ className?: string }>
}

/** Primary rail items, rendered top-to-bottom above the Thinking item. */
export const DEEP_RESEARCH_RAIL_ITEMS: DeepResearchRailItem[] = [
  { id: 'data-sources', label: 'Data Sources', Icon: Globe },
  { id: 'citations', label: 'Citations', Icon: DocumentCheckmark },
  { id: 'research', label: 'Research', Icon: Document },
]

/** Thinking is pinned to the bottom of the rail, separated from the group. */
export const DEEP_RESEARCH_THINKING_ITEM: DeepResearchRailItem = {
  id: 'thinking',
  label: 'Thinking',
  Icon: ChartFlow,
}

const ALL_RAIL_ITEMS: DeepResearchRailItem[] = [
  ...DEEP_RESEARCH_RAIL_ITEMS,
  DEEP_RESEARCH_THINKING_ITEM,
]

/** Human label for a rail-backed right panel (used by the panel header). */
export function getRailPanelLabel(panel: RightPanelType): string {
  return ALL_RAIL_ITEMS.find((item) => item.id === panel)?.label ?? ''
}

interface DeepResearchRailProps {
  /** Whether the user is authenticated; items are disabled otherwise. */
  isAuthenticated?: boolean
}

interface RailButtonProps {
  item: DeepResearchRailItem
  isActive: boolean
  isAuthenticated: boolean
  onSelect: (id: DeepResearchRailItem['id']) => void
}

const RailButton: FC<RailButtonProps> = ({ item, isActive, isAuthenticated, onSelect }) => {
  const { Icon, label, id } = item
  return (
    <button
      type="button"
      onClick={() => onSelect(id)}
      disabled={!isAuthenticated}
      aria-pressed={isActive}
      aria-label={label}
      title={isAuthenticated ? label : `Sign in to open ${label}`}
      data-testid={`deep-research-rail-item-${id}`}
      className={cn(
        'flex w-full items-center gap-3 rounded-lg px-3 py-2.5 text-left transition-colors',
        isActive
          ? 'bg-surface-sunken text-primary'
          : 'text-secondary hover:bg-surface-raised-50 hover:text-primary',
        isAuthenticated && 'cursor-pointer',
        !isAuthenticated && 'cursor-not-allowed opacity-50 hover:bg-transparent'
      )}
    >
      <Icon className="h-4 w-4 shrink-0" />
      <Text kind="label/regular/md" className="truncate">
        {label}
      </Text>
    </button>
  )
}

/**
 * The persistent Deep Research navigation rail.
 */
export const DeepResearchRail: FC<DeepResearchRailProps> = memo(function DeepResearchRail({
  isAuthenticated = false,
}) {
  const rightPanel = useLayoutStore((s) => s.rightPanel)
  const openRightPanel = useLayoutStore((s) => s.openRightPanel)
  const closeRightPanel = useLayoutStore((s) => s.closeRightPanel)
  const isDeepResearchStreaming = useChatStore((s) => s.isDeepResearchStreaming)

  const handleSelect = useCallback(
    (id: DeepResearchRailItem['id']) => {
      if (!isAuthenticated) return
      if (useLayoutStore.getState().rightPanel === id) {
        closeRightPanel()
      } else {
        openRightPanel(id)
      }
    },
    [isAuthenticated, openRightPanel, closeRightPanel]
  )

  return (
    <nav
      aria-label="Deep Research"
      className="border-base bg-surface-base flex w-[180px] shrink-0 flex-col border-l"
    >
      <Flex
        align="center"
        gap="2"
        className="border-base h-[var(--header-height)] shrink-0 border-b px-4"
      >
        <Text kind="label/semibold/md" className="text-primary whitespace-nowrap">
          Deep Research
        </Text>
        {isDeepResearchStreaming && (
          <span
            data-testid="deep-research-activity"
            role="status"
            aria-label="Deep research in progress"
            title="Deep research in progress"
            className="h-2 w-2 shrink-0 animate-pulse rounded-full bg-[color:var(--color-brand)] motion-reduce:animate-none"
          />
        )}
      </Flex>

      <Flex direction="col" justify="between" className="min-h-0 flex-1 p-3">
        <Flex direction="col" gap="1">
          {DEEP_RESEARCH_RAIL_ITEMS.map((item) => (
            <RailButton
              key={item.id}
              item={item}
              isActive={rightPanel === item.id}
              isAuthenticated={isAuthenticated}
              onSelect={handleSelect}
            />
          ))}
        </Flex>

        <Flex direction="col" gap="1" className="shrink-0">
          <RailButton
            item={DEEP_RESEARCH_THINKING_ITEM}
            isActive={rightPanel === DEEP_RESEARCH_THINKING_ITEM.id}
            isAuthenticated={isAuthenticated}
            onSelect={handleSelect}
          />
        </Flex>
      </Flex>
    </nav>
  )
})
