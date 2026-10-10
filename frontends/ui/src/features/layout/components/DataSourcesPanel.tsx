// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * DataSourcesPanel Component
 *
 * Right-side panel for managing data sources and file uploads.
 * Contains two tabs: Data Connections (API sources) and File Sources (uploaded files).
 */

'use client'

import { type FC, memo, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Flex, Text, SegmentedControl, Switch, Button, Banner } from '@/adapters/ui'
import { createMcpAuthClient, openAuthPopupAndWait } from '@/adapters/api'
import { useShallow } from 'zustand/react/shallow'
import { Close, Globe, LoadingSpinner } from '@/adapters/ui/icons'
import { useAuth } from '@/adapters/auth'
import { useReducedMotion } from '@/hooks/use-reduced-motion'
import { useLayoutStore } from '../store'
import { useIsCurrentSessionBusy, useChatStore } from '@/features/chat'
import { type DataSource, getDataSourceDisplay } from '../data-sources'
import { cn } from '@/shared/lib/cn'
import { DataConnectionCard } from './DataConnectionCard'
import { FileSourcesTab } from './FileSourcesTab'
import { UploadOrchestrator } from '@/features/documents'
import type { DataSourcesPanelTab } from '../types'

interface DataSourcesPanelProps {
  /** Callback when source enabled state changes */
  onSourceToggle?: (sourceId: string, enabled: boolean) => void
  /** Callback when a file is deleted */
  onDeleteFile?: (id: string) => void
}

/**
 * Panel for managing data sources and file uploads.
 * Opens from the right side of the screen.
 */
export const DataSourcesPanel: FC<DataSourcesPanelProps> = memo(function DataSourcesPanel({ onSourceToggle, onDeleteFile }) {
  const { idToken, authRequired } = useAuth()
  const saveDataSourcesToConversation = useChatStore(
    (state) => state.saveDataSourcesToConversation
  )

  const isOpen = useLayoutStore((s) => s.rightPanel === 'data-sources')
  const {
    dataSourcesPanelTab,
    enabledDataSourceIds,
    availableDataSources,
    dataSourcesLoading,
    dataSourcesError,
  } = useLayoutStore(useShallow((s) => ({
    dataSourcesPanelTab: s.dataSourcesPanelTab,
    enabledDataSourceIds: s.enabledDataSourceIds,
    availableDataSources: s.availableDataSources,
    dataSourcesLoading: s.dataSourcesLoading,
    dataSourcesError: s.dataSourcesError,
  })))

  const panelRef = useRef<HTMLDivElement>(null)
  const openerRef = useRef<HTMLElement | null>(null)

  const closeRightPanel = useLayoutStore((s) => s.closeRightPanel)
  const setDataSourcesPanelTab = useLayoutStore((s) => s.setDataSourcesPanelTab)
  const toggleDataSource = useLayoutStore((s) => s.toggleDataSource)
  const setEnabledDataSources = useLayoutStore((s) => s.setEnabledDataSources)
  const fetchDataSources = useLayoutStore((s) => s.fetchDataSources)
  const refreshDataSourceStatus = useLayoutStore((s) => s.refreshDataSourceStatus)

  // Refresh per-source auth status each time the panel opens so a token that was
  // invalidated server-side (e.g. expired and dropped at job time) shows as
  // Reconnect instead of a stale "connected". Selection-preserving, so it won't
  // reset the user's enabled sources.
  useEffect(() => {
    if (isOpen) {
      void refreshDataSourceStatus(idToken)
    }
  }, [isOpen, idToken, refreshDataSourceStatus])

  useEffect(() => {
    const el = panelRef.current
    if (isOpen) {
      openerRef.current = document.activeElement as HTMLElement | null
      el?.removeAttribute('inert')
      return
    }
    el?.setAttribute('inert', '')
    const opener = openerRef.current
    openerRef.current = null
    opener?.focus?.()
  }, [isOpen])

  // Check if current session is busy with operations
  const isBusy = useIsCurrentSessionBusy()

  // Error surfaced when a protected-source connect attempt fails.
  const [connectError, setConnectError] = useState<string | null>(null)

  // Check if user has valid auth token
  const hasValidToken = !!idToken

  // Convert array to Set for efficient lookups
  const enabledSourcesSet = new Set(enabledDataSourceIds)

  // Convert API data sources to UI format. Names/descriptions come from the
  // frontend display-override; per-user auth metadata is preserved for the
  // per-user OAuth connect flow.
  const displaySources: DataSource[] = useMemo(() => {
    if (!availableDataSources || availableDataSources.length === 0) {
      return []
    }
    return availableDataSources.map((source) => {
      const display = getDataSourceDisplay(source)
      return {
        id: source.id,
        name: display.name,
        description: display.description,
        category: source.category ?? 'enterprise',
        defaultEnabled: source.default_enabled ?? true,
        requiresAuth: source.requires_auth ?? false,
        perUserAuth: source.per_user_auth
          ? {
              required: source.per_user_auth.required,
              provider: source.per_user_auth.provider,
              mcpServerId: source.per_user_auth.mcp_server_id,
              status: source.per_user_auth.status,
              connectUrl: source.per_user_auth.connect_url,
              expiresAt: source.per_user_auth.expires_at,
              lastError: source.per_user_auth.last_error,
            }
          : undefined,
      }
    })
  }, [availableDataSources])

  // Check if any sources require authentication
  const hasAuthenticatedSources = useMemo(() => {
    return displaySources.some((source) => source.requiresAuth)
  }, [displaySources])

  const prefersReducedMotion = useReducedMotion()

  const handleToggle = useCallback(
    (sourceId: string, enabled: boolean) => {
      const updatedIds = enabled
        ? [...enabledDataSourceIds, sourceId]
        : enabledDataSourceIds.filter((id) => id !== sourceId)
      toggleDataSource(sourceId)
      saveDataSourcesToConversation(updatedIds)
      onSourceToggle?.(sourceId, enabled)
    },
    [toggleDataSource, enabledDataSourceIds, saveDataSourcesToConversation, onSourceToggle]
  )

  // Start (or resume) the per-user OAuth flow for a protected source: get a
  // provider login URL, open it in a popup, then refresh statuses. On failure
  // (network/popup errors) surface a banner and still resync in `finally`, so
  // the card never sticks in "Connecting…".
  const handleConnect = useCallback(
    async (sourceId: string) => {
      setConnectError(null)
      const sourceName = displaySources.find((s) => s.id === sourceId)?.name ?? sourceId
      try {
        const client = createMcpAuthClient({ authToken: idToken })
        const { status, auth_url } = await client.connect(sourceId)
        if (status === 'auth_required' && auth_url) {
          // Pass a status probe so the popup resolves once the backend records
          // the connection, even when the provider's COOP headers sever the
          // opener and the postMessage / popup-close signals never arrive.
          await openAuthPopupAndWait(auth_url, sourceId, {
            pollStatus: () => client.getStatus(sourceId).then((s) => s.status),
          })
        }
      } catch (err) {
        console.error('[DataSourcesPanel] Failed to connect data source', sourceId, err)
        const detail = err instanceof Error ? err.message : 'Please try again.'
        setConnectError(`Couldn't connect ${sourceName}. ${detail}`)
      } finally {
        try {
          await fetchDataSources(idToken)
        } catch (err) {
          console.error('[DataSourcesPanel] Failed to refresh data sources', err)
        }
      }
    },
    [idToken, fetchDataSources, displaySources]
  )

  const handleTabChange = useCallback(
    (value: string) => {
      setDataSourcesPanelTab(value as DataSourcesPanelTab)

      // Refresh files from backend when switching to the files tab
      // to detect backend-side removals (e.g. TTL cleanup)
      if (value === 'files') {
        const sessionId = useChatStore.getState().currentConversation?.id
        if (sessionId) {
          UploadOrchestrator.refreshFilesForSession(sessionId)
        }
      }
    },
    [setDataSourcesPanelTab]
  )

  // Sources are available unless they require auth and the user has no token
  const availableSources = useMemo(() => {
    return displaySources.filter(
      (source) => !source.requiresAuth || hasValidToken
    )
  }, [displaySources, hasValidToken])

  // Count enabled sources from the store (only count available ones)
  const enabledAvailableCount = enabledDataSourceIds.filter((id) =>
    availableSources.some((s) => s.id === id)
  ).length
  const availableCount = availableSources.length
  // Treat this as a master on/off switch: it stays on while any available source
  // is enabled, and turns everything off when clicked.
  const anyAvailableEnabled = enabledAvailableCount > 0

  const handleToggleAll = useCallback(() => {
    // Mirror DataConnectionCard's per-card gate: a protected source must be
    // connected before it can be enabled. "Enable All" must skip protected
    // sources that aren't connected, otherwise it bypasses that gate.
    const updatedIds = anyAvailableEnabled
      ? []
      : availableSources
          .filter((s) => !(s.perUserAuth?.required && s.perUserAuth.status !== 'connected'))
          .map((s) => s.id)
    setEnabledDataSources(updatedIds)
    saveDataSourcesToConversation(updatedIds)
  }, [anyAvailableEnabled, setEnabledDataSources, availableSources, saveDataSourcesToConversation])

  return (
    <div
      ref={panelRef}
      className={cn(
        'border-base bg-surface-base h-full shrink-0 overflow-hidden',
        isOpen && 'border-l'
      )}
      style={{
        width: isOpen ? '400px' : '0px',
        minWidth: isOpen ? '400px' : '0px',
        transition: prefersReducedMotion
          ? 'none'
          : 'width 600ms ease-in-out, min-width 600ms ease-in-out',
      }}
      aria-hidden={!isOpen}
    >
      <Flex
        direction="col"
        className="h-full w-[400px]"
        style={{
          visibility: isOpen ? 'visible' : 'hidden',
          opacity: isOpen ? 1 : 0,
          transition: prefersReducedMotion
            ? 'none'
            : isOpen
              ? 'opacity 100ms ease-in-out, visibility 0ms'
              : 'opacity 100ms ease-in-out 500ms, visibility 0ms 600ms',
        }}
      >
        {/* Header with close affordance */}
        <Flex
          align="center"
          justify="between"
          className="border-base h-[var(--header-height)] shrink-0 border-b pl-6 pr-4"
        >
          <Flex align="center" gap="2">
            <Globe className="h-5 w-5" />
            <Text kind="label/semibold/md" className="text-primary">
              Data Sources
            </Text>
          </Flex>
          <Button
            kind="tertiary"
            size="small"
            onClick={closeRightPanel}
            aria-label="Close data sources panel"
            title="Close data sources"
          >
            <Close className="h-4 w-4" aria-hidden="true" />
          </Button>
        </Flex>

        <Flex direction="col" className="flex-1 overflow-hidden px-6 py-4">
          {/* Tab Navigation */}
          <Flex className="mb-4">
            <SegmentedControl
              value={dataSourcesPanelTab}
              onValueChange={handleTabChange}
              size="small"
              className="w-full"
              items={[
                { value: 'connections', children: 'Connections' },
                { value: 'files', children: 'Files' },
              ]}
            />
          </Flex>

          {/* Tab Content */}
          {dataSourcesPanelTab === 'connections' ? (
            /* Data Sources Tab */
            <Flex direction="col" className="flex-1 overflow-y-auto">
              {/* Auth Warning Banner - shown when authenticated sources exist but no valid token */}
              {hasAuthenticatedSources && !hasValidToken && (
                <Banner
                  kind="inline"
                  status={!authRequired ? 'info' : 'warning'}
                  className="mb-6 px-4 py-3"
                >
                  {!authRequired
                    ? 'Enable authentication to access additional data sources.'
                    : 'Sign in to access additional data sources.'}
                </Banner>
              )}

              {/* Connect failure feedback so a failed attempt isn't silent */}
              {connectError && (
                <Banner kind="inline" status="error" className="mb-6 px-4 py-3">
                  {connectError}
                </Banner>
              )}

              {/* All Connections Toggle */}
              <Text
                kind="label/semibold/xs"
                className="text-subtle mb-3 font-mono uppercase tracking-[0.08em]"
              >
                All Connections
              </Text>
              <Flex
                align="center"
                justify="between"
                className={cn(
                  'surface-card mb-4 border p-3',
                  isBusy
                    ? 'border-base opacity-50'
                    : anyAvailableEnabled
                      ? 'brand-tint border'
                      : 'border-base'
                )}
                title={
                  isBusy ? 'Data source changes disabled during active operations' : undefined
                }
              >
                <Text kind="label/semibold/sm" className="text-primary">
                  Disable / Enable All
                </Text>
                <Switch
                  size="small"
                  checked={anyAvailableEnabled}
                  onCheckedChange={handleToggleAll}
                  disabled={isBusy}
                  aria-label={
                    isBusy
                      ? 'Toggle all connections (disabled)'
                      : anyAvailableEnabled
                        ? 'Disable all connections'
                        : 'Enable all connections'
                  }
                />
              </Flex>

              {/* Individual Connections */}
              <Text
                kind="label/semibold/xs"
                className="text-subtle mb-3 font-mono uppercase tracking-[0.08em]"
              >
                Individual Connections ({displaySources.length})
              </Text>

              {dataSourcesLoading ? (
                <Flex align="center" justify="center" className="py-8">
                  <LoadingSpinner size="medium" aria-label="Loading data sources" />
                </Flex>
              ) : dataSourcesError ? (
                <Flex direction="col" align="center" className="py-4">
                  <Text kind="body/regular/sm" className="text-error mb-2">
                    Unable to load data sources
                  </Text>
                  <Text kind="body/regular/xs" className="text-subtle mb-3">
                    {dataSourcesError}
                  </Text>
                  <Button
                    kind="secondary"
                    size="small"
                    onClick={() => fetchDataSources(idToken)}
                    aria-label="Retry loading data sources"
                  >
                    Retry
                  </Button>
                </Flex>
              ) : displaySources.length === 0 ? (
                <Flex direction="col" align="center" className="py-4">
                  <Text kind="body/regular/sm" className="text-subtle">
                    No data sources available
                  </Text>
                </Flex>
              ) : (
                <Flex direction="col" gap="2">
                  {displaySources.map((source) => {
                    const isSourceAvailable = !source.requiresAuth || hasValidToken
                    return (
                      <DataConnectionCard
                        key={source.id}
                        source={source}
                        isEnabled={enabledSourcesSet.has(source.id)}
                        isAvailable={isSourceAvailable}
                        isBusy={isBusy}
                        unavailableReason={
                          !isSourceAvailable
                            ? 'Sign in required to access this data source'
                            : undefined
                        }
                        onToggle={handleToggle}
                        onConnect={handleConnect}
                      />
                    )
                  })}
                </Flex>
              )}
            </Flex>
          ) : (
            /* File Sources Tab */
            <FileSourcesTab onDeleteFile={onDeleteFile} />
          )}
        </Flex>

        {/* Footer summary */}
        <Flex direction="col" className="border-base shrink-0 border-t px-6 py-3">
          {dataSourcesPanelTab === 'connections' ? (
            <Text kind="body/regular/xs" className="text-subtle">
              {enabledAvailableCount} of {availableCount} available connections enabled. Enabled
              connections will be available to the AI assistant.
            </Text>
          ) : (
            <Text kind="body/regular/xs" className="text-subtle">
              Attached files will be always available to agents until deleted.
            </Text>
          )}
        </Flex>
      </Flex>
    </div>
  )
})
