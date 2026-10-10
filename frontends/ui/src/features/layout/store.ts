// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Layout Store
 *
 * Zustand store for managing the main app layout state.
 * Controls sidebar visibility and panel states.
 */

import { create } from 'zustand'
import { devtools } from 'zustand/middleware'
import type {
  LayoutState,
  LayoutStore,
  RightPanelType,
  ResearchPanelTab,
  DataSourcesPanelTab,
  ThemeMode,
} from './types'
import { createDataSourcesClient, type DataSourceFromAPI } from '@/adapters/api'

const initialState: LayoutState = {
  isSessionsPanelOpen: false,
  sessionsCollapsed: false,
  sessionsAutoCollapsed: false,
  rightPanel: 'data-sources',
  researchPanelTab: 'tasks',
  dataSourcesPanelTab: 'connections',
  enabledDataSourceIds: [], // Start empty, populated when data sources are fetched
  theme: 'system',
  availableDataSources: null,
  knowledgeLayerAvailable: false, // Default to false until API confirms availability
  dataSourcesLoading: false,
  dataSourcesError: null,
  promptDraft: null,
  selectedModel: undefined,
  // Deprecated aliases for backwards compatibility
  detailsPanelTab: 'report',
  dataSourcePanelTab: 'connections',
}

export const useLayoutStore = create<LayoutStore>()(
  devtools(
    (set) => ({
      ...initialState,

      toggleSessionsPanel: () =>
        set(
          (state) => ({ isSessionsPanelOpen: !state.isSessionsPanelOpen }),
          false,
          'toggleSessionsPanel'
        ),

      setSessionsPanelOpen: (open: boolean) =>
        set({ isSessionsPanelOpen: open }, false, 'setSessionsPanelOpen'),

      toggleSessionsSidebar: () =>
        set(
          (state) => ({
            sessionsCollapsed: !state.sessionsCollapsed,
            sessionsAutoCollapsed: false,
          }),
          false,
          'toggleSessionsSidebar'
        ),

      setSessionsCollapsed: (collapsed: boolean) =>
        set(
          { sessionsCollapsed: collapsed, sessionsAutoCollapsed: false },
          false,
          'setSessionsCollapsed'
        ),

      openRightPanel: (panel: RightPanelType) =>
        set(
          (state) => {
            const collapsesSidebar = panel !== null && panel !== 'settings'
            if (collapsesSidebar && !state.sessionsCollapsed) {
              return { rightPanel: panel, sessionsCollapsed: true, sessionsAutoCollapsed: true }
            }
            if (!collapsesSidebar && state.sessionsAutoCollapsed) {
              return { rightPanel: panel, sessionsCollapsed: false, sessionsAutoCollapsed: false }
            }
            return { rightPanel: panel }
          },
          false,
          'openRightPanel'
        ),

      closeRightPanel: () =>
        set(
          (state) =>
            state.sessionsAutoCollapsed
              ? { rightPanel: null, sessionsCollapsed: false, sessionsAutoCollapsed: false }
              : { rightPanel: null },
          false,
          'closeRightPanel'
        ),

      setResearchPanelTab: (tab: ResearchPanelTab) =>
        set({ researchPanelTab: tab }, false, 'setResearchPanelTab'),

      setDataSourcesPanelTab: (tab: DataSourcesPanelTab) =>
        set({ dataSourcesPanelTab: tab }, false, 'setDataSourcesPanelTab'),

      toggleDataSource: (id: string) =>
        set(
          (state) => {
            const isEnabled = state.enabledDataSourceIds.includes(id)
            return {
              enabledDataSourceIds: isEnabled
                ? state.enabledDataSourceIds.filter((sourceId) => sourceId !== id)
                : [...state.enabledDataSourceIds, id],
            }
          },
          false,
          'toggleDataSource'
        ),

      setEnabledDataSources: (ids: string[]) =>
        set({ enabledDataSourceIds: ids }, false, 'setEnabledDataSources'),

      setTheme: (theme: ThemeMode) => set({ theme }, false, 'setTheme'),

      setPromptDraft: (value: string | null) =>
        set({ promptDraft: value }, false, 'setPromptDraft'),

      setSelectedModel: (model: string | undefined) =>
        set({ selectedModel: model }, false, 'setSelectedModel'),

      resetComposerState: () =>
        set({ promptDraft: null, selectedModel: undefined }, false, 'resetComposerState'),

      fetchDataSources: async (authToken?: string) => {
        set({ dataSourcesLoading: true, dataSourcesError: null }, false, 'fetchDataSources/start')

        try {
          const client = createDataSourcesClient({ authToken })
          const response = await client.getDataSources()

          // Start with every returned source enabled, EXCEPT protected per-user
          // sources that aren't connected yet: enabling those would put an
          // unusable source into the selection (shown in "Selected Data Sources"
          // and submitted), which the card toggle and "Enable All" already refuse
          // to do. The user connects such a source and then enables it. Auth-aware
          // cleanup still runs through disableAuthRequiredSources on access loss.
          const enabledIds = response.data_sources
            .filter(
              (source) =>
                !(source.per_user_auth?.required && source.per_user_auth.status !== 'connected')
            )
            .map((source) => source.id)

          set(
            {
              availableDataSources: response.data_sources,
              knowledgeLayerAvailable: response.knowledge_layer,
              enabledDataSourceIds: enabledIds,
              dataSourcesLoading: false,
              dataSourcesError: null,
            },
            false,
            'fetchDataSources/success'
          )
        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : 'Failed to fetch data sources'
          set(
            {
              dataSourcesLoading: false,
              dataSourcesError: errorMessage,
            },
            false,
            'fetchDataSources/error'
          )
        }
      },

      refreshDataSourceStatus: async (authToken?: string) => {
        // Selection-preserving refresh: update the source list/auth status without
        // touching loading/error flags or resetting non-protected selections. The
        // one exception is a protected source whose status is no longer
        // 'connected' (e.g. the token expired since it was enabled) — it can no
        // longer be used, so drop it from the selection here rather than leaving it
        // shown in "Selected Data Sources" and submitted while unusable.
        try {
          const client = createDataSourcesClient({ authToken })
          const response = await client.getDataSources()
          set(
            (state) => {
              const stillUsable = new Set(
                response.data_sources
                  .filter((s) => !(s.per_user_auth?.required && s.per_user_auth.status !== 'connected'))
                  .map((s) => s.id)
              )
              return {
                availableDataSources: response.data_sources,
                knowledgeLayerAvailable: response.knowledge_layer,
                // Only ever removes now-unusable protected ids; other selections
                // (incl. sources absent from this response) are preserved.
                enabledDataSourceIds: state.enabledDataSourceIds.filter((id) => {
                  const src = response.data_sources.find((s) => s.id === id)
                  return !(src?.per_user_auth?.required) || stillUsable.has(id)
                }),
              }
            },
            false,
            'refreshDataSourceStatus'
          )
        } catch {
          // Best-effort: keep the previously loaded state on failure.
        }
      },

      disableAuthRequiredSources: () =>
        set(
          (state) => ({
            enabledDataSourceIds: state.enabledDataSourceIds.filter((id) => {
              const source = state.availableDataSources?.find((s) => s.id === id)
              return !source?.requires_auth
            }),
          }),
          false,
          'disableAuthRequiredSources'
        ),

      setAvailableDataSources: (sources: DataSourceFromAPI[]) =>
        set({ availableDataSources: sources }, false, 'setAvailableDataSources'),

      setKnowledgeLayerAvailable: (available: boolean) =>
        set({ knowledgeLayerAvailable: available }, false, 'setKnowledgeLayerAvailable'),

      // Deprecated actions - delegate to new ones
      setDetailsPanelTab: (tab: ResearchPanelTab) =>
        set(
          { researchPanelTab: tab, detailsPanelTab: tab },
          false,
          'setDetailsPanelTab'
        ),

      setDataSourcePanelTab: (tab: DataSourcesPanelTab) =>
        set(
          { dataSourcesPanelTab: tab, dataSourcePanelTab: tab },
          false,
          'setDataSourcePanelTab'
        ),
    }),
    { name: 'LayoutStore' }
  )
)
