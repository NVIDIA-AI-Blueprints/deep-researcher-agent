// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Layout Feature Types
 *
 * Type definitions for the main app layout including sidebars and panels.
 */

import type { DataSourceFromAPI } from '@/adapters/api'

/** Theme mode options */
export type ThemeMode = 'light' | 'dark' | 'system'

/**
 * Panels that can be opened on the right side. The first four map one-to-one to
 * the Deep Research rail items; `settings` remains for the standalone
 * SettingsPanel and is not surfaced in the rail.
 */
export type RightPanelType =
  | 'data-sources'
  | 'citations'
  | 'research'
  | 'thinking'
  | 'settings'
  | null

/** Tabs within the Research panel */
export type ResearchPanelTab = 'tasks' | 'thinking' | 'report'

/** Tabs within the DataSources panel */
export type DataSourcesPanelTab = 'connections' | 'files'

/** Layout state for managing panels */
export interface LayoutState {
  /** Whether the sessions panel is open (left side) */
  isSessionsPanelOpen: boolean
  /**
   * Whether the sessions sidebar is collapsed (newer collapse model). Added
   * alongside {@link isSessionsPanelOpen}; a later PR migrates components to it.
   */
  sessionsCollapsed: boolean
  /** True when the sidebar was auto-collapsed by opening a right panel (so it can auto-restore) */
  sessionsAutoCollapsed: boolean
  /** Currently open right panel (null = closed) */
  rightPanel: RightPanelType
  /** Active tab in the research panel */
  researchPanelTab: ResearchPanelTab
  /** Active tab in the data sources panel */
  dataSourcesPanelTab: DataSourcesPanelTab
  /** IDs of enabled data sources (array for zustand serialization) */
  enabledDataSourceIds: string[]
  /** Current theme mode */
  theme: ThemeMode
  /** Dynamic data sources from API (null = not loaded yet) */
  availableDataSources: DataSourceFromAPI[] | null
  /** Whether the knowledge layer (file upload) is available */
  knowledgeLayerAvailable: boolean
  /** Whether data sources are being fetched */
  dataSourcesLoading: boolean
  /** Error message if data sources fetch failed */
  dataSourcesError: string | null
  /** In-progress composer text, preserved across panel/route changes (null = none) */
  promptDraft: string | null
  /** Final-response model the user selected (undefined = use the backend default) */
  selectedModel?: string
  /**
   * @deprecated Use researchPanelTab instead
   */
  detailsPanelTab: ResearchPanelTab
  /**
   * @deprecated Use dataSourcesPanelTab instead
   */
  dataSourcePanelTab: DataSourcesPanelTab
}

/** Layout actions for state management */
export interface LayoutActions {
  /** Toggle sessions panel open/closed */
  toggleSessionsPanel: () => void
  /** Set sessions panel state */
  setSessionsPanelOpen: (open: boolean) => void
  /** Toggle the sessions sidebar collapsed state (collapse model, clears auto-collapse) */
  toggleSessionsSidebar: () => void
  /** Set the sessions sidebar collapsed state (collapse model, clears auto-collapse) */
  setSessionsCollapsed: (collapsed: boolean) => void
  /** Open a specific right panel (closes any existing) */
  openRightPanel: (panel: RightPanelType) => void
  /** Close the right panel */
  closeRightPanel: () => void
  /** Set the active research panel tab */
  setResearchPanelTab: (tab: ResearchPanelTab) => void
  /** Set the active data sources panel tab */
  setDataSourcesPanelTab: (tab: DataSourcesPanelTab) => void
  /** Toggle a data source enabled/disabled by ID */
  toggleDataSource: (id: string) => void
  /** Set all enabled data sources */
  setEnabledDataSources: (ids: string[]) => void
  /** Set the theme mode */
  setTheme: (theme: ThemeMode) => void
  /** Set the in-progress composer draft (null clears it) */
  setPromptDraft: (value: string | null) => void
  /** Set the final-response model for subsequent turns (undefined restores the backend default) */
  setSelectedModel: (model: string | undefined) => void
  /** Clear composer state (draft and selected model) on logout or account switch */
  resetComposerState: () => void
  /** Fetch data sources from API. Only web_search is enabled by default */
  fetchDataSources: (authToken?: string) => Promise<void>
  /**
   * Refresh per-source metadata/auth status WITHOUT changing the user's enabled
   * selection. Used to pick up server-side status changes (e.g. a token that was
   * invalidated at job time) so a stale "connected" card flips to Reconnect.
   */
  refreshDataSourceStatus: (authToken?: string) => Promise<void>
  /** Disable sources that require authentication */
  disableAuthRequiredSources: () => void
  /** Set available data sources (from API) */
  setAvailableDataSources: (sources: DataSourceFromAPI[]) => void
  /** Set knowledge layer availability */
  setKnowledgeLayerAvailable: (available: boolean) => void
  /**
   * @deprecated Use setResearchPanelTab instead
   */
  setDetailsPanelTab: (tab: ResearchPanelTab) => void
  /**
   * @deprecated Use setDataSourcesPanelTab instead
   */
  setDataSourcePanelTab: (tab: DataSourcesPanelTab) => void
}

/** Combined layout store type */
export type LayoutStore = LayoutState & LayoutActions
