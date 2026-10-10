// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Layout Feature Public API
 *
 * Exports all layout components and state management.
 */

// Main layout components
export {
  MainLayout,
  AppBar,
  DeepResearchRail,
  SessionsPanel,
  ChatArea,
  InputArea,
  SettingsPanel,
} from './components'

// Research panel and related components
export {
  ResearchPanel,
  TasksTab,
  ThinkingTab,
  ReportTab,
  ReportCard,
  ExportFooter,
} from './components'

// Thinking sub-tabs and cards
export { AgentsTab, AgentCard, ThoughtCard, SourceCard } from './components'
export type { AgentInfo } from './components'
export type { ThoughtInfo } from './components'
export type { FileInfo } from './components'
export type { SourceInfo } from './components'

// Data sources panel and related components
export {
  DataSourcesPanel,
  DataConnectionsTab,
  DataConnectionCard,
  FileSourcesTab,
  FileSourceCard,
} from './components'

// Confirmation modals
export {
  DeleteFileConfirmationModal,
  DeleteSessionConfirmationModal,
} from './components'

// Data Sources Types
export type { DataSource, DataSourceCategory } from './data-sources'

// Store
export { useLayoutStore } from './store'

// Types
export type {
  LayoutState,
  LayoutActions,
  LayoutStore,
  RightPanelType,
  ResearchPanelTab,
  DataSourcesPanelTab,
  ThemeMode,
} from './types'
