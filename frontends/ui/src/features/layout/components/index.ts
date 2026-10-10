// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Layout Components Barrel Export
 */

// Main layout components
export { MainLayout } from './MainLayout'
export { AppBar } from './AppBar'
export { DeepResearchRail } from './DeepResearchRail'
export { SessionsPanel } from './SessionsPanel'
export { ChatArea } from './ChatArea'
export { InputArea } from './InputArea'
export { SettingsPanel } from './SettingsPanel'

// Research panel and tabs
export { ResearchPanel } from './ResearchPanel'
export { TasksTab } from './TasksTab'
export { ThinkingTab } from './ThinkingTab'
export { ReportTab } from './ReportTab'

// Thinking sub-tabs and cards
export { AgentsTab } from './AgentsTab'
export { AgentCard } from './AgentCard'
export type { AgentInfo } from './AgentCard'
export { ThoughtCard } from './ThoughtCard'
export type { ThoughtInfo } from './ThoughtCard'
export { FileCard } from './FileCard'
export type { FileInfo } from './FileCard'

// Citations components
export { SourceCard } from './SourceCard'
export type { SourceInfo } from './SourceCard'

// Report components
export { ReportCard } from './ReportCard'
export { ExportFooter } from './ExportFooter'

// Data sources panel and tabs
export { DataSourcesPanel } from './DataSourcesPanel'
export { DataConnectionsTab } from './DataConnectionsTab'
export { DataConnectionCard } from './DataConnectionCard'
export { FileSourcesTab } from './FileSourcesTab'
export { FileSourceCard } from './FileSourceCard'

// Confirmation modals
export { DeleteFileConfirmationModal } from './DeleteFileConfirmationModal'
export { DeleteSessionConfirmationModal } from './DeleteSessionConfirmationModal'
