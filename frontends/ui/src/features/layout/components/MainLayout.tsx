// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * MainLayout Component
 *
 * The main application layout container that orchestrates:
 * - AppBar (top)
 * - SessionsPanel (left, collapsible push rail)
 * - ChatArea + InputArea (center, responsive width)
 * - ResearchPanel (right, pushes content)
 * - DataSourcesPanel (right, push panel)
 *
 * Handles auth state to show different UI for logged-in vs logged-out users.
 */

'use client'

import { type FC, useCallback, useMemo } from 'react'
import { useShallow } from 'zustand/react/shallow'
import { Flex } from '@/adapters/ui'
import { AppBar } from './AppBar'
import { SessionsPanel } from './SessionsPanel'
import { ChatArea } from './ChatArea'
import { InputArea } from './InputArea'
import { ResearchPanel } from './ResearchPanel'
import { DataSourcesPanel } from './DataSourcesPanel'
import { DeepResearchRail } from './DeepResearchRail'
import { useChatStore, useDeepResearch, NoSourcesBanner } from '@/features/chat'
import {
  hasActiveDeepResearchJob,
  hasCompletedDeepResearchReport,
  hasExpiredDeepResearchReport,
  sortConversationsByLastUserMessage,
} from '@/features/chat/lib/session-activity'
import { useLayoutStore } from '../store'
import { useSessionUrl } from '@/hooks/use-session-url'

interface MainLayoutProps {
  /** Whether the user is authenticated */
  isAuthenticated?: boolean
  /** Whether the initial auth/session bootstrap is still loading */
  isLoading?: boolean
  /** Whether authentication is required (false = using default user) */
  authRequired?: boolean
  /** User information for AppBar */
  user?: {
    name?: string
    email?: string
    image?: string
  }
  /** Callback when sign in is clicked */
  onSignIn?: () => void
  /** Callback when sign out is clicked */
  onSignOut?: () => void
}

/**
 * Main application layout with all panels and regions.
 * Manages the overall structure and panel states.
 * Chat state is managed via the useChatStore.
 */
export const MainLayout: FC<MainLayoutProps> = ({
  isAuthenticated = false,
  isLoading = false,
  authRequired = false,
  user,
  onSignIn,
  onSignOut,
}) => {
  const {
    currentConversation,
    conversations,
    isStreaming,
    pendingInteraction,
    isDeepResearchStreaming,
    deepResearchOwnerConversationId,
    currentUserId,
  } = useChatStore(
    useShallow((s) => ({
      currentConversation: s.currentConversation,
      conversations: s.conversations,
      isStreaming: s.isStreaming,
      pendingInteraction: s.pendingInteraction,
      isDeepResearchStreaming: s.isDeepResearchStreaming,
      deepResearchOwnerConversationId: s.deepResearchOwnerConversationId,
      currentUserId: s.currentUserId,
    }))
  )

  const selectConversation = useChatStore((s) => s.selectConversation)
  const startNewSessionDraft = useChatStore((s) => s.startNewSessionDraft)
  const deleteConversation = useChatStore((s) => s.deleteConversation)
  const deleteAllConversations = useChatStore((s) => s.deleteAllConversations)
  const updateConversationTitle = useChatStore((s) => s.updateConversationTitle)

  const openRightPanel = useLayoutStore((s) => s.openRightPanel)

  // Deep research SSE hook - manages connection when deep research starts
  useDeepResearch()

  // Sync session state with URL query parameters
  const { updateSessionUrl, clearSessionUrl } = useSessionUrl({ isAuthenticated })

  // Wrap selectConversation to also update URL
  const handleSelectSession = useCallback(
    (sessionId: string) => {
      selectConversation(sessionId)
      updateSessionUrl(sessionId)
    },
    [selectConversation, updateSessionUrl]
  )

  // Start a new unsaved draft session and clear URL until first interaction.
  // Open Data Sources panel so it stays visible (default panel for new sessions).
  const handleNewSession = useCallback(() => {
    startNewSessionDraft()
    clearSessionUrl()
    if (isAuthenticated) {
      openRightPanel('data-sources')
    }
  }, [startNewSessionDraft, clearSessionUrl, openRightPanel, isAuthenticated])

  // Wrap deleteConversation to clear URL if deleting current session
  const handleDeleteSession = useCallback(
    (sessionId: string) => {
      const wasCurrentSession = currentConversation?.id === sessionId
      deleteConversation(sessionId)
      if (wasCurrentSession) {
        clearSessionUrl()
      }
    },
    [deleteConversation, currentConversation?.id, clearSessionUrl]
  )

  // Delete all sessions for the current user
  const handleDeleteAllSessions = useCallback(() => {
    deleteAllConversations()
    clearSessionUrl()
  }, [deleteAllConversations, clearSessionUrl])

  const isNavigationBlocked = isStreaming || pendingInteraction !== null

  const userConversations = useMemo(
    () =>
      sortConversationsByLastUserMessage(
        currentUserId ? conversations.filter((c) => c.userId === currentUserId) : []
      ),
    [conversations, currentUserId]
  )

  const sessions = useMemo(
    () =>
      userConversations.map((conv) => ({
        id: conv.id,
        title: conv.title,
        date: conv.updatedAt,
        hasActiveDeepResearch:
          hasActiveDeepResearchJob(conv.messages) ||
          (isDeepResearchStreaming && deepResearchOwnerConversationId === conv.id),
        hasCompletedReport: hasCompletedDeepResearchReport(conv.messages),
        hasExpiredReport: hasExpiredDeepResearchReport(conv.messages),
      })),
    [userConversations, isDeepResearchStreaming, deepResearchOwnerConversationId]
  )

  return (
    <Flex direction="col" className="h-screen min-w-[768px] overflow-x-auto overflow-y-hidden">
      {/* AppBar - Fixed at top */}
      <AppBar
        sessionTitle={currentConversation?.title}
        isAuthenticated={isAuthenticated}
        authRequired={authRequired}
        user={user}
        onNewSession={handleNewSession}
        isNewSessionDisabled={isNavigationBlocked}
        onSignIn={onSignIn}
        onSignOut={onSignOut}
      />

      {/* Main content area: in-flow panels reflow the center column (push, not overlay) */}
      <div
        data-testid="main-content-row"
        className="relative flex flex-1 overflow-x-auto overflow-y-hidden"
      >
        {/* Sessions Panel (Left) - collapsible push rail, only when authenticated */}
        {isAuthenticated && (
          <SessionsPanel
            sessions={sessions}
            selectedSessionId={currentConversation?.id}
            onSelectSession={handleSelectSession}
            onNewSession={handleNewSession}
            onDeleteSession={handleDeleteSession}
            onDeleteAllSessions={handleDeleteAllSessions}
            onRenameSession={updateConversationTitle}
          />
        )}

        {/* Center Content: Chat + Input */}
        <div className="flex min-w-[360px] flex-1 flex-col overflow-hidden">
          {/* Chat Area - Scrollable */}
          <ChatArea isAuthenticated={isAuthenticated} isLoading={isLoading} onSignIn={onSignIn} />

          {/* No sources warning - shown when no data sources or files available */}
          <NoSourcesBanner isAuthenticated={isAuthenticated} />

          {/* Input Area - Fixed at bottom of chat */}
          {/* Using WebSocket mode for full HITL (human-in-the-loop) support */}
          <InputArea isAuthenticated={isAuthenticated} connectionMode="websocket" />
        </div>

        {/* Data Sources Panel (Right) - push panel */}
        {isAuthenticated && <DataSourcesPanel />}

        {/* Deep Research content panel (Right) - pushes content, opens left of the rail */}
        <ResearchPanel isAuthenticated={isAuthenticated} />

        {/* Deep Research rail (far Right) - persistent navigation */}
        <DeepResearchRail isAuthenticated={isAuthenticated} />
      </div>
    </Flex>
  )
}
