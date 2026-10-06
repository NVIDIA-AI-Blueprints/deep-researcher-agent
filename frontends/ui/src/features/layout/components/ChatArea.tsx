// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * ChatArea Component
 *
 * Main chat display area showing messages between user and assistant.
 * Includes the message list and is positioned in the center of the layout.
 *
 * Shows different welcome states based on authentication:
 * - Logged out: Prompt to sign in with CTA button
 * - Logged in: Ready to start chatting
 */

'use client'

import { type FC, type ReactNode, memo, useRef, useEffect, useCallback, useMemo } from 'react'
import { motion } from 'motion/react'
import { Flex, Text, Button, Spinner } from '@/adapters/ui'
import { Document, Lock } from '@/adapters/ui/icons'
import { useShallow } from 'zustand/react/shallow'
import {
  useChatStore,
  AgentPrompt,
  AgentResponse,
  ErrorBanner,
  FileUploadBanner,
  DeepResearchBanner,
  UserMessage,
  ChatThinking,
} from '@/features/chat'
import type { ChatMessage } from '@/features/chat'
import { StarfieldAnimation } from '@/shared/components/StarfieldAnimation'
import { cn } from '@/shared/lib/cn'
import { isPinnedToBottom } from '@/shared/lib/scroll'

interface ChatAreaProps {
  /** Whether the user is authenticated */
  isAuthenticated?: boolean
  /** Whether the initial auth/session bootstrap is still loading */
  isLoading?: boolean
  /** Callback when sign in is clicked */
  onSignIn?: () => void
}

/** A user message plus the assistant messages that answer it. */
interface ConversationTurn {
  id: string
  user?: ChatMessage
  assistant: ChatMessage[]
}

/**
 * Main chat area container with scrollable message list.
 * Shows welcome state when no messages exist.
 */
export const ChatArea: FC<ChatAreaProps> = memo(function ChatArea({
  isAuthenticated = false,
  isLoading = false,
  onSignIn,
}) {
  const { currentConversation, isStreaming, currentUserMessageId } = useChatStore(
    useShallow((s) => ({
      currentConversation: s.currentConversation,
      isStreaming: s.isStreaming,
      currentUserMessageId: s.currentUserMessageId,
    }))
  )

  const respondToPrompt = useChatStore((s) => s.respondToPrompt)
  const getThinkingStepsForMessage = useChatStore((s) => s.getThinkingStepsForMessage)
  const dismissErrorCard = useChatStore((s) => s.dismissErrorCard)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  /** The scrollable viewport that holds the message list. */
  const scrollContainerRef = useRef<HTMLDivElement>(null)
  /** The growing content wrapper inside the viewport (observed for size changes). */
  const contentRef = useRef<HTMLDivElement>(null)
  /**
   * Whether the user is currently pinned to the bottom. Kept in a ref (not
   * state) so reading/updating it from scroll + ResizeObserver handlers never
   * triggers a re-render. Defaults to true so a freshly opened conversation
   * lands at the latest message.
   */
  const stickToBottomRef = useRef(true)

  const conversationId = currentConversation?.id ?? null
  const messages = currentConversation?.messages

  const displayableMessages = useMemo(
    () =>
      (messages ?? []).filter((msg) => {
        const messageType = msg.messageType || (msg.role === 'user' ? 'user' : 'assistant')
        return (
          messageType === 'user' ||
          messageType === 'status' ||
          messageType === 'prompt' ||
          messageType === 'agent_response' ||
          messageType === 'file' ||
          messageType === 'file_upload_status' ||
          messageType === 'error' ||
          messageType === 'deep_research_banner'
        )
      }),
    [messages]
  )

  const isEmpty = displayableMessages.length === 0

  // Group the flat message list into turns: each user message starts a turn and
  // the assistant messages that follow it belong to that turn.
  const turns = useMemo<ConversationTurn[]>(() => {
    const groupedTurns: ConversationTurn[] = []
    let activeTurn: ConversationTurn | null = null

    displayableMessages.forEach((message) => {
      const isUserMessage = message.messageType === 'user' || message.role === 'user'

      if (isUserMessage) {
        activeTurn = { id: message.id, user: message, assistant: [] }
        groupedTurns.push(activeTurn)
        return
      }

      if (!activeTurn) {
        activeTurn = { id: `assistant-${message.id}`, assistant: [] }
        groupedTurns.push(activeTurn)
      }

      activeTurn.assistant.push(message)
    })

    return groupedTurns
  }, [displayableMessages])

  /**
   * Helper to get thinking steps for a user message.
   * First checks ephemeral store (for active session), then falls back
   * to persisted steps embedded in the message (for restored sessions).
   * Deep-research steps are included so the inline thinking trace mirrors the
   * live run; the Research Panel still shows the complete picture (plan, sources,
   * report) from its own dedicated state.
   */
  const getStepsForUserMessage = (messageId: string) => {
    const storeSteps = getThinkingStepsForMessage(messageId)
    if (storeSteps.length > 0) return storeSteps

    const message = currentConversation?.messages.find((m) => m.id === messageId)
    return message?.thinkingSteps || []
  }

  /**
   * Track whether the user is pinned to the bottom. Updated on every scroll so
   * the auto-follow below knows whether to keep up with new content or leave a
   * user who scrolled up to read history exactly where they are.
   */
  useEffect(() => {
    const el = scrollContainerRef.current
    if (!el) return
    const onScroll = (): void => {
      stickToBottomRef.current = isPinnedToBottom(el.scrollTop, el.scrollHeight, el.clientHeight)
    }
    el.addEventListener('scroll', onScroll, { passive: true })
    return () => el.removeEventListener('scroll', onScroll)
  }, [])

  /**
   * Auto-follow content growth. The assistant answer and the thinking trace
   * stream into existing messages (the message count does not change), so a
   * count-based scroll never fires mid-stream. A ResizeObserver on the content
   * wrapper catches every height change (streamed tokens, expanding thinking
   * steps, newly appended turns) and pins to the bottom only when the user
   * already was, so it never yanks someone reading earlier history.
   */
  useEffect(() => {
    const el = scrollContainerRef.current
    const content = contentRef.current
    if (!el || !content) return
    const followIfPinned = (): void => {
      if (stickToBottomRef.current) el.scrollTop = el.scrollHeight
    }
    const observer = new ResizeObserver(followIfPinned)
    observer.observe(content)
    return () => observer.disconnect()
  }, [isEmpty])

  // On conversation switch, re-pin to the latest message and jump to the bottom.
  useEffect(() => {
    const el = scrollContainerRef.current
    if (!el) return
    stickToBottomRef.current = true
    el.scrollTop = el.scrollHeight
  }, [conversationId])

  /**
   * On sending a new message, always re-pin and jump to the bottom, even if the
   * user had scrolled up to read history, so their new message and the incoming
   * response are visible. The ResizeObserver above then follows the streamed
   * answer (stick is now true). Deferred a frame so the new turn is in the DOM.
   */
  useEffect(() => {
    if (!currentUserMessageId) return
    stickToBottomRef.current = true
    const el = scrollContainerRef.current
    if (el) el.scrollTop = el.scrollHeight
    const raf = requestAnimationFrame(() => {
      const node = scrollContainerRef.current
      if (node) node.scrollTop = node.scrollHeight
    })
    return () => cancelAnimationFrame(raf)
  }, [currentUserMessageId])

  const handlePromptRespond = useCallback(
    (promptId: string, response: string) => {
      respondToPrompt(promptId, response)
    },
    [respondToPrompt]
  )

  const handleFileRetry = useCallback((_messageId: string) => {}, [])

  return (
    <Flex
      ref={scrollContainerRef}
      direction="col"
      className="scrollbar-hide flex-1 overflow-y-auto"
      role="log"
      aria-live="polite"
      aria-relevant="additions text"
      aria-label="Chat messages"
    >
      {isEmpty ? (
        <WelcomeState isAuthenticated={isAuthenticated} isLoading={isLoading} onSignIn={onSignIn} />
      ) : (
        <Flex
          ref={contentRef}
          direction="col"
          gap="8"
          className="mx-auto w-full max-w-4xl px-6 pb-28 pt-6"
        >
          {turns.map((turn) => {
            const userMessage = turn.user
            const messageSteps = userMessage ? getStepsForUserMessage(userMessage.id) : []
            const hasThinkingSteps = messageSteps.length > 0

            const isCurrentlyStreaming =
              !!userMessage && isStreaming && userMessage.id === currentUserMessageId
            const shouldCheckPostState = !!userMessage && hasThinkingSteps && !isCurrentlyStreaming

            const isWaiting =
              shouldCheckPostState &&
              turn.assistant.some((m) => m.messageType === 'prompt' && !m.isPromptResponded)

            const hasResponse = turn.assistant.some(
              (m) => m.messageType === 'assistant' || m.messageType === 'agent_response'
            )
            const isInterrupted = shouldCheckPostState && !isWaiting && !hasResponse
            const hasAssistantRun = hasThinkingSteps || turn.assistant.length > 0
            const completedResponse = isCurrentlyStreaming
              ? undefined
              : [...turn.assistant].reverse().find((message) => {
                  if (message.messageType !== 'agent_response') return false
                  if (!message.deepResearchJobId) return true
                  return (
                    message.deepResearchJobStatus === 'success' ||
                    message.deepResearchJobStatus === 'failure' ||
                    message.deepResearchJobStatus === 'interrupted'
                  )
                })

            return (
              <motion.div
                key={turn.id}
                layout
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.24, ease: [0.22, 1, 0.36, 1] }}
                className="flex flex-col gap-3"
              >
                {userMessage && (
                  <MessageRenderer
                    message={userMessage}
                    onPromptRespond={handlePromptRespond}
                    onFileRetry={handleFileRetry}
                    onErrorDismiss={dismissErrorCard}
                  />
                )}

                {hasAssistantRun && (
                  <AssistantRun
                    isActive={isCurrentlyStreaming}
                    isWaiting={isWaiting}
                    isInterrupted={isInterrupted}
                  >
                    {userMessage && hasThinkingSteps && (
                      <ChatThinking
                        steps={messageSteps}
                        isThinking={isStreaming && userMessage.id === currentUserMessageId}
                        isWaiting={isWaiting}
                        isInterrupted={isInterrupted}
                        enabledDataSources={userMessage.enabledDataSources}
                        messageFiles={userMessage.messageFiles}
                        model={userMessage.selectedModel}
                        responseStartedAt={userMessage.timestamp}
                        responseCompletedAt={
                          completedResponse?.responseCompletedAt ?? completedResponse?.timestamp
                        }
                      />
                    )}

                    {turn.assistant.map((message) => (
                      <MessageRenderer
                        key={message.id}
                        message={message}
                        inline
                        onPromptRespond={handlePromptRespond}
                        onFileRetry={handleFileRetry}
                        onErrorDismiss={dismissErrorCard}
                      />
                    ))}
                  </AssistantRun>
                )}
              </motion.div>
            )
          })}

          {/* Invisible scroll anchor */}
          <div ref={messagesEndRef} />
        </Flex>
      )}
    </Flex>
  )
})

/**
 * Wraps an assistant turn in the "run lane": a quiet left spine that lights up
 * while the turn is live (active / waiting) and turns danger when interrupted.
 */
const AssistantRun: FC<{
  children: ReactNode
  isActive?: boolean
  isWaiting?: boolean
  isInterrupted?: boolean
}> = ({ children, isActive = false, isWaiting = false, isInterrupted = false }) => (
  <Flex justify="start" className="w-full">
    <div
      className={cn(
        'assistant-turn flex w-full max-w-[88%]',
        isActive && 'assistant-turn-active',
        isWaiting && 'assistant-turn-waiting',
        isInterrupted && 'assistant-turn-interrupted'
      )}
    >
      <Flex
        direction="col"
        gap="3.5"
        className="assistant-lane-glow min-w-0 flex-1 rounded-[var(--radius-card)] px-3 py-2"
      >
        {children}
      </Flex>
    </div>
  </Flex>
)

/**
 * Message renderer that dispatches to the correct component based on message type
 */
interface MessageRendererProps {
  message: ChatMessage
  inline?: boolean
  onPromptRespond: (promptId: string, response: string) => void
  onFileRetry?: (messageId: string) => void
  onFileCancel?: (messageId: string) => void
  onFileDelete?: (messageId: string) => void
  onErrorDismiss?: (messageId: string) => void
}

const MessageRenderer: FC<MessageRendererProps> = ({
  message,
  inline = false,
  onPromptRespond,
  onFileRetry: _onFileRetry,
  onFileCancel: _onFileCancel,
  onFileDelete: _onFileDelete,
  onErrorDismiss,
}) => {
  const messageType = message.messageType || (message.role === 'user' ? 'user' : 'assistant')

  switch (messageType) {
    case 'user':
      return <UserMessage content={message.content} timestamp={message.timestamp} />

    case 'status':
      if (!message.statusType) {
        return null
      }
      return (
        <Flex align="center" gap="2" className="px-1 py-1" role="status">
          <Text kind="body/regular/sm" className="text-subtle">
            {message.statusType}: {message.content}
          </Text>
        </Flex>
      )

    case 'prompt':
      if (!message.promptType) {
        return null
      }
      return (
        <AgentPrompt
          id={message.id}
          interactionId={message.promptId}
          type={message.promptType}
          content={message.content}
          options={message.promptOptions}
          placeholder={message.promptPlaceholder}
          isResponded={message.isPromptResponded}
          response={message.promptResponse}
          onRespond={onPromptRespond}
          variant={inline ? 'inline' : 'default'}
        />
      )

    case 'agent_response':
      return (
        <AgentResponse
          content={message.content}
          timestamp={message.timestamp}
          showViewReport={message.showViewReport}
          jobId={message.deepResearchJobId}
          isDeepResearchActive={message.isDeepResearchActive}
          deepResearchJobStatus={message.deepResearchJobStatus}
          variant={inline ? 'inline' : 'default'}
        />
      )

    case 'file':
      if (!message.fileData) {
        return null
      }
      return (
        <Flex
          align="center"
          gap="2"
          className="bg-surface-raised-30 border-base rounded-lg border px-4 py-2"
          role="status"
        >
          <Document className="text-subtle h-4 w-4" />
          <Text kind="body/regular/sm" className="text-subtle">
            {message.fileData.fileName} ({message.fileData.fileStatus})
          </Text>
        </Flex>
      )

    case 'file_upload_status':
      if (!message.fileUploadStatusData) {
        return null
      }
      return (
        <FileUploadBanner
          type={message.fileUploadStatusData.type}
          fileCount={message.fileUploadStatusData.fileCount}
          onDismiss={onErrorDismiss ? () => onErrorDismiss(message.id) : undefined}
        />
      )

    case 'error':
      if (!message.errorData) {
        return null
      }
      return (
        <ErrorBanner
          code={message.errorData.errorCode}
          message={message.errorData.errorMessage}
          details={message.errorData.errorDetails}
          onDismiss={onErrorDismiss ? () => onErrorDismiss(message.id) : undefined}
        />
      )

    case 'deep_research_banner':
      if (!message.deepResearchBannerData) {
        return null
      }
      return (
        <DeepResearchBanner
          bannerType={message.deepResearchBannerData.bannerType}
          jobId={message.deepResearchBannerData.jobId}
          totalTokens={message.deepResearchBannerData.totalTokens}
          toolCallCount={message.deepResearchBannerData.toolCallCount}
        />
      )

    case 'assistant':
      return null

    default:
      return null
  }
}

/**
 * Welcome state shown when no messages exist
 * Shows different content based on authentication state
 */
interface WelcomeStateProps {
  isAuthenticated?: boolean
  isLoading?: boolean
  onSignIn?: () => void
}

const WelcomeState: FC<WelcomeStateProps> = ({
  isAuthenticated = false,
  isLoading = false,
  onSignIn,
}) => {
  if (!isAuthenticated) {
    if (isLoading) {
      return (
        <Flex direction="col" align="center" justify="center" className="flex-1 p-8">
          <Spinner size="large" aria-label="Loading" />
        </Flex>
      )
    }
    return (
      <Flex direction="col" align="center" justify="center" className="relative flex-1 p-8">
        {/* Ambient starfield backdrop */}
        <div className="pointer-events-none absolute inset-0 flex items-center justify-center opacity-30">
          <div className="h-[500px] w-[500px]">
            <StarfieldAnimation particleCount={300} maxRadius={220} rotationSpeed={0.0005} />
          </div>
        </div>

        {/* Sign-in call to action */}
        <Flex direction="col" align="center" gap="6" className="relative z-10 max-w-md text-center">
          <span className="text-brand text-6xl">
            <Lock />
          </span>
          <Text kind="title/lg" className="text-primary">
            Welcome to Deep Researcher Agent
          </Text>
          <Text kind="body/regular/md" className="text-subtle">
            Sign in with your account to start your AI-powered research session.
          </Text>
          <Button
            kind="primary"
            size="large"
            onClick={onSignIn}
            aria-label="Sign in with NVIDIA SSO"
            className="mt-2"
          >
            <Flex align="center" gap="2">
              <Text kind="label/semibold/md">Sign In with SSO</Text>
            </Flex>
          </Button>
        </Flex>
      </Flex>
    )
  }

  return (
    <Flex direction="col" align="center" justify="center" className="relative flex-1 p-8">
      {/* Ambient starfield backdrop */}
      <div className="pointer-events-none absolute inset-0 flex items-center justify-center opacity-30">
        <div className="h-[500px] w-[500px]">
          <StarfieldAnimation particleCount={300} maxRadius={220} rotationSpeed={0.001} />
        </div>
      </div>

      {/* Ready-to-chat prompt */}
      <Flex direction="col" align="center" gap="5" className="relative z-10 max-w-xl text-center">
        <Text kind="title/lg" className="text-primary">
          What do you want to know?
        </Text>
        <Text kind="body/regular/md" className="text-subtle">
          Ask a question about your connected data sources, or commission a deep research report.
        </Text>
      </Flex>
    </Flex>
  )
}
