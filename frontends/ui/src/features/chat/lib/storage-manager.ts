// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Storage Manager
 *
 * Manages localStorage capacity by tracking size and automatically cleaning up
 * old sessions when approaching quota limits (5MB in most browsers).
 *
 * Cleanup uses a tiered priority:
 *  1. Sessions older than 1 day (any user) — stale sessions first
 *  2. Oldest sessions of the current user
 *
 * Protected sessions (current + actively busy) are never deleted.
 */

import type { Conversation } from '../types'
import { hasActiveDeepResearchJob } from './session-activity'
import { logStorageCapacity, logStorageWarning, logStorageCleanup } from './storage-logger'

/** localStorage quota limit in MB (conservative estimate across browsers) */
const STORAGE_QUOTA_MB = 4.8

/** Warning threshold in MB — triggers cleanup when exceeded (0.6 MB headroom to quota) */
const WARNING_THRESHOLD_MB = 4.2

/** Target size after cleanup in MB */
const TARGET_SIZE_MB = 3.5

/** Storage key for chat store */
const STORAGE_KEY = 'deep-researcher-chat-store'

/** Sessions older than this are eligible for Tier 1 cleanup */
const STALE_SESSION_MS = 24 * 60 * 60 * 1000 // 1 day

/** Maximum sessions to delete in a single cleanup pass */
const MAX_CLEANUP_DELETIONS = 10

/**
 * Calculate the size of a specific localStorage key in bytes
 */
const getKeySize = (key: string): number => {
  try {
    const value = localStorage.getItem(key)
    if (!value) return 0
    return value.length * 2
  } catch {
    return 0
  }
}

/**
 * Calculate total localStorage usage in bytes
 */
export const calculateTotalStorageSize = (): number => {
  try {
    let total = 0
    for (let i = 0; i < localStorage.length; i++) {
      const key = localStorage.key(i)
      if (key) {
        total += getKeySize(key)
      }
    }
    return total
  } catch {
    return 0
  }
}

/**
 * Calculate the size of the chat store in bytes
 */
export const calculateChatStoreSize = (): number => {
  return getKeySize(STORAGE_KEY)
}

const bytesToMB = (bytes: number): number => {
  return bytes / (1024 * 1024)
}

/**
 * Check if storage is healthy (below warning threshold)
 */
export const checkStorageHealth = (): {
  isHealthy: boolean
  currentMB: number
  percentUsed: number
} => {
  const totalBytes = calculateTotalStorageSize()
  const currentMB = bytesToMB(totalBytes)
  const percentUsed = (currentMB / STORAGE_QUOTA_MB) * 100

  return {
    isHealthy: currentMB < WARNING_THRESHOLD_MB,
    currentMB,
    percentUsed,
  }
}

/**
 * Get chat store data from localStorage
 */
const getChatStoreData = (): {
  conversations: Conversation[]
  currentConversationId: string | null
} | null => {
  try {
    const stored = localStorage.getItem(STORAGE_KEY)
    if (!stored) return null

    const parsed = JSON.parse(stored)
    const currentConversationId: string | null = parsed.state?.currentConversation ?? null

    return {
      conversations: parsed.state?.conversations ?? [],
      currentConversationId,
    }
  } catch {
    return null
  }
}

/**
 * Save chat store data back to localStorage
 */
const saveChatStoreData = (
  conversations: Conversation[],
  currentConversationId: string | null
): void => {
  try {
    const stored = localStorage.getItem(STORAGE_KEY)
    if (!stored) return

    const parsed = JSON.parse(stored)

    // Store only the ID reference (matches prunePersistedChatState format)
    parsed.state = {
      ...parsed.state,
      conversations,
      currentConversation: currentConversationId,
    }

    localStorage.setItem(STORAGE_KEY, JSON.stringify(parsed))
  } catch (error) {
    console.error('[SessionsStore] Failed to save after cleanup:', error)
  }
}

/**
 * Collect IDs of sessions that have active deep research jobs.
 * These are protected from cleanup deletion.
 */
export const getBusySessionIds = (conversations: Conversation[]): Set<string> => {
  const busy = new Set<string>()
  for (const conv of conversations) {
    if (hasActiveDeepResearchJob(conv.messages)) {
      busy.add(conv.id)
    }
  }
  return busy
}

const getUpdatedAtMs = (conv: Conversation): number =>
  new Date(conv.updatedAt as unknown as string).getTime()

/**
 * Get the oldest eligible session from a candidate list.
 */
const pickOldest = (candidates: Conversation[]): Conversation | null => {
  if (candidates.length === 0) return null
  return candidates.reduce((oldest, current) =>
    getUpdatedAtMs(current) < getUpdatedAtMs(oldest) ? current : oldest
  )
}

/**
 * Get the oldest session by updatedAt timestamp, excluding protected sessions.
 * @deprecated Use cleanupOldSessions with tiered priority instead.
 */
export const getOldestSession = (
  conversations: Conversation[],
  protectedIds: Set<string>
): Conversation | null => {
  const eligible = conversations.filter((c) => !protectedIds.has(c.id))
  return pickOldest(eligible)
}

/**
 * Clean up old sessions using tiered priority until storage is below target.
 *
 * Tier 1: Delete sessions older than 1 day from ANY user (stale first).
 * Tier 2: Delete oldest sessions of the CURRENT user (regardless of age).
 *
 * Protected sessions (current session + sessions with active deep research
 * jobs) are never deleted in either tier.
 *
 * @param currentConversationId - ID of current session to protect
 * @param currentUserId - ID of current user (for Tier 2 scoping)
 * @returns Number of sessions deleted
 */
export const cleanupOldSessions = (
  currentConversationId: string | null,
  currentUserId: string | null
): number => {
  const data = getChatStoreData()
  if (!data) return 0

  let { conversations } = data
  const deletedSessionIds: string[] = []
  const beforeMB = bytesToMB(calculateTotalStorageSize())

  const protectedIds = getBusySessionIds(conversations)
  if (currentConversationId) protectedIds.add(currentConversationId)

  const isAboveTarget = () => bytesToMB(calculateTotalStorageSize()) > TARGET_SIZE_MB
  const hitLimit = () => deletedSessionIds.length >= MAX_CLEANUP_DELETIONS

  const deleteSession = (session: Conversation) => {
    conversations = conversations.filter((c) => c.id !== session.id)
    deletedSessionIds.push(session.id)
    saveChatStoreData(conversations, currentConversationId)
  }

  // --- Tier 1: stale sessions (>1 day old) from any user, oldest first ---
  const now = Date.now()
  while (isAboveTarget() && !hitLimit()) {
    const staleCandidates = conversations.filter(
      (c) => !protectedIds.has(c.id) && now - getUpdatedAtMs(c) > STALE_SESSION_MS
    )
    const oldest = pickOldest(staleCandidates)
    if (!oldest) break
    deleteSession(oldest)
  }

  // --- Tier 2: oldest sessions of the current user ---
  if (currentUserId) {
    while (isAboveTarget() && !hitLimit()) {
      const userCandidates = conversations.filter(
        (c) => !protectedIds.has(c.id) && c.userId === currentUserId
      )
      const oldest = pickOldest(userCandidates)
      if (!oldest) break
      deleteSession(oldest)
    }
  }

  if (deletedSessionIds.length > 0) {
    const afterMB = bytesToMB(calculateTotalStorageSize())
    const freedMB = beforeMB - afterMB
    logStorageCleanup(deletedSessionIds, freedMB, beforeMB, afterMB)
  }

  return deletedSessionIds.length
}

/**
 * Ensure storage capacity by cleaning up old sessions if needed.
 * Uses tiered cleanup priority (stale > current user oldest).
 *
 * @param currentConversationId - ID of current session to protect
 * @param currentUserId - ID of current user for Tier 2 scoping
 */
export const ensureStorageCapacity = (
  currentConversationId: string | null,
  currentUserId: string | null
): void => {
  const health = checkStorageHealth()

  logStorageCapacity(health.currentMB, STORAGE_QUOTA_MB, health.percentUsed, health.isHealthy)

  if (health.isHealthy) return

  const data = getChatStoreData()
  const sessionCount = data?.conversations.length ?? 0

  logStorageWarning(health.currentMB, WARNING_THRESHOLD_MB, sessionCount)

  const deletedCount = cleanupOldSessions(currentConversationId, currentUserId)

  if (deletedCount === 0 && !checkStorageHealth().isHealthy) {
    console.warn(
      '[SessionsStore] ⚠️ Current session is too large - message pruning will occur on next save',
      {
        currentMB: health.currentMB,
        sessionCount,
      }
    )
  }
}
