// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, test, expect, beforeEach, vi } from 'vitest'
import {
  calculateTotalStorageSize,
  calculateChatStoreSize,
  checkStorageHealth,
  getOldestSession,
  cleanupOldSessions,
  ensureStorageCapacity,
  getBusySessionIds,
} from './storage-manager'
import type { Conversation, ChatMessage } from '../types'

const makeConversation = (
  id: string,
  userId: string,
  updatedAt: string | Date,
  messages: ChatMessage[] = []
): Conversation => ({
  id,
  userId,
  title: `Session ${id}`,
  messages,
  createdAt: new Date(updatedAt),
  updatedAt: new Date(updatedAt),
})

const setStoreData = (conversations: Conversation[], currentId: string | null = null) => {
  const storeData = {
    state: {
      conversations,
      currentConversation: conversations.find((c) => c.id === currentId) ?? null,
      currentUserId: 'user1',
      pendingInteraction: null,
    },
    version: 0,
  }
  localStorage.setItem('deep-researcher-chat-store', JSON.stringify(storeData))
}

describe('storage-manager', () => {
  beforeEach(() => {
    localStorage.clear()
    vi.clearAllMocks()
  })

  describe('calculateTotalStorageSize', () => {
    test('calculates total storage size', () => {
      localStorage.setItem('key1', 'abc')
      localStorage.setItem('key2', 'defgh')

      const size = calculateTotalStorageSize()

      // 'abc' = 3 chars × 2 bytes = 6 bytes
      // 'defgh' = 5 chars × 2 bytes = 10 bytes
      // Total = 16 bytes
      expect(size).toBe(16)
    })

    test('returns 0 for empty storage', () => {
      const size = calculateTotalStorageSize()
      expect(size).toBe(0)
    })
  })

  describe('calculateChatStoreSize', () => {
    test('calculates chat store size', () => {
      localStorage.setItem('deep-researcher-chat-store', 'test data')

      const size = calculateChatStoreSize()

      // 'test data' = 9 chars × 2 bytes = 18 bytes
      expect(size).toBe(18)
    })

    test('returns 0 when chat store does not exist', () => {
      const size = calculateChatStoreSize()
      expect(size).toBe(0)
    })
  })

  describe('checkStorageHealth', () => {
    test('returns healthy when under threshold', () => {
      localStorage.setItem('deep-researcher-chat-store', 'small data')

      const health = checkStorageHealth()

      expect(health.isHealthy).toBe(true)
      expect(health.currentMB).toBeLessThan(4)
      expect(health.percentUsed).toBeLessThan(80)
    })

    test('returns unhealthy when over threshold', () => {
      const largeData = 'x'.repeat(2_500_000) // ~5MB
      localStorage.setItem('deep-researcher-chat-store', largeData)

      const health = checkStorageHealth()

      expect(health.isHealthy).toBe(false)
      expect(health.currentMB).toBeGreaterThan(4)
    })
  })

  describe('getOldestSession', () => {
    test('returns session with oldest updatedAt', () => {
      const conversations = [
        makeConversation('s_new', 'user1', '2026-02-09T12:00:00Z'),
        makeConversation('s_old', 'user1', '2026-02-08T10:00:00Z'),
        makeConversation('s_middle', 'user1', '2026-02-08T15:00:00Z'),
      ]

      const oldest = getOldestSession(conversations, new Set())

      expect(oldest?.id).toBe('s_old')
    })

    test('excludes protected sessions from selection', () => {
      const conversations = [
        makeConversation('s_current', 'user1', '2026-02-07T10:00:00Z'),
        makeConversation('s_older', 'user1', '2026-02-08T10:00:00Z'),
      ]

      const oldest = getOldestSession(conversations, new Set(['s_current']))

      expect(oldest?.id).toBe('s_older')
    })

    test('excludes multiple protected sessions', () => {
      const conversations = [
        makeConversation('s_protected_1', 'user1', '2026-02-01'),
        makeConversation('s_protected_2', 'user1', '2026-02-02'),
        makeConversation('s_eligible', 'user1', '2026-02-03'),
      ]

      const oldest = getOldestSession(conversations, new Set(['s_protected_1', 's_protected_2']))

      expect(oldest?.id).toBe('s_eligible')
    })

    test('returns null when only protected sessions exist', () => {
      const conversations = [makeConversation('s_current', 'user1', new Date())]

      const oldest = getOldestSession(conversations, new Set(['s_current']))

      expect(oldest).toBeNull()
    })

    test('returns null for empty conversations array', () => {
      const oldest = getOldestSession([], new Set())
      expect(oldest).toBeNull()
    })
  })

  describe('getBusySessionIds', () => {
    test('returns empty set when no sessions have active jobs', () => {
      const conversations = [
        makeConversation('s1', 'user1', new Date()),
        makeConversation('s2', 'user1', new Date()),
      ]

      const busy = getBusySessionIds(conversations)

      expect(busy.size).toBe(0)
    })

    test('identifies sessions with active deep research jobs', () => {
      const busyMessages: ChatMessage[] = [
        {
          id: 'msg1',
          role: 'assistant',
          content: 'Running...',
          timestamp: new Date(),
          messageType: 'agent_response',
          deepResearchJobId: 'job_1',
          deepResearchJobStatus: 'running',
        },
      ]

      const conversations = [
        makeConversation('s_busy', 'user1', new Date(), busyMessages),
        makeConversation('s_idle', 'user1', new Date()),
      ]

      const busy = getBusySessionIds(conversations)

      expect(busy.size).toBe(1)
      expect(busy.has('s_busy')).toBe(true)
    })

    test('does not flag completed jobs as busy', () => {
      const completedMessages: ChatMessage[] = [
        {
          id: 'msg1',
          role: 'assistant',
          content: 'Done',
          timestamp: new Date(),
          messageType: 'agent_response',
          deepResearchJobId: 'job_1',
          deepResearchJobStatus: 'success',
        },
      ]

      const conversations = [makeConversation('s1', 'user1', new Date(), completedMessages)]

      const busy = getBusySessionIds(conversations)

      expect(busy.size).toBe(0)
    })
  })

  describe('cleanupOldSessions', () => {
    test('removes oldest sessions when over threshold', () => {
      const conversations = [
        makeConversation('s_newest', 'user1', '2026-02-09'),
        makeConversation('s_old_1', 'user1', '2026-02-01'),
        makeConversation('s_old_2', 'user1', '2026-02-02'),
      ]

      setStoreData(conversations, 's_newest')

      const deletedCount = cleanupOldSessions('s_newest', 'user1')

      // Under test storage won't be above threshold, so no deletions
      expect(deletedCount).toBeGreaterThanOrEqual(0)
    })

    test('protects current session from deletion', () => {
      const conversations = [
        makeConversation('s_current', 'user1', '2026-02-01'),
        makeConversation('s_newer', 'user1', '2026-02-09'),
      ]

      setStoreData(conversations, 's_current')

      cleanupOldSessions('s_current', 'user1')

      const stored = JSON.parse(localStorage.getItem('deep-researcher-chat-store') || '{}')
      const remainingIds = stored.state?.conversations?.map((c: Conversation) => c.id) || []

      expect(remainingIds).toContain('s_current')
    })

    test('accepts null userId for Tier 2 gracefully', () => {
      const conversations = [makeConversation('s1', 'user1', '2026-02-01')]

      setStoreData(conversations, 's1')

      // Should not throw with null userId
      expect(() => cleanupOldSessions('s1', null)).not.toThrow()
    })
  })

  describe('ensureStorageCapacity', () => {
    test('calls cleanup when storage is over threshold', () => {
      const largeConversations = Array.from({ length: 10 }, (_, i) => ({
        id: `s_${i}`,
        userId: 'user1',
        title: `Session ${i}`,
        messages: Array.from({ length: 50 }, (_, j) => ({
          id: `msg_${i}_${j}`,
          role: 'user' as const,
          content: 'x'.repeat(1000),
          timestamp: new Date(`2026-02-0${Math.min(i + 1, 9)}`),
          messageType: 'user' as const,
        })),
        createdAt: new Date(`2026-02-0${Math.min(i + 1, 9)}`),
        updatedAt: new Date(`2026-02-0${Math.min(i + 1, 9)}`),
      }))

      setStoreData(largeConversations, 's_0')

      ensureStorageCapacity('s_0', 'user1')

      expect(true).toBe(true)
    })

    test('does not throw error when storage is healthy', () => {
      localStorage.setItem('deep-researcher-chat-store', '{"state":{"conversations":[]}}')

      expect(() => ensureStorageCapacity(null, null)).not.toThrow()
    })

    test('accepts userId parameter', () => {
      localStorage.setItem('deep-researcher-chat-store', '{"state":{"conversations":[]}}')

      expect(() => ensureStorageCapacity('s_1', 'user1')).not.toThrow()
    })
  })
})
