// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, test, expect, vi, beforeEach, afterEach } from 'vitest'
import {
  logStorageWrite,
  logQuotaExceededPruning,
  logCriticalSessionsClear,
  logPruningFailure,
  logExternalStorageEvent,
  logStoreHydration,
  logStorageAvailability,
  logStorageCleanup,
  logStorageCapacity,
  logStorageWarning,
} from './storage-logger'
import type { Conversation } from '../types'

describe('storage-logger', () => {
  let consoleDebugSpy: ReturnType<typeof vi.spyOn>
  let consoleWarnSpy: ReturnType<typeof vi.spyOn>
  let consoleErrorSpy: ReturnType<typeof vi.spyOn>

  beforeEach(() => {
    consoleDebugSpy = vi.spyOn(console, 'debug').mockImplementation(() => {})
    consoleWarnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {})
    consoleErrorSpy = vi.spyOn(console, 'error').mockImplementation(() => {})
  })

  afterEach(() => {
    consoleDebugSpy.mockRestore()
    consoleWarnSpy.mockRestore()
    consoleErrorSpy.mockRestore()
    vi.unstubAllEnvs()
  })

  describe('logStorageWrite', () => {
    test('logs in development mode', () => {
      vi.stubEnv('NODE_ENV', 'development')

      const conversations: Conversation[] = [
        {
          id: 's_test_1',
          userId: 'user123',
          title: 'Test Session',
          messages: [],
          createdAt: new Date(),
          updatedAt: new Date(),
        },
      ]

      logStorageWrite(conversations, 'user123')

      expect(consoleDebugSpy).toHaveBeenCalledWith(
        expect.stringContaining('[SessionsStore]'),
        expect.objectContaining({
          userId: 'user123',
          sessionIds: ['s_test_1'],
        })
      )
    })

    test('does not log in production mode', () => {
      vi.stubEnv('NODE_ENV', 'production')

      const conversations: Conversation[] = []
      logStorageWrite(conversations, 'user123')

      expect(consoleDebugSpy).not.toHaveBeenCalled()
    })
  })

  describe('logQuotaExceededPruning', () => {
    test('logs pruning attempt in development', () => {
      vi.stubEnv('NODE_ENV', 'development')

      logQuotaExceededPruning(5, 3, 100, 60)

      expect(consoleWarnSpy).toHaveBeenCalledWith(
        expect.stringContaining('QUOTA EXCEEDED'),
        expect.objectContaining({
          before: { sessions: 5, sizeKB: 100 },
          after: { sessions: 3, sizeKB: 60 },
        })
      )
    })

    test('does not log in production', () => {
      vi.stubEnv('NODE_ENV', 'production')

      logQuotaExceededPruning(5, 3, 100, 60)

      expect(consoleWarnSpy).not.toHaveBeenCalled()
    })
  })

  describe('logCriticalSessionsClear', () => {
    test('ALWAYS logs critical data loss (even in production)', () => {
      vi.stubEnv('NODE_ENV', 'production')

      const lostSessions = ['s_test_1', 's_test_2', 's_test_3']
      const error = new Error('QuotaExceededError')

      logCriticalSessionsClear('user123', lostSessions, error)

      expect(consoleErrorSpy).toHaveBeenCalledWith(
        expect.stringContaining('CRITICAL'),
        expect.objectContaining({
          userId: 'user123',
          lostSessions,
          sessionCount: 3,
          error: 'QuotaExceededError',
        })
      )
    })

    test('logs in development mode', () => {
      vi.stubEnv('NODE_ENV', 'development')

      logCriticalSessionsClear('user123', ['s_test_1'], new Error('Test'))

      expect(consoleErrorSpy).toHaveBeenCalled()
    })
  })

  describe('logPruningFailure', () => {
    test('ALWAYS logs pruning failure (even in production)', () => {
      vi.stubEnv('NODE_ENV', 'production')

      const error = new Error('Failed to prune')
      logPruningFailure(error)

      expect(consoleErrorSpy).toHaveBeenCalledWith(
        expect.stringContaining('Pruning failed'),
        expect.objectContaining({
          error: 'Failed to prune',
        })
      )
    })
  })

  describe('logExternalStorageEvent', () => {
    test('logs storage events in development', () => {
      vi.stubEnv('NODE_ENV', 'development')

      logExternalStorageEvent('deep-researcher-chat-store', 'oldValue', null)

      expect(consoleWarnSpy).toHaveBeenCalledWith(
        expect.stringContaining('Storage event detected'),
        expect.objectContaining({
          key: 'deep-researcher-chat-store',
          cleared: true,
        })
      )
    })

    test('does not log in production', () => {
      vi.stubEnv('NODE_ENV', 'production')

      logExternalStorageEvent('deep-researcher-chat-store', 'old', 'new')

      expect(consoleWarnSpy).not.toHaveBeenCalled()
    })
  })

  describe('logStoreHydration', () => {
    test('logs successful hydration in development', () => {
      vi.stubEnv('NODE_ENV', 'development')

      logStoreHydration(true, 3, 'user123')

      expect(consoleDebugSpy).toHaveBeenCalledWith(
        expect.stringContaining('Store hydrated'),
        expect.objectContaining({
          userId: 'user123',
        })
      )
    })

    test('logs failed hydration in development', () => {
      vi.stubEnv('NODE_ENV', 'development')

      logStoreHydration(false, 0, null)

      expect(consoleWarnSpy).toHaveBeenCalledWith(
        expect.stringContaining('hydration failed'),
        expect.any(Object)
      )
    })
  })

  describe('logStorageAvailability', () => {
    test('logs when storage is unavailable in development', () => {
      vi.stubEnv('NODE_ENV', 'development')

      logStorageAvailability(false)

      expect(consoleWarnSpy).toHaveBeenCalledWith(
        expect.stringContaining('localStorage not available'),
        expect.any(Object)
      )
    })

    test('does not log when storage is available', () => {
      vi.stubEnv('NODE_ENV', 'development')

      logStorageAvailability(true)

      expect(consoleWarnSpy).not.toHaveBeenCalled()
    })
  })

  describe('logStorageCleanup', () => {
    test('ALWAYS logs cleanup (even in production)', () => {
      vi.stubEnv('NODE_ENV', 'production')

      const deletedSessions = ['s_old_1', 's_old_2']
      logStorageCleanup(deletedSessions, 1.5, 4.5, 3.0)

      expect(consoleWarnSpy).toHaveBeenCalledWith(
        expect.stringContaining('Auto-cleanup'),
        expect.objectContaining({
          deletedSessionIds: deletedSessions,
          freedSpaceMB: '1.50',
          beforeStorageMB: '4.50',
          afterStorageMB: '3.00',
        })
      )
    })

    test('logs in development mode', () => {
      vi.stubEnv('NODE_ENV', 'development')

      logStorageCleanup(['s_old'], 0.5, 4.1, 3.6)

      expect(consoleWarnSpy).toHaveBeenCalled()
    })
  })

  describe('logStorageCapacity', () => {
    test('logs capacity in development when healthy', () => {
      vi.stubEnv('NODE_ENV', 'development')

      logStorageCapacity(2.5, 5, 50, true)

      expect(consoleDebugSpy).toHaveBeenCalledWith(
        expect.stringContaining('✓ Storage'),
        expect.any(Object)
      )
    })

    test('logs capacity in development when unhealthy', () => {
      vi.stubEnv('NODE_ENV', 'development')

      logStorageCapacity(4.2, 5, 84, false)

      expect(consoleDebugSpy).toHaveBeenCalledWith(
        expect.stringContaining('⚠️ Storage'),
        expect.any(Object)
      )
    })

    test('does not log in production', () => {
      vi.stubEnv('NODE_ENV', 'production')

      logStorageCapacity(3.0, 5, 60, true)

      expect(consoleDebugSpy).not.toHaveBeenCalled()
    })
  })

  describe('logStorageWarning', () => {
    test('ALWAYS logs warning when over threshold', () => {
      vi.stubEnv('NODE_ENV', 'production')

      logStorageWarning(4.3, 4.0, 8)

      expect(consoleWarnSpy).toHaveBeenCalledWith(
        expect.stringContaining('Storage approaching limit'),
        expect.objectContaining({
          currentStorageMB: '4.30',
          thresholdMB: 4.0,
          sessionCount: 8,
        })
      )
    })

    test('logs in development mode', () => {
      vi.stubEnv('NODE_ENV', 'development')

      logStorageWarning(4.1, 4.0, 5)

      expect(consoleWarnSpy).toHaveBeenCalled()
    })
  })
})
