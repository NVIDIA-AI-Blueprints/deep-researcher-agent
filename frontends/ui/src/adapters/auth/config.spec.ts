// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { afterEach, describe, expect, test, vi } from 'vitest'
import type { AuthProviderConfig } from './providers/types'

const loadConfig = async () => {
  vi.resetModules()
  return import('./config')
}

/**
 * Load config with a custom provider config (for testing hooks).
 * Mocks the providers module before config.ts is imported.
 */
const loadConfigWithProvider = async (overrides: Partial<AuthProviderConfig>) => {
  vi.resetModules()
  const base: AuthProviderConfig = {
    provider: { id: 'test-provider', name: 'Test' },
    providerId: 'test-provider',
    refreshToken: async () => ({ access_token: 'new', expires_in: 3600 }),
    ...overrides,
  }
  vi.doMock('./providers', () => ({
    getAuthProviderConfig: () => base,
  }))
  return import('./config')
}

describe('isAuthRequired', () => {
  afterEach(() => {
    vi.unstubAllEnvs()
    vi.resetModules()
  })

  test('returns true when REQUIRE_AUTH is lowercase true', async () => {
    vi.stubEnv('REQUIRE_AUTH', 'true')
    const { isAuthRequired } = await loadConfig()
    expect(isAuthRequired()).toBe(true)
  })

  test('returns true when REQUIRE_AUTH is uppercase TRUE', async () => {
    vi.stubEnv('REQUIRE_AUTH', 'TRUE')
    const { isAuthRequired } = await loadConfig()
    expect(isAuthRequired()).toBe(true)
  })

  test('returns false when REQUIRE_AUTH is set to non-true value', async () => {
    vi.stubEnv('REQUIRE_AUTH', 'false')
    const { isAuthRequired } = await loadConfig()
    expect(isAuthRequired()).toBe(false)
  })
})

describe('auth timing config', () => {
  afterEach(() => {
    vi.unstubAllEnvs()
    vi.resetModules()
  })

  test('uses defaults when timing env vars are unset', async () => {
    const { TOKEN_REFRESH_BUFFER_SECONDS, SESSION_MAX_AGE_SECONDS } = await loadConfig()

    expect(TOKEN_REFRESH_BUFFER_SECONDS).toBe(15 * 60)
    expect(SESSION_MAX_AGE_SECONDS).toBe(24 * 60 * 60)
  })

  test('falls back to defaults when timing env vars are invalid', async () => {
    vi.stubEnv('TOKEN_REFRESH_BUFFER_MINUTES', 'abc')
    vi.stubEnv('SESSION_MAX_AGE_HOURS', 'NaN')

    const { TOKEN_REFRESH_BUFFER_SECONDS, SESSION_MAX_AGE_SECONDS } = await loadConfig()

    expect(TOKEN_REFRESH_BUFFER_SECONDS).toBe(15 * 60)
    expect(SESSION_MAX_AGE_SECONDS).toBe(24 * 60 * 60)
  })

  test('falls back to defaults when timing env vars are non-positive', async () => {
    vi.stubEnv('TOKEN_REFRESH_BUFFER_MINUTES', '0')
    vi.stubEnv('SESSION_MAX_AGE_HOURS', '-1')

    const { TOKEN_REFRESH_BUFFER_SECONDS, SESSION_MAX_AGE_SECONDS } = await loadConfig()

    expect(TOKEN_REFRESH_BUFFER_SECONDS).toBe(15 * 60)
    expect(SESSION_MAX_AGE_SECONDS).toBe(24 * 60 * 60)
  })

  test('uses configured timing env vars when valid', async () => {
    vi.stubEnv('TOKEN_REFRESH_BUFFER_MINUTES', '30')
    vi.stubEnv('SESSION_MAX_AGE_HOURS', '12')

    const { TOKEN_REFRESH_BUFFER_SECONDS, SESSION_MAX_AGE_SECONDS } = await loadConfig()

    expect(TOKEN_REFRESH_BUFFER_SECONDS).toBe(30 * 60)
    expect(SESSION_MAX_AGE_SECONDS).toBe(12 * 60 * 60)
  })
})

describe('auth jwt refresh behavior', () => {
  afterEach(() => {
    vi.unstubAllEnvs()
    vi.resetModules()
  })

  test('does not force refresh when refresh token exists but expiresAt is absent', async () => {
    const { authOptions } = await loadConfig()
    const token = {
      accessToken: 'access-token',
      refreshToken: 'refresh-token',
      userId: 'user-1',
    }

    const result = await authOptions.callbacks!.jwt!({
      token,
      account: null,
      user: {
        id: 'user-1',
        name: null,
        email: null,
        image: null,
      },
      profile: undefined,
      trigger: undefined,
      isNewUser: false,
      session: undefined,
    })

    expect(result).toEqual(token)
    expect(result.error).toBeUndefined()
  })
})

describe('provider lifecycle hooks', () => {
  afterEach(() => {
    vi.unstubAllEnvs()
    vi.resetModules()
    vi.restoreAllMocks()
  })

  test('onSignIn hook merges extra claims into JWT on initial sign-in', async () => {
    vi.stubEnv('REQUIRE_AUTH', 'true')
    vi.stubEnv('NEXTAUTH_SECRET', 'test-secret')

    const onSignIn = vi.fn().mockResolvedValue({ hasAccess: true, groupName: 'testers' })
    const { authOptions } = await loadConfigWithProvider({ onSignIn })

    const result = await authOptions.callbacks!.jwt!({
      token: {},
      account: {
        access_token: 'at',
        id_token: 'it',
        refresh_token: 'rt',
        expires_at: 9999999999,
        provider: 'test-provider',
        type: 'oauth',
        providerAccountId: 'pa1',
      },
      user: { id: 'u1', name: 'Test', email: 'test@example.com', image: null },
      profile: undefined,
      trigger: undefined,
      isNewUser: false,
      session: undefined,
    })

    expect(onSignIn).toHaveBeenCalledOnce()
    expect(result.hasAccess).toBe(true)
    expect(result.groupName).toBe('testers')
    expect(result.accessToken).toBe('at')
    expect(result.userId).toBe('u1')
  })

  test('onSignIn hook cannot override core JWT fields', async () => {
    vi.stubEnv('REQUIRE_AUTH', 'true')
    vi.stubEnv('NEXTAUTH_SECRET', 'test-secret')

    const onSignIn = vi.fn().mockResolvedValue({
      accessToken: 'hook-at',
      refreshToken: 'hook-rt',
      expiresAt: 1,
      userId: 'hook-user',
      hasAccess: true,
    })
    const { authOptions } = await loadConfigWithProvider({ onSignIn })

    const result = await authOptions.callbacks!.jwt!({
      token: {},
      account: {
        access_token: 'at',
        id_token: 'it',
        refresh_token: 'rt',
        expires_at: 9999999999,
        provider: 'test-provider',
        type: 'oauth',
        providerAccountId: 'pa1',
      },
      user: { id: 'u1', name: 'Test', email: 'test@example.com', image: null },
      profile: undefined,
      trigger: undefined,
      isNewUser: false,
      session: undefined,
    })

    expect(result.accessToken).toBe('at')
    expect(result.refreshToken).toBe('rt')
    expect(result.expiresAt).toBe(9999999999)
    expect(result.userId).toBe('u1')
    expect(result.hasAccess).toBe(true)
  })

  test('onSignIn hook failure falls back to base JWT fields', async () => {
    vi.stubEnv('REQUIRE_AUTH', 'true')
    vi.stubEnv('NEXTAUTH_SECRET', 'test-secret')

    const consoleError = vi.spyOn(console, 'error').mockImplementation(() => {})
    const onSignIn = vi.fn().mockRejectedValue(new Error('group lookup failed'))
    const { authOptions } = await loadConfigWithProvider({ onSignIn })

    const result = await authOptions.callbacks!.jwt!({
      token: {},
      account: {
        access_token: 'at',
        id_token: 'it',
        refresh_token: 'rt',
        expires_at: 9999999999,
        provider: 'test-provider',
        type: 'oauth',
        providerAccountId: 'pa1',
      },
      user: { id: 'u1', name: 'Test', email: 'test@example.com', image: null },
      profile: undefined,
      trigger: undefined,
      isNewUser: false,
      session: undefined,
    })

    expect(onSignIn).toHaveBeenCalledOnce()
    expect(consoleError).toHaveBeenCalledWith('[Auth] onSignIn hook failed:', expect.any(Error))
    expect(result.accessToken).toBe('at')
    expect(result.idToken).toBe('it')
    expect(result.refreshToken).toBe('rt')
    expect(result.expiresAt).toBe(9999999999)
    expect(result.userId).toBe('u1')
    expect(result.hasAccess).toBeUndefined()
  })

  test('onSession hook merges extra fields into session', async () => {
    vi.stubEnv('REQUIRE_AUTH', 'true')
    vi.stubEnv('NEXTAUTH_SECRET', 'test-secret')

    const onSession = vi.fn().mockReturnValue({ hasAccess: true, dlGroup: 'deep-researcher-users' })
    const { authOptions } = await loadConfigWithProvider({ onSession })

    const result = await authOptions.callbacks!.session!({
      session: { user: { name: 'Test' }, expires: '2099-01-01' },
      token: {
        accessToken: 'at',
        idToken: 'it',
        expiresAt: 9999999999,
        userId: 'u1',
        hasAccess: true,
        dlGroup: 'deep-researcher-users',
      },
      user: { id: 'u1', name: 'Test', email: 'test@example.com', image: null, emailVerified: null },
      trigger: 'update',
      newSession: undefined,
    })

    expect(onSession).toHaveBeenCalledOnce()
    // Session return type doesn't include provider-specific fields in its
    // static type — access via bracket notation (they exist at runtime)
    const sessionObj = result as unknown as Record<string, unknown>
    expect(sessionObj.hasAccess).toBe(true)
    expect(sessionObj.dlGroup).toBe('deep-researcher-users')
    expect(sessionObj.idToken).toBe('it')
    expect(sessionObj.idTokenExpiresAt).toBe(9999999999)
  })

  test('onSession hook cannot override core session fields', async () => {
    vi.stubEnv('REQUIRE_AUTH', 'true')
    vi.stubEnv('NEXTAUTH_SECRET', 'test-secret')

    const onSession = vi.fn().mockReturnValue({
      accessToken: 'hook-at',
      idToken: 'hook-it',
      userId: 'hook-user',
      error: 'hook-error',
      hasAccess: true,
    })
    const { authOptions } = await loadConfigWithProvider({ onSession })

    const result = await authOptions.callbacks!.session!({
      session: { user: { name: 'Test' }, expires: '2099-01-01' },
      token: {
        accessToken: 'at',
        idToken: 'it',
        expiresAt: 9999999999,
        userId: 'u1',
        error: undefined,
      },
      user: { id: 'u1', name: 'Test', email: 'test@example.com', image: null, emailVerified: null },
      trigger: 'update',
      newSession: undefined,
    })

    const sessionObj = result as unknown as Record<string, unknown>
    expect(sessionObj.accessToken).toBe('at')
    expect(sessionObj.idToken).toBe('it')
    expect(sessionObj.idTokenExpiresAt).toBe(9999999999)
    expect(sessionObj.userId).toBe('u1')
    expect(sessionObj.error).toBeUndefined()
    expect(sessionObj.hasAccess).toBe(true)
  })

  test('config works without hooks (backward compatible)', async () => {
    const { authOptions } = await loadConfigWithProvider({})

    // Initial sign-in without onSignIn hook
    const jwtResult = await authOptions.callbacks!.jwt!({
      token: {},
      account: {
        access_token: 'at',
        id_token: 'it',
        refresh_token: 'rt',
        expires_at: 9999999999,
        provider: 'test-provider',
        type: 'oauth',
        providerAccountId: 'pa1',
      },
      user: { id: 'u1', name: 'Test' },
      profile: undefined,
      trigger: undefined,
      isNewUser: false,
      session: undefined,
    })

    expect(jwtResult.accessToken).toBe('at')
    expect(jwtResult.userId).toBe('u1')

    // Session without onSession hook
    const sessionResult = await authOptions.callbacks!.session!({
      session: { user: { name: 'Test' }, expires: '2099-01-01' },
      token: { accessToken: 'at', idToken: 'it', expiresAt: 9999999999, userId: 'u1' },
      user: { id: 'u1', name: 'Test', email: 'test@example.com', emailVerified: null },
      trigger: 'update',
      newSession: undefined,
    }) as unknown as Record<string, unknown>

    expect(sessionResult.idToken).toBe('it')
    expect(sessionResult.idTokenExpiresAt).toBe(9999999999)
    // No extra fields from hooks
    expect(sessionResult.hasAccess).toBeUndefined()
  })

  test('tokenRefreshBufferSeconds overrides env var default', async () => {
    vi.stubEnv('TOKEN_REFRESH_BUFFER_MINUTES', '20')
    const { TOKEN_REFRESH_BUFFER_SECONDS } = await loadConfigWithProvider({
      tokenRefreshBufferSeconds: 42 * 60,
    })

    // Provider override takes precedence over env var
    expect(TOKEN_REFRESH_BUFFER_SECONDS).toBe(42 * 60)
  })

  test('env var is used when provider does not set tokenRefreshBufferSeconds', async () => {
    vi.stubEnv('TOKEN_REFRESH_BUFFER_MINUTES', '20')
    const { TOKEN_REFRESH_BUFFER_SECONDS } = await loadConfigWithProvider({})

    expect(TOKEN_REFRESH_BUFFER_SECONDS).toBe(20 * 60)
  })
})
