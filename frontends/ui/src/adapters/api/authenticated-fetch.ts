// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Authenticated Fetch Wrapper
 *
 * Provides a fetch wrapper that automatically adds authentication headers
 * to requests sent to the Deep Researcher Agent backend.
 */

import { getSession } from 'next-auth/react'
import { trackAuthEvent } from '@/shared/utils/rum'

export interface AuthenticatedFetchOptions extends RequestInit {
  /** Skip authentication header (for public endpoints) */
  skipAuth?: boolean
}

/**
 * Fetch wrapper that automatically adds authentication headers
 *
 * Uses the ID token from the NextAuth session for backend authentication.
 * The token is passed via the Authorization header as a Bearer token
 *
 * @param url - The URL to fetch
 * @param options - Fetch options with optional auth overrides
 * @returns The fetch response
 *
 * @example
 * ```typescript
 * // Automatically adds auth header
 * const response = await authenticatedFetch('/api/chat', {
 *   method: 'POST',
 *   body: JSON.stringify({ message: 'Hello' }),
 * })
 *
 * // Skip auth for public endpoints
 * const response = await authenticatedFetch('/api/health', { skipAuth: true })
 * ```
 */
export const authenticatedFetch = async (
  url: string,
  options: AuthenticatedFetchOptions = {}
): Promise<Response> => {
  const { skipAuth, ...fetchOptions } = options

  // Prepare headers
  const headers = new Headers(fetchOptions.headers)

  // Add auth header if not skipped
  if (!skipAuth) {
    try {
      const session = await getSession()
      const token = session?.idToken || session?.accessToken

      if (token) {
        headers.set('Authorization', `Bearer ${token}`)
      }
    } catch (error) {
      console.warn('[authenticatedFetch] Failed to get session:', error)
    }
  }

  // Ensure content-type is set for JSON requests
  if (!headers.has('Content-Type') && fetchOptions.body) {
    headers.set('Content-Type', 'application/json')
  }

  // Return fetch with auth headers and credentials
  const response = await fetch(url, {
    ...fetchOptions,
    headers,
    credentials: fetchOptions.credentials || 'include', // Important for CORS
  })

  // Emit auth error for observability (Datadog RUM or any APM).
  if (response.status === 401) {
    try {
      const body = await response.clone().json()
      const errorCode = body?.error || 'unknown'
      trackAuthEvent(errorCode, { path: url })
    } catch {
      // Response may not be JSON (e.g. HTML error page) — skip silently
    }
  }

  return response
}

/**
 * Create an authenticated fetch function with a pre-set token
 *
 * Useful for components that already have the token from useAuth()
 * to avoid async session lookups on each request.
 *
 * @param token - The ID token or access token to use
 * @returns A fetch function that uses the provided token
 *
 * @example
 * ```typescript
 * const { idToken } = useAuth()
 * const authFetch = createAuthenticatedFetch(idToken)
 *
 * const response = await authFetch('/api/chat', {
 *   method: 'POST',
 *   body: JSON.stringify({ message: 'Hello' }),
 * })
 * ```
 */
export const createAuthenticatedFetch = (token?: string) => {
  return async (url: string, options: RequestInit = {}): Promise<Response> => {
    const headers = new Headers(options.headers)

    if (token) {
      headers.set('Authorization', `Bearer ${token}`)
    }

    if (!headers.has('Content-Type') && options.body) {
      headers.set('Content-Type', 'application/json')
    }

    return fetch(url, {
      ...options,
      headers,
      credentials: options.credentials || 'include',
    })
  }
}

/**
 * Hook-friendly version that returns the auth token for manual use
 *
 * @param session - The session object from useAuth or getSession
 * @returns The best available token for backend auth
 */
export const getAuthToken = (
  session: {
    idToken?: string
    accessToken?: string
  } | null
): string | undefined => {
  return session?.idToken || session?.accessToken
}
