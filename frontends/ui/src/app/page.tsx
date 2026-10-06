// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Home Page
 *
 * Main chat interface using the MainLayout component.
 * Displays the full Deep Researcher Agent experience.
 * Chat state is managed via the useChatStore.
 *
 * Passes auth state to MainLayout for conditional UI rendering.
 */

'use client'

import { type ReactNode, Suspense } from 'react'
import { useAuth } from '@/adapters/auth'
import { MainLayout } from '@/features/layout'

const HomeContent = (): ReactNode => {
  const { user, isAuthenticated, isLoading, authRequired, signIn, signOut } = useAuth()

  return (
    <MainLayout
      isAuthenticated={isAuthenticated}
      isLoading={isLoading}
      authRequired={authRequired}
      user={
        isAuthenticated
          ? {
              name: user?.name || undefined,
              email: user?.email || undefined,
              image: user?.image || undefined,
            }
          : undefined
      }
      onSignIn={signIn}
      onSignOut={signOut}
    />
  )
}

const HomePage = (): ReactNode => {
  return (
    <Suspense fallback={null}>
      <HomeContent />
    </Suspense>
  )
}

export default HomePage
