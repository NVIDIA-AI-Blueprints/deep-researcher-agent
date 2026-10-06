// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Root Layout
 *
 * The root layout for the entire application.
 * Sets up the HTML structure, global styles, and providers.
 *
 * Server-side environment variables are read here and passed to client
 * providers, enabling runtime configuration without rebuilding.
 */

import { type ReactNode } from 'react'
import { type Metadata } from 'next'
import { connection } from 'next/server'
import { Providers } from './providers'
import type { AppConfig } from '@/shared/context'
import { getFileUploadConfigFromEnv } from '@/shared/config/file-upload'
import { isAuthRequired, AUTH_PROVIDER_ID, TOKEN_REFRESH_BUFFER_SECONDS } from '@/adapters/auth/config'
import './globals.css'

export const metadata: Metadata = {
  title: 'Deep Researcher Agent',
  description: 'AI-powered research assistant',
  icons: {
    icon: '/favicon.ico',
  },
}

/**
 * Runtime configuration from server-side environment variables.
 * These values can be changed at runtime without rebuilding the container.
 */
const getAppConfig = (): AppConfig => ({
  authRequired: isAuthRequired(),
  authProviderId: AUTH_PROVIDER_ID,
  sessionRefreshIntervalSeconds: Math.max(60, TOKEN_REFRESH_BUFFER_SECONDS - 60),
  fileUpload: getFileUploadConfigFromEnv(process.env),
})

interface RootLayoutProps {
  children: ReactNode
}

const RootLayout = async ({ children }: RootLayoutProps): Promise<ReactNode> => {
  await connection()
  const config = getAppConfig()

  return (
    <html lang="en" id="style-root" suppressHydrationWarning>
      <head>
        {/* CDN SVG icon loader - inlines <svg data-src="..."> elements */}
        <script
          src="https://unpkg.com/external-svg-loader@1.6.8/svg-loader.min.js"
          async
        />
      </head>
      <body className="bg-surface-base">
        <Providers config={config}>{children}</Providers>
      </body>
    </html>
  )
}

export default RootLayout
