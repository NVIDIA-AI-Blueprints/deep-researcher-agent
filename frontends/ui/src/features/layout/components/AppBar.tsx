// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * AppBar Component
 *
 * Top navigation bar with menu toggle, logo, session title,
 * and action buttons (Add Sources, User Avatar).
 *
 * Shows different states based on authentication:
 * - Auth disabled: Default User avatar with info tooltip (no sign in/out)
 * - Logged out: Sign In button, disabled action buttons
 * - Logged in: User avatar with dropdown menu
 */

'use client'

import { type FC, memo, useCallback, useEffect, useState } from 'react'
import { Flex, Text, Button, Logo, Avatar, Popover, Divider } from '@/adapters/ui'
import { Globe, Book, Lock, Logout, OpenExternal, Info, Moon, Sun } from '@/adapters/ui/icons'
import { useLayoutStore } from '../store'
import { cn } from '@/shared/lib/cn'
import type { ThemeMode } from '../types'

interface AppBarProps {
  /** Current session title to display */
  sessionTitle?: string
  /** Whether the user is authenticated */
  isAuthenticated?: boolean
  /** Whether authentication is required (false = using default user) */
  authRequired?: boolean
  /** User info for avatar */
  user?: {
    name?: string
    email?: string
    image?: string
  }
  /** Callback when a new session is requested */
  onNewSession?: () => void
  /** Disable creating a new session while shallow research/HITL is active */
  isNewSessionDisabled?: boolean
  /** Callback when sign in is clicked */
  onSignIn?: () => void
  /** Callback when sign out is clicked */
  onSignOut?: () => void
}

/**
 * Main navigation bar at the top of the application.
 * Controls sidebar toggles and navigation actions.
 */
export const AppBar: FC<AppBarProps> = memo(function AppBar({
  sessionTitle = '',
  isAuthenticated = false,
  authRequired = false,
  user,
  onNewSession,
  isNewSessionDisabled = false,
  onSignIn,
  onSignOut,
}) {
  const rightPanel = useLayoutStore((s) => s.rightPanel)
  const isDataSourcesOpen = rightPanel === 'data-sources'
  const theme = useLayoutStore((s) => s.theme)
  const setTheme = useLayoutStore((s) => s.setTheme)
  const [isUserMenuOpen, setIsUserMenuOpen] = useState(false)
  const [systemPrefersDark, setSystemPrefersDark] = useState(false)

  useEffect(() => {
    if (typeof window === 'undefined' || !window.matchMedia) return
    const mq = window.matchMedia('(prefers-color-scheme: dark)')
    setSystemPrefersDark(mq.matches)
    const handler = (e: MediaQueryListEvent): void => setSystemPrefersDark(e.matches)
    mq.addEventListener?.('change', handler)
    return () => mq.removeEventListener?.('change', handler)
  }, [])

  const isDarkMode = theme === 'dark' || (theme === 'system' && systemPrefersDark)
  const toggleTheme = useCallback(() => {
    setTheme(isDarkMode ? 'light' : 'dark')
  }, [isDarkMode, setTheme])

  const handleAddSourcesClick = useCallback(() => {
    if (!isAuthenticated) return
    const { rightPanel, closeRightPanel, openRightPanel } = useLayoutStore.getState()
    if (rightPanel === 'data-sources') {
      closeRightPanel()
    } else {
      openRightPanel('data-sources')
    }
  }, [isAuthenticated])

  const handleNewSessionClick = useCallback(() => {
    if (!isAuthenticated || isNewSessionDisabled) return
    onNewSession?.()
  }, [isAuthenticated, isNewSessionDisabled, onNewSession])

  const handleSignOut = useCallback(() => {
    setIsUserMenuOpen(false)
    onSignOut?.()
  }, [onSignOut])

  return (
    <header className="border-b border-base">
      <Flex align="center" justify="between" className="h-[var(--header-height)] gap-4 px-4">
        {/* Left section: New session button + session title */}
        <Flex align="center" gap="2" className="min-w-0 flex-1">
          <Button
            kind="tertiary"
            size="small"
            onClick={handleNewSessionClick}
            disabled={!isAuthenticated || isNewSessionDisabled}
            aria-label="Create new session"
            title={
              isNewSessionDisabled
                ? 'Cannot create new session while shallow research is active'
                : 'Create new session'
            }
          >
            <Flex align="center" gap="density-lg">
              <Logo kind="logo-only" size="small" />

              <Text kind="label/semibold/lg" className="text-primary whitespace-nowrap">
                Deep Researcher Agent
              </Text>
            </Flex>
          </Button>
          {sessionTitle && (
            <Flex justify="start">
              <Divider orientation="vertical" />
            </Flex>
          )}
          {isAuthenticated && (
            <div className="ml-4 hidden min-w-0 flex-1 items-center md:flex">
              <Text
                kind="body/regular/md"
                className="block w-full max-w-[360px] truncate text-subtle lg:max-w-[480px] xl:max-w-[560px]"
              >
                {sessionTitle}
              </Text>
            </div>
          )}
        </Flex>

        {/* Right section: Actions + User */}
        <Flex align="center" gap="2" className="shrink-0">
          <Button
            kind="tertiary"
            size="small"
            onClick={handleAddSourcesClick}
            disabled={!isAuthenticated}
            aria-label="Add data sources"
            aria-pressed={isDataSourcesOpen}
            title="Add data sources"
            className={cn(isDataSourcesOpen && 'brand-tint')}
          >
            <Flex align="center" gap="1">
              <Globe className="h-4 w-4" />
              <Text kind="label/regular/md">Data Sources</Text>
            </Flex>
          </Button>

          {/* Theme toggle: quick dark/light switch */}
          <Button
            kind="tertiary"
            size="small"
            onClick={toggleTheme}
            aria-label={isDarkMode ? 'Switch to light mode' : 'Switch to dark mode'}
            title={isDarkMode ? 'Switch to light mode' : 'Switch to dark mode'}
          >
            {isDarkMode ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
          </Button>

          {/* User section: Auth not required notice, Avatar with dropdown, or Sign In button */}
          {!authRequired ? (
            <Popover
              open={isUserMenuOpen}
              onOpenChange={setIsUserMenuOpen}
              side="bottom"
              align="end"
              className="bg-surface-base"
              slotContent={<AuthDisabledContent />}
            >
              <Button
                kind="tertiary"
                size="small"
                aria-label="Default User - Authentication Not Configured"
                title="Default User set. Authentication Not Configured."
                className="ml-2"
              >
                <Avatar size="small" fallback="D" />
              </Button>
            </Popover>
          ) : isAuthenticated ? (
            <Popover
              open={isUserMenuOpen}
              onOpenChange={setIsUserMenuOpen}
              side="bottom"
              align="end"
              className="bg-surface-base"
              slotContent={<UserDropdownContent user={user} onSignOut={handleSignOut} />}
            >
              <Button
                kind="tertiary"
                size="small"
                aria-label={`User menu for ${user?.name || user?.email || 'User'}`}
                title="User menu"
                className="ml-2"
              >
                <Avatar
                  size="small"
                  src={user?.image}
                  fallback={(user?.name || user?.email || 'U').charAt(0).toUpperCase()}
                />
              </Button>
            </Popover>
          ) : (
            <Button
              kind="primary"
              size="small"
              onClick={onSignIn}
              aria-label="Sign in with NVIDIA SSO"
              title="Sign in with NVIDIA SSO"
              className="ml-2 bg-[#76b900] hover:bg-[#5a8f00]"
            >
              <Flex align="center" gap="1">
                <Lock className="h-4 w-4" />
                <Text kind="label/semibold/sm">Sign In</Text>
              </Flex>
            </Button>
          )}
        </Flex>
      </Flex>
    </header>
  )
})

/**
 * User dropdown content with profile info and sign out button
 */
interface UserDropdownContentProps {
  user?: {
    name?: string
    email?: string
    image?: string
  }
  onSignOut?: () => void
}

const APPEARANCE_SEGMENTS: { mode: ThemeMode; label: string }[] = [
  { mode: 'system', label: 'System' },
  { mode: 'dark', label: 'Dark' },
  { mode: 'light', label: 'Light' },
]

const DOCS_URL = 'https://github.com/NVIDIA-AI-Blueprints/deep-researcher-agent/tree/develop/docs'

const AppearanceThemeControl: FC = () => {
  const theme = useLayoutStore((s) => s.theme)
  const setTheme = useLayoutStore((s) => s.setTheme)

  return (
    <Flex direction="col" gap="2">
      <Text kind="label/regular/sm" className="text-subtle">
        Appearance
      </Text>
      <Flex
        align="center"
        gap="1"
        className="p-1"
        role="radiogroup"
        aria-label="Theme"
        style={{
          background: 'var(--color-component-track-background, #FFFFFF33)',
          borderRadius: 'var(--radius-lg)',
        }}
      >
        {APPEARANCE_SEGMENTS.map(({ mode, label }) => {
          const selected = theme === mode
          return (
            <Button
              key={mode}
              type="button"
              role="radio"
              aria-checked={selected}
              aria-label={`${label} theme`}
              kind="tertiary"
              size="small"
              onClick={() => setTheme(mode)}
              className={`h-auto min-h-9 flex-1 rounded-[var(--radius-md)] border-0 px-2 py-1.5 shadow-none transition-colors focus-visible:ring-2 focus-visible:ring-[var(--color-border-focus,#76b900)] ${
                selected ? '!bg-black !text-white hover:!bg-black' : 'bg-transparent hover:bg-white/10'
              }`}
            >
              <Flex align="center" justify="center" gap="1" className="w-full">
                {mode === 'dark' ? (
                  <Moon
                    className={`h-4 w-4 shrink-0 ${selected ? '!text-white' : 'text-primary'}`}
                    width={16}
                    height={16}
                  />
                ) : null}
                {mode === 'light' ? (
                  <Sun
                    className={`h-4 w-4 shrink-0 ${selected ? '!text-white' : 'text-primary'}`}
                    width={16}
                    height={16}
                  />
                ) : null}
                <Text
                  kind={selected ? 'label/semibold/sm' : 'label/regular/sm'}
                  className={selected ? 'text-white' : 'text-primary'}
                >
                  {label}
                </Text>
              </Flex>
            </Button>
          )
        })}
      </Flex>
    </Flex>
  )
}

const DocumentationSection: FC = () => {
  return (
    <Flex direction="col" gap="2">
      <Text kind="label/regular/sm" className="text-subtle">
        Documentation
      </Text>
      <a
        href={DOCS_URL}
        target="_blank"
        rel="noopener noreferrer"
        aria-label="Open GitHub Docs"
        className="flex w-full items-center justify-between rounded-[var(--radius-md)] px-2 py-2 text-primary transition-colors hover:bg-surface-raised-50 focus-visible:outline focus-visible:outline-2 focus-visible:outline-[var(--color-border-focus,#76b900)]"
      >
        <Flex align="center" gap="2">
          <Book className="h-4 w-4 shrink-0" />
          <Text kind="label/regular/sm">GitHub Docs</Text>
        </Flex>
        <OpenExternal className="h-4 w-4 shrink-0 text-subtle" />
      </a>
    </Flex>
  )
}

const UserDropdownContent: FC<UserDropdownContentProps> = ({ user, onSignOut }) => {
  return (
    <Flex direction="col" gap="3" className="min-w-[240px] p-4">
      {/* User info section */}
      <Flex align="center" gap="3">
        <Avatar
          size="medium"
          src={user?.image}
          fallback={(user?.name || user?.email || 'U').charAt(0).toUpperCase()}
        />
        <Flex direction="col" gap="1">
          <Text kind="label/bold/md" className="text-primary">
            {user?.name || 'User'}
          </Text>
          {user?.email && (
            <Text kind="body/regular/sm" className="text-subtle">
              {user.email}
            </Text>
          )}
        </Flex>
      </Flex>

      <Divider />

      <AppearanceThemeControl />

      <Divider />

      <DocumentationSection />

      <Divider />

      {/* Sign out button */}
      <Button
        kind="secondary"
        size="small"
        onClick={onSignOut}
        className="w-full"
        aria-label="Sign out"
        title="Sign out"
      >
        <Flex align="center" justify="center" gap="2">
          <Logout className="h-4 w-4" />
          <Text kind="label/regular/sm">Sign Out</Text>
        </Flex>
      </Button>
    </Flex>
  )
}

/**
 * Content shown when authentication is disabled
 * Displays info message instead of sign out option
 */
const AuthDisabledContent: FC = () => {
  return (
    <Flex direction="col" gap="3" className="min-w-[240px] p-4">
      {/* User info section */}
      <Flex align="center" gap="3">
        <Avatar size="medium" fallback="D" />
        <Flex direction="col" gap="1">
          <Text kind="label/bold/md" className="text-primary">
            Default User
          </Text>
        </Flex>
      </Flex>

      {/* Info message */}
      <Flex align="center" gap="2" className="rounded border border-base p-3">
        <Info className="h-4 w-4 shrink-0 text-[var(--text-color-subtle)]" />
        <Text kind="body/regular/sm" className="text-subtle">
          Authentication Not Configured
        </Text>
      </Flex>

      <Divider />

      <AppearanceThemeControl />

      <Divider />

      <DocumentationSection />
    </Flex>
  )
}
