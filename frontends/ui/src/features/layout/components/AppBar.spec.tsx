// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { render, screen } from '@/test-utils'
import userEvent from '@testing-library/user-event'
import { vi, describe, test, expect, beforeEach } from 'vitest'
import { AppBar } from './AppBar'

const mockOpenRightPanel = vi.fn()
const mockCloseRightPanel = vi.fn()
const mockSetTheme = vi.fn()

const mockState = () => ({
  rightPanel: null as string | null,
  openRightPanel: mockOpenRightPanel,
  closeRightPanel: mockCloseRightPanel,
  theme: 'system',
  setTheme: mockSetTheme,
})

const expectElementBefore = (first: Element, second: Element) => {
  expect(Boolean(first.compareDocumentPosition(second) & Node.DOCUMENT_POSITION_FOLLOWING)).toBe(
    true
  )
}

vi.mock('../store', () => ({
  useLayoutStore: Object.assign(
    vi.fn((selector?: (s: any) => any) => {
      const state = mockState()
      return selector ? selector(state) : state
    }),
    { getState: () => mockState() }
  ),
}))

describe('AppBar', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  test('renders logo and title', () => {
    render(<AppBar />)

    expect(screen.getByText('Deep Researcher Agent')).toBeInTheDocument()
  })

  test('shows Sign In button when not authenticated', () => {
    render(<AppBar isAuthenticated={false} authRequired={true} />)

    expect(screen.getByRole('button', { name: /sign in/i })).toBeInTheDocument()
  })

  test('calls onSignIn when Sign In is clicked', async () => {
    const user = userEvent.setup()
    const onSignIn = vi.fn()

    render(<AppBar isAuthenticated={false} authRequired={true} onSignIn={onSignIn} />)

    await user.click(screen.getByRole('button', { name: /sign in/i }))

    expect(onSignIn).toHaveBeenCalledOnce()
  })

  test('shows session title when authenticated', () => {
    render(<AppBar isAuthenticated={true} sessionTitle="My Research Session" />)

    expect(screen.getByText('My Research Session')).toBeInTheDocument()
  })

  test('disables action buttons when not authenticated', () => {
    render(<AppBar isAuthenticated={false} />)

    expect(screen.getByRole('button', { name: /create new session/i })).toBeDisabled()
    expect(screen.getByRole('button', { name: /add data sources/i })).toBeDisabled()
    expect(screen.queryByRole('button', { name: /open documentation/i })).not.toBeInTheDocument()
  })

  test('enables action buttons when authenticated', () => {
    render(<AppBar isAuthenticated={true} />)

    expect(screen.getByRole('button', { name: /create new session/i })).not.toBeDisabled()
    expect(screen.getByRole('button', { name: /add data sources/i })).not.toBeDisabled()
    expect(screen.queryByRole('button', { name: /open documentation/i })).not.toBeInTheDocument()
  })

  test('calls onNewSession when logo button clicked', async () => {
    const user = userEvent.setup()
    const onNewSession = vi.fn()

    render(<AppBar isAuthenticated={true} onNewSession={onNewSession} />)

    await user.click(screen.getByRole('button', { name: /create new session/i }))

    expect(onNewSession).toHaveBeenCalledOnce()
  })

  test('disables new session button when shallow navigation is blocked', () => {
    render(<AppBar isAuthenticated={true} isNewSessionDisabled={true} />)

    expect(screen.getByRole('button', { name: /create new session/i })).toBeDisabled()
    expect(screen.getByRole('button', { name: /add data sources/i })).not.toBeDisabled()
  })

  test('opens data-sources panel when Add Sources clicked', async () => {
    const user = userEvent.setup()

    render(<AppBar isAuthenticated={true} />)

    await user.click(screen.getByRole('button', { name: /add data sources/i }))

    expect(mockOpenRightPanel).toHaveBeenCalledWith('data-sources')
  })

  test('toggles theme to dark from the app-bar toggle when in light/system mode', async () => {
    const user = userEvent.setup()

    render(<AppBar isAuthenticated={true} />)

    await user.click(screen.getByRole('button', { name: /switch to dark mode/i }))

    expect(mockSetTheme).toHaveBeenCalledWith('dark')
  })

  test('does not render Documentation in the top navigation', () => {
    render(<AppBar />)

    expect(screen.queryByRole('button', { name: /open documentation/i })).not.toBeInTheDocument()
    expect(screen.queryByText('Documentation')).not.toBeInTheDocument()
  })

  test('shows user avatar when authenticated', () => {
    render(
      <AppBar
        isAuthenticated={true}
        authRequired={true}
        user={{ name: 'John Doe', email: 'john@example.com' }}
      />
    )

    expect(screen.getByRole('button', { name: /user menu for john doe/i })).toBeInTheDocument()
  })

  test('shows Documentation section in authenticated user menu', async () => {
    const user = userEvent.setup()

    render(
      <AppBar
        isAuthenticated={true}
        authRequired={true}
        user={{ name: 'John Doe', email: 'john@example.com' }}
      />
    )

    await user.click(screen.getByRole('button', { name: /user menu for john doe/i }))

    expect(screen.getByText('Appearance')).toBeInTheDocument()
    expect(screen.getByText('Documentation')).toBeInTheDocument()
    expect(screen.getByRole('link', { name: /github docs/i })).toBeInTheDocument()
  })

  describe('auth disabled mode', () => {
    test('shows Default User avatar button when auth is disabled', () => {
      render(<AppBar isAuthenticated={true} authRequired={false} />)

      const avatarButton = screen.getByRole('button', {
        name: /default user.*authentication not configured/i,
      })
      expect(avatarButton).toBeInTheDocument()
    })

    test('does not show Sign In button when auth is disabled', () => {
      render(<AppBar isAuthenticated={true} authRequired={false} />)

      expect(screen.queryByRole('button', { name: /sign in/i })).not.toBeInTheDocument()
    })

    test('shows auth disabled popover with info message before appearance and docs', async () => {
      const user = userEvent.setup()

      render(<AppBar isAuthenticated={true} authRequired={false} />)

      const avatarButton = screen.getByRole('button', {
        name: /default user.*authentication not configured/i,
      })
      await user.click(avatarButton)

      expect(screen.getByText('Default User')).toBeInTheDocument()
      expect(screen.getByRole('radiogroup', { name: /theme/i })).toBeInTheDocument()
      const authNotice = screen.getByText('Authentication Not Configured')
      const appearanceSection = screen.getByText('Appearance')
      const docsSection = screen.getByText('Documentation')
      expect(authNotice).toBeInTheDocument()
      expect(docsSection).toBeInTheDocument()
      expect(screen.getByRole('link', { name: /github docs/i })).toBeInTheDocument()
      expectElementBefore(authNotice, appearanceSection)
      expectElementBefore(appearanceSection, docsSection)
    })

    test('uses the KUI popover with the app background surface for user settings', async () => {
      const user = userEvent.setup()

      render(<AppBar isAuthenticated={true} authRequired={false} />)

      await user.click(
        screen.getByRole('button', {
          name: /default user.*authentication not configured/i,
        })
      )

      const popoverContent = screen.getByTestId('nv-popover-content')
      expect(popoverContent).toHaveClass('nv-popover-content')
      expect(popoverContent).toHaveClass('bg-surface-base')
      expect(popoverContent).not.toHaveClass('!bg-transparent')
      expect(popoverContent).not.toHaveClass('!shadow-none')
      expect(popoverContent).not.toHaveStyle({ backgroundColor: 'transparent' })
    })

    test('does not show Sign Out button when auth is disabled', async () => {
      const user = userEvent.setup()

      render(<AppBar isAuthenticated={true} authRequired={false} />)

      const avatarButton = screen.getByRole('button', {
        name: /default user.*authentication not configured/i,
      })
      await user.click(avatarButton)

      expect(screen.queryByRole('button', { name: /sign out/i })).not.toBeInTheDocument()
    })

    test('action buttons are enabled when auth is disabled (user is authenticated)', () => {
      render(<AppBar isAuthenticated={true} authRequired={false} />)

      expect(screen.getByRole('button', { name: /create new session/i })).not.toBeDisabled()
      expect(screen.getByRole('button', { name: /add data sources/i })).not.toBeDisabled()
      expect(screen.queryByRole('button', { name: /open documentation/i })).not.toBeInTheDocument()
    })

    test('shows session title when auth is disabled', () => {
      render(
        <AppBar isAuthenticated={true} authRequired={false} sessionTitle="My Research Session" />
      )

      expect(screen.getByText('My Research Session')).toBeInTheDocument()
    })
  })
})
