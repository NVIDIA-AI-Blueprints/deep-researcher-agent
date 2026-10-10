// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { render, screen, waitFor } from '@/test-utils'
import userEvent from '@testing-library/user-event'
import { vi, describe, test, expect, beforeEach } from 'vitest'
import { DataSourcesPanel } from './DataSourcesPanel'

// Mock the layout store
const mockCloseRightPanel = vi.fn()
const mockOpenRightPanel = vi.fn()
const mockSetDataSourcesPanelTab = vi.fn()
const mockToggleDataSource = vi.fn()
const mockSetEnabledDataSources = vi.fn()
const mockFetchDataSources = vi.fn()
const mockRefreshDataSourceStatus = vi.fn()

const mockDataSources = [
  { id: 'web_search', name: 'Web Search', description: 'Search the web', requires_auth: false },
  { id: 'knowledge_base', name: 'Knowledge Base', description: 'Wiki docs', requires_auth: true },
  { id: 'bug_tracker', name: 'Bug Tracker', description: 'Bug tracking', requires_auth: true },
]

vi.mock('../store', () => ({
  useLayoutStore: vi.fn((selector?: (s: any) => any) => {
    const state = {
      rightPanel: 'data-sources',
      closeRightPanel: mockCloseRightPanel,
      openRightPanel: mockOpenRightPanel,
      dataSourcesPanelTab: 'connections',
      setDataSourcesPanelTab: mockSetDataSourcesPanelTab,
      enabledDataSourceIds: ['web_search', 'knowledge_base'],
      toggleDataSource: mockToggleDataSource,
      setEnabledDataSources: mockSetEnabledDataSources,
      availableDataSources: mockDataSources,
      dataSourcesLoading: false,
      dataSourcesError: null,
      fetchDataSources: mockFetchDataSources,
        refreshDataSourceStatus: mockRefreshDataSourceStatus,
    }
    return selector ? selector(state) : state
  }),
}))

// Mock useAuth hook
const mockSignIn = vi.fn()
vi.mock('@/adapters/auth', () => ({
  useAuth: vi.fn(() => ({
    idToken: 'valid-token',
    signIn: mockSignIn,
  })),
}))

// Mock the MCP auth client + popup so we can drive the connect flow.
const mockConnect = vi.fn()
const mockGetStatus = vi.fn()
const mockOpenAuthPopupAndWait = vi.fn()
vi.mock('@/adapters/api', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@/adapters/api')>()),
  createMcpAuthClient: vi.fn(() => ({ connect: mockConnect, getStatus: mockGetStatus })),
  openAuthPopupAndWait: (...args: unknown[]) => mockOpenAuthPopupAndWait(...args),
}))

// Mock child components. The connect button lets tests trigger handleConnect.
vi.mock('./DataConnectionCard', () => ({
  DataConnectionCard: ({
    source,
    isEnabled,
    isAvailable,
    onConnect,
  }: {
    source: { id: string; name: string }
    isEnabled: boolean
    isAvailable: boolean
    onConnect?: (id: string) => void
  }) => (
    <div data-testid={`connection-card-${source.id}`}>
      {source.name} - {isEnabled ? 'enabled' : 'disabled'} - {isAvailable ? 'available' : 'unavailable'}
      <button data-testid={`connect-${source.id}`} onClick={() => onConnect?.(source.id)}>
        connect
      </button>
    </div>
  ),
}))

vi.mock('./FileSourcesTab', () => ({
  FileSourcesTab: () => <div data-testid="file-sources-tab">File Sources Tab</div>,
}))

import { useLayoutStore } from '../store'
import { useAuth } from '@/adapters/auth'

/** Either auth banner string from DataSourcesPanel (depends on authRequired). */
const AUTH_DATA_SOURCES_BANNER_TEXT =
  /enable authentication to access additional data sources|sign in to access additional data sources/i

describe('DataSourcesPanel', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    // Reset mock to default open state with authenticated user
    vi.mocked(useLayoutStore).mockImplementation((selector?: (s: any) => any) => {
      const state = {
        rightPanel: 'data-sources',
        closeRightPanel: mockCloseRightPanel,
        openRightPanel: mockOpenRightPanel,
        dataSourcesPanelTab: 'connections',
        setDataSourcesPanelTab: mockSetDataSourcesPanelTab,
        enabledDataSourceIds: ['web_search', 'knowledge_base'],
        toggleDataSource: mockToggleDataSource,
        setEnabledDataSources: mockSetEnabledDataSources,
        availableDataSources: mockDataSources,
        dataSourcesLoading: false,
        dataSourcesError: null,
        fetchDataSources: mockFetchDataSources,
        refreshDataSourceStatus: mockRefreshDataSourceStatus,
      }
      return selector ? selector(state) : state
    })

    vi.mocked(useAuth).mockReturnValue({
      idToken: 'valid-token',
      signIn: mockSignIn,
    } as unknown as ReturnType<typeof useAuth>)
  })

  test('renders panel when open', () => {
    render(<DataSourcesPanel />)

    expect(screen.getByText('Data Sources')).toBeInTheDocument()
  })

  test('header uses the shared header-height for rail alignment', () => {
    const { container } = render(<DataSourcesPanel />)

    const header = container.querySelector('.h-\\[var\\(--header-height\\)\\]')
    expect(header).not.toBeNull()
    expect(header).toHaveClass('h-[var(--header-height)]', 'shrink-0')
  })

  test('renders connections tab by default', () => {
    render(<DataSourcesPanel />)

    expect(screen.getByText('Individual Connections (3)')).toBeInTheDocument()
    expect(screen.getByTestId('connection-card-web_search')).toBeInTheDocument()
    expect(screen.getByTestId('connection-card-knowledge_base')).toBeInTheDocument()
    expect(screen.getByTestId('connection-card-bug_tracker')).toBeInTheDocument()
  })

  test('renders tab navigation', () => {
    render(<DataSourcesPanel />)

    expect(screen.getByRole('radio', { name: /connections/i })).toBeInTheDocument()
    expect(screen.getByRole('radio', { name: /files/i })).toBeInTheDocument()
  })

  test('switches to files tab when clicked', async () => {
    const user = userEvent.setup()
    render(<DataSourcesPanel />)

    await user.click(screen.getByRole('radio', { name: /files/i }))

    expect(mockSetDataSourcesPanelTab).toHaveBeenCalledWith('files')
  })

  test('renders files tab content', () => {
    vi.mocked(useLayoutStore).mockImplementation((selector?: (s: any) => any) => {
      const state = {
        rightPanel: 'data-sources',
        closeRightPanel: mockCloseRightPanel,
        openRightPanel: mockOpenRightPanel,
        dataSourcesPanelTab: 'files',
        setDataSourcesPanelTab: mockSetDataSourcesPanelTab,
        enabledDataSourceIds: ['web_search', 'knowledge_base'],
        toggleDataSource: mockToggleDataSource,
        setEnabledDataSources: mockSetEnabledDataSources,
        availableDataSources: mockDataSources,
        dataSourcesLoading: false,
        dataSourcesError: null,
        fetchDataSources: mockFetchDataSources,
        refreshDataSourceStatus: mockRefreshDataSourceStatus,
      }
      return selector ? selector(state) : state
    })

    render(<DataSourcesPanel />)

    expect(screen.getByTestId('file-sources-tab')).toBeInTheDocument()
  })

  test('shows enabled count in footer for connections tab', () => {
    render(<DataSourcesPanel />)

    expect(screen.getByText(/2 of 3 available connections enabled/i)).toBeInTheDocument()
  })

  test('shows file upload message in footer for files tab', () => {
    vi.mocked(useLayoutStore).mockImplementation((selector?: (s: any) => any) => {
      const state = {
        rightPanel: 'data-sources',
        closeRightPanel: mockCloseRightPanel,
        openRightPanel: mockOpenRightPanel,
        dataSourcesPanelTab: 'files',
        setDataSourcesPanelTab: mockSetDataSourcesPanelTab,
        enabledDataSourceIds: ['web_search', 'knowledge_base'],
        toggleDataSource: mockToggleDataSource,
        setEnabledDataSources: mockSetEnabledDataSources,
        availableDataSources: mockDataSources,
        dataSourcesLoading: false,
        dataSourcesError: null,
        fetchDataSources: mockFetchDataSources,
        refreshDataSourceStatus: mockRefreshDataSourceStatus,
      }
      return selector ? selector(state) : state
    })

    render(<DataSourcesPanel />)

    expect(screen.getByText(/attached files will be always available to agents until deleted/i)).toBeInTheDocument()
  })

  test('does not render content when panel is closed', () => {
    vi.mocked(useLayoutStore).mockImplementation((selector?: (s: any) => any) => {
      const state = {
        rightPanel: null,
        closeRightPanel: mockCloseRightPanel,
        openRightPanel: mockOpenRightPanel,
        dataSourcesPanelTab: 'connections',
        setDataSourcesPanelTab: mockSetDataSourcesPanelTab,
        enabledDataSourceIds: ['web_search', 'knowledge_base'],
        toggleDataSource: mockToggleDataSource,
        setEnabledDataSources: mockSetEnabledDataSources,
        availableDataSources: mockDataSources,
        dataSourcesLoading: false,
        dataSourcesError: null,
        fetchDataSources: mockFetchDataSources,
        refreshDataSourceStatus: mockRefreshDataSourceStatus,
      }
      return selector ? selector(state) : state
    })

    const { container } = render(<DataSourcesPanel />)

    // Push panel keeps content mounted but collapses to zero width and marks
    // its wrapper div aria-hidden when closed.
    expect(container.querySelector('div[aria-hidden="true"]')).toBeInTheDocument()
  })

  test('renders all sources toggle', () => {
    render(<DataSourcesPanel />)

    expect(screen.getByText('All Connections')).toBeInTheDocument()
  })

  test('calls setEnabledDataSources when the all-connections switch is toggled', async () => {
    const user = userEvent.setup()
    render(<DataSourcesPanel />)

    // The switch is the sole control for the bulk toggle.
    await user.click(screen.getByRole('switch'))

    expect(mockSetEnabledDataSources).toHaveBeenCalled()
  })

  test('enable all excludes protected sources that are not connected', async () => {
    const user = userEvent.setup()
    vi.mocked(useLayoutStore).mockImplementation((selector?: (s: any) => any) => {
      const state = {
        rightPanel: 'data-sources',
        closeRightPanel: mockCloseRightPanel,
        openRightPanel: mockOpenRightPanel,
        dataSourcesPanelTab: 'connections',
        setDataSourcesPanelTab: mockSetDataSourcesPanelTab,
        enabledDataSourceIds: [], // nothing enabled -> click enables all
        toggleDataSource: mockToggleDataSource,
        setEnabledDataSources: mockSetEnabledDataSources,
        availableDataSources: [
          { id: 'web_search', name: 'Web Search', description: 'Search', requires_auth: false },
          {
            id: 'gdrive',
            name: 'Google Drive',
            description: 'Drive',
            requires_auth: true,
            per_user_auth: { required: true, status: 'not_connected' },
          },
        ],
        dataSourcesLoading: false,
        dataSourcesError: null,
        fetchDataSources: mockFetchDataSources,
        refreshDataSourceStatus: mockRefreshDataSourceStatus,
      }
      return selector ? selector(state) : state
    })

    render(<DataSourcesPanel />)
    await user.click(screen.getByRole('switch'))

    // gdrive is protected + not connected, so it must be excluded from bulk enable.
    expect(mockSetEnabledDataSources).toHaveBeenCalledWith(['web_search'])
  })

  test('shows correct enabled state for data connection cards', () => {
    render(<DataSourcesPanel />)

    // Web search and knowledge_base are enabled
    expect(screen.getByTestId('connection-card-web_search')).toHaveTextContent('enabled')
    expect(screen.getByTestId('connection-card-knowledge_base')).toHaveTextContent('enabled')
    // Bug Tracker is not in the enabled list
    expect(screen.getByTestId('connection-card-bug_tracker')).toHaveTextContent('disabled')
  })

  describe('error state', () => {
    test('renders error message when API fails', () => {
      vi.mocked(useLayoutStore).mockImplementation((selector?: (s: any) => any) => {
        const state = {
          rightPanel: 'data-sources',
          closeRightPanel: mockCloseRightPanel,
          openRightPanel: mockOpenRightPanel,
          dataSourcesPanelTab: 'connections',
          setDataSourcesPanelTab: mockSetDataSourcesPanelTab,
          enabledDataSourceIds: [],
          toggleDataSource: mockToggleDataSource,
          setEnabledDataSources: mockSetEnabledDataSources,
          availableDataSources: null,
          dataSourcesLoading: false,
          dataSourcesError: 'Failed to connect to server',
          fetchDataSources: mockFetchDataSources,
        refreshDataSourceStatus: mockRefreshDataSourceStatus,
        }
        return selector ? selector(state) : state
      })

      render(<DataSourcesPanel />)

      expect(screen.getByText('Unable to load data sources')).toBeInTheDocument()
      expect(screen.getByText('Failed to connect to server')).toBeInTheDocument()
    })

    test('renders retry button on error', () => {
      vi.mocked(useLayoutStore).mockImplementation((selector?: (s: any) => any) => {
        const state = {
          rightPanel: 'data-sources',
          closeRightPanel: mockCloseRightPanel,
          openRightPanel: mockOpenRightPanel,
          dataSourcesPanelTab: 'connections',
          setDataSourcesPanelTab: mockSetDataSourcesPanelTab,
          enabledDataSourceIds: [],
          toggleDataSource: mockToggleDataSource,
          setEnabledDataSources: mockSetEnabledDataSources,
          availableDataSources: null,
          dataSourcesLoading: false,
          dataSourcesError: 'Network error',
          fetchDataSources: mockFetchDataSources,
        refreshDataSourceStatus: mockRefreshDataSourceStatus,
        }
        return selector ? selector(state) : state
      })

      render(<DataSourcesPanel />)

      expect(screen.getByRole('button', { name: /retry/i })).toBeInTheDocument()
    })

    test('retry re-fetches data sources with the auth token', async () => {
      const user = userEvent.setup()
      vi.mocked(useLayoutStore).mockImplementation((selector?: (s: any) => any) => {
        const state = {
          rightPanel: 'data-sources',
          closeRightPanel: mockCloseRightPanel,
          openRightPanel: mockOpenRightPanel,
          dataSourcesPanelTab: 'connections',
          setDataSourcesPanelTab: mockSetDataSourcesPanelTab,
          enabledDataSourceIds: [],
          toggleDataSource: mockToggleDataSource,
          setEnabledDataSources: mockSetEnabledDataSources,
          availableDataSources: null,
          dataSourcesLoading: false,
          dataSourcesError: 'Network error',
          fetchDataSources: mockFetchDataSources,
          refreshDataSourceStatus: mockRefreshDataSourceStatus,
        }
        return selector ? selector(state) : state
      })

      render(<DataSourcesPanel />)

      await user.click(screen.getByRole('button', { name: /retry loading data sources/i }))

      expect(mockFetchDataSources).toHaveBeenCalledWith('valid-token')
    })
  })

  describe('closed-panel accessibility', () => {
    test('marks the closed panel inert so its descendants leave the tab order', () => {
      vi.mocked(useLayoutStore).mockImplementation((selector?: (s: any) => any) => {
        const state = {
          rightPanel: null,
          closeRightPanel: mockCloseRightPanel,
          openRightPanel: mockOpenRightPanel,
          dataSourcesPanelTab: 'connections',
          setDataSourcesPanelTab: mockSetDataSourcesPanelTab,
          enabledDataSourceIds: ['web_search', 'knowledge_base'],
          toggleDataSource: mockToggleDataSource,
          setEnabledDataSources: mockSetEnabledDataSources,
          availableDataSources: mockDataSources,
          dataSourcesLoading: false,
          dataSourcesError: null,
          fetchDataSources: mockFetchDataSources,
          refreshDataSourceStatus: mockRefreshDataSourceStatus,
        }
        return selector ? selector(state) : state
      })

      const { container } = render(<DataSourcesPanel />)

      expect(container.querySelector('div[inert]')).toBeInTheDocument()
    })
  })

  describe('authentication state', () => {
    test('shows auth warning banner when no idToken and authenticated sources exist', () => {
      vi.mocked(useAuth).mockReturnValue({
        idToken: undefined,
        signIn: mockSignIn,
      } as unknown as ReturnType<typeof useAuth>)

      render(<DataSourcesPanel />)

      expect(screen.getByText(AUTH_DATA_SOURCES_BANNER_TEXT)).toBeInTheDocument()
    })

    test('does not show auth warning when user has valid token', () => {
      vi.mocked(useAuth).mockReturnValue({
        idToken: 'valid-token',
        signIn: mockSignIn,
      } as unknown as ReturnType<typeof useAuth>)

      render(<DataSourcesPanel />)

      expect(screen.queryByText(AUTH_DATA_SOURCES_BANNER_TEXT)).not.toBeInTheDocument()
    })

    test('does not show auth warning when only web_search is available', () => {
      vi.mocked(useLayoutStore).mockImplementation((selector?: (s: any) => any) => {
        const state = {
          rightPanel: 'data-sources',
          closeRightPanel: mockCloseRightPanel,
          openRightPanel: mockOpenRightPanel,
          dataSourcesPanelTab: 'connections',
          setDataSourcesPanelTab: mockSetDataSourcesPanelTab,
          enabledDataSourceIds: ['web_search'],
          toggleDataSource: mockToggleDataSource,
          setEnabledDataSources: mockSetEnabledDataSources,
          availableDataSources: [{ id: 'web_search', name: 'Web Search', description: 'Search' }],
          dataSourcesLoading: false,
          dataSourcesError: null,
          fetchDataSources: mockFetchDataSources,
        refreshDataSourceStatus: mockRefreshDataSourceStatus,
        }
        return selector ? selector(state) : state
      })

      vi.mocked(useAuth).mockReturnValue({
        idToken: undefined,
        signIn: mockSignIn,
      } as unknown as ReturnType<typeof useAuth>)

      render(<DataSourcesPanel />)

      expect(screen.queryByText(AUTH_DATA_SOURCES_BANNER_TEXT)).not.toBeInTheDocument()
    })

    test('shows sign-in banner copy when authRequired and no token', () => {
      vi.mocked(useAuth).mockReturnValue({
        idToken: undefined,
        signIn: mockSignIn,
        authRequired: true,
      } as unknown as ReturnType<typeof useAuth>)

      render(<DataSourcesPanel />)

      expect(screen.getByText(AUTH_DATA_SOURCES_BANNER_TEXT)).toBeInTheDocument()
    })

    test('marks authenticated sources as unavailable when no token', () => {
      vi.mocked(useAuth).mockReturnValue({
        idToken: undefined,
        signIn: mockSignIn,
      } as unknown as ReturnType<typeof useAuth>)

      render(<DataSourcesPanel />)

      // web_search should be available (doesn't require auth)
      expect(screen.getByTestId('connection-card-web_search')).toHaveTextContent('available')
      // Authenticated sources should be unavailable
      expect(screen.getByTestId('connection-card-knowledge_base')).toHaveTextContent('unavailable')
      expect(screen.getByTestId('connection-card-bug_tracker')).toHaveTextContent('unavailable')
    })

    test('marks all sources as available when user has valid token', () => {
      vi.mocked(useAuth).mockReturnValue({
        idToken: 'valid-token',
        signIn: mockSignIn,
      } as unknown as ReturnType<typeof useAuth>)

      render(<DataSourcesPanel />)

      expect(screen.getByTestId('connection-card-web_search')).toHaveTextContent('available')
      expect(screen.getByTestId('connection-card-knowledge_base')).toHaveTextContent('available')
      expect(screen.getByTestId('connection-card-bug_tracker')).toHaveTextContent('available')
    })
  })

  describe('connect flow (handleConnect)', () => {
    test('opens the auth popup when the source requires auth', async () => {
      const user = userEvent.setup()
      mockConnect.mockResolvedValue({ status: 'auth_required', auth_url: 'https://provider/oauth' })
      mockOpenAuthPopupAndWait.mockResolvedValue({ ok: true })

      render(<DataSourcesPanel />)
      await user.click(screen.getByTestId('connect-knowledge_base'))

      await waitFor(() => expect(mockOpenAuthPopupAndWait).toHaveBeenCalled())
      expect(mockOpenAuthPopupAndWait).toHaveBeenCalledWith(
        'https://provider/oauth',
        'knowledge_base',
        expect.objectContaining({ pollStatus: expect.any(Function) })
      )
      // finally: always refresh statuses after the attempt
      await waitFor(() => expect(mockFetchDataSources).toHaveBeenCalledWith('valid-token'))
    })

    test('does not open the popup when no auth is required, but still refreshes', async () => {
      const user = userEvent.setup()
      mockConnect.mockResolvedValue({ status: 'connected', auth_url: null })

      render(<DataSourcesPanel />)
      await user.click(screen.getByTestId('connect-knowledge_base'))

      await waitFor(() => expect(mockFetchDataSources).toHaveBeenCalledWith('valid-token'))
      expect(mockOpenAuthPopupAndWait).not.toHaveBeenCalled()
    })

    test('shows a failure banner and still refreshes when connect throws', async () => {
      const user = userEvent.setup()
      mockConnect.mockRejectedValue(new Error('network down'))

      render(<DataSourcesPanel />)
      await user.click(screen.getByTestId('connect-knowledge_base'))

      // connectError banner surfaces the source name + error detail
      expect(await screen.findByText(/Couldn't connect Knowledge Base\. network down/)).toBeInTheDocument()
      // finally still runs despite the failure
      await waitFor(() => expect(mockFetchDataSources).toHaveBeenCalledWith('valid-token'))
      expect(mockOpenAuthPopupAndWait).not.toHaveBeenCalled()
    })
  })
})
