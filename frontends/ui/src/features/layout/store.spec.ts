// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, test, expect, beforeEach, vi } from 'vitest'
import { useLayoutStore } from './store'

const mockGetDataSources = vi.hoisted(() => vi.fn())

vi.mock('@/adapters/api', () => ({
  createDataSourcesClient: vi.fn(() => ({
    getDataSources: mockGetDataSources,
  })),
}))

describe('useLayoutStore', () => {
  beforeEach(() => {
    mockGetDataSources.mockReset()
    // Reset store to initial state before each test (matches store.ts initialState)
    useLayoutStore.setState({
      isSessionsPanelOpen: false,
      sessionsCollapsed: false,
      sessionsAutoCollapsed: false,
      rightPanel: 'data-sources',
      researchPanelTab: 'tasks',
      dataSourcesPanelTab: 'connections',
      enabledDataSourceIds: [],
      theme: 'system',
      availableDataSources: null,
      knowledgeLayerAvailable: false,
      dataSourcesLoading: false,
      dataSourcesError: null,
      promptDraft: null,
      selectedModel: undefined,
    })
  })

  describe('initial state', () => {
    test('has correct default values', () => {
      const state = useLayoutStore.getState()

      expect(state.isSessionsPanelOpen).toBe(false)
      expect(state.rightPanel).toBe('data-sources')
      expect(state.researchPanelTab).toBe('tasks')
      expect(state.dataSourcesPanelTab).toBe('connections')
    })
  })

  describe('toggleSessionsPanel', () => {
    test('opens sessions panel when closed', () => {
      useLayoutStore.getState().toggleSessionsPanel()

      expect(useLayoutStore.getState().isSessionsPanelOpen).toBe(true)
    })

    test('closes sessions panel when open', () => {
      useLayoutStore.setState({ isSessionsPanelOpen: true })

      useLayoutStore.getState().toggleSessionsPanel()

      expect(useLayoutStore.getState().isSessionsPanelOpen).toBe(false)
    })

    test('toggles multiple times correctly', () => {
      const { toggleSessionsPanel } = useLayoutStore.getState()

      toggleSessionsPanel()
      expect(useLayoutStore.getState().isSessionsPanelOpen).toBe(true)

      toggleSessionsPanel()
      expect(useLayoutStore.getState().isSessionsPanelOpen).toBe(false)

      toggleSessionsPanel()
      expect(useLayoutStore.getState().isSessionsPanelOpen).toBe(true)
    })
  })

  describe('setSessionsPanelOpen', () => {
    test('sets sessions panel to open', () => {
      useLayoutStore.getState().setSessionsPanelOpen(true)

      expect(useLayoutStore.getState().isSessionsPanelOpen).toBe(true)
    })

    test('sets sessions panel to closed', () => {
      useLayoutStore.setState({ isSessionsPanelOpen: true })

      useLayoutStore.getState().setSessionsPanelOpen(false)

      expect(useLayoutStore.getState().isSessionsPanelOpen).toBe(false)
    })
  })

  describe('openRightPanel', () => {
    test('opens research panel', () => {
      useLayoutStore.getState().openRightPanel('research')

      expect(useLayoutStore.getState().rightPanel).toBe('research')
    })

    test('opens data-sources panel', () => {
      useLayoutStore.getState().openRightPanel('data-sources')

      expect(useLayoutStore.getState().rightPanel).toBe('data-sources')
    })

    test('opens settings panel', () => {
      useLayoutStore.getState().openRightPanel('settings')

      expect(useLayoutStore.getState().rightPanel).toBe('settings')
    })

    test.each(['citations', 'thinking'] as const)(
      'opens the %s rail panel',
      (panel) => {
        useLayoutStore.getState().openRightPanel(panel)

        expect(useLayoutStore.getState().rightPanel).toBe(panel)
      }
    )

    test('replaces existing panel', () => {
      useLayoutStore.setState({ rightPanel: 'research' })

      useLayoutStore.getState().openRightPanel('settings')

      expect(useLayoutStore.getState().rightPanel).toBe('settings')
    })
  })

  describe('closeRightPanel', () => {
    test('closes open panel', () => {
      useLayoutStore.setState({ rightPanel: 'research' })

      useLayoutStore.getState().closeRightPanel()

      expect(useLayoutStore.getState().rightPanel).toBeNull()
    })

    test('handles closing when already closed', () => {
      useLayoutStore.setState({ rightPanel: null })

      useLayoutStore.getState().closeRightPanel()

      expect(useLayoutStore.getState().rightPanel).toBeNull()
    })
  })

  describe('sessions sidebar collapse model', () => {
    test('toggleSessionsSidebar flips collapsed and clears auto-collapse', () => {
      useLayoutStore.setState({ sessionsAutoCollapsed: true })

      useLayoutStore.getState().toggleSessionsSidebar()

      expect(useLayoutStore.getState().sessionsCollapsed).toBe(true)
      expect(useLayoutStore.getState().sessionsAutoCollapsed).toBe(false)
    })

    test('setSessionsCollapsed sets collapsed and clears auto-collapse', () => {
      useLayoutStore.setState({ sessionsAutoCollapsed: true })

      useLayoutStore.getState().setSessionsCollapsed(true)

      expect(useLayoutStore.getState().sessionsCollapsed).toBe(true)
      expect(useLayoutStore.getState().sessionsAutoCollapsed).toBe(false)
    })

    test('opening a research/data-sources panel auto-collapses the sidebar', () => {
      useLayoutStore.setState({ sessionsCollapsed: false, rightPanel: null })

      useLayoutStore.getState().openRightPanel('research')

      expect(useLayoutStore.getState().rightPanel).toBe('research')
      expect(useLayoutStore.getState().sessionsCollapsed).toBe(true)
      expect(useLayoutStore.getState().sessionsAutoCollapsed).toBe(true)
    })

    test.each(['data-sources', 'citations', 'research', 'thinking'] as const)(
      'opening the %s rail panel auto-collapses the sidebar',
      (panel) => {
        useLayoutStore.setState({ sessionsCollapsed: false, rightPanel: null })

        useLayoutStore.getState().openRightPanel(panel)

        expect(useLayoutStore.getState().sessionsCollapsed).toBe(true)
        expect(useLayoutStore.getState().sessionsAutoCollapsed).toBe(true)
      }
    )

    test('opening the settings overlay does not collapse the sidebar', () => {
      useLayoutStore.setState({ sessionsCollapsed: false, rightPanel: null })

      useLayoutStore.getState().openRightPanel('settings')

      expect(useLayoutStore.getState().sessionsCollapsed).toBe(false)
      expect(useLayoutStore.getState().sessionsAutoCollapsed).toBe(false)
    })

    test('closing after an auto-collapse restores the sidebar', () => {
      useLayoutStore.setState({ sessionsCollapsed: false, rightPanel: null })
      useLayoutStore.getState().openRightPanel('research')

      useLayoutStore.getState().closeRightPanel()

      expect(useLayoutStore.getState().rightPanel).toBeNull()
      expect(useLayoutStore.getState().sessionsCollapsed).toBe(false)
      expect(useLayoutStore.getState().sessionsAutoCollapsed).toBe(false)
    })

    test('does not auto-restore a sidebar the user collapsed manually', () => {
      useLayoutStore.setState({ sessionsCollapsed: true, sessionsAutoCollapsed: false, rightPanel: 'research' })

      useLayoutStore.getState().closeRightPanel()

      expect(useLayoutStore.getState().sessionsCollapsed).toBe(true)
    })
  })

  describe('promptDraft and selectedModel', () => {
    test('setPromptDraft stores and clears the composer draft', () => {
      useLayoutStore.getState().setPromptDraft('half a question')
      expect(useLayoutStore.getState().promptDraft).toBe('half a question')

      useLayoutStore.getState().setPromptDraft(null)
      expect(useLayoutStore.getState().promptDraft).toBeNull()
    })

    test('selectedModel defaults to undefined and is settable', () => {
      expect(useLayoutStore.getState().selectedModel).toBeUndefined()

      useLayoutStore.getState().setSelectedModel('custom-model')

      expect(useLayoutStore.getState().selectedModel).toBe('custom-model')
    })

    test('setSelectedModel(undefined) restores the backend default', () => {
      useLayoutStore.getState().setSelectedModel('gpt-5.4')

      useLayoutStore.getState().setSelectedModel(undefined)

      expect(useLayoutStore.getState().selectedModel).toBeUndefined()
    })

    test('resetComposerState clears the draft and selected model', () => {
      useLayoutStore.setState({ promptDraft: 'half a question', selectedModel: 'gpt-5.4' })

      useLayoutStore.getState().resetComposerState()

      expect(useLayoutStore.getState().promptDraft).toBeNull()
      expect(useLayoutStore.getState().selectedModel).toBeUndefined()
    })
  })

  describe('setResearchPanelTab', () => {
    test('sets thinking tab', () => {
      useLayoutStore.getState().setResearchPanelTab('thinking')

      expect(useLayoutStore.getState().researchPanelTab).toBe('thinking')
    })

    test('sets report tab', () => {
      useLayoutStore.setState({ researchPanelTab: 'thinking' })

      useLayoutStore.getState().setResearchPanelTab('report')

      expect(useLayoutStore.getState().researchPanelTab).toBe('report')
    })
  })

  describe('setDataSourcesPanelTab', () => {
    test('sets connections tab', () => {
      useLayoutStore.setState({ dataSourcesPanelTab: 'files' })

      useLayoutStore.getState().setDataSourcesPanelTab('connections')

      expect(useLayoutStore.getState().dataSourcesPanelTab).toBe('connections')
    })

    test('sets files tab', () => {
      useLayoutStore.getState().setDataSourcesPanelTab('files')

      expect(useLayoutStore.getState().dataSourcesPanelTab).toBe('files')
    })
  })

  describe('setTheme', () => {
    test('sets light theme', () => {
      useLayoutStore.getState().setTheme('light')

      expect(useLayoutStore.getState().theme).toBe('light')
    })

    test('sets dark theme', () => {
      useLayoutStore.getState().setTheme('dark')

      expect(useLayoutStore.getState().theme).toBe('dark')
    })

    test('sets system theme', () => {
      useLayoutStore.setState({ theme: 'dark' })

      useLayoutStore.getState().setTheme('system')

      expect(useLayoutStore.getState().theme).toBe('system')
    })
  })

  describe('fetchDataSources', () => {
    test('enables all returned sources by default', async () => {
      mockGetDataSources.mockResolvedValueOnce({
        data_sources: [
          { id: 'web_search', name: 'Web Search', requires_auth: false },
          { id: 'knowledge_base', name: 'Knowledge Base', requires_auth: true },
        ],
        knowledge_layer: true,
      })

      await useLayoutStore.getState().fetchDataSources('token-1')

      expect(useLayoutStore.getState().enabledDataSourceIds).toEqual([
        'web_search',
        'knowledge_base',
      ])
      expect(useLayoutStore.getState().knowledgeLayerAvailable).toBe(true)
    })

    test('does not auto-enable a protected source that is not connected', async () => {
      mockGetDataSources.mockResolvedValueOnce({
        data_sources: [
          { id: 'web_search', name: 'Web Search', requires_auth: false },
          {
            id: 'gdrive',
            name: 'Google Drive',
            requires_auth: false,
            per_user_auth: { required: true, status: 'not_connected' },
          },
        ],
        knowledge_layer: false,
      })

      await useLayoutStore.getState().fetchDataSources('token-1')

      // gdrive is not connected -> excluded from the initial selection.
      expect(useLayoutStore.getState().enabledDataSourceIds).toEqual(['web_search'])
    })

    test('auto-enables a protected source once it is connected', async () => {
      mockGetDataSources.mockResolvedValueOnce({
        data_sources: [
          {
            id: 'gdrive',
            name: 'Google Drive',
            requires_auth: false,
            per_user_auth: { required: true, status: 'connected' },
          },
        ],
        knowledge_layer: false,
      })

      await useLayoutStore.getState().fetchDataSources('token-1')

      expect(useLayoutStore.getState().enabledDataSourceIds).toEqual(['gdrive'])
    })
  })

  describe('refreshDataSourceStatus', () => {
    test('drops a protected source that is no longer connected from the selection', async () => {
      useLayoutStore.setState({ enabledDataSourceIds: ['web_search', 'gdrive'] })
      mockGetDataSources.mockResolvedValueOnce({
        data_sources: [
          { id: 'web_search', name: 'Web Search', requires_auth: false },
          {
            id: 'gdrive',
            name: 'Google Drive',
            requires_auth: false,
            per_user_auth: { required: true, status: 'expired' },
          },
        ],
        knowledge_layer: false,
      })

      await useLayoutStore.getState().refreshDataSourceStatus('token-1')

      // gdrive's token expired -> reconciled out; web_search preserved.
      expect(useLayoutStore.getState().enabledDataSourceIds).toEqual(['web_search'])
    })

    test('keeps a still-connected protected source and preserves other selections', async () => {
      useLayoutStore.setState({ enabledDataSourceIds: ['gdrive', 'confluence'] })
      mockGetDataSources.mockResolvedValueOnce({
        data_sources: [
          {
            id: 'gdrive',
            name: 'Google Drive',
            requires_auth: false,
            per_user_auth: { required: true, status: 'connected' },
          },
          // 'confluence' is absent from this response -> must be preserved, not dropped.
        ],
        knowledge_layer: false,
      })

      await useLayoutStore.getState().refreshDataSourceStatus('token-1')

      expect(useLayoutStore.getState().enabledDataSourceIds).toEqual(['gdrive', 'confluence'])
    })
  })
})
