// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { afterEach, describe, expect, test, vi } from 'vitest'
import { createDeepResearchClient, getJobStatus, parseCitationVerification } from './deep-research-client'

class FakeEventSource {
  static latest: FakeEventSource | null = null
  onopen: (() => void) | null = null
  onmessage: ((event: MessageEvent) => void) | null = null
  onerror: (() => void) | null = null
  private listeners = new Map<string, (event: MessageEvent) => void>()

  constructor(_url: string) {
    FakeEventSource.latest = this
  }

  addEventListener(type: string, listener: EventListener): void {
    this.listeners.set(type, listener as (event: MessageEvent) => void)
  }

  close(): void {}

  emit(type: string, data: unknown): void {
    this.listeners.get(type)?.(new MessageEvent(type, { data: JSON.stringify(data) }))
  }
}

describe('deep research REST client', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
  })

  test('includes proxy error details when job status lookup fails', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        new Response(
          JSON.stringify({
            error: {
              code: 'PROXY_ERROR',
              message: 'fetch failed',
            },
          }),
          {
            status: 500,
            headers: { 'Content-Type': 'application/json' },
          }
        )
      )
    )

    await expect(getJobStatus('job-1')).rejects.toThrow(
      'Failed to get job status: 500 - PROXY_ERROR: fetch failed'
    )
  })

  test('maps generated binary file events to durable artifact metadata', () => {
    vi.stubGlobal('EventSource', FakeEventSource)
    const onFileUpdate = vi.fn()
    const client = createDeepResearchClient({ jobId: 'job-1', callbacks: { onFileUpdate } })
    client.connect()

    FakeEventSource.latest?.emit('artifact.update', {
      data: {
        type: 'file',
        file_path: 'chart.png',
        artifact_id: 'art_123',
        content_url: '/v1/jobs/async/job/job-1/artifacts/art_123/content',
        kind: 'image',
        mime_type: 'image/png',
        size_bytes: 2048,
        sha256: 'a'.repeat(64),
        title: 'Quarterly CapEx',
        caption: 'Comparison chart',
        inline: true,
      },
    })

    expect(onFileUpdate).toHaveBeenCalledWith({
      filename: 'chart.png',
      content: undefined,
      artifactId: 'art_123',
      contentUrl: '/api/jobs/async/job/job-1/artifacts/art_123/content',
      kind: 'image',
      mimeType: 'image/png',
      sizeBytes: 2048,
      sha256: 'a'.repeat(64),
      title: 'Quarterly CapEx',
      caption: 'Comparison chart',
      inline: true,
    })
  })

  test('forwards citation verification fields on citation_use events', () => {
    vi.stubGlobal('EventSource', FakeEventSource)
    const onCitationUpdate = vi.fn()
    const client = createDeepResearchClient({ jobId: 'job-1', callbacks: { onCitationUpdate } })
    client.connect()

    FakeEventSource.latest?.emit('artifact.update', {
      data: {
        type: 'citation_use',
        url: 'https://example.com/guess',
        content: 'https://example.com/guess',
        final_report: true,
        verified: false,
        confidence: 0.3,
        match_kind: 'host_only',
      },
    })
    FakeEventSource.latest?.emit('artifact.update', {
      data: { type: 'citation_use', url: 'https://example.com/a', content: 'https://example.com/a' },
    })

    expect(onCitationUpdate).toHaveBeenNthCalledWith(
      1,
      'https://example.com/guess',
      'https://example.com/guess',
      true,
      { verified: false, confidence: 0.3, matchKind: 'host_only' }
    )
    expect(onCitationUpdate).toHaveBeenNthCalledWith(
      2,
      'https://example.com/a',
      'https://example.com/a',
      true,
      undefined
    )
  })

  test('parseCitationVerification ignores malformed fields and clamps confidence', () => {
    expect(parseCitationVerification({})).toBeUndefined()
    expect(parseCitationVerification({ verified: 'yes', confidence: 'high', match_kind: 3 })).toBeUndefined()
    expect(parseCitationVerification({ confidence: 1.7 })).toEqual({ confidence: 1 })
    expect(parseCitationVerification({ confidence: Number.NaN, verified: true })).toEqual({ verified: true })
  })

  test('preserves legacy text file events', () => {
    vi.stubGlobal('EventSource', FakeEventSource)
    const onFileUpdate = vi.fn()
    const client = createDeepResearchClient({ jobId: 'job-1', callbacks: { onFileUpdate } })
    client.connect()

    FakeEventSource.latest?.emit('artifact.update', {
      data: { type: 'file', file_path: 'report.md', content: '# Report' },
    })

    expect(onFileUpdate).toHaveBeenCalledWith(
      expect.objectContaining({ filename: 'report.md', content: '# Report' })
    )
  })

  test('maps url-only artifacts (no artifact_id) via the content_url/url fallback', () => {
    vi.stubGlobal('EventSource', FakeEventSource)
    const onFileUpdate = vi.fn()
    const client = createDeepResearchClient({ jobId: 'job-1', callbacks: { onFileUpdate } })
    client.connect()

    // No artifact_id: filename derives from the url and contentUrl falls back to content_url.
    FakeEventSource.latest?.emit('artifact.update', {
      data: {
        type: 'file',
        url: 'https://example.com/exports/data.csv',
        content_url: 'https://example.com/exports/data.csv',
        mime_type: 'text/csv',
      },
    })

    expect(onFileUpdate).toHaveBeenCalledWith(
      expect.objectContaining({
        filename: 'data.csv',
        artifactId: undefined,
        contentUrl: 'https://example.com/exports/data.csv',
        mimeType: 'text/csv',
      })
    )
  })
})
