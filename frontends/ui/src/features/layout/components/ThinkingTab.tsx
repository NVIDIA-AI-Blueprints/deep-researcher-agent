// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * ThinkingTab Component
 *
 * The "Thinking" panel: the single authoritative reasoning/steps trace of what
 * the agent did. Tool calls are grouped under their parent agent (human label +
 * status + count) and raw model reasoning is demoted to a collapsed disclosure
 * inside that view. A thin "Generated files" disclosure appears only when files
 * exist. Sources now live entirely in the Citations panel (ResearchSourcesView).
 *
 * SSE Events (Deep Research only):
 * - workflow.start/end, tool.start/end, llm.start/end → Steps
 * - artifact.update (file) → Generated files disclosure
 */

'use client'

import { type FC, useState } from 'react'
import { Flex, Text } from '@/adapters/ui'
import { Document, ChevronDown } from '@/adapters/ui/icons'
import { useChatStore } from '@/features/chat'
import { AgentsTab } from './AgentsTab'
import { FileCard } from './FileCard'

/**
 * Thinking panel content: the steps trace plus a thin generated-files
 * disclosure. Consumes the dedicated files array from the chat store.
 */
export const ThinkingTab: FC = () => {
  const deepResearchFiles = useChatStore((s) => s.deepResearchFiles)
  const [showFiles, setShowFiles] = useState(false)

  return (
    <Flex direction="col" gap="3" className="h-full min-h-0">
      <div className="min-h-0 flex-1 overflow-y-auto">
        <AgentsTab />
      </div>

      {deepResearchFiles.length > 0 && (
        <Flex direction="col" gap="2" className="border-base shrink-0 border-t pt-3">
          <button
            type="button"
            onClick={() => setShowFiles((v) => !v)}
            aria-expanded={showFiles}
            className="text-secondary hover:text-primary flex cursor-pointer items-center gap-1.5 self-start transition-colors"
          >
            <Document className="h-4 w-4" aria-hidden="true" />
            <Text kind="body/regular/sm">Generated files ({deepResearchFiles.length})</Text>
            <ChevronDown
              className={`h-4 w-4 transition-transform duration-200 ${showFiles ? 'rotate-180' : ''}`}
              aria-hidden="true"
            />
          </button>
          {showFiles && (
            <Flex direction="col" gap="2" className="max-h-64 overflow-y-auto">
              {deepResearchFiles.map((file) => (
                <div key={file.id} className="shrink-0">
                  <FileCard file={file} />
                </div>
              ))}
            </Flex>
          )}
        </Flex>
      )}
    </Flex>
  )
}
