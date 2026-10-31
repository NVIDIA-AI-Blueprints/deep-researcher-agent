// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * API Adapters
 *
 * Re-exports all API-related functionality for use in features.
 * Features should import from '@/adapters/api' only.
 */

// Configuration
export { apiConfig, getWebSocketUrl } from './config'

// Chat Client (SSE Streaming)
export { streamGenerate } from './chat-client'
export type {
  StreamGenerateOptions,
  GenerateStreamCallbacks,
  GenerateStreamMessage,
  BackendStatusType,
  BackendPromptType,
  GenerateMessageType,
} from './chat-client'

// WebSocket Client (NAT Protocol)
export { NATWebSocketClient, createNATWebSocketClient } from './websocket-client'
export type {
  ConnectionStatus,
  NATWebSocketClientCallbacks,
  NATWebSocketClientOptions,
} from './websocket-client'
export { NATMessageType, NATSchemaType, HumanPromptType } from './websocket-client'
export type { NATHumanPrompt, NATIntermediateStepContent, NATErrorContent } from './websocket-client'

// Schemas and Types
export {
  MessageSchema,
  ChatCompletionChunkSchema,
  WorkflowConfigSchema,
  ApiErrorSchema,
  WebSocketIncomingMessageSchema,
} from './schemas'

export type {
  Message,
  ChatCompletionRequest,
  ChatCompletionChunk,
  ChatCompletionChoice,
  WebSocketConnectMessage,
  WebSocketUserMessage,
  WebSocketAgentTextMessage,
  WebSocketStatusMessage,
  WebSocketToolCallMessage,
  WebSocketErrorMessage,
  WebSocketIncomingMessage,
  WorkflowConfig,
  ApiError,
} from './schemas'

// Documents Client
export { createDocumentsClient } from './documents-client'
export type {
  DocumentsClient,
  DocumentsClientOptions,
  UploadFilesOptions,
} from './documents-client'

// Data Sources Client
export { createDataSourcesClient } from './data-sources-client'
export type {
  DataSourcesClient,
  DataSourcesClientOptions,
  DataSourceFromAPI,
  DataSourcesResponse,
  PerUserAuthInfoFromAPI,
  PerUserAuthStatus,
} from './data-sources-client'

// MCP Auth Client (per-user OAuth control plane)
export { createMcpAuthClient, openAuthPopupAndWait } from './mcp-auth-client'
export type {
  McpAuthClient,
  McpAuthClientOptions,
  SourceAuthStatusResponse,
  SourceConnectResponse,
  AuthPopupResult,
} from './mcp-auth-client'

// Documents Schemas
export {
  DocumentFileStatusSchema,
  JobStateSchema,
  CollectionInfoSchema,
  FileInfoSchema,
  FileProgressSchema,
  IngestionJobStatusSchema,
} from './documents-schemas'

export type {
  DocumentFileStatus,
  JobState,
  CollectionInfo,
  FileInfo,
  FileProgress,
  IngestionJobStatus,
} from './documents-schemas'

// Deep Research Client (SSE Streaming for async jobs)
export {
  createDeepResearchClient,
  getJobStatus,
  getJobState,
  getJobReport,
  cancelJob,
  MATCH_KINDS,
  VALID_MATCH_KINDS,
  isCitationMatchKind,
} from './deep-research-client'
export type {
  DeepResearchJobStatus,
  DeepResearchEventType,
  ArtifactType,
  DeepResearchSSEEvent,
  StreamStartEvent,
  JobStatusEvent,
  WorkflowStartEvent,
  WorkflowEndEvent,
  LLMStartEvent,
  LLMChunkEvent,
  LLMEndEvent,
  ToolStartEvent,
  ToolEndEvent,
  TodoItem,
  FileArtifactUpdate,
  ArtifactUpdateEvent,
  DeepResearchEvent,
  DeepResearchCallbacks,
  DeepResearchStreamOptions,
  DeepResearchClient,
  JobStateResponse,
  CitationMatchKind,
} from './deep-research-client'
