// Client of the server-owned app database (docs/ui/08-database.md). Every call matches a real /dara/ endpoint.
import { apiClient } from "./client.ts";
import type {
  DbStoreInfo,
  DbConversationsResponse,
  DbConversationDetail,
  DbConversationSummary,
  DbMessage,
  DbMessageAttachment,
  DbSearchResponse,
  DbArtifact,
  DbArtifactsResponse,
  DbArtifactAsset,
  DbArtifactChatResponse,
  DbArtifactChatRow,
  DbArtifactComment,
  DbArtifactCommentsResponse,
  DbArtifactVersion,
  DbArtifactVersionsResponse,
  ArtifactKind,
  SavedModelConfig,
  ChatStats
} from "./types.ts";

function optSignal(signal?: AbortSignal): { signal?: AbortSignal } {
  return signal ? { signal } : {};
}

const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

/** The server requires UUIDs for conversation and message ids. */
export function newId(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function")
    return crypto.randomUUID();
  // Fallback for very old WebViews: RFC 4122 v4 from Math.random (ids are not secrets).
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    return (c === "x" ? r : (r & 0x3) | 0x8).toString(16);
  });
}

export function isUuid(id: string): boolean {
  return UUID_RE.test(id);
}

export interface ConversationCreateBody {
  id?: string;
  title?: string;
  model?: string | null;
  system_prompt?: string;
  params?: Record<string, unknown>;
  pinned?: boolean;
}

export interface MessageAppendBody {
  id?: string;
  role: "system" | "user" | "assistant";
  content?: string;
  finish_reason?: string | null;
  error?: string | null;
  stats?: Partial<ChatStats> | Record<string, unknown> | null;
  model?: string | null;
  created_at?: number;
}

export interface ArtifactCreateBody {
  id?: string;
  kind?: ArtifactKind;
  title?: string;
  content?: string;
  pinned?: boolean;
}

export class StoreApi {
  listArtifacts(
    kind?: ArtifactKind,
    limit = 200,
    offset = 0,
    signal?: AbortSignal
  ): Promise<DbArtifactsResponse> {
    const q = `limit=${limit}&offset=${offset}${kind ? `&kind=${kind}` : ""}`;
    return apiClient.request<DbArtifactsResponse>(`/dara/artifacts?${q}`, {
      method: "GET",
      ...optSignal(signal)
    });
  }

  getArtifact(id: string, signal?: AbortSignal): Promise<DbArtifact> {
    return apiClient.request<DbArtifact>(`/dara/artifacts/${encodeURIComponent(id)}`, {
      method: "GET",
      ...optSignal(signal)
    });
  }

  createArtifact(body: ArtifactCreateBody, signal?: AbortSignal): Promise<DbArtifact> {
    return apiClient.request<DbArtifact>("/dara/artifacts", {
      method: "POST",
      body: JSON.stringify(body),
      ...optSignal(signal)
    });
  }

  patchArtifact(
    id: string,
    patch: { title?: string; content?: string; pinned?: boolean; snapshot?: string },
    signal?: AbortSignal
  ): Promise<DbArtifact> {
    return apiClient.request<DbArtifact>(`/dara/artifacts/${encodeURIComponent(id)}`, {
      method: "PATCH",
      body: JSON.stringify(patch),
      ...optSignal(signal)
    });
  }

  listArtifactVersions(id: string, signal?: AbortSignal): Promise<DbArtifactVersionsResponse> {
    return apiClient.request<DbArtifactVersionsResponse>(
      `/dara/artifacts/${encodeURIComponent(id)}/versions`,
      { method: "GET", ...optSignal(signal) }
    );
  }

  getArtifactVersion(id: string, rev: number, signal?: AbortSignal): Promise<DbArtifactVersion> {
    return apiClient.request<DbArtifactVersion>(
      `/dara/artifacts/${encodeURIComponent(id)}/versions/${rev}`,
      { method: "GET", ...optSignal(signal) }
    );
  }

  restoreArtifact(id: string, rev: number, signal?: AbortSignal): Promise<DbArtifact> {
    return apiClient.request<DbArtifact>(`/dara/artifacts/${encodeURIComponent(id)}/restore`, {
      method: "POST",
      body: JSON.stringify({ rev }),
      ...optSignal(signal)
    });
  }

  // ---- the chat that belongs to an artifact ----
  listArtifactChat(id: string, signal?: AbortSignal): Promise<DbArtifactChatResponse> {
    return apiClient.request<DbArtifactChatResponse>(
      `/dara/artifacts/${encodeURIComponent(id)}/chat`,
      { method: "GET", ...optSignal(signal) }
    );
  }

  putArtifactChatMessage(
    id: string,
    messageId: string,
    data: Record<string, unknown>,
    signal?: AbortSignal
  ): Promise<DbArtifactChatRow> {
    return apiClient.request<DbArtifactChatRow>(
      `/dara/artifacts/${encodeURIComponent(id)}/chat/${encodeURIComponent(messageId)}`,
      { method: "PUT", body: JSON.stringify({ data }), ...optSignal(signal) }
    );
  }

  clearArtifactChat(id: string, signal?: AbortSignal): Promise<{ deleted: number }> {
    return apiClient.request<{ deleted: number }>(
      `/dara/artifacts/${encodeURIComponent(id)}/chat`,
      { method: "DELETE", ...optSignal(signal) }
    );
  }

  // ---- comments ----
  listArtifactComments(id: string, signal?: AbortSignal): Promise<DbArtifactCommentsResponse> {
    return apiClient.request<DbArtifactCommentsResponse>(
      `/dara/artifacts/${encodeURIComponent(id)}/comments`,
      { method: "GET", ...optSignal(signal) }
    );
  }

  createArtifactComment(
    id: string,
    body: {
      id?: string;
      parent_id?: string;
      author?: "user" | "dara";
      body: string;
      quote?: string;
      nth?: number;
    },
    signal?: AbortSignal
  ): Promise<DbArtifactComment> {
    return apiClient.request<DbArtifactComment>(
      `/dara/artifacts/${encodeURIComponent(id)}/comments`,
      { method: "POST", body: JSON.stringify(body), ...optSignal(signal) }
    );
  }

  patchArtifactComment(
    id: string,
    commentId: string,
    patch: { body?: string; resolved?: boolean; quote?: string; nth?: number },
    signal?: AbortSignal
  ): Promise<DbArtifactComment> {
    return apiClient.request<DbArtifactComment>(
      `/dara/artifacts/${encodeURIComponent(id)}/comments/${encodeURIComponent(commentId)}`,
      { method: "PATCH", body: JSON.stringify(patch), ...optSignal(signal) }
    );
  }

  deleteArtifactComment(
    id: string,
    commentId: string,
    signal?: AbortSignal
  ): Promise<{ deleted: number }> {
    return apiClient.request<{ deleted: number }>(
      `/dara/artifacts/${encodeURIComponent(id)}/comments/${encodeURIComponent(commentId)}`,
      { method: "DELETE", ...optSignal(signal) }
    );
  }

  // ---- images and video ----
  uploadArtifactAsset(
    id: string,
    file: Blob,
    name: string,
    signal?: AbortSignal
  ): Promise<DbArtifactAsset> {
    return apiClient.request<DbArtifactAsset>(`/dara/artifacts/${encodeURIComponent(id)}/assets`, {
      method: "POST",
      body: file,
      headers: { "Content-Type": file.type, "X-Dara-Name": encodeURIComponent(name) },
      timeoutMs: 120000,
      ...optSignal(signal)
    });
  }

  getArtifactAsset(id: string, assetId: string, signal?: AbortSignal): Promise<Blob> {
    return apiClient.requestBlob(
      `/dara/artifacts/${encodeURIComponent(id)}/assets/${encodeURIComponent(assetId)}`,
      optSignal(signal)
    );
  }

  /** Copies a document with its images and videos (the server rewrites the asset ids in the copy). */
  duplicateArtifact(
    id: string,
    body: { id?: string; title?: string },
    signal?: AbortSignal
  ): Promise<DbArtifact> {
    return apiClient.request<DbArtifact>(`/dara/artifacts/${encodeURIComponent(id)}/duplicate`, {
      method: "POST",
      body: JSON.stringify(body),
      ...optSignal(signal)
    });
  }

  deleteArtifact(id: string, signal?: AbortSignal): Promise<{ deleted: boolean }> {
    return apiClient.request<{ deleted: boolean }>(`/dara/artifacts/${encodeURIComponent(id)}`, {
      method: "DELETE",
      ...optSignal(signal)
    });
  }

  getStoreInfo(signal?: AbortSignal): Promise<DbStoreInfo> {
    return apiClient.request<DbStoreInfo>("/dara/store", { method: "GET", ...optSignal(signal) });
  }

  getSettings(signal?: AbortSignal): Promise<Record<string, unknown>> {
    return apiClient.request<Record<string, unknown>>("/dara/settings", {
      method: "GET",
      ...optSignal(signal)
    });
  }

  /** Upserts the given {key: value} pairs (keys: lowercase letters, digits, "_", ".", "-"). */
  patchSettings(
    settings: Record<string, unknown>,
    signal?: AbortSignal
  ): Promise<Record<string, unknown>> {
    return apiClient.request<Record<string, unknown>>("/dara/settings", {
      method: "PATCH",
      body: JSON.stringify(settings),
      ...optSignal(signal)
    });
  }

  deleteSetting(key: string, signal?: AbortSignal): Promise<{ deleted: boolean }> {
    return apiClient.request<{ deleted: boolean }>(`/dara/settings/${encodeURIComponent(key)}`, {
      method: "DELETE",
      ...optSignal(signal)
    });
  }

  listConversations(
    limit = 100,
    offset = 0,
    signal?: AbortSignal
  ): Promise<DbConversationsResponse> {
    return apiClient.request<DbConversationsResponse>(
      `/dara/conversations?limit=${limit}&offset=${offset}`,
      {
        method: "GET",
        ...optSignal(signal)
      }
    );
  }

  createConversation(
    data: ConversationCreateBody,
    signal?: AbortSignal
  ): Promise<DbConversationSummary> {
    return apiClient.request<DbConversationSummary>("/dara/conversations", {
      method: "POST",
      body: JSON.stringify(data),
      ...optSignal(signal)
    });
  }

  getConversation(id: string, signal?: AbortSignal): Promise<DbConversationDetail> {
    return apiClient.request<DbConversationDetail>(
      `/dara/conversations/${encodeURIComponent(id)}`,
      {
        method: "GET",
        ...optSignal(signal)
      }
    );
  }

  patchConversation(
    id: string,
    patch: Partial<Omit<ConversationCreateBody, "id">>,
    signal?: AbortSignal
  ): Promise<DbConversationSummary> {
    return apiClient.request<DbConversationSummary>(
      `/dara/conversations/${encodeURIComponent(id)}`,
      {
        method: "PATCH",
        body: JSON.stringify(patch),
        ...optSignal(signal)
      }
    );
  }

  deleteConversation(id: string, signal?: AbortSignal): Promise<{ deleted: boolean }> {
    return apiClient.request<{ deleted: boolean }>(
      `/dara/conversations/${encodeURIComponent(id)}`,
      {
        method: "DELETE",
        ...optSignal(signal)
      }
    );
  }

  appendMessage(
    convId: string,
    message: MessageAppendBody,
    signal?: AbortSignal
  ): Promise<DbMessage> {
    return apiClient.request<DbMessage>(
      `/dara/conversations/${encodeURIComponent(convId)}/messages`,
      {
        method: "POST",
        body: JSON.stringify(message),
        ...optSignal(signal)
      }
    );
  }

  /** Update a stored message (used when a streamed reply finishes, and as a checkpoint while it streams). */
  patchMessage(
    msgId: string,
    patch: {
      content?: string;
      finish_reason?: string | null;
      error?: string | null;
      stats?: Record<string, unknown> | null;
    },
    signal?: AbortSignal
  ): Promise<DbMessage> {
    return apiClient.request<DbMessage>(`/dara/messages/${encodeURIComponent(msgId)}`, {
      method: "PATCH",
      body: JSON.stringify(patch),
      ...optSignal(signal)
    });
  }

  /** Deletes the messages at positions `from` and later (regenerate, edit and resend). */
  deleteMessagesFrom(
    convId: string,
    fromPosition: number,
    signal?: AbortSignal
  ): Promise<{ deleted: number }> {
    return apiClient.request<{ deleted: number }>(
      `/dara/conversations/${encodeURIComponent(convId)}/messages?from=${Math.max(0, Math.floor(fromPosition))}`,
      { method: "DELETE", ...optSignal(signal) }
    );
  }

  /** Stores an image (a `data:` URI or bare base64) as an attachment of a message. PNG, JPEG and WebP up to 20 MB. */
  addAttachment(
    messageId: string,
    dataUri: string,
    signal?: AbortSignal
  ): Promise<DbMessageAttachment> {
    return apiClient.request<DbMessageAttachment>(
      `/dara/messages/${encodeURIComponent(messageId)}/attachments`,
      {
        method: "POST",
        body: JSON.stringify({ data: dataUri }),
        timeoutMs: 120000,
        ...optSignal(signal)
      }
    );
  }

  /** Image bytes. An <img src> cannot send the Authorization header, so fetch them here and use the blob: URL. */
  getAttachmentBlob(id: string, signal?: AbortSignal): Promise<Blob> {
    return apiClient.requestBlob(`/dara/attachments/${encodeURIComponent(id)}`, optSignal(signal));
  }

  searchMessages(query: string, limit = 20, signal?: AbortSignal): Promise<DbSearchResponse> {
    return apiClient.request<DbSearchResponse>(
      `/dara/search?q=${encodeURIComponent(query)}&limit=${limit}`,
      {
        method: "GET",
        ...optSignal(signal)
      }
    );
  }

  getModelConfigs(signal?: AbortSignal): Promise<Record<string, SavedModelConfig>> {
    return apiClient.request<Record<string, SavedModelConfig>>("/dara/model-configs", {
      method: "GET",
      ...optSignal(signal)
    });
  }

  putModelConfig(
    modelId: string,
    config: {
      kv?: string;
      flash?: "auto" | "on" | "off" | boolean;
      vision?: boolean;
      context?: number | null;
    },
    signal?: AbortSignal
  ): Promise<Record<string, unknown>> {
    return apiClient.request<Record<string, unknown>>(
      `/dara/model-configs/${encodeURIComponent(modelId)}`,
      {
        method: "PUT",
        body: JSON.stringify(config),
        ...optSignal(signal)
      }
    );
  }

  deleteModelConfig(modelId: string, signal?: AbortSignal): Promise<{ deleted: boolean }> {
    return apiClient.request<{ deleted: boolean }>(
      `/dara/model-configs/${encodeURIComponent(modelId)}`,
      {
        method: "DELETE",
        ...optSignal(signal)
      }
    );
  }

  /** Everything (settings, model configs, conversations with messages) as one JSON document. */
  exportAll(signal?: AbortSignal): Promise<unknown> {
    return apiClient.request<unknown>("/dara/data/export", {
      method: "GET",
      timeoutMs: 120000,
      ...optSignal(signal)
    });
  }

  /** Deletes conversations, messages, attachment files and saved model configs (settings are kept). */
  deleteAllData(signal?: AbortSignal): Promise<{ deleted: boolean }> {
    return apiClient.request<{ deleted: boolean }>("/dara/data", {
      method: "DELETE",
      body: JSON.stringify({ confirm: "delete-all-data" }),
      timeoutMs: 120000,
      ...optSignal(signal)
    });
  }
}

export const storeApi = new StoreApi();
