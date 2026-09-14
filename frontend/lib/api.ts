// The one place that talks to the backend.
//
// Two decisions worth stating.
//
// *The token lives in sessionStorage, not a cookie.* The API authenticates
// with a bearer header, so there is nothing for the browser to attach
// automatically — which is what makes CSRF structurally impossible here.
// sessionStorage over localStorage because a demo session ending with the
// tab is the right lifetime for something seeded with a published password.
//
// *SSE is read with fetch, not EventSource.* EventSource cannot send an
// Authorization header and cannot POST, and this endpoint needs both.

import type {
  ActionResult,
  Appointment,
  ChatResponse,
  DemoAccount,
  LabResult,
  Medication,
  Page,
  PatientProfile,
  Source,
} from "./types";

export const API_URL =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

const TOKEN_KEY = "carecopilot.token";

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
  }
}

export function getToken(): string | null {
  if (typeof window === "undefined") return null;
  return window.sessionStorage.getItem(TOKEN_KEY);
}

export function setToken(token: string | null): void {
  if (typeof window === "undefined") return;
  if (token) window.sessionStorage.setItem(TOKEN_KEY, token);
  else window.sessionStorage.removeItem(TOKEN_KEY);
}

function authHeaders(): Record<string, string> {
  const token = getToken();
  return token ? { Authorization: `Bearer ${token}` } : {};
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const response = await fetch(`${API_URL}/api${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...authHeaders(),
      ...(init.headers ?? {}),
    },
  });

  if (!response.ok) {
    // The backend returns {detail: "..."} for handled errors. Anything else
    // is reported by status alone rather than by dumping a body that may be
    // an HTML error page.
    let detail = `Request failed (${response.status})`;
    try {
      const body = await response.json();
      if (typeof body?.detail === "string") detail = body.detail;
    } catch {
      /* not JSON; the status line is all we have */
    }
    throw new ApiError(detail, response.status);
  }

  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

// --- auth ----------------------------------------------------------------- //

export async function login(email: string, password: string): Promise<string> {
  const body = await request<{ access_token: string }>("/auth/login", {
    method: "POST",
    body: JSON.stringify({ email, password }),
  });
  setToken(body.access_token);
  return body.access_token;
}

export function logout(): void {
  setToken(null);
}

export function fetchDemoAccounts(): Promise<DemoAccount[]> {
  return request<DemoAccount[]>("/auth/demo-accounts");
}

// --- records -------------------------------------------------------------- //

export const fetchProfile = () => request<PatientProfile>("/me");
export const fetchAppointments = () =>
  request<Page<Appointment>>("/appointments?limit=20");
export const fetchMedications = () =>
  request<Page<Medication>>("/medications");
export const fetchLabs = () => request<Page<LabResult>>("/labs?limit=20");

// --- actions -------------------------------------------------------------- //

export function confirmAction(token: string): Promise<ActionResult> {
  return request<ActionResult>("/actions/confirm", {
    method: "POST",
    body: JSON.stringify({ token }),
  });
}

// --- chat ----------------------------------------------------------------- //

export function sendMessage(
  message: string,
  conversationId: string | null,
): Promise<ChatResponse> {
  return request<ChatResponse>("/chat", {
    method: "POST",
    body: JSON.stringify({ message, conversation_id: conversationId }),
  });
}

export interface StreamHandlers {
  onMeta?: (data: { conversation_id: string; disclaimer: string }) => void;
  onDelta?: (text: string) => void;
  onSources?: (sources: Source[]) => void;
  onDone?: (response: ChatResponse & { replaces_streamed_text: boolean }) => void;
  onError?: (detail: string) => void;
}

/**
 * Stream one turn, dispatching each SSE event to a handler.
 *
 * The turn is finished when `done` arrives, and `done` carries the
 * authoritative answer: guardrails run *after* generation, so the validated
 * text can differ from the deltas already painted. `replaces_streamed_text`
 * says whether it did, which lets the caller repaint only when something
 * actually changed instead of flickering on every turn.
 */
export async function streamMessage(
  message: string,
  conversationId: string | null,
  handlers: StreamHandlers,
  signal?: AbortSignal,
): Promise<void> {
  const response = await fetch(`${API_URL}/api/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders() },
    body: JSON.stringify({ message, conversation_id: conversationId }),
    signal,
  });

  if (!response.ok || !response.body) {
    throw new ApiError(`Stream failed (${response.status})`, response.status);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;

      // A network chunk is not an SSE frame: one read can carry half a
      // frame or three of them. Frames are separated by a blank line, so
      // the buffer is split on that and any trailing partial is kept.
      buffer += decoder.decode(value, { stream: true });
      const frames = buffer.split("\n\n");
      buffer = frames.pop() ?? "";

      for (const frame of frames) dispatch(frame, handlers);
    }
    if (buffer.trim()) dispatch(buffer, handlers);
  } finally {
    reader.releaseLock();
  }
}

function dispatch(frame: string, handlers: StreamHandlers): void {
  let event = "message";
  const dataLines: string[] = [];

  for (const line of frame.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
  }
  if (dataLines.length === 0) return;

  let data: unknown;
  try {
    data = JSON.parse(dataLines.join("\n"));
  } catch {
    // A malformed frame is dropped rather than thrown: losing one delta
    // degrades the answer, while throwing would abandon the whole turn.
    return;
  }

  switch (event) {
    case "meta":
      handlers.onMeta?.(data as { conversation_id: string; disclaimer: string });
      break;
    case "delta":
      handlers.onDelta?.((data as { text: string }).text);
      break;
    case "sources":
      handlers.onSources?.((data as { sources: Source[] }).sources);
      break;
    case "done":
      handlers.onDone?.(
        data as ChatResponse & { replaces_streamed_text: boolean },
      );
      break;
    case "error":
      handlers.onError?.((data as { detail: string }).detail);
      break;
  }
}
