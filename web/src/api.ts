// Thin client for api/server.py. Every response is already scrubbed of model
// identity on the server; this file only adds the session header.

import { useCallback, useEffect, useState } from 'react'

const SESSION_KEY = 'dhaga.session'

/** Per-browser id so reviewer corrections stay private (decision D5). */
export function sessionId(): string {
  let id: string | null = null
  try {
    id = localStorage.getItem(SESSION_KEY)
  } catch {
    /* storage blocked: fall through to an in-memory id */
  }
  if (!id || !/^[A-Za-z0-9_-]{8,64}$/.test(id)) {
    // getRandomValues works outside secure contexts too (e.g. plain-http
    // intranet hosts), unlike randomUUID.
    id = Array.from(crypto.getRandomValues(new Uint8Array(16)), (b) => b.toString(16).padStart(2, '0')).join('')
    try {
      localStorage.setItem(SESSION_KEY, id)
    } catch {
      /* ignore */
    }
  }
  return id
}

const SID = sessionId()

export class ApiError extends Error {
  status: number
  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const headers = new Headers(init?.headers)
  headers.set('X-Session-Id', SID)
  const resp = await fetch(path, { ...init, headers })
  if (!resp.ok) {
    let message = `${resp.status} ${resp.statusText}`
    try {
      const body = await resp.json()
      if (typeof body.detail === 'string') message = body.detail
      else if (Array.isArray(body.detail)) message = body.detail.map((d: { msg: string }) => d.msg).join('; ')
    } catch {
      /* not JSON */
    }
    throw new ApiError(resp.status, message)
  }
  return resp.json() as Promise<T>
}

export const api = {
  get: <T>(path: string) => request<T>(path),
  post: <T>(path: string, body: unknown) =>
    request<T>(path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }),
  del: <T>(path: string) => request<T>(path, { method: 'DELETE' }),
  upload: <T>(path: string, file: File) => {
    const form = new FormData()
    form.append('file', file)
    return request<T>(path, { method: 'POST', body: form })
  },
  exportUrl: (params: Record<string, string> = {}) =>
    `/api/export.csv?${new URLSearchParams({ session: SID, ...params })}`,
}

export interface Loaded<T> {
  data: T | null
  error: ApiError | null
  loading: boolean
  reload: () => void
}

/** Fetch `path` whenever it changes. `null` skips the request. */
export function useApi<T>(path: string | null): Loaded<T> {
  const [tick, setTick] = useState(0)
  const key = path === null ? null : `${path}#${tick}`
  const [state, setState] = useState<{ key: string | null; data: T | null; error: ApiError | null }>({
    key: null,
    data: null,
    error: null,
  })

  useEffect(() => {
    if (path === null || key === null) return
    let live = true
    api
      .get<T>(path)
      .then((data) => live && setState({ key, data, error: null }))
      .catch(
        (e: unknown) =>
          live &&
          setState((prev) => ({
            key,
            data: prev.data,
            error: e instanceof ApiError ? e : new ApiError(0, String(e)),
          })),
      )
    return () => {
      live = false
    }
  }, [path, key])

  const reload = useCallback(() => setTick((t) => t + 1), [])
  // Keep showing the previous data while a new request is in flight.
  return { data: state.data, error: state.error, loading: key !== null && state.key !== key, reload }
}

// --- shapes -------------------------------------------------------------

export interface Run {
  finished_at: string
  elapsed_seconds: number
  eligible_rows: number
  prompt_version: string
  accept_threshold_first_pass: number
  accept_threshold_second_check: number
}

export interface Readiness {
  ready: boolean
  messages: string[]
}

export interface Status {
  artifact: boolean
  classifier: Readiness
  run?: Run
  run_total_usd?: number | null
  corrections?: number
  review_pending?: number
  job?: { id: string; state: 'running' | 'done' | 'error'; progress: number; message: string } | null
}

export interface Bucket {
  label: string
  correct: number
  n: number
  accuracy_pct: number | null
  ci_low_pct: number
  ci_high_pct: number
}

export interface TraceStep {
  step: string
  route?: string
  reason?: string | null
  detail?: string
  primary?: string
  confidence?: number
  needs_review?: boolean
  sub_reason?: string
  schema_ok?: boolean
  repaired?: boolean
  error?: string | null
}

export interface ReturnRow {
  return_id: string
  sku: string
  product_name: string
  category: string
  vendor: string
  size_ordered: string | null
  original_text: string | null
  primary_reason: string | null
  sub_reason: string | null
  body_area: string | null
  confidence: number | null
  evidence_text: string | null
  status: string
  review_reason: string | null
  routing_detail?: string
  trace?: TraceStep[]
}

export interface Taxonomy {
  primary_reasons: string[]
  sub_reasons: Record<string, string[]>
  body_areas: string[]
}
