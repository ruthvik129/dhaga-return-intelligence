// Shared building blocks. Icons are inline Lucide-style SVG strokes (no emoji,
// no icon font), per the design system's pre-delivery checklist.

import type { ReactNode } from 'react'
import type { ApiError, Run } from './api'

import { fmtInt, fmtWhen, humanize } from './format'

// --- icons --------------------------------------------------------------

const PATHS: Record<string, ReactNode> = {
  overview: (
    <>
      <rect x="3" y="3" width="7" height="9" rx="1" />
      <rect x="14" y="3" width="7" height="5" rx="1" />
      <rect x="14" y="12" width="7" height="9" rx="1" />
      <rect x="3" y="16" width="7" height="5" rx="1" />
    </>
  ),
  upload: (
    <>
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
      <path d="m17 8-5-5-5 5" />
      <path d="M12 3v12" />
    </>
  ),
  insights: (
    <>
      <path d="M3 3v18h18" />
      <path d="M7 16h4" />
      <path d="M7 11h8" />
      <path d="M7 6h12" />
    </>
  ),
  review: (
    <>
      <rect x="8" y="2" width="8" height="4" rx="1" />
      <path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2" />
      <path d="m9 14 2 2 4-4" />
    </>
  ),
  evaluation: (
    <>
      <circle cx="12" cy="12" r="9" />
      <circle cx="12" cy="12" r="5" />
      <circle cx="12" cy="12" r="1" />
    </>
  ),
  cost: (
    <>
      <path d="M19 7V5a2 2 0 0 0-2-2H5a2 2 0 0 0 0 4h14a2 2 0 0 1 2 2v3" />
      <path d="M3 5v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-3" />
      <path d="M21 12h-4a2 2 0 0 0 0 4h4z" />
    </>
  ),
  thread: (
    <>
      <path d="M5 19c3-9 11-9 14 0" />
      <path d="M5 13c3-6 11-6 14 0" />
      <path d="M12 3v18" />
    </>
  ),
  info: (
    <>
      <circle cx="12" cy="12" r="9" />
      <path d="M12 16v-4" />
      <path d="M12 8h.01" />
    </>
  ),
  check: (
    <>
      <circle cx="12" cy="12" r="9" />
      <path d="m8.5 12 2.5 2.5 4.5-5" />
    </>
  ),
  alert: (
    <>
      <path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z" />
      <path d="M12 9v4" />
      <path d="M12 17h.01" />
    </>
  ),
  x: (
    <>
      <circle cx="12" cy="12" r="9" />
      <path d="m15 9-6 6" />
      <path d="m9 9 6 6" />
    </>
  ),
  chevron: <path d="m9 18 6-6-6-6" />,
  download: (
    <>
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
      <path d="m7 10 5 5 5-5" />
      <path d="M12 15V3" />
    </>
  ),
  file: (
    <>
      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
      <path d="M14 2v6h6" />
    </>
  ),
  play: <path d="m6 3 14 9-14 9z" />,
  sun: (
    <>
      <circle cx="12" cy="12" r="4" />
      <path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" />
    </>
  ),
  moon: <path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z" />,
  monitor: (
    <>
      <rect x="2" y="3" width="20" height="14" rx="2" />
      <path d="M8 21h8M12 17v4" />
    </>
  ),
}

export function Icon({ name, size = 18, label }: { name: keyof typeof PATHS | string; size?: number; label?: string }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={2}
      strokeLinecap="round"
      strokeLinejoin="round"
      role={label ? 'img' : undefined}
      aria-label={label}
      aria-hidden={label ? undefined : true}
    >
      {PATHS[name]}
    </svg>
  )
}

// --- layout pieces ------------------------------------------------------

export function PageHead({ title, lede, children }: { title: string; lede?: ReactNode; children?: ReactNode }) {
  return (
    <header className="page-head">
      <h1>{title}</h1>
      {lede && <p className="lede">{lede}</p>}
      {children}
    </header>
  )
}

export function Card({
  title,
  sub,
  actions,
  children,
}: {
  title?: ReactNode
  sub?: ReactNode
  actions?: ReactNode
  children: ReactNode
}) {
  return (
    <section className="card">
      {(title || actions) && (
        <div className="card-head">
          <div className="stack-sm" style={{ gap: 4 }}>
            {title && <h2>{title}</h2>}
            {sub && <p>{sub}</p>}
          </div>
          {actions}
        </div>
      )}
      {children}
    </section>
  )
}

export function Metric({
  label,
  value,
  hint,
  tone,
}: {
  label: string
  value: ReactNode
  hint?: ReactNode
  tone?: 'accent'
}) {
  return (
    <div className={`metric${tone ? ` tone-${tone}` : ''}`}>
      <span className="metric-label">{label}</span>
      <span className="metric-value">{value}</span>
      {hint && <span className="metric-hint">{hint}</span>}
    </div>
  )
}

type Tone = 'info' | 'success' | 'warning' | 'danger'
const TONE_ICON: Record<Tone, string> = { info: 'info', success: 'check', warning: 'alert', danger: 'x' }

export function Callout({ tone = 'info', children }: { tone?: Tone; children: ReactNode }) {
  return (
    <div className={`callout ${tone}`} role={tone === 'danger' ? 'alert' : 'status'}>
      <Icon name={TONE_ICON[tone]} />
      <div>{children}</div>
    </div>
  )
}

export function RunBanner({ run }: { run?: Run }) {
  if (!run) return null
  return (
    <div className="run-banner" aria-label="Run details">
      <span>
        Run finished <strong>{fmtWhen(run.finished_at)}</strong>
      </span>
      <span>
        <strong>{fmtInt(run.eligible_rows)}</strong> rows
      </span>
      <span>
        accept on first pass ≥ <strong>{run.accept_threshold_first_pass}</strong>
      </span>
      <span>
        accept on second check ≥ <strong>{run.accept_threshold_second_check}</strong>
      </span>
      <span>
        prompts <strong>{run.prompt_version}</strong>
      </span>
    </div>
  )
}

export function NoArtifact() {
  return (
    <Callout tone="info">
      Nothing processed yet. Open <a href="#/process">Upload &amp; Process</a>, upload a file and classify some of its
      rows to build this screen.
    </Callout>
  )
}

export function Loading({ height = 160 }: { height?: number }) {
  return <div className="skeleton" style={{ height }} aria-busy="true" aria-label="Loading" />
}

export function ErrorNote({ error }: { error: ApiError | null }) {
  if (!error) return null
  if (error.status === 409) return <NoArtifact />
  return <Callout tone="danger">Could not load this screen: {error.message}</Callout>
}

// --- data display -------------------------------------------------------

export interface BarDatum {
  key: string
  label: string
  value: number
  display?: string
}

/** Horizontal bars, sorted by the caller, value labels always visible. */
export function BarChart({
  data,
  selected,
  onSelect,
  label,
  selectVerb = 'Show only',
}: {
  data: BarDatum[]
  selected?: string | null
  onSelect?: (key: string) => void
  label: string
  /** Prefix for each bar's tooltip and screen-reader name, e.g. "Show only Fit". */
  selectVerb?: string
}) {
  const max = Math.max(1, ...data.map((d) => d.value))
  return (
    <ul className={`bars${onSelect ? ' bars-clickable' : ''}`} aria-label={label}>
      {data.map((d) => {
        const on = selected === d.key
        const inner = (
          <>
            <span className="bar-label" title={d.label}>
              {d.label}
            </span>
            <span className="bar-track" aria-hidden="true">
              <span className="bar-fill" style={{ width: `${(100 * d.value) / max}%`, display: 'block' }} />
            </span>
            <span className="bar-value">{d.display ?? fmtInt(d.value)}</span>
          </>
        )
        return (
          <li key={d.key}>
            {onSelect ? (
              <button
                type="button"
                className="bar-row"
                aria-pressed={on}
                title={on ? `Clear ${d.label} and show all` : `${selectVerb} ${d.label}`}
                onClick={() => onSelect(d.key)}
              >
                {inner}
                {/* The affordance: a bar that does something has to look like it
                    does something. Dim until hover, swapping to a clear
                    control once the bar is the active filter. */}
                <span className={`bar-cta${on ? ' on' : ''}`} aria-hidden="true">
                  {on ? <Icon name="x" size={13} /> : <Icon name="chevron" size={14} />}
                </span>
              </button>
            ) : (
              <div className="bar-row">{inner}</div>
            )}
          </li>
        )
      })}
    </ul>
  )
}

export interface Column<T> {
  key: string
  header: ReactNode
  render?: (row: T) => ReactNode
  num?: boolean
  wide?: boolean
}

export function Table<T>({
  rows,
  columns,
  scroll,
  empty = 'Nothing to show.',
  caption,
  onRowClick,
  isActive,
}: {
  rows: T[]
  columns: Column<T>[]
  scroll?: boolean
  empty?: string
  caption?: string
  /** Makes the whole row clickable for mouse users. Pair it with a real
   *  button in one column so keyboard users get a focusable control. */
  onRowClick?: (row: T) => void
  isActive?: (row: T) => boolean
}) {
  if (!rows.length) return <p className="caption">{empty}</p>
  return (
    <div className={`table-wrap${scroll ? ' scroll' : ''}`}>
      <table>
        {caption && <caption className="sr-only">{caption}</caption>}
        <thead>
          <tr>
            {columns.map((c) => (
              <th key={c.key} className={c.num ? 'num' : undefined} scope="col">
                {c.header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr
              key={i}
              className={[onRowClick && 'row-click', isActive?.(r) && 'row-active'].filter(Boolean).join(' ') || undefined}
              aria-current={isActive?.(r) ? 'true' : undefined}
              onClick={onRowClick ? () => onRowClick(r) : undefined}
            >
              {columns.map((c) => (
                <td key={c.key} className={[c.num && 'num', c.wide && 'wide'].filter(Boolean).join(' ') || undefined}>
                  {c.render ? c.render(r) : String((r as Record<string, unknown>)[c.key] ?? '—')}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export function Select({
  label,
  value,
  options,
  onChange,
  format = humanize,
}: {
  label: string
  value: string
  options: string[]
  onChange: (v: string) => void
  format?: (v: string) => string
}) {
  return (
    <label className="field">
      {label}
      <select value={value} onChange={(e) => onChange(e.target.value)}>
        {options.map((o) => (
          <option key={o} value={o}>
            {format(o)}
          </option>
        ))}
      </select>
    </label>
  )
}
