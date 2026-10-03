/**
 * The review queue as a triage inbox (approved design direction C).
 *
 * This is the only screen where a human does volume work: 24 comments the
 * agent refused to guess at. It is built like a mail client because everyone
 * already knows how to clear an inbox — folders by why it was flagged, the
 * comment in the middle, the decision docked beside it, and J/K/Enter so a
 * reviewer never has to reach for the mouse.
 *
 * No model names, no confidence scores, no field names. "Why the agent is
 * unsure" is assembled from the routing trace into plain sentences.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { api, ApiError, useApi, type ReturnRow, type Taxonomy, type TraceStep } from '../api'
import { Callout, ErrorNote, Icon, Loading, Select } from '../ui'
import { detailLabel, fmtMinutes, humanize, reasonLabel, reviewLabel } from '../format'

interface Correction {
  return_id: string
  primary_reason: string
  sub_reason: string
  body_area: string
  action: string
  note: string | null
  updated_at: string
}

interface ReviewData {
  queue: ReturnRow[]
  corrections: Correction[]
  estimated_minutes: number
  minutes_per_row: number
}

const ACTION_LABEL: Record<string, string> = {
  confirm: 'Confirmed',
  change: 'Corrected',
  mark_unclear: 'Marked unclear',
}

const ALL = '__all__'
const CLEARED = '__cleared__'

/** Plain sentences explaining the hold-up, built from the routing trace. */
/** `unclear` is a real label, but "the first read it as comments too vague to
 *  sort" is not a sentence anyone says. Inside a narration it becomes a verb. */
const readAs = (primary: string | undefined) =>
  !primary || primary === 'unclear' ? 'could not tell' : `read it as ${reasonLabel(primary)}`

function whyUnsure(row: ReturnRow): string[] {
  const trace = row.trace ?? []
  const pass = (name: string) => trace.find((s) => s.step === name)
  const first = pass('first_pass')
  const second = pass('second_check')
  const out: string[] = []

  switch (row.review_reason) {
    case 'DISAGREEMENT':
      out.push(
        `The two readings did not agree. The first ${readAs(first?.primary)}, and the second ` +
          `${readAs(second?.primary)}.`,
      )
      out.push('When two readings conflict, a person decides rather than the system picking one.')
      break
    case 'AMBIGUOUS_TEXT':
      out.push('The comment says something was wrong but not what, so any label would be a guess.')
      out.push(
        second
          ? 'A second, stronger reading looked again and agreed it could not tell.'
          : 'It was flagged as too vague on the first reading.',
      )
      break
    case 'LOW_CONFIDENCE':
      out.push(
        `The best reading was ${reasonLabel(second?.primary ?? first?.primary)}, but it was not confident ` +
          'enough to stand on its own.',
      )
      out.push('Below that bar the comment comes to you instead of going into the charts.')
      break
    case 'NO_USEFUL_TEXT':
      out.push('This return arrived with an empty comment, so there was nothing to read.')
      out.push('No reading was attempted: it went straight to you.')
      break
    case 'SCHEMA_FAILURE':
      out.push('The reading came back unreadable twice in a row, so nothing was recorded for it.')
      break
    case 'UNSUPPORTED_OUTPUT':
      out.push('The reading produced a reason pair that is not in the agreed list, so it was not accepted.')
      break
    default:
      if (row.routing_detail) out.push(row.routing_detail)
  }
  if (!row.evidence_text && row.review_reason !== 'NO_USEFUL_TEXT') {
    out.push('Nothing in the comment could be quoted as support, which is itself a reason to stop.')
  }
  return out
}

const STEP_LABEL: Record<string, string> = {
  first_pass: 'First reading',
  routing_after_first_pass: 'Decision',
  second_check: 'Second reading',
  routing_after_second_check: 'Decision',
  attributes: 'Details',
}

function traceLine(step: TraceStep): string {
  const name = STEP_LABEL[step.step] ?? humanize(step.step)
  if (step.step.startsWith('routing')) {
    return `${name}: ${step.detail ?? step.route ?? ''}`
  }
  if (step.step === 'attributes') {
    return `${name}: ${step.sub_reason ? detailLabel(step.sub_reason) : 'none recorded'}${
      step.schema_ok === false ? ' (response did not parse)' : ''
    }`
  }
  if (step.step === 'first_pass' || step.step === 'second_check') {
    return `${name}: ${readAs(step.primary)}${step.needs_review ? ', and flagged it as unclear' : ''}`
  }
  return `${name}: ${step.detail ?? ''}`
}

function Decision({
  row,
  taxonomy,
  onSaved,
  submitRef,
}: {
  row: ReturnRow
  taxonomy: Taxonomy
  onSaved: (msg: string) => void
  submitRef: React.MutableRefObject<(() => void) | null>
}) {
  const initial = row.primary_reason && taxonomy.primary_reasons.includes(row.primary_reason) ? row.primary_reason : 'unclear'
  const [primary, setPrimary] = useState(initial)
  const subs = taxonomy.sub_reasons[primary] ?? []
  const [sub, setSub] = useState(row.sub_reason && subs.includes(row.sub_reason) ? row.sub_reason : subs[0])
  const [area, setArea] = useState(row.body_area && taxonomy.body_areas.includes(row.body_area) ? row.body_area : 'none')
  const [note, setNote] = useState('')
  const [saving, setSaving] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  // Reset the form when the reviewer moves to another comment.
  useEffect(() => {
    const p = row.primary_reason && taxonomy.primary_reasons.includes(row.primary_reason) ? row.primary_reason : 'unclear'
    const s = taxonomy.sub_reasons[p] ?? []
    setPrimary(p)
    setSub(row.sub_reason && s.includes(row.sub_reason) ? row.sub_reason : s[0])
    setArea(row.body_area && taxonomy.body_areas.includes(row.body_area) ? row.body_area : 'none')
    setNote('')
    setError(null)
    setSaving(null)
  }, [row, taxonomy])

  const save = useCallback(
    async (action: 'confirm' | 'change' | 'mark_unclear') => {
      setSaving(action)
      setError(null)
      try {
        await api.post(`/api/review/${encodeURIComponent(row.return_id)}`, {
          action,
          primary_reason: primary,
          sub_reason: sub,
          body_area: area,
          note: note || null,
        })
        onSaved(`${row.return_id} ${ACTION_LABEL[action].toLowerCase()}.`)
      } catch (e) {
        setError(e instanceof ApiError ? e.message : String(e))
        setSaving(null)
      }
    },
    [row.return_id, primary, sub, area, note, onSaved],
  )

  // Enter confirms, from anywhere on the screen.
  useEffect(() => {
    submitRef.current = () => save('confirm')
    return () => {
      submitRef.current = null
    }
  }, [save, submitRef])

  const changePrimary = (p: string) => {
    setPrimary(p)
    const next = taxonomy.sub_reasons[p] ?? []
    if (!next.includes(sub)) setSub(next[0])
  }

  return (
    <div className="decide">
      <h4>Your decision</h4>
      <div className="grid grid-3" style={{ marginBottom: 'var(--space-3)' }}>
        <Select
          label="What was wrong"
          value={primary}
          options={taxonomy.primary_reasons}
          onChange={changePrimary}
          format={(v) => humanize(v)}
        />
        <Select label="More detail" value={sub} options={subs} onChange={setSub} format={(v) => humanize(v)} />
        <Select label="Where on the body" value={area} options={taxonomy.body_areas} onChange={setArea} format={(v) => humanize(v)} />
      </div>
      <label className="field" style={{ marginBottom: 'var(--space-3)' }}>
        Note (optional)
        <input type="text" value={note} maxLength={1000} onChange={(e) => setNote(e.target.value)} />
      </label>
      {error && <Callout tone="danger">{error}</Callout>}
      <div className="row" style={{ gap: 'var(--space-2)', alignItems: 'center' }}>
        <button className="btn primary" disabled={saving !== null} onClick={() => save('confirm')}>
          {saving === 'confirm' ? 'Saving…' : 'Confirm'}
        </button>
        <button className="btn" disabled={saving !== null} onClick={() => save('change')}>
          {saving === 'change' ? 'Saving…' : 'Save change'}
        </button>
        <button className="btn" disabled={saving !== null} onClick={() => save('mark_unclear')}>
          {saving === 'mark_unclear' ? 'Saving…' : 'Still unclear'}
        </button>
        <span className="kbd-hint">
          <kbd>J</kbd>
          <kbd>K</kbd> move · <kbd>↵</kbd> confirm
        </span>
      </div>
    </div>
  )
}

export default function Review({ onChange }: { onChange: () => void }) {
  const { data, error, loading, reload } = useApi<ReviewData>('/api/review')
  const taxonomy = useApi<Taxonomy>('/api/taxonomy')
  const [folder, setFolder] = useState(ALL)
  const [openId, setOpenId] = useState<string | null>(null)
  const [flash, setFlash] = useState<string | null>(null)
  const submitRef = useRef<(() => void) | null>(null)

  const queue = data?.queue ?? []
  const resolved = useMemo(() => new Set((data?.corrections ?? []).map((c) => c.return_id)), [data])

  const counts = useMemo(() => {
    const m = new Map<string, number>()
    queue.forEach((r) => m.set(r.review_reason ?? 'OTHER', (m.get(r.review_reason ?? 'OTHER') ?? 0) + 1))
    return m
  }, [queue])

  const total = queue.length
  const done = queue.filter((r) => resolved.has(r.return_id)).length
  const left = total - done

  const shown =
    folder === ALL
      ? queue
      : folder === CLEARED
        ? queue.filter((r) => resolved.has(r.return_id))
        : queue.filter((r) => r.review_reason === folder)
  const open = shown.find((r) => r.return_id === openId) ?? shown[0] ?? null
  const index = open ? shown.findIndex((r) => r.return_id === open.return_id) : -1

  const move = useCallback(
    (delta: number) => {
      if (!shown.length) return
      const next = Math.min(Math.max((index === -1 ? 0 : index) + delta, 0), shown.length - 1)
      setOpenId(shown[next].return_id)
    },
    [shown, index],
  )

  // Keyboard: only when the reviewer is not typing into a field.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement | null
      if (el && /^(INPUT|SELECT|TEXTAREA)$/.test(el.tagName)) return
      if (e.key === 'j' || e.key === 'ArrowDown') {
        e.preventDefault()
        move(1)
      } else if (e.key === 'k' || e.key === 'ArrowUp') {
        e.preventDefault()
        move(-1)
      } else if (e.key === 'Enter') {
        e.preventDefault()
        submitRef.current?.()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [move])

  const saved = (msg: string) => {
    setFlash(msg)
    // Move on before reloading: clearing a queue is a rhythm, not a round trip.
    move(1)
    reload()
    onChange()
  }

  return (
    <>
      <ErrorNote error={error} />
      {loading && !data && <Loading />}
      {data && total === 0 && (
        <div className="stack">
          <h1 style={{ margin: 0 }}>Your review</h1>
          <Callout tone="success">
            Nothing waiting. Every comment was clear enough to sort, or you have already cleared the ones that
            were not.
          </Callout>
        </div>
      )}
      {data && total > 0 && taxonomy.data && (
        <div className="review-full">
          {flash && <Callout tone="success">{flash}</Callout>}
          <div className="filter-tabs" role="tablist" aria-label="Filter the queue">
            <button
              className="filter-tab"
              role="tab"
              aria-selected={folder === ALL}
              onClick={() => setFolder(ALL)}
            >
              Everything
              <span className="count-pill">{total}</span>
            </button>
            {[...counts.entries()]
              .sort((a, b) => b[1] - a[1])
              .map(([code, n]) => (
                <button
                  key={code}
                  className="filter-tab"
                  role="tab"
                  aria-selected={folder === code}
                  onClick={() => setFolder(code)}
                >
                  {reviewLabel(code)}
                  <span className="count-pill">{n}</span>
                </button>
              ))}
            <button
              className="filter-tab"
              role="tab"
              aria-selected={folder === CLEARED}
              onClick={() => setFolder(CLEARED)}
              disabled={done === 0}
              title={done === 0 ? 'Nothing cleared yet' : undefined}
            >
              Cleared
              <span className="count-pill">{done}</span>
            </button>
          </div>
          <div className="inbox">
            <div className="inbox-list" role="listbox" aria-label="Comments to review">
              {shown.map((r) => (
                <button
                  key={r.return_id}
                  className={`inbox-msg${resolved.has(r.return_id) ? ' resolved' : ''}`}
                  aria-current={open?.return_id === r.return_id}
                  onClick={() => setOpenId(r.return_id)}
                >
                  <span className="mrow">
                    <span className="mid">{r.return_id}</span>
                    {resolved.has(r.return_id) && <span className="mdone">✓ cleared</span>}
                  </span>
                  <span className="mtext">{r.original_text || '(no comment)'}</span>
                  <span className="mmeta">
                    <span className="badge warn">{reviewLabel(r.review_reason)}</span>
                    <span>
                      {r.sku} · {r.category}
                    </span>
                  </span>
                </button>
              ))}
            </div>

            {open ? (
              <div className="inbox-open">
                <div className="crumb">
                  {folder === ALL ? 'Everything' : folder === CLEARED ? 'Cleared' : reviewLabel(folder)} › {open.return_id}
                  {index >= 0 && ` · ${index + 1} of ${shown.length}`}
                </div>
                <p className="said">{open.original_text ? <q>{open.original_text}</q> : <em>No comment was written</em>}</p>
                <p className="about">
                  {open.product_name} · {open.sku} · {open.category} · {open.vendor} · size{' '}
                  {open.size_ordered || 'not recorded'}
                </p>

                <div className="why-box">
                  <h4>Why the agent is unsure</h4>
                  {whyUnsure(open).map((line, i) => (
                    <p key={i}>{line}</p>
                  ))}
                  {open.trace && open.trace.length > 0 && (
                    <details>
                      <summary>Show every step it took</summary>
                      <ol className="trace" style={{ marginTop: 'var(--space-2)' }}>
                        {open.trace.map((s, i) => (
                          <li key={i}>{traceLine(s)}</li>
                        ))}
                      </ol>
                    </details>
                  )}
                </div>

                <Decision row={open} taxonomy={taxonomy.data} onSaved={saved} submitRef={submitRef} />
              </div>
            ) : (
              <div className="inbox-open">
                <p className="caption">Nothing in this folder. Pick another on the left.</p>
              </div>
            )}

            <div className="inbox-foot">
              <span className="pbar">
                <i style={{ width: `${total ? (100 * done) / total : 0}%` }} />
              </span>
              <span>
                <b style={{ color: 'var(--color-fg)' }}>
                  {done} of {total}
                </b>{' '}
                cleared
                {left > 0 && <> · {fmtMinutes(Math.round(left * data.minutes_per_row))} left at {data.minutes_per_row} min each</>}
              </span>
              <span className="kbd-hint">
                <Icon name="check" size={14} />
                Your answers update the briefing straight away
              </span>
            </div>
          </div>
        </div>
      )}
    </>
  )
}
