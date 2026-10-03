import { useEffect, useRef, useState, type DragEvent } from 'react'
import { api, ApiError, type Readiness, type Status } from '../api'
import { Callout, Card, Icon, Metric, PageHead } from '../ui'
import { fmtInt, fmtUsd } from '../format'

interface Report {
  total_rows: number
  other_rows: number
  other_pct: number
  unique_skus: number
  unique_vendors: number
  unique_categories: number
  non_other_rows_with_text: number
  other_rows_with_empty_text: number
  errors: string[]
  is_valid: boolean
}

interface DatasetInfo {
  source: string
  name: string
  original_name?: string
  report: Report
  has_answer_key: boolean
  classifier: Readiness
  max_rows: number
}

interface Job {
  id: string
  state: 'running' | 'done' | 'error'
  progress: number
  message: string
  error?: string
  result?: {
    rows: number
    elapsed_seconds: number
    run_total_usd: number | null
    counts: Record<string, number>
  }
}

const REQUIRED =
  'return_id, order_id, sku, product_name, category, vendor, return_reason_dropdown, return_date'

export default function Process({ status, onChange }: { status: Status | null; onChange: () => void }) {
  const [info, setInfo] = useState<DatasetInfo | null>(null)
  const [busy, setBusy] = useState<'upload' | 'start' | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [drag, setDrag] = useState(false)
  const [limit, setLimit] = useState(40)
  const [localJob, setJob] = useState<Job | null>(null)
  const fileInput = useRef<HTMLInputElement>(null)

  // Pick up a run this session already has in flight (e.g. after a reload).
  const resumed = status?.job?.state === 'running' ? (status.job as Job) : null
  const job = localJob ?? resumed

  // Poll the running job.
  useEffect(() => {
    if (!job || job.state !== 'running') return
    const t = window.setInterval(async () => {
      try {
        const next = await api.get<Job>(`/api/jobs/${job.id}`)
        setJob(next)
        if (next.state !== 'running') onChange()
      } catch (e) {
        // The server keeps jobs in memory; after a restart the job is gone.
        const message =
          e instanceof ApiError && e.status === 404
            ? 'The server lost track of this run (it may have restarted). Start it again.'
            : e instanceof ApiError
              ? e.message
              : String(e)
        setJob({ ...job, state: 'error', error: message })
        onChange()
      }
    }, 1000)
    return () => window.clearInterval(t)
  }, [job, onChange])

  const choose = (next: DatasetInfo) => {
    setInfo(next)
    setJob(null)
    setError(null)
    const other = Math.min(next.report.other_rows, next.max_rows)
    setLimit(Math.min(40, Math.max(1, other)))
  }

  const upload = async (file: File) => {
    setBusy('upload')
    setError(null)
    try {
      choose(await api.upload<DatasetInfo>('/api/upload', file))
    } catch (e) {
      setInfo(null)
      setError(e instanceof ApiError ? e.message : String(e))
    } finally {
      setBusy(null)
    }
  }

  const useSample = async () => {
    setBusy('upload')
    setError(null)
    try {
      choose(await api.get<DatasetInfo>('/api/dataset?source=sample'))
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e))
    } finally {
      setBusy(null)
    }
  }

  const start = async () => {
    if (!info) return
    setBusy('start')
    setError(null)
    try {
      setJob(await api.post<Job>('/api/jobs', { source: info.source, limit }))
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e))
    } finally {
      setBusy(null)
    }
  }

  const startOver = async () => {
    await api.del('/api/run')
    setInfo(null)
    setJob(null)
    onChange()
  }

  const onDrop = (e: DragEvent) => {
    e.preventDefault()
    setDrag(false)
    const f = e.dataTransfer.files?.[0]
    if (f) upload(f)
  }

  const report = info?.report
  const eligible = report?.other_rows ?? 0
  // The server caps one run; the slider never offers more than it will accept.
  const other = Math.min(eligible, info?.max_rows ?? eligible)
  const ready = info?.classifier.ready ?? status?.classifier.ready ?? false
  const running = job?.state === 'running'

  return (
    <>
      <PageHead
        title="Upload & process"
        lede="Upload a returns file, choose how many rows to classify, and run it. The dashboard is built only from the rows you process here."
      />

      <div className="stack">
        {status?.artifact && status.run && !job && (
          <Callout tone="success">
            <div className="row" style={{ justifyContent: 'space-between' }}>
              <span>
                Your dashboard is built from <strong>{fmtInt(status.run.eligible_rows)}</strong> classified rows
                (finished {status.run.finished_at}). Processing a new file replaces it.
              </span>
              <span className="row">
                <a className="btn primary" href="#/overview">
                  Open dashboard
                </a>
                <button className="btn" type="button" onClick={startOver}>
                  Start over
                </button>
              </span>
            </div>
          </Callout>
        )}

        <Card title="1. Choose a file" sub={`.xlsx or .csv, up to 20 MB. Required columns: ${REQUIRED}.`}>
          <div
            className={`dropzone${drag ? ' drag' : ''}`}
            onDragOver={(e) => {
              e.preventDefault()
              setDrag(true)
            }}
            onDragLeave={() => setDrag(false)}
            onDrop={onDrop}
          >
            <Icon name="file" size={28} />
            <div className="stack-sm" style={{ flex: 1, gap: 4 }}>
              <strong>{info ? (info.original_name ?? info.name) : 'Drop a returns file here'}</strong>
              <span className="caption">
                {info ? 'Ready. Choose another file to replace it.' : 'or browse for one on your computer'}
              </span>
            </div>
            <input
              ref={fileInput}
              type="file"
              accept=".xlsx,.csv"
              className="sr-only"
              tabIndex={-1}
              aria-hidden="true"
              id="file-input"
              onChange={(e) => {
                const f = e.target.files?.[0]
                if (f) upload(f)
                e.target.value = ''
              }}
            />
            <button className="btn primary" type="button" disabled={busy !== null || running} onClick={() => fileInput.current?.click()}>
              <Icon name="upload" />
              {busy === 'upload' ? 'Reading…' : 'Browse'}
            </button>
          </div>
          <p className="caption" style={{ marginTop: 12 }}>
            No file to hand?{' '}
            <button
              type="button"
              className="btn"
              style={{ minHeight: 32, padding: '0 10px', fontSize: '0.82rem' }}
              disabled={busy !== null || running}
              onClick={useSample}
            >
              Use the sample dataset
            </button>{' '}
            (1,000 synthetic returns with an answer key, so Evaluation can score it).
          </p>
        </Card>

        {error && <Callout tone="danger">{error}</Callout>}

        {report && !report.is_valid && (
          <Callout tone="danger">
            This file cannot be processed:
            <ul style={{ margin: '4px 0 0', paddingLeft: 18 }}>
              {report.errors.map((e) => (
                <li key={e}>{e}</li>
              ))}
            </ul>
          </Callout>
        )}

        {report && report.is_valid && (
          <Card title="2. Check the file">
            <div className="stack-sm">
              <div className="grid grid-4">
                <Metric label="Rows" value={fmtInt(report.total_rows)} />
                <Metric label="'Other' rows" value={fmtInt(report.other_rows)} hint={`${report.other_pct}% of all`} tone="accent" />
                <Metric label="SKUs" value={fmtInt(report.unique_skus)} />
                <Metric label="Vendors" value={fmtInt(report.unique_vendors)} />
              </div>
              <div className="row">
                {info?.has_answer_key ? (
                  <span className="badge ok">Answer key found: Evaluation will score this run</span>
                ) : (
                  <span className="badge">No answer key: Evaluation will be skipped</span>
                )}
                {report.other_rows_with_empty_text > 0 && (
                  <span className="badge warn">
                    {report.other_rows_with_empty_text} 'Other' rows have no text and go straight to review
                  </span>
                )}
              </div>
            </div>
          </Card>
        )}

        {report && report.is_valid && eligible === 0 && (
          <Callout tone="warning">This file has no rows with the dropdown set to 'Other', so there is nothing to classify.</Callout>
        )}

        {report && report.is_valid && eligible > 0 && (
          <Card title="3. Choose rows and run" sub="Rows are taken in file order from the 'Other' rows.">
            <div className="stack-sm">
              <label className="field">
                <span>
                  Rows to classify: <strong className="mono" style={{ color: 'var(--color-fg)' }}>{fmtInt(limit)}</strong>{' '}
                  of {fmtInt(eligible)}
                  {other < eligible && ` (one run can classify up to ${fmtInt(other)})`}
                </span>
                <input
                  type="range"
                  min={1}
                  max={other}
                  step={1}
                  value={limit}
                  disabled={running}
                  onChange={(e) => setLimit(Number(e.target.value))}
                  aria-valuetext={`${limit} rows`}
                />
              </label>
              <div className="row">
                {[40, 100, other].filter((n, i, a) => n <= other && a.indexOf(n) === i).map((n) => (
                  <button key={n} type="button" className="btn" style={{ minHeight: 32 }} disabled={running} onClick={() => setLimit(n)}>
                    {n === eligible ? `All ${fmtInt(n)}` : n === other ? `Max ${fmtInt(n)}` : fmtInt(n)}
                  </button>
                ))}
              </div>
              <p className="caption">
                Most rows need two classifier calls. As a guide, 440 rows took about 6 minutes and cost under 2 cents.
                Responses are cached, so re-running the same rows is free and fast.
              </p>

              {!ready && (
                <Callout tone="warning">
                  {(info?.classifier.messages ?? status?.classifier.messages ?? []).join(' ')}
                </Callout>
              )}

              <div>
                <button className="btn primary" type="button" disabled={!ready || running || busy !== null} onClick={start}>
                  <Icon name="play" size={16} />
                  {running ? 'Classifying…' : `Classify ${fmtInt(limit)} rows`}
                </button>
              </div>
            </div>
          </Card>
        )}

        {job && (
          <Card
            title={job.state === 'done' ? 'What the agent did' : job.state === 'error' ? 'The run stopped' : 'What the agent is doing'}
            sub={job.state === 'running' ? 'Each stage finishes before the next one starts. Leaving this screen does not stop it.' : undefined}
          >
            {job.state === 'error' ? (
              <Callout tone="danger">{job.error}</Callout>
            ) : (
              <div className="stack-sm" aria-live="polite">
                <RunTimeline progress={job.state === 'done' ? 1 : job.progress} counts={job.result?.counts} />
                {job.state === 'running' && (
                  <p className="caption">{pres(job.message)}</p>
                )}
                {job.state === 'done' && job.result && (
                  <>
                    <Callout tone="success">
                      Read {fmtInt(job.result.rows)} comments in {job.result.elapsed_seconds.toFixed(0)} seconds for{' '}
                      {fmtUsd(job.result.run_total_usd, 2)}. Nothing was dropped.
                    </Callout>
                    <div className="row">
                      <a className="btn primary" href="#/overview">
                        <Icon name="insights" size={16} />
                        Read this week's briefing
                      </a>
                      {(job.result.counts?.NEEDS_REVIEW ?? 0) > 0 && (
                        <a className="btn" href="#/review">
                          Review the {fmtInt(job.result.counts.NEEDS_REVIEW)} it could not sort
                        </a>
                      )}
                    </div>
                  </>
                )}
              </div>
            )}
          </Card>
        )}
      </div>
    </>
  )
}


/** Progress fractions the pipeline reports, mapped to named stages.
 *  A bare "72%" tells a reviewer nothing; "double-checking the unsure ones"
 *  tells them what the agent is busy with and why it takes a moment. */
interface Stage {
  from: number
  to: number
  label: string
  sub: string
  key: string
  handover?: boolean
}

const STAGES: Stage[] = [
  { from: 0, to: 0.1, label: 'Reading the comments', sub: 'Every return that arrived with no reason on it', key: 'rows' },
  { from: 0.1, to: 0.4, label: 'Sorting them into reasons', sub: 'Deciding what each customer was telling you', key: 'AUTO_ACCEPTED' },
  { from: 0.4, to: 0.8, label: 'Double-checking the unsure ones', sub: 'A second, stronger reading where the first was not confident', key: 'ACCEPTED_AFTER_CHECK' },
  { from: 0.8, to: 1, label: 'Handing the rest to a person', sub: 'Comments too vague to sort. Nothing was guessed.', key: 'NEEDS_REVIEW', handover: true },
]

function RunTimeline({ progress, counts }: { progress: number; counts?: Record<string, number> }) {
  return (
    <ol className="timeline">
      {STAGES.map((st) => {
        const share = Math.min(Math.max((progress - st.from) / (st.to - st.from), 0), 1)
        const state = share >= 1 ? 'done' : share > 0 ? 'active' : 'pending'
        const tally = counts?.[st.key]
        return (
          <li key={st.label} className={`${state}${st.handover ? ' handover' : ''}`}>
            <span className="pip">{state === 'done' ? (st.handover ? '!' : '✓') : state === 'active' ? '•' : ''}</span>
            <div>
              <b>{st.label}</b>
              <div className="sub">{st.sub}</div>
              <div className="track">
                <i style={{ width: `${Math.round(share * 100)}%` }} />
              </div>
            </div>
            <div className="tally">
              {tally === undefined ? '' : fmtInt(tally)}
              {tally === undefined ? '' : <small>{st.handover ? 'for you' : 'done'}</small>}
            </div>
          </li>
        )
      })}
    </ol>
  )
}

/** Progress messages arrive already free of model identity; this only drops
 *  the leading row counts that read as machine output. */
const pres = (msg: string) => msg.replace(/^\s*\[?\d+%?\]?\s*/, '')
