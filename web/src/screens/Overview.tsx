/**
 * The home screen, as a written briefing (approved design direction B).
 *
 * The audience is a category head deciding what to change this week, so the
 * screen leads with conclusions and keeps the machinery out of sight. Every
 * sentence here is built from numbers the API computed; nothing is written by
 * a model, and every claim links to the comments behind it.
 */
import { useApi, type Run } from '../api'
import { Callout, Card, ErrorNote, Icon, Loading, Metric, PageHead } from '../ui'
import { DIMENSION_NOUN, detailLabel, fmtInt, fmtMinutes, fmtUsd, fmtWhen, reasonLabel } from '../format'

interface Quote {
  return_id: string
  text: string
  size_ordered: string | null
}

export interface Finding {
  dimension: string
  group: string
  product_name: string | null
  reason: string
  matching_rows: number
  classified_rows: number
  share_pct: number
  baseline_pct: number
  lift: number | null
  share_ci_low_pct: number
  share_ci_high_pct: number
  excess_rows: number
  top_detail: string | null
  top_detail_rows: number
  top_body_area: string | null
  top_body_area_rows: number
  quotes: Quote[]
}

interface Briefing {
  artifact: boolean
  run: Run
  counts: {
    total_returns_loaded: number
    total_other: number
    pct_other: number
    auto_classified: number
    needs_review: number
    resolved_by_reviewer: number
    processing_error: number
  }
  findings: Finding[]
  queue: { rows: number; minutes: number; by_reason: { reason: string; rows: number }[]; example: string | null }
  cost: { run_total_usd: number | null; per_1000_usd: number | null }
  accuracy: { overall_pct: number | null; auto_pct: number | null; rows: number } | null
  non_other_rows_with_text: number
  validation: { errors: string[]; rows_missing_required_values: number }
}

const ORDINAL = ['ONE', 'TWO', 'THREE', 'FOUR', 'FIVE']

/** The headline sentence: one claim, no jargon, no statistics vocabulary.
 *
 *  Built from the dominant detail rather than hedged. The data says
 *  `too_small`, so the sentence says "runs small" — "tight or loose" would be
 *  technically safer and useless to act on.
 */
export function headline(f: Finding): string {
  const name = f.product_name && f.dimension === 'sku' ? `${f.group} ${f.product_name}` : f.group
  const detail = detailLabel(f.top_detail)
  const area = f.top_body_area && f.top_body_area !== 'none' ? f.top_body_area : null

  if (f.reason === 'fit' && area) {
    if (f.top_detail === 'too_small') return `The ${name} runs small at the ${area}.`
    if (f.top_detail === 'too_large') return `The ${name} runs large at the ${area}.`
    return `The ${name} does not fit right at the ${area}.`
  }
  if (f.dimension === 'vendor') {
    return `Returns from ${f.group} are mostly about ${detail ?? reasonLabel(f.reason)}.`
  }
  if (detail) {
    return `The ${name} keeps coming back for ${detail}.`
  }
  return `The ${name} comes back for ${reasonLabel(f.reason)} more than anything else.`
}

/** The supporting sentences: the arithmetic, in words read once and understood. */
export function evidence(f: Finding): string {
  const noun = DIMENSION_NOUN[f.dimension] ?? f.dimension
  const detail = detailLabel(f.top_detail)
  const area = f.top_body_area && f.top_body_area !== 'none' ? f.top_body_area : null

  const sentences = [
    `${f.matching_rows} of its ${f.classified_rows} returns are about ${reasonLabel(f.reason)}.`,
  ]
  const named: string[] = []
  if (detail && f.top_detail_rows > 1) named.push(`${f.top_detail_rows} name ${detail}`)
  if (area && f.top_body_area_rows > 1) named.push(`${f.top_body_area_rows} mention the ${area}`)
  if (named.length) sentences.push(`${named.join(', and ')}.`)

  // "every other" would be wrong: the baseline is the share across the whole
  // file, this group included. Say what the number actually is.
  sentences.push(
    `Across all ${noun}s in the file, ${Math.round(f.baseline_pct)}% of returns are about ` +
      `${reasonLabel(f.reason)}; for this one it is ${Math.round(f.share_pct)}%.`,
  )
  sentences.push(
    f.excess_rows === 1
      ? 'That is one more return than you would expect.'
      : `That is ${f.excess_rows} more returns than you would expect.`,
  )
  return sentences.join(' ')
}

export default function Overview() {
  const { data, error, loading } = useApi<Briefing>('/api/briefing')

  const n = data?.findings.length ?? 0
  const count = ['no', 'One', 'Two', 'Three', 'Four', 'Five'][n] ?? String(n)

  return (
    <>
      <PageHead
        title="This week in returns"
        lede={
          data?.artifact
            ? `Your returns agent read ${fmtInt(data.counts.total_other)} customer comments that arrived with no reason attached, ${fmtWhen(data.run.finished_at)}. It sorted ${fmtInt(data.counts.auto_classified)} on its own and left ${fmtInt(data.queue.rows)} for a person.`
            : undefined
        }
      />
      <ErrorNote error={error} />
      {loading && !data && <Loading />}
      {data?.artifact && (
        <div className="stack">
          <div className="grid grid-4">
            <Metric label="Comments read" value={fmtInt(data.counts.total_other)} />
            <Metric label="Sorted by the agent" value={fmtInt(data.counts.auto_classified)} />
            <Metric
              label="Waiting for you"
              value={fmtInt(data.queue.rows)}
              tone={data.queue.rows ? 'accent' : undefined}
              hint={data.queue.rows ? <a href="#/review">Open your review</a> : 'Nothing waiting'}
            />
            <Metric
              label="What it cost"
              value={fmtUsd(data.cost.run_total_usd, 2)}
              hint={<a href="#/cost">See the breakdown</a>}
            />
          </div>

          <Card
            title={n ? `${count} thing${n === 1 ? '' : 's'} to fix` : 'Nothing stands out this week'}
            sub={n ? 'Ranked by how many more returns these carry than the rest of the file.' : undefined}
          >
            <div className="reading">
              {n === 0 && (
                <p className="caption" style={{ fontSize: '0.9375rem' }}>
                  No style or supplier is getting noticeably more of any one complaint than the others. The returns
                  you have are spread evenly, so there is no single fix to make. Worth looking again after the next
                  batch.
                </p>
              )}
              {data.findings.map((f, i) => (
                <article className="finding" key={`${f.dimension}-${f.group}-${f.reason}`}>
                  <div className="num">{ORDINAL[i] ?? `ITEM ${i + 1}`}</div>
                  <h3>{headline(f)}</h3>
                  <p className="ev">{evidence(f)}</p>
                  {f.quotes.map((q) => (
                    <blockquote key={q.return_id}>
                      {q.text}
                      <cite>
                        {q.return_id}
                        {q.size_ordered ? ` · size ${q.size_ordered}` : ''}
                      </cite>
                    </blockquote>
                  ))}
                  <p className="range">
                    Measured on {f.classified_rows} returns, so the true figure is very likely between{' '}
                    {Math.round(f.share_ci_low_pct)}% and {Math.round(f.share_ci_high_pct)}% — still above the{' '}
                    {Math.round(f.baseline_pct)}% figure for the file as a whole, either way.
                  </p>
                  <a className="more" href="#/insights">
                    Read all {f.matching_rows} comments
                    <Icon name="chevron" size={15} />
                  </a>
                </article>
              ))}
            </div>
          </Card>

          {data.queue.rows > 0 && (
            <Card title={`${fmtInt(data.queue.rows)} comments were too unclear to sort`}>
              <div className="stack-sm">
                <Callout tone="warning">
                  The agent would have been guessing, so it stopped and left them for a person. Most read like{' '}
                  {data.queue.example ? <q>{data.queue.example}</q> : 'a few words with nothing to go on'} — nothing
                  to go on. {fmtMinutes(data.queue.minutes)} of reading.
                </Callout>
                <div>
                  <a className="btn primary" href="#/review">
                    <Icon name="review" size={16} />
                    Open your review
                  </a>
                </div>
              </div>
            </Card>
          )}

          {data.validation.errors.length > 0 && <Callout tone="danger">{data.validation.errors.join(' ')}</Callout>}

          <Card title="How this was made">
            <p className="caption" style={{ margin: 0 }}>
              Two AI models read every comment. The first sorted {fmtInt(data.counts.auto_classified)} of{' '}
              {fmtInt(data.counts.total_other)} on its own; where it was unsure a stronger second model looked
              again, and where both stayed unsure the comment came to you. Nothing was dropped and nothing was
              guessed.
              {data.accuracy?.overall_pct != null && (
                <>
                  {' '}
                  Checked against the answers already in your file, it agreed on{' '}
                  {Math.round(data.accuracy.overall_pct)}% of all {fmtInt(data.accuracy.rows)} comments and{' '}
                  {data.accuracy.auto_pct != null && `${Math.round(data.accuracy.auto_pct)}%`} of the ones it
                  decided alone.
                </>
              )}{' '}
              {data.non_other_rows_with_text > 0 && (
                <>
                  {fmtInt(data.non_other_rows_with_text)} returns that already had a reason also carried a comment;
                  those are out of scope for now.
                </>
              )}{' '}
              <a href="#/evaluation">See how it was checked</a> · <a href="#/cost">See what it cost</a>
            </p>
          </Card>
        </div>
      )}
    </>
  )
}
