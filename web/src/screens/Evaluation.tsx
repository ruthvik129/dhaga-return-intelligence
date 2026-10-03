import { useApi, type Bucket, type Run } from '../api'
import { Callout, Card, ErrorNote, Loading, Metric, PageHead, RunBanner, Table } from '../ui'
import { STATUS_LABEL, fmtInt, fmtPct, humanize } from '../format'

interface Baseline {
  rows: number
  answered: number
  abstained: number
  coverage_pct: number
  accuracy_on_answered: Bucket
  accuracy_over_all: Bucket
}

interface Cascade {
  available: boolean
  buckets: Bucket[]
  primary_accuracy_overall: Bucket
  rates: { auto_accept_pct: number; second_check_pct: number; review_pct: number }
  confusion_matrix: Record<string, Record<string, number>>
  notable_confusions: { pair: string; confused_rows: number }[]
}

interface EvalData {
  artifact: boolean
  run?: Run
  dataset?: { total_rows: number; other_rows: number }
  has_answer_key?: boolean
  baseline?: Baseline
  cascade?: Cascade
  misses?: {
    return_id: string
    original_text: string
    primary_reason: string
    expected_primary_reason: string
    confidence: number | null
    status: string
  }[]
  rows_scored?: number
}

const ci = (b: Bucket) => (b.n ? `${b.ci_low_pct.toFixed(1)}–${b.ci_high_pct.toFixed(1)}` : 'n/a')

function Matrix({ m }: { m: Record<string, Record<string, number>> }) {
  const rows = Object.keys(m).sort()
  const cols = Array.from(new Set(rows.flatMap((r) => Object.keys(m[r])))).sort()
  const max = Math.max(1, ...rows.flatMap((r) => cols.map((c) => m[r][c] ?? 0)))
  return (
    <div className="table-wrap">
      <table className="matrix">
        <caption className="sr-only">Confusion matrix: rows are the answer key, columns are predictions</caption>
        <thead>
          <tr>
            <th scope="col">Answer key ↓ / predicted →</th>
            {cols.map((c) => (
              <th key={c} scope="col" className="num">
                {humanize(c)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r}>
              <th scope="row" style={{ position: 'static' }}>
                {humanize(r)}
              </th>
              {cols.map((c) => {
                const v = m[r][c] ?? 0
                const diag = r === c
                const alpha = v ? 0.12 + 0.6 * (v / max) : 0
                const tint = diag ? 'var(--color-success)' : 'var(--color-danger)'
                return (
                  <td
                    key={c}
                    className={`num${diag ? ' diag' : ''}`}
                    style={{ background: v ? `color-mix(in srgb, ${tint} ${Math.round(alpha * 100)}%, transparent)` : undefined }}
                  >
                    {v || '·'}
                  </td>
                )
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export default function Evaluation() {
  const { data, error, loading } = useApi<EvalData>('/api/evaluation')

  return (
    <>
      <PageHead
        title="How accurate is it"
        lede="Your run marked against the answers already sitting in the file you uploaded. While it reads, the agent only ever sees the product details and what the customer wrote."
      >
        <RunBanner run={data?.run} />
      </PageHead>
      <ErrorNote error={error} />
      {loading && !data && <Loading />}

      {data && data.artifact && !data.has_answer_key && (
        <Callout tone="info">
          The file this run came from has no answer-key columns (<code>expected_primary_reason</code>,{' '}
          <code>expected_sub_reason</code>, <code>expected_body_area</code>), so there is nothing to score against. Upload a
          labelled file, or use the sample dataset, to see accuracy here.
        </Callout>
      )}

      {data && data.artifact && data.has_answer_key && data.cascade && !data.cascade.available && (
        <Callout tone="info">
          The file has answer-key columns, but none of them line up with the rows this run classified, so there is
          nothing to score. Check that the answer key is filled in for the 'Other' rows.
        </Callout>
      )}

      {data?.baseline && data.cascade?.available && (
        <div className="stack">
          <Callout tone="info">
            If the answer key was generated with the data (as in the sample), read these as self-consistency against that
            generator rather than real-world accuracy.
          </Callout>

          <Card title="Classifier vs rules baseline" sub={`Over the same ${fmtInt(data.baseline.rows)} rows you classified.`}>
            <div className="stack-sm">
              <div className="grid grid-4">
                <Metric
                  label="Classifier, all rows"
                  value={fmtPct(data.cascade.primary_accuracy_overall.accuracy_pct)}
                  hint={`${data.cascade.primary_accuracy_overall.correct}/${data.cascade.primary_accuracy_overall.n}, CI ${ci(
                    data.cascade.primary_accuracy_overall,
                  )}`}
                  tone="accent"
                />
                <Metric
                  label="Rules baseline, all rows"
                  value={fmtPct(data.baseline.accuracy_over_all.accuracy_pct)}
                  hint="Abstentions count as misses"
                />
                <Metric
                  label="Rules baseline, rows it answered"
                  value={fmtPct(data.baseline.accuracy_on_answered.accuracy_pct)}
                  hint={`${data.baseline.accuracy_on_answered.correct}/${data.baseline.accuracy_on_answered.n}`}
                />
                <Metric label="Baseline coverage" value={fmtPct(data.baseline.coverage_pct)} hint={`Abstained on ${data.baseline.abstained}`} />
              </div>
              <p className="caption">
                Both baseline figures are shown on purpose. Quoting only the answered-rows number would hide that keyword
                matching cannot read {data.baseline.abstained} of the comments at all. The rules baseline uses no
                classifier and no API key.
              </p>
            </div>
          </Card>

          <Card title="Accuracy" sub="Auto-accept precision matters most: it is the accuracy of the rows nobody looks at.">
            <Table
              rows={data.cascade.buckets}
              caption="Accuracy by bucket"
              columns={[
                { key: 'label', header: 'Metric' },
                { key: 'correct', header: 'Correct', num: true },
                { key: 'n', header: 'n', num: true },
                { key: 'acc', header: 'Accuracy', num: true, render: (b) => fmtPct(b.accuracy_pct) },
                { key: 'ci', header: '95% CI', num: true, render: ci },
              ]}
            />
          </Card>

          <div className="grid grid-3">
            <Metric label="Auto-accepted" value={fmtPct(data.cascade.rates.auto_accept_pct)} />
            <Metric label="Sent for a second check" value={fmtPct(data.cascade.rates.second_check_pct)} />
            <Metric label="Sent to review" value={fmtPct(data.cascade.rates.review_pct)} />
          </div>

          <Card title="Confusion matrix" sub="Rows are the answer key, columns are predictions. Green is correct, red is a miss.">
            <Matrix m={data.cascade.confusion_matrix} />
          </Card>

          <Card title="Confusions worth watching">
            <Table
              rows={data.cascade.notable_confusions}
              caption="Notable confusions"
              columns={[
                { key: 'pair', header: 'Pair (either direction)', render: (c) => c.pair.split(' vs ').map(humanize).join(' vs ') },
                { key: 'confused_rows', header: 'Rows', num: true },
              ]}
            />
          </Card>

          <Card
            title="Read the misses"
            sub={`${fmtInt(data.misses?.length ?? 0)} of ${fmtInt(data.rows_scored ?? 0)} rows disagree with the answer key. Your file has ${fmtInt(
              data.dataset?.total_rows ?? 0,
            )} rows, ${fmtInt(data.dataset?.other_rows ?? 0)} of them 'Other'.`}
          >
            <Table
              rows={data.misses ?? []}
              scroll
              caption="Misses"
              empty="No misses."
              columns={[
                { key: 'return_id', header: 'Return' },
                { key: 'original_text', header: 'Customer comment', wide: true },
                { key: 'primary_reason', header: 'Predicted', render: (m) => humanize(m.primary_reason) },
                { key: 'expected_primary_reason', header: 'Answer key', render: (m) => humanize(m.expected_primary_reason) },
                {
                  key: 'confidence',
                  header: 'Confidence',
                  num: true,
                  render: (m) => (m.confidence === null || m.confidence === undefined ? 'n/a' : m.confidence.toFixed(2)),
                },
                { key: 'status', header: 'Status', render: (m) => STATUS_LABEL[m.status] ?? m.status },
              ]}
            />
          </Card>
        </div>
      )}
    </>
  )
}
