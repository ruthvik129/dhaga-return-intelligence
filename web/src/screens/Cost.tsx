import { useState } from 'react'
import { useApi, type Run } from '../api'
import { Callout, Card, ErrorNote, Loading, Metric, PageHead, RunBanner, Table } from '../ui'
import { fmtInt, fmtPct, fmtUsd } from '../format'

interface StageCost {
  stage: string
  calls: number
  input_tokens: number
  output_tokens: number
  input_cost_usd: number | null
  output_cost_usd: number | null
  total_cost_usd: number | null
  priced: boolean
  price_as_of: string
  token_counts: string
}

interface CostData {
  run: Run
  cost: {
    run_total_usd: number | null
    cost_per_1000_comments_usd: number | null
    rows_classified: number
    fully_priced: boolean
    all_tokens_measured: boolean
    per_stage: StageCost[]
    unpriced_stages: string[]
  }
  escalation: { total_calls: number; second_check_calls: number; pct_calls_on_second_check: number }
  usage: (Record<string, number | string | boolean> & { stage: string })[]
  review_share: number
  minutes_per_row: number
}

const usd6 = (n: number | null) => fmtUsd(n, 6)

export default function Cost() {
  const { data, error, loading } = useApi<CostData>('/api/cost')
  const [volume, setVolume] = useState(50_000)

  const c = data?.cost
  const e = data?.escalation
  const projected =
    c?.cost_per_1000_comments_usd === null || c?.cost_per_1000_comments_usd === undefined
      ? null
      : Math.round(((c.cost_per_1000_comments_usd * volume) / 1000) * 100) / 100
  const reviewHours = data ? (data.review_share * volume * data.minutes_per_row) / 60 : 0
  const asOf = Array.from(new Set(c?.per_stage.map((s) => s.price_as_of).filter(Boolean) ?? []))

  return (
    <>
      <PageHead title="What it cost" lede="What your run actually spent, what the second opinion added, and what the same work would cost at a bigger volume.">
        <RunBanner run={data?.run} />
      </PageHead>
      <ErrorNote error={error} />
      {loading && !data && <Loading />}
      {data && c && e && (
        <div className="stack">
          <div className="grid grid-4">
            <Metric label="This run" value={fmtUsd(c.run_total_usd)} tone="accent" />
            <Metric label="Per 1,000 comments" value={fmtUsd(c.cost_per_1000_comments_usd)} />
            <Metric label="Rows classified" value={fmtInt(c.rows_classified)} />
            <Metric label="Classifier calls" value={fmtInt(e.total_calls)} />
          </div>

          {!c.fully_priced && (
            <Callout tone="warning">
              Not every stage has a configured price, so the total is incomplete ({c.unpriced_stages.join(', ')}). An
              unpriced stage is reported as unavailable rather than $0, because a silent zero makes an expensive run look
              free.
            </Callout>
          )}
          {!c.all_tokens_measured && (
            <Callout tone="warning">
              At least one call returned no usage data, so part of this is estimated rather than measured.
            </Callout>
          )}

          <Card title="By stage" sub={asOf.length ? `Prices are a snapshot taken ${asOf.join(', ')}.` : undefined}>
            <Table
              rows={c.per_stage}
              caption="Cost by stage"
              columns={[
                { key: 'stage', header: 'Stage' },
                { key: 'calls', header: 'Calls', num: true, render: (s) => fmtInt(s.calls) },
                { key: 'input_tokens', header: 'Input tokens', num: true, render: (s) => fmtInt(s.input_tokens) },
                { key: 'output_tokens', header: 'Output tokens', num: true, render: (s) => fmtInt(s.output_tokens) },
                { key: 'input_cost_usd', header: 'Input $', num: true, render: (s) => usd6(s.input_cost_usd) },
                { key: 'output_cost_usd', header: 'Output $', num: true, render: (s) => usd6(s.output_cost_usd) },
                { key: 'total_cost_usd', header: 'Total $', num: true, render: (s) => usd6(s.total_cost_usd) },
                { key: 'token_counts', header: 'Tokens' },
              ]}
            />
          </Card>

          <Card
            title="What escalation cost"
            sub="Escalation only pays for itself while the share sent for a second check stays small. If most rows need one, a single stronger pass is simpler and costs about the same."
          >
            <div className="grid grid-3">
              <Metric label="Second-check calls" value={fmtInt(e.second_check_calls)} />
              <Metric label="Share of calls on second check" value={fmtPct(e.pct_calls_on_second_check)} />
              <Metric
                label="Share of rows escalated"
                value={fmtPct(c.rows_classified ? (100 * e.second_check_calls) / c.rows_classified : null)}
              />
            </div>
          </Card>

          <Card title="Project it forward">
            <div className="stack-sm">
              <label className="field" style={{ maxWidth: 280 }}>
                Comments per month
                <input
                  type="number"
                  min={1000}
                  max={5_000_000}
                  step={1000}
                  value={volume}
                  onChange={(ev) => setVolume(Math.max(1000, Math.min(5_000_000, Number(ev.target.value) || 1000)))}
                />
              </label>
              <div className="grid grid-2">
                <Metric label={`Classifier cost at ${fmtInt(volume)} comments`} value={fmtUsd(projected, 2)} />
                <Metric label="Review hours at that volume" value={`${fmtInt(Math.round(reviewHours))} h`} tone="accent" />
              </div>
              <p className="caption">
                At your measured {fmtPct(data.review_share * 100)} review rate and an assumed {data.minutes_per_row} minutes per
                row. Reviewer time is the real cost here; classifier spend is rounding error beside it.
              </p>
            </div>
          </Card>

          <Card title="Raw token usage">
            <Table
              rows={data.usage}
              caption="Raw token usage by stage"
              columns={[
                { key: 'stage', header: 'Stage' },
                ...['calls', 'input_tokens', 'output_tokens', 'cache_read_tokens', 'cache_creation_tokens', 'reasoning_tokens', 'transport_retries', 'schema_repairs', 'schema_failures', 'errors'].map(
                  (k) => ({
                    key: k,
                    header: k.replace(/_/g, ' '),
                    num: true,
                    render: (u: CostData['usage'][number]) => fmtInt(Number(u[k] ?? 0)),
                  }),
                ),
              ]}
            />
          </Card>
        </div>
      )}
    </>
  )
}
