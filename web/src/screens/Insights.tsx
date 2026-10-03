import { useState } from 'react'
import { api, useApi, type ReturnRow, type Run, type Taxonomy } from '../api'
import { BarChart, Callout, Card, ErrorNote, Icon, Loading, PageHead, RunBanner, Select, Table } from '../ui'
import { fmtPct, humanize } from '../format'

interface Reasons {
  run: Run
  reasons: { primary_reason: string; rows: number; pct: number }[]
  sub_reasons: { sub_reason: string; rows: number }[]
  body_areas: { body_area: string; rows: number }[]
}

interface Group {
  group: string
  classified_rows: number
  matching_rows: number
  share_pct: number
  baseline_pct: number
  lift: number | null
  share_ci_low_pct: number
  share_ci_high_pct: number
}

interface GroupDetail {
  sub_reasons: { sub_reason: string; rows: number }[]
  body_areas: { body_area: string; rows: number }[]
  rows: ReturnRow[]
}

const GROUP_BY = ['sku', 'vendor', 'category']
const GROUP_LABEL: Record<string, string> = { sku: 'SKU', vendor: 'Vendor', category: 'Category' }
const GROUP_PLURAL: Record<string, string> = { sku: 'SKUs', vendor: 'vendors', category: 'categories' }

type Verdict = 'clear' | 'maybe' | 'normal' | 'none'

/** The judgment the table used to leave to the reader. "Clear" only when even
 *  the low end of the likely range sits above the average, so a handful of
 *  returns cannot produce a red badge on their own. */
function verdict(g: Group): Verdict {
  if (g.lift === null || g.baseline_pct === 0) return 'none'
  if (g.share_ci_low_pct > g.baseline_pct) return 'clear'
  if (g.share_pct > g.baseline_pct) return 'maybe'
  return 'normal'
}

const VERDICT: Record<Verdict, { label: string; tone: string }> = {
  clear: { label: 'Clear problem', tone: 'danger' },
  maybe: { label: 'Too few to tell', tone: 'warn' },
  normal: { label: 'Normal', tone: 'ok' },
  none: { label: 'No average', tone: '' },
}

const pct0 = (n: number) => `${Math.round(n)}%`
const RANGE_HELP =
  'With only a few returns, a percentage is a rough guess. The true figure is very likely (95 times in 100) somewhere in this range. Fewer returns, wider range.'

export default function Insights() {
  const [primary, setPrimary] = useState<string | null>(null)
  const reasons = useApi<Reasons>(`/api/insights/reasons${primary ? `?primary=${encodeURIComponent(primary)}` : ''}`)
  const taxonomy = useApi<Taxonomy>('/api/taxonomy')

  const [groupBy, setGroupBy] = useState('sku')
  const [target, setTarget] = useState('fit')
  const [minRows, setMinRows] = useState(5)
  const reasonWord = humanize(target).toLowerCase()
  const groups = useApi<{ group_by: string; reason: string; groups: Group[] }>(
    `/api/insights/groups?${new URLSearchParams({ group_by: groupBy, reason: target, min_rows: String(minRows) })}`,
  )

  const [picked, setValue] = useState<string | null>(null)

  /** Picking a reason anywhere on this screen moves everything that depends on
   *  it. Before this, clicking "Quality issue" narrowed the two breakdown
   *  panels but left the table below still analysing fit, so the page showed
   *  two different answers to the same question. */
  const selectReason = (key: string) => {
    setPrimary((p) => (p === key ? null : key))
    setTarget(key)
    setValue(null) // the inspected group belongs to the old reason
  }

  // While a new grouping loads, the previous response is still held; only use
  // it when it answers the current question.
  const fresh = groups.data?.group_by === groupBy && groups.data?.reason === target
  const list = fresh ? groups.data!.groups : []
  // The inspected group follows the table: keep the pick while it is listed,
  // otherwise fall back to the top row.
  const value = list.some((g) => g.group === picked) ? picked : (list[0]?.group ?? null)

  const detail = useApi<GroupDetail>(
    value !== null
      ? `/api/insights/group?${new URLSearchParams({ group_by: groupBy, value, reason: target })}`
      : null,
  )

  const r = reasons.data
  const top = list[0]

  return (
    <>
      <PageHead title="Dig deeper" lede="What the unlabelled returns were really about, which styles and suppliers it concentrates in, and the customer comments behind every number.">
        <RunBanner run={r?.run} />
      </PageHead>
      <ErrorNote error={reasons.error} />
      {reasons.loading && !r && <Loading />}
      {r && (
        <div className="stack">
          <Card
            title="What 'Other' actually was"
            actions={
              primary ? (
                <button className="btn btn-sm" onClick={() => setPrimary(null)}>
                  <Icon name="x" size={14} />
                  Clear filter: {humanize(primary)}
                </button>
              ) : undefined
            }
          >
            <div className="stack-sm">
              {/* The hint retires once they have used it: an instruction nobody
                  needs any more is just noise on every later visit. */}
              {!primary && (
                <p className="hint">
                  <span className="icon-lead">
                    <Icon name="chevron" size={14} />
                  </span>
                  <span>
                    <strong>These bars are buttons.</strong> Click one to break it down below and point the whole page
                    at that reason, tables included.
                  </span>
                </p>
              )}
              <BarChart
                label="What was wrong"
                selected={primary}
                onSelect={selectReason}
                data={r.reasons.map((x) => ({
                  key: x.primary_reason,
                  label: humanize(x.primary_reason),
                  value: x.rows,
                  display: `${x.rows} · ${fmtPct(x.pct)}`,
                }))}
              />
            </div>
          </Card>

          <p className="scope">
            {primary ? (
              <>
                <span>Breaking down</span>
                <span className="chip">
                  {humanize(primary)}
                  <button type="button" className="chip-x" onClick={() => setPrimary(null)} title="Show all reasons again">
                    <Icon name="x" size={13} label="Clear this reason" />
                  </button>
                </span>
                <span>— the two panels below cover only these returns.</span>
              </>
            ) : (
              <span>
                Showing <strong>all reasons</strong> together. Click a bar above to narrow the two panels below to one
                reason.
              </span>
            )}
          </p>

          <div className="grid grid-2">
            <Card title={primary ? `What exactly, within ${humanize(primary)}` : 'What exactly, across all reasons'}>
              <BarChart
                label="More detail"
                data={r.sub_reasons.map((x) => ({ key: x.sub_reason, label: humanize(x.sub_reason), value: x.rows }))}
              />
              {!r.sub_reasons.length && <p className="caption">No further detail was recorded for this selection.</p>}
            </Card>
            <Card title={primary ? `Body areas within ${humanize(primary)}` : 'Body areas, fit returns'}>
              <BarChart
                label="Body areas"
                data={r.body_areas.map((x) => ({ key: x.body_area, label: humanize(x.body_area), value: x.rows }))}
              />
              {!r.body_areas.length && (
                <p className="caption">
                  No stated body areas here. The classifier is instructed never to infer one the customer did not mention.
                </p>
              )}
            </Card>
          </div>

          <Card
            title={`Which ${GROUP_PLURAL[groupBy]} get more ${reasonWord} complaints than usual?`}
            sub={`Each row compares one ${GROUP_LABEL[groupBy]}'s returns with the average across all ${GROUP_PLURAL[groupBy]}. Based on returns only, not sales, so it can't tell you how often something is returned.`}
          >
            <div className="stack-sm">
              <div className="grid grid-3">
                <Select label="Group by" value={groupBy} options={GROUP_BY} onChange={setGroupBy} format={(v) => GROUP_LABEL[v]} />
                <Select
                  label="Reason"
                  value={target}
                  options={taxonomy.data?.primary_reasons ?? [target]}
                  onChange={(v) => {
                    setTarget(v)
                    setPrimary(v)
                    setValue(null)
                  }}
                />
                <label className="field">
                  Hide groups with fewer returns than
                  <input
                    type="number"
                    min={1}
                    max={100}
                    value={minRows}
                    onChange={(e) => setMinRows(Math.max(1, Math.min(100, Number(e.target.value) || 1)))}
                  />
                </label>
              </div>
              <p className="hint">
                <span className="icon-lead">
                  <Icon name="chevron" size={15} />
                </span>
                <span>
                  <strong>Click any row</strong> to see what exactly was wrong, where on the body, and the customer
                  comments behind the number.
                </span>
              </p>
              {groups.loading && !fresh ? (
                <Loading height={120} />
              ) : (
                <div className="wrap-head">
                <Table
                  rows={list}
                  empty={`No ${GROUP_LABEL[groupBy]} has enough returns to show. Lower the minimum above.`}
                  caption={`${GROUP_PLURAL[groupBy]} compared with the average for ${reasonWord}`}
                  scroll
                  onRowClick={(g) => setValue(g.group)}
                  isActive={(g) => g.group === value}
                  columns={[
                    { key: 'group', header: GROUP_LABEL[groupBy], render: (g) => <span style={{ whiteSpace: 'nowrap' }}>{g.group}</span> },
                    {
                      key: 'verdict',
                      header: 'Verdict',
                      render: (g) => {
                        const v = VERDICT[verdict(g)]
                        return <span className={`badge ${v.tone}`}>{v.label}</span>
                      },
                    },
                    { key: 'matching_rows', header: `${humanize(target)} returns`, num: true },
                    { key: 'classified_rows', header: 'All returns', num: true },
                    { key: 'share', header: `% about ${reasonWord}`, num: true, render: (g) => pct0(g.share_pct) },
                    {
                      key: 'range',
                      header: (
                        <span title={RANGE_HELP} style={{ display: 'inline-flex', gap: 4, alignItems: 'center' }}>
                          Likely range <Icon name="info" size={13} label="What is the likely range?" />
                        </span>
                      ),
                      num: true,
                      render: (g) => `${Math.round(g.share_ci_low_pct)}–${pct0(g.share_ci_high_pct)}`,
                    },
                    {
                      key: 'baseline',
                      header: <span title={`Average across all ${GROUP_PLURAL[groupBy]}`}>Average</span>,
                      num: true,
                      render: (g) => pct0(g.baseline_pct),
                    },
                    {
                      key: 'lift',
                      header: 'vs average',
                      num: true,
                      render: (g) => (g.lift === null ? '—' : `${g.lift.toFixed(1)}×`),
                    },
                    {
                      key: 'open',
                      header: <span className="sr-only">Look closer</span>,
                      render: (g) => (
                        <button type="button" className="btn btn-sm" onClick={() => setValue(g.group)}>
                          {g.group === value ? 'Showing' : 'Look closer'}
                          <Icon name="chevron" size={14} />
                        </button>
                      ),
                    },
                  ]}
                />
                </div>
              )}
              <p className="caption">
                <strong>Likely range:</strong> with only a few returns, a percentage is a rough guess, so this is where the
                true figure very likely sits. <strong>Clear problem</strong> means even the low end of that range is above
                the average. <strong>Too few to tell</strong> means it is above average, but the range dips below it, so
                it could be chance. <strong>vs average</strong>: 1.6× means 1.6 times the average share.
              </p>
              {top && <Summary top={top} reasonWord={reasonWord} plural={GROUP_PLURAL[groupBy]} single={GROUP_LABEL[groupBy]} />}
            </div>
          </Card>

          {value !== null && (
            <Card
              title={`Inside ${value}`}
              sub={`Everything below is just this ${GROUP_LABEL[groupBy] === 'SKU' ? 'SKU' : GROUP_LABEL[groupBy].toLowerCase()}. Pick another from the table above or the menu here.`}
              actions={
                <div style={{ minWidth: 220 }}>
                  <Select
                    label={GROUP_LABEL[groupBy]}
                    value={value}
                    options={list.map((g) => g.group)}
                    onChange={setValue}
                    format={(v) => v}
                  />
                </div>
              }
            >
              {detail.loading && !detail.data ? (
                <Loading />
              ) : (
                detail.data && (
                  <div className="stack">
                    <div className="grid grid-2">
                      <div className="stack-sm">
                        <h3>
                          What exactly, for {reasonWord} in {value}
                        </h3>
                        <Table
                          rows={detail.data.sub_reasons}
                          empty="No further detail recorded."
                          columns={[
                            { key: 'sub_reason', header: 'What exactly', render: (x) => humanize(x.sub_reason) },
                            { key: 'rows', header: 'Rows', num: true },
                          ]}
                        />
                      </div>
                      <div className="stack-sm">
                        <h3>Body areas in {value} (fit rows)</h3>
                        <Table
                          rows={detail.data.body_areas}
                          empty="No stated body areas in this group."
                          columns={[
                            { key: 'body_area', header: 'Body area', render: (x) => humanize(x.body_area) },
                            { key: 'rows', header: 'Rows', num: true },
                          ]}
                        />
                      </div>
                    </div>
                    <p className="caption">
                      A sizing fix becomes actionable here: a SKU whose fit complaints concentrate in one body area is a
                      pattern, not a pile of anecdotes. Body area is recorded only where the customer stated it.
                    </p>
                    <div className="stack-sm">
                      <div className="row" style={{ justifyContent: 'space-between' }}>
                        <h3>The comments behind it</h3>
                        {detail.data.rows.length > 0 && (
                          <a
                            className="btn"
                            href={api.exportUrl({ group_by: groupBy, value, reason: target })}
                            download
                          >
                            <Icon name="download" size={16} />
                            Download these rows (CSV)
                          </a>
                        )}
                      </div>
                      <Table
                        rows={detail.data.rows}
                        scroll
                        empty="Nothing matches that combination."
                        caption="Customer comments"
                        columns={[
                          { key: 'return_id', header: 'Return' },
                          { key: 'original_text', header: 'Customer comment', wide: true },
                          { key: 'sub_reason', header: 'What exactly', render: (x) => humanize(x.sub_reason) },
                          { key: 'body_area', header: 'Body area', render: (x) => humanize(x.body_area) },
                          { key: 'size_ordered', header: 'Size' },
                          {
                            key: 'evidence_text',
                            header: 'Evidence (verbatim)',
                            render: (x) => (x.evidence_text ? <q>{x.evidence_text}</q> : '—'),
                          },
                        ]}
                      />
                    </div>
                  </div>
                )
              )}
            </Card>
          )}

          <div>
            <a className="btn" href={api.exportUrl()} download>
              <Icon name="download" size={16} />
              Download all classifications (CSV)
            </a>
          </div>
        </div>
      )}
    </>
  )
}

/** One plain sentence about the top row, matching its verdict. */
function Summary({ top, reasonWord, plural, single }: { top: Group; reasonWord: string; plural: string; single: string }) {
  const v = verdict(top)
  const facts = (
    <>
      {top.matching_rows} of its {top.classified_rows} returns ({pct0(top.share_pct)}) are about {reasonWord}, against{' '}
      {pct0(top.baseline_pct)} for {plural} on average
      {top.lift !== null && <>: {top.lift.toFixed(1)}× the average</>}.
    </>
  )
  if (v === 'clear')
    return (
      <Callout tone="danger">
        <strong>
          {top.group} has a clear {reasonWord} problem.
        </strong>{' '}
        {facts}
      </Callout>
    )
  if (v === 'maybe')
    return (
      <Callout tone="warning">
        <strong>
          {top.group} gets the most {reasonWord} complaints, but has too few returns to be sure.
        </strong>{' '}
        {facts}
      </Callout>
    )
  return (
    <Callout tone="success">
      No {single === 'SKU' ? single : single.toLowerCase()} gets noticeably more {reasonWord} complaints than average.
    </Callout>
  )
}
