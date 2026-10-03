// Formatting and public vocabulary. Status and review-reason codes arrive
// from the API already free of model identity; these are their display labels.

export const fmtInt = (n: number | null | undefined) =>
  n === null || n === undefined ? 'n/a' : n.toLocaleString('en-IN')

export const fmtPct = (n: number | null | undefined, digits = 1) =>
  n === null || n === undefined ? 'n/a' : `${n.toFixed(digits)}%`

export const fmtUsd = (n: number | null | undefined, digits = 4) =>
  n === null || n === undefined ? 'not priced' : `$${n.toFixed(digits)}`

/** `damaged_defective` -> `Damaged defective` */
export const humanize = (s: string | null | undefined) => {
  if (!s) return '—'
  const t = s.replace(/_/g, ' ')
  return t.charAt(0).toUpperCase() + t.slice(1)
}

export const STATUS_LABEL: Record<string, string> = {
  AUTO_ACCEPTED: 'Auto-accepted',
  ACCEPTED_AFTER_CHECK: 'Accepted after second check',
  NEEDS_REVIEW: 'Needs review',
  RESOLVED_BY_REVIEWER: 'Resolved by you',
  PROCESSING_ERROR: 'Processing error',
  INVALID_INPUT: 'Invalid input',
}

export const REVIEW_REASON: Record<string, { label: string; help: string }> = {
  AMBIGUOUS_TEXT: { label: 'Ambiguous text', help: 'The comment itself was flagged as ambiguous or contradictory.' },
  DISAGREEMENT: { label: 'Conflicting readings', help: 'The first pass and the second check read it differently.' },
  LOW_CONFIDENCE: { label: 'Low confidence', help: 'Confidence stayed below the accept threshold.' },
  NO_USEFUL_TEXT: { label: 'No useful text', help: 'Nothing to classify, so no classifier call was made.' },
  SCHEMA_FAILURE: { label: 'Unreadable response', help: 'The response would not parse twice in a row.' },
  UNSUPPORTED_OUTPUT: { label: 'Unsupported output', help: 'The reason pair returned is not in the taxonomy.' },
}

export const reviewLabel = (code: string | null | undefined) =>
  code ? (REVIEW_REASON[code]?.label ?? humanize(code)) : '—'


// --- plain language ----------------------------------------------------
// The team reading this screen does not know what `primary_reason` or
// `sub_reason` mean, and should not have to. These are the words a
// merchandiser would use out loud. Anything missing falls back to humanize()
// rather than showing a raw code.

export const REASON_LABEL: Record<string, string> = {
  fit: 'fit and sizing',
  quality_issue: 'fabric and quality',
  appearance_issue: 'colour and print',
  wrong_item: 'the wrong item arriving',
  damaged_defective: 'damage and faults',
  changed_mind: 'customers changing their minds',
  delivery_issue: 'delivery',
  product_not_as_expected: 'the product not matching expectations',
  unclear: 'comments too vague to sort',
}

export const DETAIL_LABEL: Record<string, string> = {
  too_small: 'the item running small',
  too_large: 'the item running large',
  too_long: 'the item being too long',
  too_short: 'the item being too short',
  unspecified_fit: 'fit, without saying how',
  fabric_quality: 'the fabric feeling poor',
  material_mismatch: 'the material not matching the listing',
  comfort: 'the item being uncomfortable to wear',
  stitching_quality: 'weak stitching',
  other_quality: 'quality, unspecified',
  colour_mismatch: 'the colour being different',
  shade_mismatch: 'the shade being off',
  print_mismatch: 'the print not matching the photo',
  other_appearance: 'appearance, unspecified',
  hole: 'holes',
  tear: 'tears',
  zip: 'broken zips',
  button: 'missing buttons',
  stitching: 'seams coming apart',
  general_defect: 'faults',
  transparency: 'the fabric being see-through',
  style_mismatch: 'the style being different',
  shrinkage_concern: 'worries about shrinking',
  occasion_unsuitable: 'not suiting the occasion',
  late_delivery: 'late delivery',
  package_damage: 'damaged packaging',
  wrong_product_or_variant: 'the wrong product being sent',
  preference_or_no_longer_needed: 'simply not wanting it',
  insufficient_information: 'nothing specific',
}

export const reasonLabel = (code: string | null | undefined) =>
  code ? (REASON_LABEL[code] ?? humanize(code).toLowerCase()) : 'something'

export const detailLabel = (code: string | null | undefined) =>
  code ? (DETAIL_LABEL[code] ?? humanize(code).toLowerCase()) : null

/** `sku` -> `style`, because nobody outside ops says SKU out loud. */
export const DIMENSION_NOUN: Record<string, string> = {
  sku: 'style',
  vendor: 'supplier',
  category: 'category',
}

/** "about an hour", "about 25 minutes" — a claim on someone's day, not a count. */
export const fmtMinutes = (m: number | null | undefined) => {
  if (!m) return 'no time at all'
  if (m < 50) return `about ${Math.round(m / 5) * 5} minutes`
  const h = m / 60
  if (h < 1.25) return 'about an hour'
  if (h < 8) return `about ${h % 1 < 0.25 || h % 1 > 0.75 ? Math.round(h) : `${Math.floor(h)}.5`} hours`
  return `${Math.round(h / 8)} working days`
}

/** A machine timestamp as a person would say it.
 *
 *  The API returns `2026-10-02T18:56:12+0530`. Nobody reads that; worse, it
 *  makes the screen look like a log file. Recent runs get a relative time
 *  because "20 minutes ago" is what you actually want to know about a run.
 */
export const fmtWhen = (iso: string | null | undefined): string => {
  if (!iso) return 'at an unknown time'
  const then = new Date(iso)
  if (Number.isNaN(then.getTime())) return iso
  const mins = Math.round((Date.now() - then.getTime()) / 60000)
  if (mins < 1) return 'just now'
  if (mins < 60) return `${mins} minute${mins === 1 ? '' : 's'} ago`
  const hours = Math.round(mins / 60)
  if (hours < 24) return `${hours} hour${hours === 1 ? '' : 's'} ago`
  if (hours < 48) return 'yesterday'
  return then.toLocaleString('en-IN', {
    day: 'numeric',
    month: 'short',
    hour: 'numeric',
    minute: '2-digit',
    hour12: true,
  })
}
