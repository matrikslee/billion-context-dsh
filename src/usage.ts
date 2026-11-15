/**
 * Provider-anchored usage — what the provider actually billed, and what a
 * compression costs to undo.
 *
 * Every compress is a durable surface replace (design decision 1), so the next
 * request re-reads everything AFTER the rewrite point: the cheaper rewrite point
 * is the later one, and the expensive one is the oldest range. Nothing in the
 * engine could say that before, because the billing lived in the host projection
 * this module reads — `sessionProjections.snapshot(session).values.tokenUsage`,
 * the same read face src/nudge.ts and src/window.ts already use for
 * contextPressure. No new service, no kernel change, no persisted member (so rule
 * 15 is untouched); the per-compress baseline lives in a WeakMap and simply
 * disappears when the session object is dropped.
 * @module billion-context-dsh/usage
 */

import type { Agent } from '@deepseek-ai/dsh-agent'
import type { Session, SessionEvent } from '@deepseek-ai/dsh-session'

/** Cumulative per-session billing, in the host token-meter's own buckets. */
export interface ProviderUsage {
  readonly uncachedInputTokens: number
  readonly cacheReadTokens: number
  readonly cacheWriteTokens: number
  readonly outputTokens: number
}

/** The minimal projection read face (mirrors src/window.ts ProjectionProbe). */
type UsageProbe = { snapshot?: (session: unknown) => { values?: { tokenUsage?: Partial<ProviderUsage> } } } | undefined

/**
 * Cumulative provider usage, or undefined when the host has no tokenUsage unit
 * (a minimal host without dsh-token-meter). Never throws.
 */
export function providerUsage(agent: Agent): ProviderUsage | undefined {
  const projections = agent.ctx?.get?.('sessionProjections') as UsageProbe
  const raw = projections?.snapshot?.(agent.session)?.values?.tokenUsage
  if (raw === undefined || typeof raw.uncachedInputTokens !== 'number') return undefined
  return {
    uncachedInputTokens: raw.uncachedInputTokens ?? 0,
    cacheReadTokens: raw.cacheReadTokens ?? 0,
    cacheWriteTokens: raw.cacheWriteTokens ?? 0,
    outputTokens: raw.outputTokens ?? 0,
  }
}

/** Total request input billed so far (the total-length caliber). */
export function totalRequestInput(usage: ProviderUsage): number {
  return usage.uncachedInputTokens + usage.cacheReadTokens + usage.cacheWriteTokens
}

/** Snapshot-identity cache, same pattern as the block ledger (rule 14). */
const peakCache = new WeakMap<readonly unknown[], { len: number; peak: number }>()


/**
 * Peak single-request input, scanned from the append-only log: the projection
 * keeps only the newest request, so there is no historical max to read.
 * Cached per snapshot identity (a frozen array while unchanged, fresh after an
 * append), so the scan runs once per snapshot instead of per status call.
 */
export function peakRequestInput(session: Session): number {
  const events = session.snapshotEvents() as readonly SessionEvent[]
  const asKey = events as readonly unknown[]
  const cached = peakCache.get(asKey)
  if (cached !== undefined && cached.len === events.length) return cached.peak
  let peak = 0
  for (const event of events) {
    if (event.type !== 'assistant/message' && event.type !== 'assistant/attempt') continue
    const usage = (event.data as { usage?: { inputTokens?: number; cacheReadTokens?: number; cacheWriteTokens?: number } }).usage
    if (usage === undefined) continue
    const total = (usage.inputTokens ?? 0) + (usage.cacheReadTokens ?? 0) + (usage.cacheWriteTokens ?? 0)
    if (total > peak) peak = total
  }
  peakCache.set(asKey, { len: events.length, peak })
  return peak
}

/**
 * The NEWEST request’s input tokens — the live working set the next request
 * re-prefills from. Read from the append-only log (the projection only keeps
 * cumulative totals), so it needs no tokenMeter.
 *
 * This is the paper’s PEAK-vs-total distinction turned into an actionable
 * number (arXiv 2611.05306 §2.4): a fold re-prefills everything the prompt
 * carries AFTER the rewrite point, so live + room is what decides whether a fold
 * pays at all.
 */
export function lastRequestInput(session: Session): number {
  return lastRequestUsage(session)?.tokens ?? 0
}

/**
 * The newest billed row AND the surface seq it was appended at. A calibration pair is only
 * sound when the billed request happened AFTER the anchor was set: after a restart the log
 * still carries the pre-restart billed number while the anchor is the post-restart surface,
 * and pairing them forges a tokenizer ratio the engine never observed (round-12 review Q2).
 */
export function lastRequestUsage(session: Session): { tokens: number; seq: number } | undefined {
  const events = session.snapshotEvents() as readonly SessionEvent[]
  for (let index = events.length - 1; index >= 0; index -= 1) {
    const event = events[index]!
    if (event.type !== 'assistant/message' && event.type !== 'assistant/attempt') continue
    const usage = (event.data as { usage?: { inputTokens?: number; cacheReadTokens?: number; cacheWriteTokens?: number } }).usage
    if (usage === undefined) continue
    return {
      tokens: (usage.inputTokens ?? 0) + (usage.cacheReadTokens ?? 0) + (usage.cacheWriteTokens ?? 0),
      seq: event.seq,
    }
  }
  return undefined
}

/**
 * A PAIRED calibration sample: what the provider billed for a request and what
 * this plugin estimated for the SAME surface at the same moment. The emergency
 * fit check needs this pairing rather than a ratio of two different moments
 * (cross-review Q3): 'billed(now) / surface(now)' mixes the fixed host overhead
 * into the tokenizer ratio, which inflates it on a small surface (a false refusal
 * that lasts forever) while hiding the overhead entirely on the live-after side
 * (a false acceptance that pays the re-prefill and still overflows).
 *
 * In-memory only: the durable event shapes are frozen (rule 15), so a restarted
 * session has no sample and the caller falls back to a conservative constant.
 */
interface RequestPair { readonly billed: number; readonly surface: number }

const requestPairs = new WeakMap<Session, RequestPair>()

/**
 * Ceiling for the tokenizer ratio. It guards a catastrophic estimator mismatch,
 * NOT a heavy step: what a step carries beyond the surface is `requestOverhead`,
 * added ADDITIVELY (cross-review Q3), so clamping the ratio cannot hide a large
 * read — a 3x pair still shows up whole in the overhead term. 2.0 leaves headroom
 * for a surface our 1-char/token rule underprices, and the fit check LOGS the
 * value it used so a bad sample is visible instead of silent.
 */
export const MAX_TOKENIZER_INFLATION = 2

/** Floor: even with no measurable overhead the engine tokenizer is not cheaper than ours. */
export const MIN_TOKENIZER_INFLATION = 1.05

/**
 * What the emergency fit check assumes with no paired sample (a restarted session
 * keeps nothing in memory): conservative, because a false acceptance is the worse
 * failure.
 */
export const DEFAULT_OVERFLOW_INFLATION = 1.15

/**
 * The surface estimate taken at the PREVIOUS `agent/pre-step`, i.e. the moment
 * BEFORE the request whose usage row is the newest one on the log.
 *
 * Round-10 review S2: pairing the newest billed request with the surface of the
 * CURRENT step compares two different moments — the surface already carries the
 * previous step's assistant text and tool results, so `billed` is a snapshot of
 * a smaller prompt. Both derived quantities then move the WRONG way for the fit
 * check: the ratio drifts toward 1 and `billed - surface` shrinks (or clamps to
 * zero), which under-quotes a fold that still overflows — the expensive failure.
 * Carrying the previous step's surface makes the pair describe ONE request.
 */
const surfaceAtPrevStep = new WeakMap<Session, { tokens: number; seq: number }>()

/** The (billed, anchor-seq) pair already recorded. A request that never produced a usage row
 *  (abort, rate limit, provider rejection) leaves the newest billed row unchanged, so pairing
 *  that stale number with THIS step's anchor compares a request against a surface it never
 *  sent (round-11 Q4) — a repeat records nothing. But a genuinely new request whose total
 *  happens to equal the previous one IS a fresh sample, so the guard requires BOTH components
 *  to repeat, not just the billed number (round-12 Q2). */
const lastPaired = new WeakMap<Session, { billed: number; anchorSeq: number }>()

/**
 * The newest seq the anchor can be dated by. `surface.nodes` is NOT monotonic: a
 * `surfaceOp: replace` splices its checkpoint/tombstone seq IN PLACE (region.ts documents
 * this), so the tail node can carry a seq LOWER than the log's newest event. And the newest
 * billed row can be an `assistant/attempt`, which is never a surface node at all. Taking the
 * surface tail alone let a pre-fold `attempt` (seq 105) look NEWER than an anchor set by a fold
 * that rewrote the head (surface tail still 104), pairing the two again — exactly the forgery
 * this dating exists to stop (round-13 review Q1/Q4, both halves).
 */
function lastRecordedSeq(session: Session): number {
  const nodes = session.surface.nodes
  const surfaceTail = nodes.length > 0 ? nodes[nodes.length - 1]! : 0
  const events = session.snapshotEvents() as readonly SessionEvent[]
  const logTail = events.length > 0 ? events[events.length - 1]!.seq : 0
  return Math.max(surfaceTail, logTail)
}

/**
 * Record a calibration pair from a pre-step: the newest billed request against
 * the surface measured at the step that preceded it, then store this step's
 * surface as the anchor for the next request. The first call has no anchor and
 * records nothing, so a restarted session keeps the conservative fallback. A
 * billed row older than the anchor is never paired (see `lastRequestUsage`).
 */
export function calibrateRequestPair(session: Session, billed: number, billedSeq: number, surfaceNow: number): void {
  const previous = surfaceAtPrevStep.get(session)
  if (previous !== undefined && billed > 0 && billedSeq > previous.seq) {
    const seen = lastPaired.get(session)
    if (seen === undefined || seen.billed !== billed || seen.anchorSeq !== previous.seq) {
      recordRequestPair(session, billed, previous.tokens)
      lastPaired.set(session, { billed, anchorSeq: previous.seq })
    }
  }
  if (surfaceNow > 0) surfaceAtPrevStep.set(session, { tokens: surfaceNow, seq: lastRecordedSeq(session) })
}

/**
 * Point the anchor at a surface the engine itself just rewrote. The overflow path folds the
 * surface and RETRIES the same request, so the billed number of that retry belongs to the
 * FOLDED surface, while the carried anchor still holds the pre-fold one — pairing them reads a
 * tokenizer inflation of ~2× on an engine that changed nothing (round-11 Q4).
 */
export function setCalibrationAnchor(session: Session, surface: number): void {
  if (surface > 0) surfaceAtPrevStep.set(session, { tokens: surface, seq: lastRecordedSeq(session) })
}
/**
 * The route the calibration samples belong to. A provider/model switch changes BOTH derived
 * quantities at once — the tokenizer ratio and the fixed request overhead were measured on the
 * PREVIOUS route — so a stale pair can read a huge `requestOverhead` on a smaller-window route
 * and trip `requestOverhead >= window.limit`, stopping emergency recovery after one pass
 * (round-14 review Q8a, rank 1). A route change drops the samples: the fallback
 * (`DEFAULT_OVERFLOW_INFLATION`, overhead 0) is the conservative reading, and one completed turn
 * on the new route pairs honestly again. First call records the route without dropping anything.
 */
const calibrationRoute = new WeakMap<Session, string>()

export function setCalibrationRoute(session: Session, routeKey: string): void {
  const previous = calibrationRoute.get(session)
  if (previous === routeKey) return
  calibrationRoute.set(session, routeKey)
  if (previous === undefined) return
  requestPairs.delete(session)
  surfaceAtPrevStep.delete(session)
  lastPaired.delete(session)
}

/** Record the newest billed request against the surface estimate of that same moment. */
export function recordRequestPair(session: Session, billed: number, surface: number): void {
  if (billed <= 0 || surface <= 0) return
  requestPairs.set(session, { billed, surface })
}

/** billed / surface for the paired sample, clamped; undefined when there is no sample. */
export function pairedInflation(session: Session): number | undefined {
  const pair = requestPairs.get(session)
  if (pair === undefined) return undefined
  return Math.min(MAX_TOKENIZER_INFLATION, Math.max(MIN_TOKENIZER_INFLATION, pair.billed / pair.surface))
}

/**
 * What the provider billed beyond our surface estimate — the fixed per-request
 * overhead (system prompt, tool registry, host rows). It stays live after a fold,
 * so a post-fold projection must carry it, not fold it away.
 */
export function requestOverhead(session: Session): number {
  const pair = requestPairs.get(session)
  if (pair === undefined) return 0
  return Math.max(0, pair.billed - pair.surface)
}

interface CompressBaseline { readonly uncached: number; readonly count: number }

const compressBaselines = new WeakMap<Session, CompressBaseline>()

/** Remember the billed uncached total at the moment of a successful compress. */
export function recordCompressCost(session: Session, usage: ProviderUsage | undefined): void {
  if (usage === undefined) return
  const previous = compressBaselines.get(session)
  compressBaselines.set(session, { uncached: usage.uncachedInputTokens, count: (previous?.count ?? 0) + 1 })
}

/**
 * Uncached input billed since the last recorded compress — the price the model
 * just paid for the invalidated prefix. null when no compress is recorded yet
 * (a fresh session) or the host reports no usage, so the nudge stays untouched.
 */
export function costSinceLastCompress(
  session: Session,
  usage: ProviderUsage | undefined,
): { delta: number; count: number } | null {
  const baseline = compressBaselines.get(session)
  if (baseline === undefined || usage === undefined) return null
  return { delta: Math.max(0, usage.uncachedInputTokens - baseline.uncached), count: baseline.count }
}
