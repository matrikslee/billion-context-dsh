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
import type { Agent } from '@deepseek-ai/dsh-agent';
import type { Session } from '@deepseek-ai/dsh-session';
/** Cumulative per-session billing, in the host token-meter's own buckets. */
export interface ProviderUsage {
    readonly uncachedInputTokens: number;
    readonly cacheReadTokens: number;
    readonly cacheWriteTokens: number;
    readonly outputTokens: number;
}
/**
 * Cumulative provider usage, or undefined when the host has no tokenUsage unit
 * (a minimal host without dsh-token-meter). Never throws.
 */
export declare function providerUsage(agent: Agent): ProviderUsage | undefined;
/** Total request input billed so far (the total-length caliber). */
export declare function totalRequestInput(usage: ProviderUsage): number;
/**
 * Peak single-request input, scanned from the append-only log: the projection
 * keeps only the newest request, so there is no historical max to read.
 * Cached per snapshot identity (a frozen array while unchanged, fresh after an
 * append), so the scan runs once per snapshot instead of per status call.
 */
export declare function peakRequestInput(session: Session): number;
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
export declare function lastRequestInput(session: Session): number;
/**
 * The newest billed row AND the surface seq it was appended at. A calibration pair is only
 * sound when the billed request happened AFTER the anchor was set: after a restart the log
 * still carries the pre-restart billed number while the anchor is the post-restart surface,
 * and pairing them forges a tokenizer ratio the engine never observed (round-12 review Q2).
 */
export declare function lastRequestUsage(session: Session): {
    tokens: number;
    seq: number;
} | undefined;
/**
 * Ceiling for the tokenizer ratio. It guards a catastrophic estimator mismatch,
 * NOT a heavy step: what a step carries beyond the surface is `requestOverhead`,
 * added ADDITIVELY (cross-review Q3), so clamping the ratio cannot hide a large
 * read — a 3x pair still shows up whole in the overhead term. 2.0 leaves headroom
 * for a surface our 1-char/token rule underprices, and the fit check LOGS the
 * value it used so a bad sample is visible instead of silent.
 */
export declare const MAX_TOKENIZER_INFLATION = 2;
/** Floor: even with no measurable overhead the engine tokenizer is not cheaper than ours. */
export declare const MIN_TOKENIZER_INFLATION = 1.05;
/**
 * What the emergency fit check assumes with no paired sample (a restarted session
 * keeps nothing in memory): conservative, because a false acceptance is the worse
 * failure.
 */
export declare const DEFAULT_OVERFLOW_INFLATION = 1.15;
/**
 * Record a calibration pair from a pre-step: the newest billed request against
 * the surface measured at the step that preceded it, then store this step's
 * surface as the anchor for the next request. The first call has no anchor and
 * records nothing, so a restarted session keeps the conservative fallback. A
 * billed row older than the anchor is never paired (see `lastRequestUsage`).
 */
export declare function calibrateRequestPair(session: Session, billed: number, billedSeq: number, surfaceNow: number): void;
/**
 * Point the anchor at a surface the engine itself just rewrote. The overflow path folds the
 * surface and RETRIES the same request, so the billed number of that retry belongs to the
 * FOLDED surface, while the carried anchor still holds the pre-fold one — pairing them reads a
 * tokenizer inflation of ~2× on an engine that changed nothing (round-11 Q4).
 */
export declare function setCalibrationAnchor(session: Session, surface: number): void;
export declare function setCalibrationRoute(session: Session, routeKey: string): void;
/** Record the newest billed request against the surface estimate of that same moment. */
export declare function recordRequestPair(session: Session, billed: number, surface: number): void;
/** billed / surface for the paired sample, clamped; undefined when there is no sample. */
export declare function pairedInflation(session: Session): number | undefined;
/**
 * What the provider billed beyond our surface estimate — the fixed per-request
 * overhead (system prompt, tool registry, host rows). It stays live after a fold,
 * so a post-fold projection must carry it, not fold it away.
 */
export declare function requestOverhead(session: Session): number;
/** Remember the billed uncached total at the moment of a successful compress. */
export declare function recordCompressCost(session: Session, usage: ProviderUsage | undefined): void;
/**
 * Uncached input billed since the last recorded compress — the price the model
 * just paid for the invalidated prefix. null when no compress is recorded yet
 * (a fresh session) or the host reports no usage, so the nudge stays untouched.
 */
export declare function costSinceLastCompress(session: Session, usage: ProviderUsage | undefined): {
    delta: number;
    count: number;
} | null;
