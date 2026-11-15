/**
 * M3 — the four model tools: compress / decompress / search_context /
 * acp_status, registered through `ctx.tools` (defineTool).
 *
 * compress is the heart of ACP: the model writes the summary and the tool
 * lands it as a durable surface replacement (no second LLM summarization
 * call). decompress recovers shadowed content read-only from the log (DSH
 * keeps the originals — V5). search_context scores blocks rebuilt from the
 * log. acp_status reports the block ledger and pressure.
 * @module billion-context-dsh/tools
 */
import { type ToolDefinition } from '@deepseek-ai/dsh-tools';
import { type CompressionCore, type SearchDoc } from 'acp-kernel';
import type { Agent } from '@deepseek-ai/dsh-agent';
import type { Session } from '@deepseek-ai/dsh-session';
import type { AcpStateStore } from './state.ts';
import { type KernelConfigInput } from './config.ts';
import { type AcpWindow } from './window.ts';
import { type SeqCompressibleRange, type FoldPricing } from './region.ts';
import { type ResolvedPrompts } from './prompts.ts';
import { type ResolvedCallStyle } from './ptc.ts';
import type { SettingsCommandSurface } from './settings.ts';
import type { PresetName } from './presets.ts';
export interface ToolEnvironment extends KernelConfigInput {
    readonly kernel: CompressionCore;
    readonly store: AcpStateStore;
    /** Display-only: the named preset that produced the nudge thresholds above, if any (`/acp-prune status` names it). Never read by the kernel path. */
    readonly preset?: PresetName;
    /** Resolve the effective context window for an agent (optional: status falls back to modelContextLimit). */
    readonly windowFor?: (agent: Agent) => Promise<AcpWindow>;
    /** Resolved prompt templates (optional: falls back to DEFAULT_RESOLVED). */
    readonly prompts?: ResolvedPrompts;
    /** How the model can REACH these tools in the current step (src/ptc.ts). The
     *  engine swaps in the PTC variant of `prompts` together with this, so the two
     *  never disagree; a stand-in env (tests, plain compositions) defaults to
     *  `native` and keeps the direct-call copy byte-identical. */
    readonly toolCallStyle?: ResolvedCallStyle;
    /**
     * Call ids of compress invocations that created a durable block. The engine
     * listens for the matching `tool/result` and hides the call/result pair from
     * the surface, preventing the compaction summary from sitting between them
     * (strict providers reject that sequence with HTTP 400).
     */
    readonly compressCallIdsToHide?: Set<string>;
    /**
     * Read/write access to the runtime settings layer for `/acp-prune config`.
     * Absent surfaces (never expected — the engine always builds one) would
     * degrade the command to advice text.
     */
    readonly settingsCommand?: SettingsCommandSurface;
}
/**
 * Resolve the effective context window for a tool or command run: probe the
 * agent's real window via `windowFor` when provided, otherwise fall back to
 * the environment's `modelContextLimit`. Shared by the compress and
 * acp_status tool handlers and the `/acp-prune` command so the resolution logic
 * lives in exactly one place (issue #63 — the tools used the 128K fallback
 * for pressure decisions even when auto-detection had found a larger window).
 */
export declare function resolveEffectiveWindow(env: ToolEnvironment, agent: Agent): Promise<AcpWindow>;
export declare const compressParameters: {
    readonly arguments: {
        readonly type: 'json';
        readonly description: 'Tolerated wrapped-arguments form (model-generated); unwrapped in handleCompress. Prefer passing content directly.';
    };
    readonly topic: {
        readonly type: 'string';
        readonly description: 'Fallback topic for entries without their own.';
    };
    readonly content: {
        readonly type: 'array';
        readonly description: 'One or more ranges to compress, each with startSeq/endSeq boundaries (surface seqs) and a dense summary. Required — pass it directly, not wrapped in an arguments key.';
        readonly items: {
            readonly type: 'object';
            readonly properties: {
                readonly startSeq: {
                    readonly required: true;
                    readonly oneOf: readonly [{
                        readonly type: 'integer';
                        readonly description: 'First surface seq of the range.';
                    }, {
                        readonly type: 'string';
                        readonly description: 'Seq as text; a trailing #callId fragment is ignored. A drilldown mN ref ("m00306") or a kernel block id ("b1", as acp_status lists) is also accepted — a block id resolves to that block\'s checkpoint seq, which makes the fold a tier 2/3 distillation.';
                    }];
                };
                readonly endSeq: {
                    readonly required: true;
                    readonly oneOf: readonly [{
                        readonly type: 'integer';
                        readonly description: 'Inclusive last surface seq of the range.';
                    }, {
                        readonly type: 'string';
                        readonly description: 'Seq as text; a trailing #callId fragment is ignored. Also accepts a drilldown mN ref or a kernel block id ("b3") — for a multi-block condense name the LAST block id (tier 3).';
                    }];
                };
                readonly summary: {
                    readonly type: 'string';
                    readonly required: true;
                    readonly description: 'Complete technical summary replacing the range; keep paths, decisions, values verbatim. Minimum 50 characters.';
                };
                readonly topic: {
                    readonly type: 'string';
                    readonly description: 'Short label (3-5 words) for this range.';
                };
                readonly verifiedReadings: {
                    readonly type: 'array';
                    readonly items: {
                        readonly type: 'string';
                    };
                    readonly description: 'Optional: acceptance readings that are already green before this compression (e.g. "t0-fastpath 8/8", "closedloop 414/414"). Stored structurally on the compaction/summary event and recovered by verifiedReadingsOf, so later steps need not re-run the checks.';
                };
            };
            readonly additionalProperties: false;
        };
    };
    readonly force: {
        readonly type: 'boolean';
        readonly description: 'Override the working-set floor guard on BOTH paths — an explicit content range whose fold leaves less than foldFloorPct of the window visible, and a condense election. Legitimate only for imminent overflow: a shallow working set is where open-weight accuracy collapses (arXiv 2611.05306 Appendix D).';
    };
    readonly condense: {
        readonly type: 'object';
        readonly description: 'Engine-picked fold: the engine elects the LARGEST eligible continuous run of the CURRENT surface (walls: the newest injected instruction copy of every scope, the newest skill catalog, checkpoint carriers, the system node, the active user turn) and folds it in ONE durable replace — you write the handoff note. Mutually exclusive with content. The note must carry "## Goal" (what the stretch was doing and what remains) and "## Next" (the immediate next command plus the completion condition): a note without them turns one fold into a re-run of the work it replaced. Price: the next request re-prefills every token AFTER the rewrite point ONCE, so the rewrite point IS the lever — a fold near the tail barely re-reads anything, one near the head re-reads almost everything. Prefer the LATEST consumed stretch, fold in bulk, and do not fold at all when nothing consumed is worth reclaiming.';
        readonly properties: {
            readonly summary: {
                readonly type: 'string';
                readonly required: true;
                readonly description: 'The handoff note replacing the elected stretch. Minimum 50 characters; must carry ## Goal and ## Next. Best shape (arXiv 2611.05306 Appendix G): ## Goal, ## Established facts, ## Applied changes, ## Verification (what you ran, what passed, what failed and the workaround), ## Dead ends (tried and why they failed — so nobody retries them), ## Remaining plan with the immediate next command AND the completion condition, carried verbatim. Only Goal/Next are enforced; the rest is checked and named as a warning.';
            };
            readonly topic: {
                readonly type: 'string';
                readonly description: 'Short label (3-5 words) for the folded block.';
            };
            readonly force: {
                readonly type: 'boolean';
                readonly description: 'Fold even though it leaves less than the working-set floor visible (foldFloorPct of the window). Only for imminent overflow: a shallow working set is where open-weight models lose accuracy.';
            };
            readonly verifiedReadings: {
                readonly type: 'array';
                readonly items: {
                    readonly type: 'string';
                };
                readonly description: 'Optional: acceptance readings already green before this fold (e.g. "t0-fastpath 8/8").';
            };
        };
        readonly additionalProperties: false;
    };
};
/** Exported so the ref-width contract test can pin it without a 100K-message fixture. */
export declare function mnRefIndex(value: string): number | null;
/**
 * Pure gate helpers for the compress tool's CURRENT-instruction-row rejection.
 *
 * Decision history (issue #71 review): the first draft only WARNED when a
 * manual compress range swallowed a current injected row (F7), because the
 * compression is safe and self-healing. The owner reversed that during PR1
 * review: compressing a CURRENT row has NO legitimate outcome — the host
 * re-injects the newest AGENTS.md copy unconditionally the moment it leaves
 * the surface (presence gate, deepseek-harness
 * packages/context/agent-instructions/src/index.ts:137/:163), so the tokens
 * come straight back and the call is pure waste — and a hard reject keeps the
 * manual path consistent with the system-side GC's iron rule (PR2: never
 * clear a group's newest row). STALE copies stay compressible: removing them
 * while the newest stays visible is the actual cleanup and triggers no
 * re-injection. Skill-catalog rows joined the guard set for a DIFFERENT
 * reason (issue #185): they are not self-healing at all — the host resends
 * the catalog only when its digest changes, so a folded catalog stays gone
 * until the catalog actually changes. The range table
 * (buildCompressibleSeqRanges) never offers these rows, so the gate only fires
 * on hand-built ranges.
 *
 * `guardedRowsInSpan` is the overlap probe. It takes the POSITIONAL span the
 * transaction will actually shadow (`shadowedSeqsOf`), never a numeric
 * `start <= seq <= end` interval: the surface is locally non-monotonic after
 * earlier replacements (a checkpoint seq spliced ahead of older residual
 * nodes), so a tier-2 distill of two checkpoints can carry a CURRENT
 * instruction row numerically inside its edges while the sliced span excludes
 * it — the interval probe rejected exactly the call the nudge hands the model
 * (issue #71 review B1). Probing the slice also keeps guard and effect in
 * agreement: `shadowedSeqsOf` is what the transaction prices and
 * `assertProvenance` verifies.
 * `protectedRowRejectionNote` renders the rejection the model sees: it names
 * the offending seqs AND the compressible slices left in the span, so the model
 * can re-cut (or split into two calls) instead of retrying the same call.
 * `guardedSurfaceSeqsOf` supplies the protected set.
 */
export declare function guardedRowsInSpan(guarded: ReadonlySet<number>, shadowed: readonly number[]): number[];
/**
 * Advisory note for a plain range that would fold a LIVE block's checkpoint
 * carrier (see `liveCheckpointCarriersInSpan`). Names the block ids the model
 * can use instead — a block-ref boundary is the call that legitimately folds
 * carriers (tier 2/3 distillation).
 */
export declare function liveCarrierRejectionNote(start: number, end: number, carriers: readonly {
    seq: number;
    kernelBlockId: string;
}[]): string;
export declare function protectedRowRejectionNote(start: number, end: number, hits: readonly number[], shadowed: readonly number[], session?: Session): string;
/**
 * Kernel ref for one RESOLVED range edge's surface node (issue #155).
 * Resolved edges are tool-pairing-balanced and anchorable, but NO LONGER
 * guaranteed to carry a bare-`${seq}` ref: a multi-tool-call assistant
 * projects to `${seq}#${callId}` sub-ids, and an empty tool result projects
 * to nothing. Three tiers, in order:
 *
 *  1. the bare-seq id (user turns, single-call assistants, text-bearing results);
 *  2. the node's sub-ids in projection (= content) order — FIRST for a start
 *     edge (the kernel must consume the whole node from its first sub-message),
 *     LAST for an end edge (unreachable in practice: a node with open calls
 *     makes every cut after it unbalanced, so it can never be a resolved end
 *     edge);
 *  3. the nearest message-bearing LIVE node between this edge and the opposite
 *     edge (inclusive), walked along the SURFACE — never raw-id space, where
 *     shadowed nodes keep their refs and would anchor onto already-compressed
 *     messages. Walking inward only also guarantees the returned ref names a
 *     message the transaction actually shadows; a span whose interior carries
 *     no message at all (two adjacent empty results) yields undefined and the
 *     caller raises the existing "no assigned ref" error.
 */
export declare function edgeRefForSeq(session: Session, byRaw: Readonly<Record<string, string>>, seq: number, role: 'start' | 'end', oppositeSeq: number): string | undefined;
/** One continuous run a condense may fold, priced with the fold-cost column. */
export interface CondensePick {
    readonly start: number;
    readonly end: number;
    readonly count: number;
    readonly tokens: number;
    readonly foldCostTokens: number;
    /** Tokens that stay visible after this fold. */
    readonly liveAfterFoldTokens: number;
    /** The elected run would leave less than the working-set floor visible. */
    readonly belowFloor: boolean;
    /**
     * Whether ANY eligible run (not just the elected one) clears the floor. A
     * refusal telling the model to fold WIDER when nothing wider is eligible is a
     * loop generator: every larger span is guarded, so the model gets the
     * guarded-row refusal back with no third move (cross-review Q3).
     */
    readonly aboveFloorAvailable: boolean;
    /**
     * True when the PAYBACK gate stepped aside because the session is already close to
     * the window: refusing a fold there does not save the session — the provider rejects
     * the next request outright, and a costly fold beats a hard failure.
     */
    readonly pressureRelief: boolean;
    /**
     * The window limit behind `pressureRelief` (0 when no pressure was supplied). The write-time
     * gate needs it: relief must buy REAL headroom, and what counts as real is window-relative
     * (round-13 review Q2 — `netSaving > 0` let a 1-token reclaim spend a permanent barrier and a
     * full re-prefill at ≥85% of the window).
     */
    readonly reliefLimitTokens: number;
    /**
     * The largest note (stored form, frame prefix included) that still pays back inside
     * `CONDENSE_PAYBACK_REQUESTS` for the elected span; `Infinity` under pressure relief, where the
     * horizon stepped aside. The write-time refusal quotes it so a model that over-wrote its note
     * shortens it to a number that works instead of guessing (round-14 review Q2).
     */
    readonly noteBudgetTokens: number;
    /**
     * How many surface rows the guards refuse (current instruction copies, the
     * newest skill catalog, checkpoint carriers, the active user turn). A session
     * that is wide but walled is NOT stuck: telling it to fold wider sends it into
     * the guard, and telling it to stop compressing hides the fact that the
     * walls shrink on their own as the session continues (cross-review Q4).
     */
    readonly guardedRowCount: number;
    /** How many eligible ranges the geometry produced at all (0 = nothing clears the kernel minimum). */
    readonly eligibleRangeCount: number;
    /**
     * Set on the synthesized content item of a DEGRADED election. The explicit-range
     * floor guard reads it instead of a mutable flag, so the two guards share one
     * decision (cross-review Q2: an ambient bypass is how the elected fold got
     * re-refused by its own guard).
     */
    readonly degraded: boolean;
}
export interface CondenseMiss {
    readonly reason: string;
    readonly candidates: SeqCompressibleRange[];
}
/**
 * The largest CONTINUOUS eligible run of the current surface, for condense.
 *
 * Geometry is the range table’s own (buildCompressibleSeqRanges, rule 3), so a
 * condense can never shadow a row the durable surface forbids: the CURRENT
 * injected copy of every scope, the newest visible skill catalog, checkpoint
 * carriers, the system node and the active user turn are the walls. Superseded
 * injection copies are NOT walls (foldStaleInstructionRows) — the host re-sends a
 * group’s newest row only, so an older AGENTS.md or catalog row folded away stays
 * gone, and that residue is most of what a long session carries.
 *
 * Ranked by NET saving (`tokens − foldCostTokens`), not by volume. Volume was the
 * old rule and it is backwards on a durable surface: compression summary nodes are
 * appended AFTER the rows they replace, so nearly every older row sits in front of a
 * large resident summary block and re-prefills it. Measured 2026-11-15: the volume
 * election removed 23,588 tokens and the very next request re-prefilled 155,927 —
 * about 65 seconds on this engine for a 23.6K-per-request saving.
 *
 * The saving IS recurring while the re-prefill is paid ONCE, so a slow payback is still a
 * payback: a row with `foldCost >= tokens` is NOT hard-refused (round-10 review B2), it is an
 * investment that turns positive after foldCost / (tokens − note) further requests. The engine
 * cannot know how many requests are left, so the election REFUSES only when
 * `foldCost > tokens * CONDENSE_PAYBACK_REQUESTS` — beyond the horizon — printing that
 * break-even count and leaving the call to the model through an explicit `content`
 * range (rule 3: the engine never folds by itself). The one genuinely unrecoverable case,
 * `note >= tokens`, is refused at write time by condenseNotePayback below.
 */
export declare function pickCondenseSpan(session: Session, turn: ReturnType<CompressionCore['processTurn']>, preserveRecent: number, foldPricing?: FoldPricing,
/** live billed input and window limit, for the pressure-relief bypass. */
pressure?: {
    readonly liveTokens: number;
    readonly limitTokens: number;
}): CondensePick | CondenseMiss;
/**
 * Whether a below-floor condense may still fold. The ONLY compliant reason is
 * that NO eligible row could keep the floor: while one exists, condense must fold
 * that row instead — degrading when a compliant row is available throws away the
 * working set for nothing (round-4 Q1: the barrier-count gate could fire while a
 * 20K row above the floor was still sitting there eligible).
 *
 * The barrier COUNT is not the gate, because it never measured what it claimed:
 * a fold leaves a summary node (a few hundred tokens of note), not a floor-sized
 * hole, so `n folds × floor` is not a debt the window carries. Fragmentation is
 * observable directly in the row set, which needs no new measurement.
 */
export declare function condenseDegradable(pick: CondensePick, minimum: number): boolean;
/**
 * Refusal copy for a below-floor condense. The THREE branches ARE the point
 * (cross-review Q4): two-branch copy conflated "nothing wider is above the floor"
 * with "nothing is eligible at all", so a session whose reclaimable area is merely
 * FRAGMENTED by the guards was told to stop compressing — and a session with
 * nothing to fold was told to widen into a guard. Both are loops. Exported so
 * every branch is testable without building three fixtures.
 */
export declare function condenseFloorRefusal(pick: CondensePick, floor: number): string;
export declare function missingCondenseHeadings(note: string): string[];
export declare function condenseNoteGaps(note: string): string[];
/**
 * Does the note actually COMPRESS? Three cross-reviews killed the earlier
 * post-fold duplication warning: it fires AFTER the fold lands, so obeying it
 * means rewriting the note and paying a second full re-prefill — costlier to act
 * on than to ignore — and lexical shingles flag exactly the verbatim identifiers,
 * paths and error strings a good note SHOULD carry.
 *
 * The replacement is deterministic, pre-fold and language-independent: compare
 * the note’s weight against the stretch it replaces. A note weighing a third of
 * the folded span is not a briefing, it is a second copy that stays resident in
 * every later request — the fold never earns its one-time re-prefill back.
 */
export declare const MAX_NOTE_WEIGHT_RATIO = 0.35;
/**
 * What a handoff note actually weighs on this host, measured 2026-11-15: a real model-written
 * `condense` note stored ~915 tokens while missing FOUR of the six Appendix-G fields. That is the
 * number the span floor has to clear, not a guess: a floor derived from a note the model will
 * never write lets a span into the election the weight gate then refuses (round-15 review Q7.1:
 * at `MAX_NOTE_WEIGHT_RATIO` a 2000-token span admits only ~700 stored tokens, so every span
 * between 2000 and ~2614 tokens was elected, paid for by a turn of writing, and then refused).
 */
export declare const TYPICAL_HANDOFF_NOTE_TOKENS = 900;
/**
 * Smallest stretch condense will elect: the smallest span that can carry a note this host's
 * models actually write inside the weight cap — `TYPICAL_HANDOFF_NOTE_TOKENS / MAX_NOTE_WEIGHT_RATIO`,
 * so a copy or ratio change follows it instead of drifting. The kernel minimum (`minCompressRange`)
 * is a CHARACTER threshold (5000 chars ≈ 1250 tokens), which filters nothing; the real floor is
 * the note's weight (round-13 review Q5: a 750-token election, then the weight gate).
 */
export declare const MIN_CONDENSE_SPAN_TOKENS: number;
/**
 * The smallest handoff note (stored form, frame prefix included) that can still brief a resumed
 * session: the Appendix-G headings plus one line each. The election uses it as the floor of the
 * NOTE BUDGET it solves for, so a span that cannot carry a real briefing inside the payback
 * horizon is refused before the model writes anything (round-14 review Q2/Q5).
 */
export declare const MIN_HANDOFF_NOTE_TOKENS = 250;
/**
 * How much net headroom a relief fold must buy, as a fraction of the window. Relief exists to
 * avoid a provider rejection, so a fold that reclaims a few hundred tokens while spending a
 * permanent barrier and a full re-prefill does not relieve anything — it just re-arms the same
 * cliff one step later (round-13 review Q2: cliff thrashing). Window-relative for the same
 * reason the emergency minimum is: a hard constant silently disables relief on a small window.
 */
export declare const RELIEF_MIN_NET_SAVING_RATIO = 0.02;
/**
 * The net reclaim relief still demands (0 outside relief). Exported so the arithmetic is
 * testable without engineering a session whose pressure, suffix and note all line up.
 */
/**
 * Cap on the relief floor, deliberately NOT `MIN_DEGRADED_FOLD_TOKENS`. That constant is a GROSS
 * span-size floor (`pick.tokens >= minimum`), while `reliefMinSavingTokens` gates a NET reclaim
 * (`span − note`); sharing one value meant a net-reclaim floor of 4096 forced a ~6.3K-token span
 * whenever the model wrote a 35% note, and any change to one silently moved the other
 * (round-14 review Q3).
 */
export declare const RELIEF_MIN_NET_SAVING_MAX_TOKENS = 2048;
export declare function reliefMinSavingTokens(pressureRelief: boolean, limitTokens: number): number;
/** How many further requests a condense must pay back inside to be worth electing. */
export declare const CONDENSE_PAYBACK_REQUESTS = 4;
/**
 * The largest note (stored form, frame prefix included) that still pays back inside the horizon:
 * `foldCost + note <= N × (tokens − note)`  ⇒  `note <= (N × tokens − foldCost) / (N + 1)`.
 * Exported so the arithmetic is testable without engineering a fixture whose span, suffix and
 * note all line up (round-14 review Q2 replaced the worst-case-note price with this).
 */
export declare function condenseNoteBudget(spanTokens: number, foldCostTokens: number): number;
/**
 * Whether a relief span can reach the relief floor at all — with the smallest note this tool
 * accepts (the bare summary frame prefix). Exported for the same reason (round-14 review Q3).
 */
export declare function reliefSpanReachable(spanTokens: number, limitTokens: number): boolean;
/**
 * Pressure at which the payback gate steps aside (live billed input ÷ window).
 * Below it a poor-payback fold is a choice; at or above it, NOT folding is the
 * riskier option because the next request can be rejected by the provider.
 */
export declare const CONDENSE_PRESSURE_RELIEF_RATIO = 0.85;
/**
 * Is this condense worth running once the note is counted? The folded volume is the saving
 * side, but the note stays resident on EVERY later request, so the honest saving is
 * `span − note`, and the one-time cost is the re-prefill. Exported so the arithmetic is
 * testable without engineering a fixture whose stretch, suffix and note all line up.
 */
export declare function condenseNotePayback(spanTokens: number, foldCostTokens: number, noteTokens: number): {
    readonly worth: boolean;
    readonly payback: number;
    readonly netSaving: number;
};
/**
 * Note weight, priced as the note is STORED. Two gates read this number — the weight cap here
 * and the payback re-pricing in `handleCompress` — and they must not disagree: measuring the
 * raw note here while pricing the stored node there let a note just under 35% raw exceed 35%
 * once the frame prefix (and the degrade stamp) landed (round-13 review Q1).
 * `storedTokens` lets a caller that already counted the stored form pass it in.
 */
export declare function noteWeightRatio(note: string, spanTokens: number, storedTokens?: number): number;
export declare function noteWeightRejection(note: string, spanTokens: number, storedTokens?: number, includesDegradeLine?: boolean): string;
/**
 * Build the unified SearchDoc[] from the log: one block doc per ledger entry
 * (ref = compactionId, so `decompress({ blockId })` closes the loop) plus one
 * message doc per shadowed ORIGINAL (expanded through distilled parents; each
 * seq is claimed by the earliest/innermost block that covered it, mirroring
 * pi's owner map — decompress on that block recovers the original).
 * Cached per log snapshot (see searchDocsCache). Exported for the issue #133
 * regression tests (not part of the public API — index.ts re-exports only).
 */
export declare function buildSearchDocs(session: Session): SearchDoc[];
/** Build the four ACP model tools bound to one engine. */
export declare function makeTools(env: ToolEnvironment): ToolDefinition[];
