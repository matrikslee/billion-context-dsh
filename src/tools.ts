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

import { defineTool, ToolArgsError, type ToolDefinition, type ToolRunContext } from '@deepseek-ai/dsh-tools'
import { buildStatusReport, defaultCountTokens, searchBlocks, type CompressionCore, type MessageRole, type SearchDoc } from 'acp-kernel'
import type { Agent } from '@deepseek-ai/dsh-agent'
import type { Session, SessionEvent } from '@deepseek-ai/dsh-session'
import type { AcpStateStore } from './state.ts'
import { foldPricingFor, kernelConfigFor, MIN_DEGRADED_FOLD_TOKENS, type KernelConfigInput } from './config.ts'
import { resolveTokenCount } from './nudge.ts'
import { routeFor, type AcpWindow } from './window.ts'
import {
  AlreadyCompressedRangeError,
  buildCompressibleSeqRanges,
  blockIdOfKernelRef,
  blockRefForSummarySeq,
  blockRegistry,
  compactionIdsOfKernelBlocks,
  expandShadowedSeqs,
  guardedSurfaceSeqsOf,
  type KernelRangeView,
  type SeqCompressibleRange,
  isExactSurfaceSpan,
  rebuildBlockLedger,
  resolveSurfaceRange,
  runCompactionTransaction,
  liveCheckpointCarriersInSpan,
  shadowedSeqsOf,
  stripOrphanedSurfaceToolMessages,
  summarySeqOfKernelBlock,
  openToolCallIds,
  sliceDecompressPage,
  surfaceSummary,
  DEFAULT_DECOMPRESS_PAGE,
  DEFAULT_DECOMPRESS_PAGE_CHARS,
  foldSpanPricing,
  surfaceTokens,
  type FoldPricing,
  type ResolvedSurfaceRange,
} from './region.ts'
import { allLogMessages, attachmentsOfEvent, buildToolCallIndex, eventsToCoreMessages, extractEventText, isAgentInstructionsRow, isCheckpointNode, isRealUserTurn, isSkillCatalogRow, surfaceEventsOf, toolCallsOf, withSummaryFramePrefix } from './messages.ts'
import { shadowedTokensViaMeter } from './host-tokens.ts'
import { DEFAULT_OVERFLOW_INFLATION, lastRequestInput, pairedInflation, peakRequestInput, providerUsage, recordCompressCost, requestOverhead, totalRequestInput } from './usage.ts'
import { eventAtOf, sessionEventsOf } from './session-events.ts'
import { DEFAULT_RESOLVED, type ResolvedPrompts } from './prompts.ts'
import { callHint, type ResolvedCallStyle } from './ptc.ts'
import type { SettingsCommandSurface } from './settings.ts'
import type { PresetName } from './presets.ts'

export interface ToolEnvironment extends KernelConfigInput {
  readonly kernel: CompressionCore
  readonly store: AcpStateStore
  /** Display-only: the named preset that produced the nudge thresholds above, if any (`/acp-prune status` names it). Never read by the kernel path. */
  readonly preset?: PresetName
  /** Resolve the effective context window for an agent (optional: status falls back to modelContextLimit). */
  readonly windowFor?: (agent: Agent) => Promise<AcpWindow>
  /** Resolved prompt templates (optional: falls back to DEFAULT_RESOLVED). */
  readonly prompts?: ResolvedPrompts
  /** How the model can REACH these tools in the current step (src/ptc.ts). The
   *  engine swaps in the PTC variant of `prompts` together with this, so the two
   *  never disagree; a stand-in env (tests, plain compositions) defaults to
   *  `native` and keeps the direct-call copy byte-identical. */
  readonly toolCallStyle?: ResolvedCallStyle
  /**
   * Call ids of compress invocations that created a durable block. The engine
   * listens for the matching `tool/result` and hides the call/result pair from
   * the surface, preventing the compaction summary from sitting between them
   * (strict providers reject that sequence with HTTP 400).
   */
  readonly compressCallIdsToHide?: Set<string>
  /**
   * Read/write access to the runtime settings layer for `/acp-prune config`.
   * Absent surfaces (never expected — the engine always builds one) would
   * degrade the command to advice text.
   */
  readonly settingsCommand?: SettingsCommandSurface
}

interface TextOutput {
  text: string
}

function textOutput(): {
  schema: { type: 'object'; properties: { text: { type: 'string' } }; additionalProperties: boolean }
  render: (args: unknown, value: TextOutput) => import('@deepseek-ai/dsh-llm').ContentBlock[]
} {
  return {
    schema: {
      type: 'object',
      properties: { text: { type: 'string' } },
      additionalProperties: false,
    },
    render: (_args, value) => [{ type: 'text', text: value.text }],
  }
}

function requireAgent(exec: ToolRunContext): Agent {
  if (exec.agent === undefined) {
    throw new Error('billion-context-dsh: tool requires an agent execution context')
  }
  return exec.agent
}

/**
 * Resolve the effective context window for a tool or command run: probe the
 * agent's real window via `windowFor` when provided, otherwise fall back to
 * the environment's `modelContextLimit`. Shared by the compress and
 * acp_status tool handlers and the `/acp-prune` command so the resolution logic
 * lives in exactly one place (issue #63 — the tools used the 128K fallback
 * for pressure decisions even when auto-detection had found a larger window).
 */
export async function resolveEffectiveWindow(env: ToolEnvironment, agent: Agent): Promise<AcpWindow> {
  return env.windowFor === undefined
    ? { limit: env.modelContextLimit, source: 'explicit' as const }
    : await env.windowFor(agent)
}

export const compressParameters = {
  // Tolerated wrapped-arguments form: some models emit
  // `{ "arguments": "{\"content\": [...]}" }` (double-nested) or
  // `{ "arguments": { "content": [...] } }` instead of the unwrapped
  // `{ "content": [...] }`. The old DSH validator surfaced this as
  // `invalid arguments: "arguments" must be an object` and the model retried
  // forever. `arguments` is accepted as an optional JSON node so the wrapped
  // shape passes schema validation; `handleCompress` unwraps it and falls back
  // to a clear runtime error when neither form carries content. `content` is
  // intentionally NOT `required: true` — a required property would reject the
  // wrapped shape before `handleCompress` can see it. The tool description
  // still tells the model content is mandatory.
  //
  // The items fields are the opposite case: startSeq/endSeq/summary MUST be
  // `required: true`. Without that, a model call that omits `summary` (only
  // startSeq/endSeq/topic present) passed schema validation and failed late
  // inside the kernel with "Summary is empty" — and live sessions showed the
  // model retrying the identical broken call in a loop. With the fields
  // required, the same call is rejected at the schema gate with
  // `missing required property "content[0].summary"`, which tells the model
  // exactly which field to add (same pattern as decompress's required
  // blockId / search_context's required query).
  arguments: { type: 'json', description: 'Tolerated wrapped-arguments form (model-generated); unwrapped in handleCompress. Prefer passing content directly.' },
  topic: { type: 'string' as const, description: 'Fallback topic for entries without their own.' },
  content: {
    type: 'array' as const,
    description: 'One or more ranges to compress, each with startSeq/endSeq boundaries (surface seqs) and a dense summary. Required — pass it directly, not wrapped in an arguments key.',
    items: {
      type: 'object' as const,
      properties: {
        startSeq: {
          required: true,
          oneOf: [
            { type: 'integer' as const, description: 'First surface seq of the range.' },
            { type: 'string' as const, description: 'Seq as text; a trailing #callId fragment is ignored. A drilldown mN ref ("m00306") or a kernel block id ("b1", as acp_status lists) is also accepted — a block id resolves to that block\'s checkpoint seq, which makes the fold a tier 2/3 distillation.' },
          ],
        },
        endSeq: {
          required: true,
          oneOf: [
            { type: 'integer' as const, description: 'Inclusive last surface seq of the range.' },
            { type: 'string' as const, description: 'Seq as text; a trailing #callId fragment is ignored. Also accepts a drilldown mN ref or a kernel block id ("b3") — for a multi-block condense name the LAST block id (tier 3).' },
          ],
        },
        summary: { type: 'string' as const, required: true, description: 'Complete technical summary replacing the range; keep paths, decisions, values verbatim. Minimum 50 characters.' },
        topic: { type: 'string' as const, description: 'Short label (3-5 words) for this range.' },
        // B3 (2026-09-08 governance plan): the handler and region.ts have
        // accepted verifiedReadings since the plan landed, but the declared
        // parameter schema did not list it — `additionalProperties: false`
        // then rejected every live call that carried it
        // (`invalid arguments: "content[0].verifiedReadings" is not a declared
        // property`), so the structured-loss-stopping field was unreachable
        // from the model's tool interface. Declared here; additionalProperties
        // stays false so unknown fields are still rejected.
        verifiedReadings: {
          type: 'array' as const,
          items: { type: 'string' as const },
          description: 'Optional: acceptance readings that are already green before this compression (e.g. "t0-fastpath 8/8", "closedloop 414/414"). Stored structurally on the compaction/summary event and recovered by verifiedReadingsOf, so later steps need not re-run the checks.',
        },
      },
      additionalProperties: false,
    },
  },
  force: {
    type: 'boolean' as const,
    description: 'Override the working-set floor guard on BOTH paths — an explicit content range whose fold leaves less than foldFloorPct of the window visible, and a condense election. Legitimate only for imminent overflow: a shallow working set is where open-weight accuracy collapses (arXiv 2611.05306 Appendix D).',
  },
  condense: {
    type: 'object' as const,
    description: 'Engine-picked fold: the engine elects the LARGEST eligible continuous run of the CURRENT surface (walls: the newest injected instruction copy of every scope, the newest skill catalog, checkpoint carriers, the system node, the active user turn) and folds it in ONE durable replace — you write the handoff note. Mutually exclusive with content. The note must carry "## Goal" (what the stretch was doing and what remains) and "## Next" (the immediate next command plus the completion condition): a note without them turns one fold into a re-run of the work it replaced. Price: the next request re-prefills every token AFTER the rewrite point ONCE, so the rewrite point IS the lever — a fold near the tail barely re-reads anything, one near the head re-reads almost everything. Prefer the LATEST consumed stretch, fold in bulk, and do not fold at all when nothing consumed is worth reclaiming.',
    properties: {
      summary: { type: 'string' as const, required: true, description: 'The handoff note replacing the elected stretch. Minimum 50 characters; must carry ## Goal and ## Next. Best shape (arXiv 2611.05306 Appendix G): ## Goal, ## Established facts, ## Applied changes, ## Verification (what you ran, what passed, what failed and the workaround), ## Dead ends (tried and why they failed — so nobody retries them), ## Remaining plan with the immediate next command AND the completion condition, carried verbatim. Only Goal/Next are enforced; the rest is checked and named as a warning.' },
      topic: { type: 'string' as const, description: 'Short label (3-5 words) for the folded block.' },
      force: { type: 'boolean' as const, description: 'Fold even though it leaves less than the working-set floor visible (foldFloorPct of the window). Only for imminent overflow: a shallow working set is where open-weight models lose accuracy.' },
      verifiedReadings: { type: 'array' as const, items: { type: 'string' as const }, description: 'Optional: acceptance readings already green before this fold (e.g. "t0-fastpath 8/8").' },
    },
    additionalProperties: false,
  },
} as const

/** Normalize a seq arg: number, "295", or "295#call_00_xxx" → 295. */
function parseSeq(value: number | string): number {
  const text = String(value).split('#')[0]!.trim()
  const seq = Number(text)
  if (!Number.isInteger(seq) || seq < 0) {
    throw new Error(`billion-context-dsh: invalid seq "${String(value)}" — use a surface seq like 295`)
  }
  return seq
}

/**
 * Match a drilldown mN ref: "m00306" / "m306" (kernel `refToIndex` semantics,
 * `m0*(\d{1,7})`), tolerating a trailing `#callId` fragment (symmetric with
 * `parseSeq`'s `#` handling). Returns the ref index, or null for non-mN input.
 *
 * The width AND the bound mirror the kernel's `REF_PATTERN` / `MAX_INDEX`
 * (0.0.101: `\d{1,7}` and `9999999`; through 0.0.63 both were 5 digits /
 * `99999`). They must be re-checked on every kernel bump (AGENTS.md §4b): a
 * session numbers refs upward and never reuses one, so once it outgrows the
 * old width the kernel keeps handing out `m100000`+ — and a host parser capped
 * at 5 digits rejects acp_status's OWN rows as "not a ref", in exactly the long
 * sessions a ref-width bump exists to rescue. `tests/tools.test.ts` round-trips
 * the kernel's `indexToRef` boundaries, so a future widening turns that test
 * red instead of degrading silently.
 */
const MN_RE = /^m0*(\d{1,7})(?:#.*)?$/i

/** Exported so the ref-width contract test can pin it without a 100K-message fixture. */
export function mnRefIndex(value: string): number | null {
  const match = MN_RE.exec(value.trim())
  if (match === null) return null
  const index = Number(match[1])
  return index >= 1 && index <= 9999999 ? index : null
}

/**
 * Resolve a compress boundary arg to a surface seq. Accepts:
 *  - a bare surface seq (number, "295", "295#call_00_x" — `parseSeq`);
 *  - a drilldown mN ref ("m00306" / "m306") — reverse-mapped via the CURRENT
 *    turn's `messageRefs.byRef` (CoreMessage.id = seq or "seq#callId" → split
 *    on "#"). Unknown mN (never assigned on the current surface) fails with
 *    guidance; a valid mN whose span was already compressed falls through to
 *    the existing recover-stale / already-compressed semantics (rule 7);
 *  - a kernel block ref ("b1") — the id `acp_status` prints per active block and
 *    `decompress` already accepts — resolved to that block's checkpoint seq
 *    (`summarySeqOfKernelBlock`). Naming the checkpoint node is what makes the
 *    fold a tier 2/3 distillation: the resolver then sees the edge ON the
 *    carrier, `blockRefForSummarySeq` hands the kernel the block ref back, and
 *    `liveCheckpointCarriersInSpan`'s guard stands down (that guard exists to
 *    refuse a PLAIN seq range swallowing a live carrier). Before this, the
 *    reject note's own advice — "distill it with the block ids" — led to
 *    `invalid seq "b1"`, i.e. the tool told the model to make a call no parser
 *    accepted. A block id that is not an ACTIVE block (superseded, or never
 *    existed) fails with guidance.
 * `byRef` MUST come from `turn.state.messageRefs` (after `processTurn`), not
 * the persisted store state: acp_status's turn is never persisted, so mN refs
 * shown in a drilldown (including refs for messages that arrived since the
 * last nudge/compress) only exist on the current turn's ref map — a lookup
 * against the stored state would report a false "unknown mN" and dead-loop
 * the model between acp_status and compress.
 */
function parseBoundary(value: number | string, byRef: Record<string, string>, session: Session): number {
  const text = String(value).trim()
  if (/^b\d+$/.test(text)) {
    const seq = summarySeqOfKernelBlock(session, text)
    if (seq === null) {
      throw new Error(
        `billion-context-dsh: block "${text}" is not an active block — run acp_status for the live block ids `
        + '(a block-id boundary distills that block, tier 2/3)',
      )
    }
    return seq
  }
  const index = mnRefIndex(text)
  if (index === null) return parseSeq(value)
  // Normalize to the kernel's padded key ("m00306") — byRef holds exact keys.
  const ref = `m${String(index).padStart(5, '0')}`
  const raw = byRef[ref]
  if (raw === undefined) {
    throw new Error(
      `billion-context-dsh: mN "${text}" not found on the current surface — re-run acp_status for fresh refs (the surface may have moved)`,
    )
  }
  const seq = Number(String(raw).split('#')[0]!)
  if (!Number.isInteger(seq) || seq < 0) {
    throw new Error(
      `billion-context-dsh: mN "${text}" maps to a non-seq id "${raw}" — re-run acp_status`,
    )
  }
  return seq
}

interface CompressArgs {
  /** Tolerated wrapped-arguments form (model-generated double-nesting). */
  arguments?: string | { content?: CompressArgs['content'] }
  topic?: string
  content?: Array<{ startSeq: number | string; endSeq: number | string; summary: string; topic?: string; verifiedReadings?: string[] }>
  /** Engine-picked fold: the engine names the span, the model writes the handoff note. */
  force?: boolean
  condense?: { summary: string; topic?: string; verifiedReadings?: string[]; force?: boolean }
}

/**
 * Unwrap the tolerated wrapped-arguments forms back to the canonical shape:
 * `{ arguments: "{\"content\": [...]}" }` or `{ arguments: { content: [...] } }`
 * → `{ content: [...] }`. The direct `{ content: [...] }` form passes through
 * untouched. Returns null when no form carries content (caller raises).
 */
function unwrapCompressArgs(args: CompressArgs): CompressArgs | null {
  if (args.content !== undefined || args.condense !== undefined) return args
  if (args.arguments === undefined) return null
  let inner: unknown = args.arguments
  if (typeof inner === 'string') {
    try {
      inner = JSON.parse(inner)
    } catch {
      return null
    }
  }
  if (typeof inner !== 'object' || inner === null || Array.isArray(inner)) return null
  const inner2 = inner as { content?: unknown; condense?: unknown }
  if (inner2.content === undefined && inner2.condense === undefined) return null
  return {
    ...args,
    content: inner2.content as CompressArgs['content'],
    condense: inner2.condense as CompressArgs['condense'],
  }
}

/**
 * Peel the tolerated wrapped-arguments envelope `{ arguments: {…} }` that some
 * model channels emit for ANY tool — the same double-nesting that birthed
 * `unwrapCompressArgs` (live-verified on acp_status: a drilldown call arrived
 * as `{"arguments":{"scope":"compressed"}}` and was silently dropped, since
 * only compress unwrapped). The envelope may be an object or a JSON string;
 * inner keys win over outer duplicates. Args without an envelope pass through
 * untouched.
 */
function unwrapEnvelope<T extends object>(args: T): T {
  const envelope = (args as { arguments?: unknown }).arguments
  if (envelope === undefined) return args
  let inner: unknown = envelope
  if (typeof inner === 'string') {
    try {
      inner = JSON.parse(inner)
    } catch {
      return args
    }
  }
  if (typeof inner !== 'object' || inner === null || Array.isArray(inner)) return args
  return { ...args, ...(inner as object) } as T
}

/**
 * Enforce the items-level `required` contract on the EFFECTIVE content, after
 * the wrapped-arguments envelope has been peeled. The DSH schema gate only
 * sees the model's top-level arguments object — when the call arrives wrapped
 * as `{ arguments: { content: [...] } }`, the top-level `content` property is
 * absent there (it lives inside the envelope), so the gate never checks the
 * items and a missing `summary`/`startSeq`/`endSeq` sailed through to the
 * kernel, which fails late with a field-less "Summary is empty" and sent live
 * sessions into a retry loop (the same failure mode the schema gate fix for
 * the direct form closed). Running the SAME check on the unwrapped content
 * closes that window for both forms, and produces the identical
 * `invalid arguments: missing required property "content[0].summary"` surface
 * by reusing the host's `ToolArgsError` instead of a hand-rolled format.
 * An empty/whitespace-only summary counts as missing (the kernel would
 * reject it anyway — fail early with the field name instead).
 */
function validateContentItems(content: NonNullable<CompressArgs['content']>): void {
  const violations: string[] = []
  content.forEach((item, index) => {
    const path = `content[${index}]`
    if (item.startSeq === undefined) violations.push(`missing required property "${path}.startSeq"`)
    if (item.endSeq === undefined) violations.push(`missing required property "${path}.endSeq"`)
    if (typeof item.summary !== 'string' || item.summary.trim().length === 0) {
      violations.push(`missing required property "${path}.summary"`)
    }
  })
  if (violations.length > 0) throw new ToolArgsError(violations)
}

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
export function guardedRowsInSpan(guarded: ReadonlySet<number>, shadowed: readonly number[]): number[] {
  const inSpan = new Set(shadowed)
  return [...guarded].filter((seq) => inSpan.has(seq)).sort((a, b) => a - b)
}

/**
 * Advisory note for a plain range that would fold a LIVE block's checkpoint
 * carrier (see `liveCheckpointCarriersInSpan`). Names the block ids the model
 * can use instead — a block-ref boundary is the call that legitimately folds
 * carriers (tier 2/3 distillation).
 */
export function liveCarrierRejectionNote(
  start: number,
  end: number,
  carriers: readonly { seq: number; kernelBlockId: string }[],
): string {
  const list = carriers.map((carrier) => `seq ${carrier.seq} (${carrier.kernelBlockId})`).join(', ')
  const first = carriers[0]!
  // Name the LAST carrier as the end edge too: a valid multi-carrier span is a
  // tier-3 condense, and suggesting `b1..b1` would silently distill only the
  // first parent.
  const last = carriers[carriers.length - 1]!
  return `  seqs ${start}..${end} rejected — checkpoint ${list} carries the visible summary of a still-active block, and a plain seq range never supersedes one. Distill it with the block ids instead (compress({ content: [{ startSeq: "${first.kernelBlockId}", endSeq: "${last.kernelBlockId}", summary }] })), or cut the span around those seqs.`
}

export function protectedRowRejectionNote(
  start: number,
  end: number,
  hits: readonly number[],
  shadowed: readonly number[],
  session?: Session,
): string {
  const preview = hits.slice(0, 4).join(', ')
  const more = hits.length > 4 ? ` +${hits.length - 4} more` : ''
  const first = shadowed.indexOf(hits[0]!)
  const last = shadowed.indexOf(hits[hits.length - 1]!)
  const before = first > 0 ? shadowed.slice(0, first) : []
  const after = last >= 0 && last < shadowed.length - 1 ? shadowed.slice(last + 1) : []
  const slices = [before, after]
    .filter((slice) => slice.length > 0)
    .map((slice) => `${slice[0]}..${slice[slice.length - 1]}`)
  const recovery = slices.length === 0
    ? 'no part of this span is compressible while those rows are current — pick an OLDER span instead (acp_status lists the live ranges)'
    : `the compressible part of this span is seq ${slices.join(' and ')} — submit them as separate content entries (or two compress calls), each with its own summary`
  // Why the guarded rows may not be folded, per channel. Without a session the
  // hits cannot be classified, so the wording stays the instruction-only text
  // this note has always carried (pinned by tests).
  let hasInstructions = false
  let hasCatalog = false
  let hasUserTurn = false
  if (session !== undefined) {
    for (const seq of hits) {
      const event = eventAtOf(session, seq)
      if (event === undefined) continue
      if (!hasInstructions && isAgentInstructionsRow(event)) hasInstructions = true
      if (!hasCatalog && isSkillCatalogRow(event)) hasCatalog = true
      if (!hasUserTurn && isRealUserTurn(event)) hasUserTurn = true
      if (hasInstructions && hasCatalog && hasUserTurn) break
    }
  }
  const reasons: string[] = []
  if (session === undefined) {
    reasons.push('the host re-injects the newest AGENTS.md copy the moment it leaves the surface, so compressing it reclaims nothing')
  } else if (!hasInstructions && !hasCatalog && !hasUserTurn) {
    // A session that identifies none of the hits must NOT claim an AGENTS.md
    // re-injection it cannot confirm: the rows stay guarded, the mechanism
    // stays unnamed.
    reasons.push('these rows must stay visible on the surface')
  } else {
    // Per-channel WHY (issue #185): the guarded channels fail differently when
    // folded — AGENTS.md comes straight back (host presence gate); a skill
    // catalog never does (resend keyed on the catalog digest, not visibility);
    // and the active user turn would vanish from the surface entirely.
    if (hasInstructions) {
      reasons.push('the host re-injects the newest AGENTS.md copy the moment it leaves the surface, so compressing it reclaims nothing')
    }
    if (hasCatalog) {
      reasons.push('a folded skill catalog is never re-sent — its resend gate is the catalog digest, which folding cannot change')
    }
    if (hasUserTurn) {
      reasons.push('the active user message must stay live to preserve conversation intent')
    }
  }
  // A hit set that is ONLY the active user turn must not be explained with the
  // AGENTS.md re-injection story (or its stale-copy escape): neither applies to
  // a user message. `guardedSurfaceSeqsOf` may also protect the newest real user
  // turn alongside the policy rows, hence the two labels.
  const userOnly = hasUserTurn && !hasInstructions && !hasCatalog
  const rowLabel = hasUserTurn ? 'CURRENT guarded row(s)' : 'CURRENT injected instruction row(s)'
  const staleCopyTail = userOnly ? '' : ' (older/stale copies of the same channel are fine to compress)'
  return `  seqs ${start}..${end} rejected — the span covers ${hits.length} ${rowLabel} (seq ${preview}${more}); ${reasons.join('; ')} — ${recovery}${staleCopyTail}`
}

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
export function edgeRefForSeq(
  session: Session,
  byRaw: Readonly<Record<string, string>>,
  seq: number,
  role: 'start' | 'end',
  oppositeSeq: number,
): string | undefined {
  const nodes = session.surface.nodes
  let index = -1
  let oppositeIndex = -1
  for (let i = 0; i < nodes.length; i += 1) {
    if (nodes[i] === seq) index = i
    if (nodes[i] === oppositeSeq) oppositeIndex = i
  }
  if (index < 0 || oppositeIndex < 0) return undefined
  const direct = anchorRefForNode(session, byRaw, seq, role)
  if (direct !== undefined) return direct
  const step = role === 'start' ? 1 : -1
  // Walk from just past this edge toward the opposite edge, INCLUSIVE of it:
  // the opposite edge's own ref is a legitimate answer when nothing in between
  // carries one (e.g. [empty result, text result]).
  for (let i = index + step; i !== oppositeIndex + step; i += step) {
    const ref = anchorRefForNode(session, byRaw, nodes[i]!, role)
    if (ref !== undefined) return ref
  }
  return undefined
}

/** Bare-seq ref for one node, else its multi-call sub-id ref (see edgeRefForSeq). */
function anchorRefForNode(
  session: Session,
  byRaw: Readonly<Record<string, string>>,
  seq: number,
  role: 'start' | 'end',
): string | undefined {
  const direct = byRaw[String(seq)]
  if (direct !== undefined) return direct
  const event = eventAtOf(session, seq)
  if (event?.type !== 'assistant/message') return undefined
  const content = (event.data as { message?: { content?: unknown } }).message?.content
  // The projection emits one sub-message PER CALL, including calls whose id is
  // absent (keyed `${seq}#`), so the gate must mirror that trigger exactly:
  // fewer than 2 calls means this node projected to a bare seq (tier 1 above).
  const ids = toolCallsOf(content).map((call) => call.id ?? '')
  if (ids.length < 2) return undefined
  // Projection order IS content order (projectEvent maps the calls in sequence).
  const ordered = role === 'start' ? ids : [...ids].reverse()
  for (const id of ordered) {
    const ref = byRaw[`${seq}#${id}`]
    if (ref !== undefined) return ref
  }
  return undefined
}

/** One continuous run a condense may fold, priced with the fold-cost column. */
export interface CondensePick {
  readonly start: number
  readonly end: number
  readonly count: number
  readonly tokens: number
  readonly foldCostTokens: number
  /** Tokens that stay visible after this fold. */
  readonly liveAfterFoldTokens: number
  /** The elected run would leave less than the working-set floor visible. */
  readonly belowFloor: boolean
  /**
   * Whether ANY eligible run (not just the elected one) clears the floor. A
   * refusal telling the model to fold WIDER when nothing wider is eligible is a
   * loop generator: every larger span is guarded, so the model gets the
   * guarded-row refusal back with no third move (cross-review Q3).
   */
  readonly aboveFloorAvailable: boolean
  /**
   * True when the PAYBACK gate stepped aside because the session is already close to
   * the window: refusing a fold there does not save the session — the provider rejects
   * the next request outright, and a costly fold beats a hard failure.
   */
  readonly pressureRelief: boolean
  /**
   * The window limit behind `pressureRelief` (0 when no pressure was supplied). The write-time
   * gate needs it: relief must buy REAL headroom, and what counts as real is window-relative
   * (round-13 review Q2 — `netSaving > 0` let a 1-token reclaim spend a permanent barrier and a
   * full re-prefill at ≥85% of the window).
   */
  readonly reliefLimitTokens: number
  /**
   * The largest note (stored form, frame prefix included) that still pays back inside
   * `CONDENSE_PAYBACK_REQUESTS` for the elected span; `Infinity` under pressure relief, where the
   * horizon stepped aside. The write-time refusal quotes it so a model that over-wrote its note
   * shortens it to a number that works instead of guessing (round-14 review Q2).
   */
  readonly noteBudgetTokens: number
  /**
   * How many surface rows the guards refuse (current instruction copies, the
   * newest skill catalog, checkpoint carriers, the active user turn). A session
   * that is wide but walled is NOT stuck: telling it to fold wider sends it into
   * the guard, and telling it to stop compressing hides the fact that the
   * walls shrink on their own as the session continues (cross-review Q4).
   */
  readonly guardedRowCount: number
  /** How many eligible ranges the geometry produced at all (0 = nothing clears the kernel minimum). */
  readonly eligibleRangeCount: number
  /**
   * Set on the synthesized content item of a DEGRADED election. The explicit-range
   * floor guard reads it instead of a mutable flag, so the two guards share one
   * decision (cross-review Q2: an ambient bypass is how the elected fold got
   * re-refused by its own guard).
   */
  readonly degraded: boolean
}

export interface CondenseMiss {
  readonly reason: string
  readonly candidates: SeqCompressibleRange[]
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
export function pickCondenseSpan(
  session: Session,
  turn: ReturnType<CompressionCore['processTurn']>,
  preserveRecent: number,
  foldPricing?: FoldPricing,
  /** live billed input and window limit, for the pressure-relief bypass. */
  pressure?: { readonly liveTokens: number; readonly limitTokens: number },
): CondensePick | CondenseMiss {
  const byRaw = turn.state.messageRefs.byRaw
  const nodes = session.surface.nodes
  let firstSeq: number | undefined
  let lastSeq: number | undefined
  let refCount = 0
  for (const seq of nodes) {
    if (byRaw[String(seq)] === undefined) continue
    refCount += 1
    firstSeq ??= seq
    lastSeq = seq
  }
  if (firstSeq === undefined || lastSeq === undefined) {
    return {
      reason: 'the kernel holds no ref for any of the ' + nodes.length + ' live surface nodes \u2014 nothing has been projected yet',
      candidates: [],
    }
  }
  const view: KernelRangeView = {
    ranges: [{ startRef: byRaw[String(firstSeq)]!, endRef: byRaw[String(lastSeq)]! }],
    refs: { byRef: turn.state.messageRefs.byRef },
  }
  const ranges = buildCompressibleSeqRanges(session, view, {
    preserveRecent,
    foldStaleInstructionRows: true,
    ...(foldPricing === undefined ? {} : { foldPricing }),
  })
  let best: SeqCompressibleRange | undefined
  let bestNet = Number.NEGATIVE_INFINITY
  for (const range of ranges) {
    const net = range.tokens - range.foldCostTokens
    // Equal net, unequal safety: a run whose tool output the host already wrote to a FILE costs
    // nothing to re-read after the fold, so it wins the tie (round-15 review Q4).
    if (net > bestNet || (net === bestNet && best !== undefined && range.spills > best.spills)) {
      bestNet = net
      best = range
    }
  }
  if (best === undefined) {
    return {
      reason: 'no continuous run clears the kernel minimum range (counted in CHARACTERS, compress.minCompressRange) \u2014 ' + refCount + ' projected nodes, nothing contiguous enough to fold',
      candidates: ranges.slice(0, 6),
    }
  }
  // A fold pays its re-prefill ONCE and saves `span − note` on EVERY later request, so the
  // honest gate is a payback horizon computed WITH the note. The engine cannot know how many
  // requests this session still has, so it assumes the session keeps running for
  // CONDENSE_PAYBACK_REQUESTS more of them and refuses below that; the refusal prints the
  // real payback count so the model can override with an explicit content range.
  const relief =
    pressure !== undefined &&
    pressure.limitTokens > 0 &&
    pressure.liveTokens >= pressure.limitTokens * CONDENSE_PRESSURE_RELIEF_RATIO
  // A span this small CANNOT carry a handoff: the 35% weight cap allows ~420 note tokens at
  // the floor and less below it, which is not a briefing — electing it spends a turn on a note
  // the write-time gate would reject anyway (round-13 review Q5: a 750-token span was elected,
  // the model wrote a note, and the note gate refused it, burning a turn and a re-prefill).
  if (best.tokens < MIN_CONDENSE_SPAN_TOKENS) {
    return {
      reason: 'the largest eligible run is ~' + best.tokens + ' tokens, under the ' + MIN_CONDENSE_SPAN_TOKENS
        + '-token floor for a condense handoff: a note that fits the ' + Math.round(MAX_NOTE_WEIGHT_RATIO * 100)
        + '% weight cap on a span this small is a second copy, not a briefing. Keep working so consumed stretches accumulate, or send that span as an explicit content range with a one-line summary — that path carries no handoff requirement',
      candidates: ranges.slice(0, 6),
    }
  }
  // The election side cannot see the note yet, so it solves for the LARGEST note that can still
  // pay back inside the horizon instead of assuming the worst the weight cap allows:
  // `foldCost + note <= N × (tokens − note)`  ⇒  `note <= (N × tokens − foldCost) / (N + 1)`.
  // Pricing the worst-case 35% note here refused spans a real handoff pays back (round-14 review
  // Q2: a 40K span against a 96K re-prefill breaks even in 3 requests with an ~85-token note, yet
  // the worst-case price read 5 and refused the election BEFORE the model could write anything).
  // Nothing lands unchecked: the write-time gate still enforces the 35% weight cap and re-prices
  // the payback against the REAL note (`condenseNotePayback`).
  const noteBudget = condenseNoteBudget(best.tokens, best.foldCostTokens)
  if (!relief && noteBudget < MIN_HANDOFF_NOTE_TOKENS) {
    return {
      reason: [
        'every eligible row costs MORE to re-prefill (once) than it removes: the best row removes ' +
          best.tokens + ' tokens and re-prefills ~' + best.foldCostTokens + ' after its rewrite point',
        '(one-time net ' + bestNet + '). A handoff note has to be at least ~' + MIN_HANDOFF_NOTE_TOKENS +
          ' tokens to brief a resumed session, and this row allows a note of only ~' +
          Math.max(0, noteBudget) + ' stored tokens (frame prefix included) inside the ' +
          CONDENSE_PAYBACK_REQUESTS + '-request horizon' +
          (noteBudget <= 0 ? ' — what it removes never earns the re-prefill back at ANY note size' : '') +
          '. Send that span as an explicit content range.',
        'Otherwise stop compressing and keep working: a fold that removes little right in front of a',
        'big resident summary is a loss. The payback gate steps aside only at ≥'
        + Math.round(CONDENSE_PRESSURE_RELIEF_RATIO * 100) + '% of the window (pressure relief, and',
        'even there only for a fold that reclaims real headroom); the working-set floor and the',
        'note-weight gate never step aside.',
      ].join(' '),
      candidates: ranges.slice(0, 6),
    }
  }
  // Relief exists to buy off a provider rejection, so it must buy REAL headroom even with the
  // smallest note this tool accepts. A span that cannot reach `reliefFloor` with a bare note is
  // refused HERE, not after the model spends a turn writing one (round-14 review Q3: the election
  // still offered 2000..4096-token rows on a 262K window that `handleCompress` then refused
  // unconditionally).
  const reliefFloor = reliefMinSavingTokens(relief, pressure?.limitTokens ?? 0)
  if (relief && !reliefSpanReachable(best.tokens, pressure?.limitTokens ?? 0)) {
    return {
      reason: [
        'pressure relief demands a real reclaim: the best row removes only ' + best.tokens +
          ' tokens, and even with the smallest note this tool accepts (~' + SUMMARY_FRAME_TOKENS +
          ' tokens of summary frame prefix) that leaves under ~' + reliefFloor + ' tokens of headroom',
        '(2% of the ~' + (pressure?.limitTokens ?? 0) + '-token window). A fold that saves less spends a',
        'permanent barrier and a full re-prefill and re-arms the same cliff one step later. Fold a',
        'bigger stretch — the nudge range table and acp_status list the eligible rows.',
      ].join(' '),
      candidates: ranges.slice(0, 6),
    }
  }
  return {
    start: best.start,
    end: best.end,
    count: best.count,
    tokens: best.tokens,
    foldCostTokens: best.foldCostTokens,
    liveAfterFoldTokens: best.liveAfterFoldTokens,
    belowFloor: best.belowFloor,
    aboveFloorAvailable: ranges.some((range) => !range.belowFloor),
    guardedRowCount: guardedSurfaceSeqsOf(session).size,
    eligibleRangeCount: ranges.length,
    degraded: false,
    pressureRelief: relief,
    reliefLimitTokens: pressure?.limitTokens ?? 0,
    noteBudgetTokens: relief ? Number.POSITIVE_INFINITY : noteBudget,
  }
}

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
export function condenseDegradable(pick: CondensePick, minimum: number): boolean {
  return pick.belowFloor && !pick.aboveFloorAvailable && pick.tokens >= minimum
}

/**
 * Refusal copy for a below-floor condense. The THREE branches ARE the point
 * (cross-review Q4): two-branch copy conflated "nothing wider is above the floor"
 * with "nothing is eligible at all", so a session whose reclaimable area is merely
 * FRAGMENTED by the guards was told to stop compressing — and a session with
 * nothing to fold was told to widen into a guard. Both are loops. Exported so
 * every branch is testable without building three fixtures.
 */
export function condenseFloorRefusal(pick: CondensePick, floor: number): string {
  const head = `compress: condense refused seqs ${pick.start}..${pick.end} — it removes ~${pick.tokens} tokens and leaves only ~${pick.liveAfterFoldTokens}, under the working-set floor ~${floor}.\n`

  if (pick.aboveFloorAvailable) {
    return head + `AN ELIGIBLE ROW STILL LEAVES THE FLOOR: fold THAT row instead. A fold leaves exactly as much room as it does NOT remove, so the widest row is the one that strands the working set. Batch only spans you would fold anyway: a batch pays ONE re-prefill measured from its EARLIEST rewrite point, so folding an early row together with a later one pays the early row's whole suffix (see the foldCost column). Pass force:true only when you accept a shallow working set (e.g. an imminent overflow).`
  }
  if (pick.eligibleRangeCount > 0) {
    return head + `NO ELIGIBLE ROW LEAVES THE FLOOR: ${pick.eligibleRangeCount} row(s) remain and every one of them ends under ~${floor} tokens, while ${pick.guardedRowCount} row(s) are refused by the guards (current instruction copies, the newest skill catalog, the active user turn). Do NOT widen blindly — no call can reclaim a guarded row. The engine folds below the floor on its own only when nothing else can keep it, and that is decided from these rows, not from a fold count. Fold nothing for now, or name ONE explicit range with force:true if an overflow is imminent.`
  }
  return head + `NOTHING ELSE IS ELIGIBLE AT ALL: no range clears the kernel minimum, so there is no fold to make here. Do NOT retry — stop compressing and keep working.`
}

/**
 * A condense note replaces a stretch the model never named, so it must carry the
 * handoff the model would have written by hand: what the work IS, and what runs
 * next. A note missing either heading turns one fold into a re-run of consumed
 * exploration, which costs more than the fold saved.
 */
// Tolerant on purpose: `## Goal`, `# Goal`, `**Goal**`, `Goal:` and the Chinese
// spellings all pass. A gate that rejects a formatting variant is a production
// break on a model whose note is otherwise correct (cross-review Q1).
const CONDENSE_GOAL = /(^|\n)\s*(?:#{1,4}|\*{1,3})?\s*[{[【]?\s*(goal|objective|\u4efb\u52a1|\u76ee\u6807)\s*[}\]】]?\s*[:：]?/i
const CONDENSE_NEXT = /(^|\n)\s*(?:#{1,4}|\*{1,3})?\s*[{[【]?\s*(next|next step|remaining plan|\u4e0b\u4e00\u6b65|\u5b8c\u6210\u6761\u4ef6|\u5269\u4f59\u8ba1\u5212)\s*[}\]】]?\s*[:：]?/i

export function missingCondenseHeadings(note: string): string[] {
  const missing: string[] = []
  if (!CONDENSE_GOAL.test(note)) missing.push('## Goal \u2014 what this stretch was doing and what remains')
  if (!CONDENSE_NEXT.test(note)) missing.push('## Next \u2014 the immediate next command and the completion condition')
  return missing
}

/**
 * The six fields of arXiv 2611.05306 Appendix G — the shape whose absence makes
 * a resumed session re-derive work it already did. ADVISORY, never a gate: a
 * hard six-heading gate costs resident prompt text plus a retry loop, and on an
 * open-weight model that trade measured worse than the missing fields (both
 * cross-reviews rejected the hard gate). Only Goal and Next are enforced above;
 * a note missing the rest folds with a one-line warning.
 */
const CONDENSE_FIELDS: ReadonlyArray<readonly [RegExp, string]> = [
  [/^\s*#{1,3}\s*(objective|goal|\u4efb\u52a1|\u76ee\u6807)/im, 'Objective'],
  [/^\s*#{1,3}\s*(established facts|facts|\u5df2\u786e\u8ba4\u4e8b\u5b9e|\u4e8b\u5b9e)/im, 'Established facts'],
  [/^\s*#{1,3}\s*(applied changes|changes|\u5df2\u505a\u6539\u52a8|\u6539\u52a8)/im, 'Applied changes'],
  [/^\s*#{1,3}\s*(verification|verified|\u9a8c\u8bc1|\u5df2\u9a8c\u8bc1)/im, 'Verification'],
  [/^\s*#{1,3}\s*(dead ends|failed|\u6b7b\u8def|\u5931\u8d25\u5c1d\u8bd5)/im, 'Dead ends'],
  [/^\s*#{1,3}\s*(remaining plan|next|next step|\u8ba1\u5212|\u4e0b\u4e00\u6b65|\u5b8c\u6210\u6761\u4ef6)/im, 'Remaining plan'],
]

export function condenseNoteGaps(note: string): string[] {
  return CONDENSE_FIELDS.filter(([pattern]) => !pattern.test(note)).map(([, name]) => name)
}

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
export const MAX_NOTE_WEIGHT_RATIO = 0.35

/**
 * What the summary frame prefix costs on its own. The stored note is ALWAYS
 * `withSummaryFramePrefix(note)` (region.ts writes it, messages.ts projects it), so every
 * weight and payback number must include it — a bare note count under-priced the resident
 * node (round-12 Q3 caveat) and made the weight gate disagree with the payback gate
 * (round-13 Q1). Derived from the prefix itself, not hardcoded, so a copy change follows it.
 */
const SUMMARY_FRAME_TOKENS = defaultCountTokens(withSummaryFramePrefix('x')) - defaultCountTokens('x')

/**
 * What a handoff note actually weighs on this host, measured 2026-11-15: a real model-written
 * `condense` note stored ~915 tokens while missing FOUR of the six Appendix-G fields. That is the
 * number the span floor has to clear, not a guess: a floor derived from a note the model will
 * never write lets a span into the election the weight gate then refuses (round-15 review Q7.1:
 * at `MAX_NOTE_WEIGHT_RATIO` a 2000-token span admits only ~700 stored tokens, so every span
 * between 2000 and ~2614 tokens was elected, paid for by a turn of writing, and then refused).
 */
export const TYPICAL_HANDOFF_NOTE_TOKENS = 900

/**
 * Smallest stretch condense will elect: the smallest span that can carry a note this host's
 * models actually write inside the weight cap — `TYPICAL_HANDOFF_NOTE_TOKENS / MAX_NOTE_WEIGHT_RATIO`,
 * so a copy or ratio change follows it instead of drifting. The kernel minimum (`minCompressRange`)
 * is a CHARACTER threshold (5000 chars ≈ 1250 tokens), which filters nothing; the real floor is
 * the note's weight (round-13 review Q5: a 750-token election, then the weight gate).
 */
export const MIN_CONDENSE_SPAN_TOKENS = Math.ceil(TYPICAL_HANDOFF_NOTE_TOKENS / MAX_NOTE_WEIGHT_RATIO)

/**
 * The smallest handoff note (stored form, frame prefix included) that can still brief a resumed
 * session: the Appendix-G headings plus one line each. The election uses it as the floor of the
 * NOTE BUDGET it solves for, so a span that cannot carry a real briefing inside the payback
 * horizon is refused before the model writes anything (round-14 review Q2/Q5).
 */
export const MIN_HANDOFF_NOTE_TOKENS = 250

/**
 * How much net headroom a relief fold must buy, as a fraction of the window. Relief exists to
 * avoid a provider rejection, so a fold that reclaims a few hundred tokens while spending a
 * permanent barrier and a full re-prefill does not relieve anything — it just re-arms the same
 * cliff one step later (round-13 review Q2: cliff thrashing). Window-relative for the same
 * reason the emergency minimum is: a hard constant silently disables relief on a small window.
 */
export const RELIEF_MIN_NET_SAVING_RATIO = 0.02

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
export const RELIEF_MIN_NET_SAVING_MAX_TOKENS = 2048

export function reliefMinSavingTokens(pressureRelief: boolean, limitTokens: number): number {
  if (!pressureRelief || limitTokens <= 0) return 0
  return Math.min(RELIEF_MIN_NET_SAVING_MAX_TOKENS, Math.max(1, Math.floor(limitTokens * RELIEF_MIN_NET_SAVING_RATIO)))
}

/** How many further requests a condense must pay back inside to be worth electing. */
export const CONDENSE_PAYBACK_REQUESTS = 4

/**
 * The largest note (stored form, frame prefix included) that still pays back inside the horizon:
 * `foldCost + note <= N × (tokens − note)`  ⇒  `note <= (N × tokens − foldCost) / (N + 1)`.
 * Exported so the arithmetic is testable without engineering a fixture whose span, suffix and
 * note all line up (round-14 review Q2 replaced the worst-case-note price with this).
 */
export function condenseNoteBudget(spanTokens: number, foldCostTokens: number): number {
  return Math.floor((CONDENSE_PAYBACK_REQUESTS * spanTokens - foldCostTokens) / (CONDENSE_PAYBACK_REQUESTS + 1))
}

/**
 * Whether a relief span can reach the relief floor at all — with the smallest note this tool
 * accepts (the bare summary frame prefix). Exported for the same reason (round-14 review Q3).
 */
export function reliefSpanReachable(spanTokens: number, limitTokens: number): boolean {
  return spanTokens - SUMMARY_FRAME_TOKENS >= reliefMinSavingTokens(true, limitTokens)
}

/**
 * Pressure at which the payback gate steps aside (live billed input ÷ window).
 * Below it a poor-payback fold is a choice; at or above it, NOT folding is the
 * riskier option because the next request can be rejected by the provider.
 */
export const CONDENSE_PRESSURE_RELIEF_RATIO = 0.85

/**
 * Is this condense worth running once the note is counted? The folded volume is the saving
 * side, but the note stays resident on EVERY later request, so the honest saving is
 * `span − note`, and the one-time cost is the re-prefill. Exported so the arithmetic is
 * testable without engineering a fixture whose stretch, suffix and note all line up.
 */
export function condenseNotePayback(spanTokens: number, foldCostTokens: number, noteTokens: number):
  { readonly worth: boolean; readonly payback: number; readonly netSaving: number } {
  const netSaving = spanTokens - noteTokens
  // A note as big as the stretch it replaces is a second copy, not a briefing: nothing is
  // ever saved, so the fold can NEVER pay back — and that must not hinge on how cheap the
  // re-prefill happens to be. The earlier Math.max(1, netSaving) turned "no saving" into
  // "1 token per request", which made a tail fold (foldCost 0..4) look worth doing (round-11 Q7).
  if (netSaving <= 0) return { worth: false, payback: Number.POSITIVE_INFINITY, netSaving }
  // The note itself also sits AT the rewrite point, so it is not inside the suffix price
  // (surfaceSuffixTokens[endIndex + 1] starts AFTER the END edge) and the next request
  // prefills it uncached once (round-11 Q1).
  const cost = foldCostTokens + noteTokens
  return { worth: cost <= netSaving * CONDENSE_PAYBACK_REQUESTS, payback: Math.ceil(cost / netSaving), netSaving }
}

/**
 * Note weight, priced as the note is STORED. Two gates read this number — the weight cap here
 * and the payback re-pricing in `handleCompress` — and they must not disagree: measuring the
 * raw note here while pricing the stored node there let a note just under 35% raw exceed 35%
 * once the frame prefix (and the degrade stamp) landed (round-13 review Q1).
 * `storedTokens` lets a caller that already counted the stored form pass it in.
 */
export function noteWeightRatio(note: string, spanTokens: number, storedTokens?: number): number {
  // A zero-priced span cannot carry ANY note — returning 0 here read as "weightless" and
  // admitted a note onto a span with nothing to save (round-11 Q7).
  if (spanTokens <= 0) return Number.POSITIVE_INFINITY
  return (storedTokens ?? defaultCountTokens(withSummaryFramePrefix(note))) / spanTokens
}

export function noteWeightRejection(
  note: string,
  spanTokens: number,
  storedTokens?: number,
  includesDegradeLine = false,
): string {
  const tokens = storedTokens ?? defaultCountTokens(withSummaryFramePrefix(note))
  const ratio = noteWeightRatio(note, spanTokens, tokens)
  if (!Number.isFinite(ratio)) {
    return (
      'compress: condense refused — the elected span holds ~0 tokens, so a note of any size can only ADD resident text.\n'
        + 'Nothing lands. Fold a stretch that actually carries consumed work (see the nudge range table or acp_status), or skip the fold.'
    )
  }
  return (
    'compress: condense refused — your note is ~' + Math.round(ratio * 100) + '% of the ' + spanTokens + ' tokens it replaces (limit ' + Math.round(MAX_NOTE_WEIGHT_RATIO * 100) + '%), counted as it is stored (' + tokens + ' tokens including the summary frame prefix'
      + (includesDegradeLine ? ' and the [dsh:degraded] line' : '') + ').\n'
      + 'That is a second copy, not a briefing: it stays resident in every later request, so the fold never pays back its one-time re-prefill.\n'
      + 'Cut it to what a resumed session cannot re-derive: ## Goal, Established facts, Applied changes, Verification, Dead ends, Remaining plan (next command + completion condition). Nothing lands — rewrite the note and send it again.'
  )
}

async function handleCompress(env: ToolEnvironment, args: CompressArgs, exec: ToolRunContext): Promise<TextOutput> {
  const agent = requireAgent(exec)
  const session = agent.session
  // Clean orphan tool messages before any range solve: a single orphan result
  // corrupts the pairing balance cache and rejects every large range (issue
  // #18). Every call still in flight — the compress call itself AND any
  // sibling tool called in the same assistant message — must be excluded from
  // orphan pruning: its tool/result lands at the end of the step, and pruning
  // the call now would orphan that result.
  stripOrphanedSurfaceToolMessages(session, openToolCallIds(session))
  const state = env.store.stateFor(session)
  // The kernel gets the FULL log (visible + shadowed): syncBlocks deactivates
  // a block whose consumed messages are absent, and resolveBoundaries refuses
  // to anchor a block ref it cannot find, so tier-2/3 distillation needs the
  // originals present. The token count uses the same priority chain as the
  // nudge (projectedTokens → surfaceTokens → character heuristic).
  const coreMessages = allLogMessages(session)
  const surfaceMessages = eventsToCoreMessages(surfaceEventsOf(session))
  const tokenCount = resolveTokenCount(agent, surfaceMessages)
  const window = await resolveEffectiveWindow(env, agent)
  const config = kernelConfigFor({ ...env, modelContextLimit: window.limit })

  // Assign refs / advance state exactly like a turn would.
  const turn = env.kernel.processTurn({ messages: coreMessages, state, config, tokenCount })
  env.store.set(session, turn.state)
  const byRaw = turn.state.messageRefs.byRaw
  // mN drilldown refs resolve against the CURRENT turn's ref map (not the
  // stored state) — acp_status's turn is never persisted, so its mN rows only
  // exist here; the deterministic re-assignment yields the same mN for the
  // same messages (see parseBoundary).
  const byRef = turn.state.messageRefs.byRef

  // Tolerate the wrapped-arguments forms some models emit (double-nested
  // `{ arguments: "..." }`), which the old DSH validator surfaced as
  // `"arguments" must be an object` and sent the model into a retry loop.
  const unwrapped = unwrapCompressArgs(args)
  if (unwrapped === null) {
    return {
      text: `compress: missing content — pass the content array directly: ${callHint(env.toolCallStyle ?? 'native', 'compress', '{ content: [{ startSeq, endSeq, summary }] }')}`,
    }
  }
  args = unwrapped
  // condense: the engine names the span, the model writes the handoff note. From
  // here on the write is IDENTICAL to an explicit range — guarded rows, live
  // checkpoint carriers, pairing balance and the provenance gate all still apply,
  // because the engine picks from the range table geometry the guards produced.
  let condensePick: CondensePick | undefined
  if (args.condense !== undefined) {
    if (args.content !== undefined) {
      return { text: 'compress: condense and content are mutually exclusive — pass one or the other' }
    }
    const note = args.condense.summary ?? ''
    const missing = missingCondenseHeadings(note)
    if (missing.length > 0) {
      return {
        text: 'compress: condense needs a handoff note, not a topical summary — add ' + missing.join(' and ') + '.\n'
          + 'Re-send the SAME condense call with the completed note: the engine re-picks the span from the CURRENT surface, so nothing you wrote is invalidated.',
      }
    }
    // Pressure decides whether the payback gate may step aside (round-10 review B2).
    // lastRequestInput is a LAGGING quantity (round-11 Q6): it carries none of that request's
    // own output and none of the rows that landed since it, so a session that crossed the cliff
    // AFTER the last billed request would still enforce the gate — and the very next request is
    // then rejected by the provider. Project the CURRENT surface with the same calibrated factors
    // the emergency projection uses and take the larger side.
    const liveTokens = Math.max(
      lastRequestInput(session),
      surfaceTokens(session) * (pairedInflation(session) ?? DEFAULT_OVERFLOW_INFLATION) + requestOverhead(session),
    )
    const pick = pickCondenseSpan(session, turn, 5, foldPricingFor(env), {
      liveTokens,
      limitTokens: window.limit,
    })
    if ('reason' in pick) {
      const rows = pick.candidates.map((range) =>
        '  - seq ' + range.start + '..' + range.end + ' — ' + range.count + ' messages, ~' + range.tokens + ' tokens — fold costs ~' + range.foldCostTokens + ' once',
      )
      return {
        text: 'compress: condense folded nothing — ' + pick.reason + '.\n'
          + 'Compare against the automatic checkpoint it replaces: a host checkpoint costs a fully cold prefill (~130 s here), so waiting is cheaper only while nothing is consumed.\n'
          + (rows.length > 0
            ? 'Eligible runs you can fold explicitly:\n' + rows.join('\n')
            : 'No eligible run yet — keep working; consumed spans reappear in the next nudge.'),
      }
    }
    // Working-set guard (Appendix D of arXiv 2611.05306): on an open-weight
    // engine a fold that leaves a shallow working set is the regime where
    // benchmark scores collapse, and the failure is not recoverable by a better
    // note. Advisory first, hard only here — the model can force it, because
    // overflow pressure is a legitimate reason to accept a thin context.
    const forced = args.force === true || args.condense.force === true
    const floor = foldPricingFor(env).floorTokens ?? 0
    let degradedNote = ''
    let degradedPick: CondensePick | undefined

    if (pick.belowFloor && !forced) {
      const minimum = Math.max(MIN_DEGRADED_FOLD_TOKENS, Math.floor(floor / 2))
      if (!condenseDegradable(pick, minimum)) {
        return { text: condenseFloorRefusal(pick, floor) }
      }
      // DEGRADED ELECTION (round-4 Q1/Q4): NO eligible row could keep the floor, so
      // refusing is a deadlock with no compliant move — a real reclaim beats a
      // permanent refusal. The stamp is ONE delimited, machine-readable line so a
      // later floor statistic, a search hit or a distill can recognise it instead
      // of reading this as a compliant fold.
      degradedPick = { ...pick, degraded: true }
      degradedNote = '\n[dsh:degraded] no eligible row could keep the working-set floor: this fold leaves ~'
        + pick.liveAfterFoldTokens + ' of ~' + floor + ' tokens. Provenance of a desperate fold, not a proposal.'
    }

    // ONE number for both gates, priced as the note is STORED: the summary frame prefix the
    // checkpoint node carries plus the degrade stamp, counted over the concatenated text the
    // transaction actually writes (`note + degradedNote`) rather than two separate counts, so
    // the two `defaultCountTokens` calls cannot drift by a token at the join (round-13 review
    // Q1). A bare `defaultCountTokens(note)` under-priced the resident node by ~16-35 tokens.
    const storedSummary = note + degradedNote
    const noteTokens = defaultCountTokens(withSummaryFramePrefix(storedSummary))
    // A note that is a third the size of the stretch is a second copy, not a
    // briefing — refuse BEFORE anything lands, so the model can rewrite and resend
    // without paying a second re-prefill (nothing was written).
    const weight = noteWeightRatio(note, pick.tokens, noteTokens)
    if (weight > MAX_NOTE_WEIGHT_RATIO) {
      return { text: noteWeightRejection(note, pick.tokens, noteTokens, degradedNote !== '') }
    }

    // PAYBACK RE-PRICED WITH THE REAL NOTE (round-9 review, GLM): the note stays resident on
    // every later request, so the saving side is the folded volume MINUS the note, and it also
    // sits AT the rewrite point so the next request prefills it uncached once.
    // UNDER WINDOW PRESSURE the payback horizon steps aside (round-12 review Q5.1) — but it
    // steps aside for a fold that buys REAL headroom, not for any positive saving: at ≥85% of
    // the window a fold that reclaims a few hundred tokens spends a permanent barrier and a
    // full re-prefill and leaves the session on the same cliff one step later (round-13 review
    // Q2). What NEVER steps aside: the working-set floor and the note-weight gate above.
    const payback = condenseNotePayback(pick.tokens, pick.foldCostTokens, noteTokens)
    const reliefFloor = reliefMinSavingTokens(pick.pressureRelief, pick.reliefLimitTokens)
    if (!payback.worth && !(pick.pressureRelief && payback.netSaving >= reliefFloor)) {
      const horizon = Number.isFinite(payback.payback)
        ? `Payback needs ~${payback.payback} requests and the horizon is ${CONDENSE_PAYBACK_REQUESTS}. Fold a bigger stretch, write a shorter note, or send the span as an explicit content range when you know this session has longer to run.`
        : `Nothing is saved at any horizon: a note this size leaves ${payback.netSaving} tokens per request, so NO session length and NO explicit content range makes this fold positive — the only correct move is a materially shorter note or a bigger stretch.`
      // Q2 (round-14): the model has already written its note, so the refusal must hand it the
      // number that works — the largest note that still pays back inside the horizon.
      const budgetHint = Number.isFinite(pick.noteBudgetTokens) && pick.noteBudgetTokens >= MIN_HANDOFF_NOTE_TOKENS
        ? ` A note of at most ~${pick.noteBudgetTokens} stored tokens (frame prefix included) still pays back inside ${CONDENSE_PAYBACK_REQUESTS} requests.`
        : ''
      const reliefLine = pick.pressureRelief
        ? ` You are past the relief threshold, but relief still requires a net reclaim of at least ~${reliefFloor} tokens (2% of the window) — a fold that saves less spends a permanent barrier and a full re-prefill for nothing.`
        : ''
      return { text:
        `condense refused: the note (~${noteTokens} tokens, stored form including the summary frame prefix${degradedNote !== '' ? ' and the [dsh:degraded] line' : ''}) stays resident on every later request, so ${payback.netSaving} tokens are really saved against a ~${pick.foldCostTokens}-token re-prefill. ${horizon}${budgetHint}${reliefLine}`,
      }
    }
    condensePick = degradedPick ?? pick
    args = {
      ...args,
      content: [{
        startSeq: pick.start,
        endSeq: pick.end,
        summary: note + degradedNote,
        topic: args.condense.topic,
        verifiedReadings: args.condense.verifiedReadings,
      }],
    }
  }
  // Items-level required check AFTER the envelope peel (see
  // validateContentItems for why the schema gate alone cannot do this).
  validateContentItems(args.content!)

  const ranges: Array<
    ResolvedSurfaceRange & {
      startSeq: number
      endSeq: number
      startRef: string
      endRef: string
      summary: string
      topic?: string
      /** B3：本段压缩时已绿的验收读数（结构化落盘）。 */
      verifiedReadings?: string[]
    }
  > = []
  // Ranges whose whole span was already shadowed by earlier compressions.
  // They land as advisory warnings, never as errors or phantom blocks.
  const alreadyCompressedNotes: string[] = []
  // Ranges rejected because they cover a CURRENT injected instruction row —
  // hard reject before the kernel apply (supersedes the F7 warn-only draft,
  // see protectedRowRejectionNote): the kernel never sees these ranges, so no
  // phantom block can exist. Computed once here: the surface is stable from
  // the orphan strip onward, the deferred compress-pair hide only touches tool
  // events, and every accepted range lands in ONE applyCompression call at the
  // end of the loop — so the set cannot go stale mid-batch.
  const rejectedNotes: string[] = []
  // Range keys (`startRef::endRef`) already claimed by an accepted range in
  // THIS call: several requested ranges can resolve to ONE balanced span
  // (single tool results expand outward to the enclosing pair), and feeding
  // the same span twice makes the kernel stamp a block per request while only
  // one durable transaction can ever land for it (issue #201).
  const seenRangeKeys = new Set<string>()
  const duplicateRangeNotes: string[] = []
  // Reversed edges (`start > end`) are a model slip, not a real span — the same
  // span with the boundaries swapped is what the caller meant (measured
  // 2026-11-01 in one live window: 8 `reversed range` rejections, every one
  // recoverable by a swap). Swapping HERE rather than inside
  // `resolveSurfaceRange` keeps that resolver's `start ≤ end` invariant intact
  // (its stale recovery and index arithmetic both assume ordered edges) and
  // lets the advisory line name the ORIGINAL edges.
  const swappedNotes: string[] = []
  const guardedSeqs = guardedSurfaceSeqsOf(session)
  for (const range of args.content!) {
    let startSeq = parseBoundary(range.startSeq, byRef, session)
    let endSeq = parseBoundary(range.endSeq, byRef, session)
    if (startSeq > endSeq) {
      swappedNotes.push(
        `  seqs ${startSeq}..${endSeq} were reversed — swapped to ${endSeq}..${startSeq}`,
      )
      const mirrored = startSeq
      startSeq = endSeq
      endSeq = mirrored
    }
    let resolved: ResolvedSurfaceRange
    try {
      // Balance edges FIRST: the requested edges may sit mid-pair or on nodes
      // that project no bare-`${seq}` id (multi-tool-call assistants project to
      // `${seq}#${callId}` sub-ids; empty tool results project to nothing —
      // issue #155). resolveSurfaceRange shifts them to clean
      // tool-pairing-balanced cuts that ANCHOR (bare ref, sub-id, or the
      // empty-result fallback — see edgeRefForSeq), so the resolved refs exist
      // and the shadowed span matches the returned range. Edges shadowed by an
      // earlier compression (stale nudge table / old compress result) are
      // remapped to the still-live content of the span.
      resolved = resolveSurfaceRange(session, startSeq, endSeq)
    } catch (error) {
      if (error instanceof AlreadyCompressedRangeError) {
        const covering = error.coveringBlockIds
        const blockNote = covering.length === 0
          ? ''
          : ` (block ${covering[0]!.slice(0, 8)}${covering.length > 1 ? ` +${covering.length - 1} more` : ''})`
        alreadyCompressedNotes.push(
          `  seqs ${error.start}..${error.end} already compressed${blockNote} — nothing to reclaim; decompress to recover the originals`,
        )
        continue
      }
      throw error
    }
    // Hard reject BEFORE the kernel: a span covering a CURRENT injected policy
    // row has no legitimate outcome — the host re-injects the newest AGENTS.md
    // copy the moment it leaves the surface (compress → re-inject loop fuel,
    // issue #71), and a folded skill catalog never comes back at all (resend
    // keyed on the digest, issue #185). Stale copies pass: removing them while
    // the newest stays visible is the real cleanup and triggers no re-injection.
    // Probe the set that will ACTUALLY be shadowed (`shadowedSeqsOf`, the
    // positional slice the transaction prices) rather than a numeric interval —
    // see guardedRowsInSpan for why the interval false-positives on a locally
    // non-monotonic surface.
    const shadowedSpan = shadowedSeqsOf(session, resolved.start, resolved.end)
    const instructionHits = guardedRowsInSpan(guardedSeqs, shadowedSpan)
    if (instructionHits.length > 0) {
      rejectedNotes.push(protectedRowRejectionNote(resolved.start, resolved.end, instructionHits, shadowedSpan, session))
      continue
    }
    // The working-set floor applies to a model-NAMED range too (cross-review Q1:
    // all three reviewers). The guard protects the post-fold working set, and an
    // explicit range that folds down to a sliver violates it exactly as hard as an
    // elected one — while still paying the full one-time re-prefill. Same pricing
    // implementation as the table rows (foldSpanPricing), so a row marked
    // "below floor" and the guard can never disagree.
    // `forced` covers the content range condense synthesized above, and so does the
    // plan's own `degraded` flag: the guard reads the decision the election already
    // made instead of a mutable bypass, so the two guards cannot disagree.
    const priced = foldSpanPricing(session, resolved.start, resolved.end, foldPricingFor(env))
    if (priced.belowFloor && args.force !== true && args.condense?.force !== true && condensePick?.degraded !== true) {
      rejectedNotes.push(
        '  seq ' + resolved.start + '..' + resolved.end + ' would leave ~' + priced.liveAfterTokens + ' tokens visible, under the working-set floor ~' + (foldPricingFor(env).floorTokens ?? 0) + ' — one fold costs the SAME single re-prefill, so this span is too thin to earn it back. Do NOT just widen blindly: current instruction copies, the newest skill catalog and the active user turn are refused by a DIFFERENT guard (they appear as their own rejection line when hit), so widen only across rows that stay eligible, or fold several disjoint slices in ONE call. Read consumed bytes back from files instead of folding. force:true bypasses THIS floor only — never the guarded rows. Other ranges in this batch already landed, so do not resend this one unchanged.',
      )
      continue
    }
    // An edge on an ACTIVE block's checkpoint summary node resolves to the
    // kernel block ref (bN) — the boundary that makes applyCompression distill
    // (tier 2/3) instead of folding the summary as a plain message.
    const startBlockRef = blockRefForSummarySeq(session, resolved.start)
    const endBlockRef = blockRefForSummarySeq(session, resolved.end)
    const startRef = startBlockRef ?? edgeRefForSeq(session, byRaw, resolved.start, 'start', resolved.end)
    const endRef = endBlockRef ?? edgeRefForSeq(session, byRaw, resolved.end, 'end', resolved.start)
    if (startRef === undefined || endRef === undefined) {
      throw new Error(
        `billion-context-dsh: seq ${resolved.start}..${resolved.end} has no assigned ref — `
        + 'the range must be on the current surface (run acp_status for the live seq list)',
      )
    }
    // A plain (message-ref) range must not fold a LIVE block's checkpoint
    // carrier. The kernel keeps that carrier visible — our projection marks it
    // via `summaryOfBlockId`, upstream #335 — but this host transaction replaces
    // the WHOLE span in one `surfaceOp`, so the carrier would leave the surface
    // anyway: the block's summary would vanish while the kernel still counts the
    // block as active. Block-ref boundaries (bN..bM) are the call that SHOULD
    // fold carriers (tier 2/3), so the reject fires only when NEITHER edge is a
    // block ref — the same rule the kernel applies (`resolveBoundaries` reports
    // boundary kind `block` as soon as one edge is a block ref).
    if (startBlockRef == null && endBlockRef == null) {
      const carriers = liveCheckpointCarriersInSpan(session, shadowedSpan)
      if (carriers.length > 0) {
        rejectedNotes.push(liveCarrierRejectionNote(resolved.start, resolved.end, carriers))
        continue
      }
    }
    const rangeKey = `${startRef}::${endRef}`
    if (seenRangeKeys.has(rangeKey)) {
      // An earlier range in this call already owns this exact span: its summary
      // lands for the whole span, and a second block here could never get its
      // own durable transaction (the span leaves the surface after the first).
      duplicateRangeNotes.push(
        `  seqs ${range.startSeq}..${range.endSeq} resolve to the same span as an earlier range in this call (${resolved.start}..${resolved.end}) — skipped`,
      )
      continue
    }
    seenRangeKeys.add(rangeKey)
    ranges.push({
      ...resolved,
      startSeq,
      endSeq,
      startRef,
      endRef,
      // B3：把该段声明的已绿验收读数带上（缺位=不写键）
      ...(Array.isArray(range.verifiedReadings) && range.verifiedReadings.length > 0
        ? { verifiedReadings: range.verifiedReadings.map(String) }
        : {}),
      summary: range.summary,
      ...(range.topic ?? args.topic) === undefined ? {} : { topic: range.topic ?? args.topic },
    })
  }

  // Nothing to do: every requested range was already compressed, rejected, or
  // a duplicate of an earlier range in the same call.
  if (ranges.length === 0) {
    const text = ['Compressed 0 block(s), ~0 tokens reclaimed.', ...swappedNotes, ...alreadyCompressedNotes, ...duplicateRangeNotes, ...rejectedNotes]
    if (alreadyCompressedNotes.length > 0) {
      text.push('  (all requested ranges were already compressed — decompress a block to recover its originals)')
    } else if (duplicateRangeNotes.length > 0 && rejectedNotes.length === 0) {
      text.push('  (nothing compressed — every range resolved to a span an earlier range in this call already covers)')
    } else if (rejectedNotes.length > 0) {
      text.push('  (nothing compressed — every range covered a current guarded row, an injected policy row or the active user turn; see the rejections above)')
    }
    return { text: text.join('\n') }
  }

  const applied = env.kernel.applyCompression({
    ranges: ranges.map(({ startRef, endRef, summary, topic }) => ({ startRef, endRef, summary, topic })),
    messages: coreMessages,
    state: turn.state,
    config,
    // Deliberately NOT overriding protectedMessageIds: with the full log the
    // kernel's recent/last-user protection is computed over the same
    // non-block-covered messages as the visible feed, so default behavior is
    // preserved. Any 'Excluded N protected message(s)' warning is surfaced.
  })
  // A kernel error for ONE range must not poison the whole call: the other
  // ranges still created blocks. This matters for issue #18's "phantom range"
  // — messages absorbed into an earlier block's effectiveMessageIds (kernel
  // boundary adjustment) but still live on the surface resolve fine but make
  // the kernel throw "Range contains no compressible messages". Fail only
  // when NOTHING landed; otherwise land the successes and surface the
  // failures as advisory lines below.
  if (applied.result.errors.length > 0 && applied.result.blocksCreated === 0) {
    return { text: `compress failed: ${applied.result.errors.join('; ')}` }
  }
  env.store.set(session, applied.state)
  if (applied.result.blocksCreated > 0) {
    // Hide this compress call/result after the tool result lands, so the
    // compaction summary never sits between an assistant tool_calls block and
    // its tool response (strict providers reject that sequence).
    env.compressCallIdsToHide?.add(exec.callId)
  }

  // Match freshly created kernel blocks to the requested ranges by their
  // range key (the kernel stamps startRef/endRef onto each new block).
  const previousIds = new Set(turn.state.blocks.map((block) => block.blockId))
  const newBlocks = applied.state.blocks.filter((block) => !previousIds.has(block.blockId))
  const blockByRangeKey = new Map(newBlocks.map((block) => [`${block.startRef}::${block.endRef}`, block]))
  // Warnings carry two shapes: range-prefixed ("Skipped range (a..b) — …")
  // attributable to a specific range, and free-form ("Excluded N protected
  // message(s) …") attributable to the call as a whole.
  const warningByRangeKey = new Map<string, string[]>()
  const freeWarnings: string[] = []
  for (const warning of applied.result.warnings) {
    const match = /^Skipped range \((.+?)\.\.(.+?)\)/.exec(warning)
    if (match !== null) {
      const key = `${match[1]}::${match[2]}`
      const list = warningByRangeKey.get(key) ?? []
      list.push(warning)
      warningByRangeKey.set(key, list)
    } else {
      freeWarnings.push(warning)
    }
  }

  const lines: string[] = []
  let skippedRanges = 0
  for (const range of ranges) {
    const key = `${range.startRef}::${range.endRef}`
    const block = blockByRangeKey.get(key)
    if (block === undefined) {
      // The kernel skipped this range (already compressed / overlapped): no
      // kernel block was created, so no durable transaction is landed — the
      // ledger must never record a block the kernel does not know.
      skippedRanges += 1
      const warnings = warningByRangeKey.get(key) ?? []
      for (const warning of warnings) lines.push(`  ${warning}`)
      continue
    }
    // The edges were already balanced above; shadow exactly that span.
    const { start, end } = range
    const shadowed = shadowedSeqsOf(session, start, end)
    // Last line of defense before the durable write: if an earlier landed
    // transaction in this batch removed this span from the surface (the kernel
    // can stamp one block per requested range even when several share refs),
    // skip instead of transacting — a summary whose shadowedSeqs no longer
    // name an exact current surface span bricks the session log on next load
    // (issue #201). runCompactionTransaction re-validates and would throw;
    // skipping keeps the rest of the batch landing.
    if (!isExactSurfaceSpan(session, start, end, shadowed)) {
      skippedRanges += 1
      lines.push(`  skipped seqs ${start}..${end}: already shadowed by an earlier range in this call`)
      continue
    }
    // Price the reclaimed tokens in the HOST's token vocabulary (rule 12):
    // prefer the live meter's per-node prices, fall back to the exact mirror.
    // NEVER defaultCountTokens — that overdraws the meter on CJK (issue #54).
    const shadowedTokens = shadowedTokensViaMeter(session, shadowed, agent.ctx)
    const tier = block.tier === 2 || block.tier === 3 ? block.tier : 1
    const parentBlockIds = compactionIdsOfKernelBlocks(session, block.directBlockIds)
    // Provenance follows the LIVE route, not `agent.options`: after a mid-session
    // model switch the latter is a stale snapshot (the PREVIOUS route), so the
    // summary node would be stamped with a route the summary did not come from.
    const { provider, model } = routeFor(agent)
    const { compactionId } = runCompactionTransaction(session, {
      start,
      end,
      shadowedSeqs: shadowed,
      summary: [{ type: 'text', text: range.summary }],
      shadowedTokenCount: shadowedTokens,
      provider,
      model,
      tier,
      kernelBlockId: block.blockId,
      ...(range.topic === undefined ? {} : { topic: range.topic }),
      ...(parentBlockIds.length === 0 ? {} : { parentBlockIds }),
      // Record the kernel block's raw coverage so a restarted engine
      // rehydrates the SAME effective messages (a tier-2 block's coverage is
      // its parents' originals, not the checkpoint node).
      directMessageIds: block.directMessageIds,
      effectiveMessageIds: block.effectiveMessageIds,
      // B3：已绿验收读数随压缩块落盘（缺位=不写键）
      ...(range.verifiedReadings === undefined ? {} : { verifiedReadings: range.verifiedReadings }),
    })
    const adjusted = start !== range.startSeq || end !== range.endSeq
    // Always report the tier, even tier 1: a silently-downgraded distill
    // (boundary moved off the checkpoint seq → the kernel folds a plain
    // message) must be visible to the model immediately, or the model keeps
    // believing the distillation landed (issue #60, failure mode 2).
    const tierLabel = `, tier ${tier}`
    // B3 must be READABLE, not just durable: echo the recorded readings back in the
    // compress result, otherwise "later steps need not re-run them" is unreachable
    // (the field had no other production reader).
    const readingsLabel = range.verifiedReadings !== undefined && range.verifiedReadings.length > 0
      ? `, verified: ${range.verifiedReadings.join('; ')}`
      : ''
    const note = range.recovered === true
      ? ` (seqs ${range.startSeq}..${range.endSeq} were already shadowed — compressed the live remainder ${start}..${end})`
      : adjusted
        ? ` (adjusted from ${range.startSeq}..${range.endSeq} to balanced edges)`
        : ''
    lines.push(
      `  block ${compactionId.slice(0, 8)}: seqs ${start}..${end}, ${shadowed.length} messages shadowed${tierLabel}${readingsLabel}${note}`,
    )
  }

  // Baseline for the nudge cost line: the request after a replace re-prefills
  // almost the whole surface once, and the delta is what that fold actually cost.
  if (applied.result.blocksCreated > 0) recordCompressCost(session, providerUsage(agent))

  const summaryLine = `Compressed ${applied.result.blocksCreated} block(s), ~${applied.result.tokensCompressed} tokens reclaimed.`
  // What the elected fold costs the NEXT request, in the currency the model
  // optimizes with (tokens, not seconds — the prefill rate is host-specific and
  // unverified here). Measured 2026-11-15: the slot reuses the prompt prefix up to
  // the rewrite point and re-prefills EVERYTHING after it — and nothing before it,
  // which is where the fixed request head (system prompt, tool registry, injected
  // rows) lives. Adding the paired request overhead here double-counted that head
  // and pushed the payback gate above what the host bills (round-10 review B1); the
  // overhead now belongs only to the emergency projection in src/index.ts.
  const condenseLine = condensePick === undefined || applied.result.blocksCreated === 0
    ? ''
        : (() => {
            const noteTok = defaultCountTokens(withSummaryFramePrefix(args.content?.[0]?.summary ?? ''))
            const priced = condenseNotePayback(condensePick.tokens, condensePick.foldCostTokens, noteTok)
            const even = Number.isFinite(priced.payback) ? ` — it breaks even after ~${priced.payback} further requests` : ''
            return `  condense elected seqs ${condensePick.start}..${condensePick.end} (~${condensePick.tokens} tokens removed); the next request re-prefills ~${condensePick.foldCostTokens} tokens ONCE — every token AFTER the rewrite point, because the fixed request head (system prompt, tool registry) sits BEFORE it and a warm cache reuses it (an injected row that landed AFTER the rewrite point is inside this price, not reused) — and this note (~${noteTok} tokens) stays resident on every request after it, saving ~${priced.netSaving} each${even}.\n  Batching spans into ONE call pays a single re-prefill, measured from the EARLIEST rewrite point of that batch; splitting them across calls pays one re-prefill each, so batch only what you would fold anyway — folding an early span to reach a later one pays the early span's whole suffix.`
          })()
  const totalSkipped = skippedRanges + alreadyCompressedNotes.length + duplicateRangeNotes.length + rejectedNotes.length
  const failedLines = applied.result.errors.map((error) => `  ${error}`)
  const condenseAdvisories = condensePick === undefined || applied.result.blocksCreated === 0
    ? []
    : [
        ...(condenseNoteGaps(args.condense?.summary ?? '').length > 0
          ? ['  condense note is missing Appendix G field(s): ' + condenseNoteGaps(args.condense?.summary ?? '').join(', ') + ' — Verification and Dead ends are the ones a resumed session re-derives most expensively.']
          : []),
        ...(condensePick?.degraded === true
          ? ['  DEGRADED election: no eligible row could keep the working-set floor, so this fold went below it on purpose. The STORED note carries the [dsh:degraded] line — that line is the single source; this line only quotes it so you know now, in this turn, instead of discovering it after the next re-prefill.']
          : []),
      ]
  const warningLines = [
    ...condenseAdvisories,
    ...freeWarnings.map((warning) => `  ${warning}`),
    ...failedLines,
    ...alreadyCompressedNotes,
    ...duplicateRangeNotes,
    ...rejectedNotes,
    ...swappedNotes,
    ...lines,
  ]
  const footer = totalSkipped > 0
    ? `  (${totalSkipped} range(s) skipped or failed — see above)`
    : ''
  return { text: `${summaryLine}\n${[...warningLines, condenseLine, footer].filter((line) => line !== '').join('\n')}` }
}

const decompressParameters = {
  blockId: { type: 'string' as const, required: true, description: 'Block id: the kernel block ref `bN` shown by acp_status (e.g. b1), or a compaction id / prefix from search_context.' },
  offset: { type: 'integer' as const, description: 'Start position in the block\'s message list (default 0). Blocks are paged by size — each page stays under the host tool-result trim budget (up to 100 messages) — so follow the continue hint in the result to walk the rest.' },
  limit: { type: 'integer' as const, description: 'Messages per page (default 100; values above 100 are capped to 100). Pages are also bounded by a character budget, so long messages return fewer than this per call.' },
} as const

interface DecompressArgs {
  blockId: string
  offset?: number
  limit?: number
}

/** Resolve a block arg to its durable compaction id: exact `bN` kernel ref
 *  first (acp_status shows `bN`), then the compaction-id prefix match that
 *  search_context and /acp-prune have always used. The `bN` branch is exact
 *  (`/^b\d+$/` with `$`), so a UUID that happens to start with `b1` cannot be
 *  shadowed — full UUIDs and 8-char prefixes never match the anchored regex. */
function resolveBlockId(session: Session, arg: string): string | null {
  const byKernelRef = blockIdOfKernelRef(session, arg)
  if (byKernelRef !== null) return byKernelRef
  // A canonical `bN` form is NEVER read as a compaction-id prefix. `blockIdOfKernelRef` already
  // refused it (no active block carries that ref), and compaction ids are UUIDs whose leading
  // chars are hex — `b01` is itself a valid UUID prefix, so falling through to the prefix scan
  // resolved a malformed ref to a real block with probability 1/16^len (round-14 review Q6: the
  // one-off `M3` flake, unreproducible by construction).
  if (/^b\d+$/.test(arg)) return null
  const ledger = rebuildBlockLedger(sessionEventsOf(session))
  const byPrefix = ledger.find((entry) => entry.blockId.startsWith(arg))
  return byPrefix?.blockId ?? null
}

function handleDecompress(_env: ToolEnvironment, rawArgs: DecompressArgs, exec: ToolRunContext): TextOutput {
  const args = unwrapEnvelope<DecompressArgs>(rawArgs)
  const session = requireAgent(exec).session
  const blockId = resolveBlockId(session, args.blockId)
  if (blockId === null) {
    return { text: `decompress: block "${args.blockId}" not found (see acp_status for the block list)` }
  }
  const ledger = rebuildBlockLedger(sessionEventsOf(session))
  const block = ledger.find((entry) => entry.blockId === blockId)
  if (block === undefined) {
    return { text: `decompress: block "${args.blockId}" not found (see acp_status for the block list)` }
  }
  // Tier-2/3 blocks shadow parent checkpoint nodes: expand to the originals.
  // Page by BOTH message count and rendered chars so a normal page stays under
  // the host's tool-result pruner threshold (DEFAULT_DECOMPRESS_PAGE_CHARS);
  // renderLen prices each message exactly as it will appear on this page.
  const expanded = expandShadowedSeqs(session, block.blockId)
  const page = sliceDecompressPage(
    expanded,
    args.offset ?? 0,
    args.limit ?? DEFAULT_DECOMPRESS_PAGE,
    DEFAULT_DECOMPRESS_PAGE_CHARS,
    (seq) => {
      const event = eventAtOf(session, seq)
      const text = event === undefined ? '' : extractEventText(event)
      return text.length === 0 ? 0 : `[seq ${seq}] ${text}`.length
    },
  )
  if (page.total === 0 || page.seqs.length === 0) {
    const where = page.total === 0 ? '' : ` has ${page.total} messages; offset ${page.offset} is past the end — use an offset below ${page.total}, or omit it`
    return { text: page.total === 0 ? `Block ${block.blockId} — ${block.summary}\n\n(no recoverable content)` : `decompress: block ${block.blockId}${where}` }
  }
  const parts: string[] = []
  for (const seq of page.seqs) {
    const event = eventAtOf(session, seq)
    const text = event === undefined ? '' : extractEventText(event)
    if (text.length > 0) parts.push(`[seq ${seq}] ${text}`)
  }
  const tierNote = block.tier > 1 ? ` (tier ${block.tier}, distills ${block.parentBlockIds.length} block(s))` : ''
  // Marker + continue hint LEAD the payload, not trail it: if a page is ever
  // oversized and the host drops its middle, the "this page is partial" line
  // survives up top rather than being the exact line that gets trimmed away.
  const lines: string[] = []
  lines.push(`[messages ${page.offset + 1}..${page.offset + page.seqs.length} of ${page.total}]`)
  if (!page.exhausted) lines.push(`More available — continue with decompress({ blockId: "${block.blockId}", offset: ${page.offset + page.seqs.length} })`)
  return {
    text: `Block ${block.blockId} — ${block.summary}${tierNote}\n\n${lines.join('\n')}\n\n${parts.join('\n\n') || '(no text content on this page)'}`,
  }
}

const searchParameters = {
  query: { type: 'string' as const, required: true, description: 'Search terms to find inside compressed blocks.' },
  limit: { type: 'integer' as const, description: 'Maximum results (default 5).' },
} as const

interface SearchArgs {
  query: string
  limit?: number
}

/** Event type → kernel message role (drives hybrid role weighting). */
function roleOfEvent(event: SessionEvent): MessageRole | null {
  switch (event.type) {
    case 'user/message': return 'user'
    case 'assistant/message': return 'assistant'
    case 'tool/result': return 'tool'
    default: return null
  }
}

// The search corpus (block summaries + all shadowed originals) is a pure
// function of the append-only log: rebuild it once per log snapshot and reuse
// across searches until the next append (issue #133 — the per-call full
// rebuild re-extracted and re-counted every shadowed message on every search;
// the snapshot array is stable until the next append, see sessionEventsOf).
const searchDocsCache = new WeakMap<readonly SessionEvent[], SearchDoc[]>()

/**
 * Build the unified SearchDoc[] from the log: one block doc per ledger entry
 * (ref = compactionId, so `decompress({ blockId })` closes the loop) plus one
 * message doc per shadowed ORIGINAL (expanded through distilled parents; each
 * seq is claimed by the earliest/innermost block that covered it, mirroring
 * pi's owner map — decompress on that block recovers the original).
 * Cached per log snapshot (see searchDocsCache). Exported for the issue #133
 * regression tests (not part of the public API — index.ts re-exports only).
 */
export function buildSearchDocs(session: Session): SearchDoc[] {
  const events = sessionEventsOf(session)
  const cached = searchDocsCache.get(events)
  if (cached !== undefined) return cached
  const ledger = rebuildBlockLedger(events)
  const docs: SearchDoc[] = []
  const claimed = new Set<number>()
  for (const block of ledger) {
    docs.push({
      kind: 'block',
      ref: block.blockId,
      text: block.summary,
      title: block.summary.slice(0, 60) || block.blockId,
      blockId: block.blockId,
      tier: block.tier,
      tokens: defaultCountTokens(block.summary),
    })
    for (const seq of expandShadowedSeqs(session, block.blockId)) {
      if (claimed.has(seq)) continue
      claimed.add(seq)
      const event = eventAtOf(session, seq)
      if (event === undefined) continue
      const role = roleOfEvent(event)
      const text = extractEventText(event)
      if (role === null || text.length === 0) continue
      docs.push({
        kind: 'message',
        ref: `seq ${seq}`,
        text,
        title: `${role}: ${text.slice(0, 60)}`,
        role,
        blockId: block.blockId,
        tier: block.tier,
        tokens: defaultCountTokens(text),
      })
    }
  }
  searchDocsCache.set(events, docs)
  return docs
}

function handleSearch(env: ToolEnvironment, rawArgs: SearchArgs, exec: ToolRunContext): TextOutput {
  const args = unwrapEnvelope<SearchArgs>(rawArgs)
  const session = requireAgent(exec).session
  if (args.query.trim() === '') return { text: 'search_context: empty query (no matches)' }
  const docs = buildSearchDocs(session)
  // Trust the kernel: hybrid (0.7×BM25 stemmed + 0.3×fuzzy n-gram) is the
  // algorithm contract — no engine-side gate or threshold re-implements
  // search policy. Scores are surfaced so the model can judge a weak hit
  // (fuzzy-only tops out near 0.3).
  const results = searchBlocks(docs, args.query, { limit: args.limit ?? 5, previewLength: 160 })
  if (results.length === 0) return { text: `search_context: no matches for "${args.query}"` }
  const lines = results.map((r) => {
    const kind = r.kind === 'block' ? `block ${r.ref}` : `message ${r.ref} (${r.role ?? '?'}, in block ${r.blockId ?? '?'})`
    return `  - ${kind} (score ${r.score.toFixed(2)}): ${r.preview}`
  })
  return {
    text: `Matches for "${args.query}":\n${lines.join('\n')}\n\nDecompress with: ${callHint(env.toolCallStyle ?? 'native', 'decompress', '{ blockId }')}`,
  }
}

/** acp_status drilldown passthrough (kernel buildStatusReport options). All
 *  keys optional — no args = overview. `view`/`tool`/`sort`/`limit` only have
 *  meaning under `scope:"uncompressed"` (`tool` narrows to `view:"messages"`;
 *  `sort:"age"` applies to `scope:"compressed"`); the kernel ignores them in
 *  overview mode (upstream status-tool docstring documented the same scope).
 *  DSH schema compiler: `string` + `enum` supported, no `required: true`
 *  anywhere → all optional (schema.js:192-210). */
const statusParameters = {
  scope: {
    type: 'string' as const,
    enum: ['compressed', 'uncompressed'] as const,
    description: 'Drilldown scope: "compressed" lists compressed blocks, "uncompressed" lists visible messages. Omit for the overview.',
  },
  view: {
    type: 'string' as const,
    enum: ['ranges', 'messages'] as const,
    description: 'Drilldown view under scope:"uncompressed": "ranges" merges visible messages into ranges (default), "messages" lists every message.',
  },
  tool: {
    type: 'string' as const,
    description: 'Filter drilldown rows to one tool name (scope:"uncompressed" + view:"messages" only).',
  },
  sort: {
    type: 'string' as const,
    enum: ['size', 'time', 'tool', 'age'] as const,
    description: 'Row order: size (default, most tokens first), time, tool; "age" applies to compressed blocks.',
  },
  limit: {
    type: 'integer' as const,
    description: 'Cap on rows or blocks shown (default 30).',
  },
}

interface StatusArgs {
  scope?: 'compressed' | 'uncompressed'
  view?: 'ranges' | 'messages'
  tool?: string
  sort?: 'size' | 'time' | 'tool' | 'age'
  limit?: number
}


async function handleStatus(env: ToolEnvironment, rawArgs: StatusArgs, exec: ToolRunContext): Promise<TextOutput> {
  // The model channel may wrap ANY tool's args under `{ arguments: {…} }`;
  // peel it or drilldown params never reach buildStatusReport (live-verified
  // `{"arguments":{"scope":"compressed"}}` silently rendered the overview).
  const args = unwrapEnvelope<StatusArgs>(rawArgs)
  const agent = requireAgent(exec)
  const session = agent.session
  const state = env.store.stateFor(session)
  const surface = surfaceEventsOf(session)
  // One tool-call index for both projections below (P2-5): tool/result
  // toolName/toolCallId are backfilled from the assistant tool-calls.
  const toolNames = buildToolCallIndex(surface)
  const coreMessages = allLogMessages(session)
  const surfaceMessages = eventsToCoreMessages(surface, toolNames)
  const tokenCount = resolveTokenCount(agent, surfaceMessages)
  const window = await resolveEffectiveWindow(env, agent)
  const config = kernelConfigFor({ ...env, modelContextLimit: window.limit })
  // Run the same pipeline the context transform runs, so what acp_status
  // reports matches what the model actually receives. The returned turn.state
  // carries the freshly assigned refs; it is NOT persisted — acp_status is a
  // read-only view, and env.store.set would advance the nudge baseline a
  // second time in the same turn (design §6.1 P2-2).
  const turn = env.kernel.processTurn({ messages: coreMessages, state, config, tokenCount })
  // Status messages = visible surface EXCLUDING checkpoint summary nodes (P1-3).
  const statusMessages = eventsToCoreMessages(
    surface.filter((event) => isCheckpointNode(event) === false),
    toolNames,
  )
  // Upstream-aligned: the kernel renders the breakdown (percentages of the
  // VISIBLE total — no window semantics; drilldown scope/view/tool/sort/limit
  // pass through verbatim); the engine only appends the nudge decision line,
  // the DSH Surface anchor, and — in drilldown mode — the mN-vs-seq note.
  const report = buildStatusReport(turn.state, statusMessages, defaultCountTokens, args)
  const lines = [report]
  // Mirror upstream pi (`if (args.scope) return base`): a drilldown request
  // answers with the kernel report alone — the nudge decision line is an
  // overview concept. The Surface anchor stays in ALL modes: it is the model's
  // compressible-ref locator (design P2-1).
  if (args.scope === undefined) {
    const nudge = turn.nudge
    if (nudge !== undefined) {
      lines.push('', `Nudge: ${nudge.shouldInject ? 'ACTIVE' : 'idle'} — ${nudge.reason}`)
    }
    // Issue #60 P2: the model's only route to T2/T3 distillation is a LIVE
    // checkpoint seq — but acp_status (kernel buildStatusReport) is blind to
    // summary nodes (they are excluded as messages, rule 9) and shows only bN
    // refs. Append an engine-side mapping bN → checkpoint seq for ACTIVE
    // blocks (only active blocks are distillable). Appending is the
    // kernel-alignment contract: the kernel owns the report text, the engine
    // owns the wiring — this row is wiring, never a rewrite of the report.
    const checkpointRows = blockRegistry(session)
      .filter((entry) => entry.active && entry.summarySeq !== null)
      .map((entry) => `${entry.kernelBlockId} → seq ${entry.summarySeq}`)
    if (checkpointRows.length > 0) {
      lines.push('', `Checkpoint seqs (active blocks — compress a checkpoint seq to distill it): ${checkpointRows.join(', ')}`)
    }
    // Two calibers meet in this report: the pressure/nudge line is
    // provider-anchored (the session projection, where images and files carry
    // the live route's price) while the breakdown above is a text-only
    // estimate. They legitimately disagree on media-heavy sessions, and a model
    // reads that disagreement as a broken number unless it is told which is
    // which (issue #117). Emitted only when the surface actually carries media,
    // so a text-only session keeps the kernel report untouched.
    const mediaOnSurface = surface.some((event) => {
      const counts = attachmentsOfEvent(event)
      return counts.images + counts.files > 0
    })
    if (mediaOnSurface) {
      lines.push(
        '',
        'Note: the pressure line is provider-anchored (images/files priced by the live route); the breakdown above is a text-only estimate. They can differ on media-heavy sessions.',
      )
    }
  }
    // Provider-anchored billing: an engine append, never a rewrite of the kernel
    // report (rule 9). The two calibers a compression decision needs: cumulative
    // uncached vs cache-read, and peak vs total request input. A replace invalidates
    // the prefix from its rewrite point, so a compress that rewrites early re-reads
    // the whole remainder on the next request — that is the cost the model cannot
    // see otherwise.
    const usage = providerUsage(agent)
    if (usage !== undefined) {
      lines.push(
        '',
        `Provider usage (cumulative): ${usage.uncachedInputTokens} uncached input | ${usage.cacheReadTokens} cache read | ${usage.cacheWriteTokens} cache write | ${usage.outputTokens} output`,
      )
      lines.push('', `Total request input: ${totalRequestInput(usage)}`)
    }
    // Live vs peak vs room — the paper’s measurement vocabulary (arXiv 2611.05306
    // §2.4 separates total length from PEAK length, and names a "compaction
    // threshold"), made actionable here: a durable replace costs a near-full
    // re-prefill of whatever stays live, so the model cannot decide whether a
    // fold pays without knowing the live size and the remaining room. Read from
    // the log, so it works without a tokenMeter. Rule 9 used to keep window
    // semantics human-side only; measured fold pricing moved them to the model.
    const live = lastRequestInput(session)
    const peak = peakRequestInput(session)
    const contextLimit = env.windowFor !== undefined ? (await env.windowFor(agent)).limit : env.modelContextLimit
    const sizeParts: string[] = []
    if (live > 0) sizeParts.push(`live ${live}`)
    if (peak > 0) sizeParts.push(`peak ${peak}`)
    // Every fold leaves a permanent un-foldable summary barrier, so fold count is a
    // resource the model spends: more barriers means smaller eligible stretches
    // later, which is how a long session ends up refusing everything (Q5). Cheap
    // because the ledger is snapshot-cached (rule 14).
    const folds = rebuildBlockLedger(sessionEventsOf(session)).length
    if (folds > 0) {
      // A spent resource: more barriers means smaller eligible stretches later,
      // which is how a long session ends up refusing everything. No unlock number is
      // printed here — the round-4 reviews showed a fold count never measured the
      // debt it claimed (a barrier is a note, not a floor-sized hole), and whether
      // the engine may fold below the floor is read from the live rows instead.
      lines.push('', `Fold barriers: ${folds} (each splits the surface and can never be folded again) — fewer, larger folds preserve room for later folds`)
    }
    if (contextLimit > 0 && live > 0) sizeParts.push(`window ${contextLimit}`, `room ${Math.max(0, contextLimit - live)}`)
    if (sizeParts.length > 0) lines.push('', `Request input sizes: ${sizeParts.join(' | ')}`)
    // The working-set floor is what makes a fold REFUSE, but it was only visible inside a nudge —
    // and a nudge only fires above the pressure threshold, so on a mid-pressure session the model
    // sees neither the range table nor the floor and calls `condense` blind (round-15 review Q5).
    // Printed from the SAME `foldPricingFor` the gate uses, so the number the model reads is the
    // number the gate applies. The top candidate is deliberately NOT printed: computing it runs
    // `buildCompressibleSeqRanges`, whose orphan strip APPENDS durable prune events, and this tool
    // is read-only — a status call must not change the surface it is reporting on.
    const floorTokens = foldPricingFor({ ...env, modelContextLimit: contextLimit }).floorTokens
    if (floorTokens !== undefined && floorTokens > 0) {
      const close = live > 0 && live < floorTokens * 2
      // The direction matters: a fold leaves EXACTLY the room it does NOT remove, so near the
      // floor a WIDER span is the one that strands the working set — the same reasoning
      // `condenseFloorRefusal` carries (round-16 review Q1: this line told the model to widen
      // into the refusal the guard then produced).
      lines.push('', `Fold floor: a fold must leave at least ~${floorTokens} live tokens, below which compress and condense both refuse${live > 0 ? ` (live is ${live}${close ? ' — close to the floor: a fold leaves exactly the room it does NOT remove, so the SMALLER span is the one that stays above the floor; better still, stop compressing' : ''})` : ''}`)
    }
  lines.push('', `Surface: ${surfaceSummary(session)}`)
  // Drilldown rows carry kernel refs (mN, dense log-order ids) — compress
  // accepts them directly (handleCompress reverse-maps mN → live surface seq
  // via the current turn's messageRefs.byRef; issue #31). The Surface anchor
  // remains the model's compressible-seq locator for nudge-style ranges.
  if (args.scope === 'uncompressed') {
    lines.push('', 'Note: drilldown rows are kernel refs (mN) — feed them straight to compress (auto-mapped to the live surface seq); an unknown mN fails with guidance.')
  }
  return { text: lines.join('\n') }
}

/** Build the four ACP model tools bound to one engine. */
export function makeTools(env: ToolEnvironment): ToolDefinition[] {
  const prompts = env.prompts ?? DEFAULT_RESOLVED
  return [
    defineTool({
      name: 'compress',
      description: prompts.tools.compress,
      parameters: compressParameters,
      output: textOutput(),
      async execute(args, exec) {
        return handleCompress(env, args as CompressArgs, exec)
      },
    }),
    defineTool({
      name: 'decompress',
      description: prompts.tools.decompress,
      parameters: decompressParameters,
      output: textOutput(),
      execute(args, exec) {
        return Promise.resolve(handleDecompress(env, args as DecompressArgs, exec))
      },
    }),
    defineTool({
      name: 'search_context',
      description: prompts.tools.searchContext,
      parameters: searchParameters,
      output: textOutput(),
      execute(args, exec) {
        return Promise.resolve(handleSearch(env, args as SearchArgs, exec))
      },
    }),
    defineTool({
      name: 'acp_status',
      description: prompts.tools.acpStatus,
      parameters: statusParameters,
      output: textOutput(),
      execute(args, exec) {
        return handleStatus(env, args as StatusArgs, exec)
      },
    }),
  ]
}
