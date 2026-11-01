/**
 * PTC / code-mode awareness (src/ptc.ts).
 *
 * Two contracts, both cheap to break and expensive to break in the field:
 * 1. native mode stays BYTE-IDENTICAL — the PTC work must not touch a single
 *    byte of the copy a native session reads (the same anchor tests/prompts.test.ts
 *    holds for the template migration);
 * 2. in PTC mode every model-visible call shape is the `run_code` SDK form, the
 *    kernel nudge path is still the kernel path, and the rewrite is idempotent.
 */

import { test } from 'node:test'
import assert from 'node:assert/strict'
import { createCore, type CompressionCore, type NudgeDecision } from 'acp-kernel'
import { buildTextSession, wholeSurfaceRangeView } from './helpers.ts'
import { buildNudgeText } from '../src/nudge.ts'
import { DEFAULT_PROMPTS, renderSystemPrompt, resolvePrompts } from '../src/prompts.ts'
import {
  PTC_FOOTER_NOTICE,
  PTC_MARKERS,
  PTC_SYSTEM_LINE,
  TOOL_CALL_MARK,
  callHint,
  detectPtcMode,
  looksLikeLeakedAcpCall,
  ptcText,
  toPtcCallSyntax,
  toPtcPrompts,
} from '../src/ptc.ts'

/** The host's two PTC texts, quoted verbatim from `dsh-tool-executor-ptc`
 *  (lib/index.js:1704 / :916 TS SDK / :1639 Python SDK). */
const PTC_ONLY = 'Because `run_code` is the only tool you can call directly — a tool call naming any other tool fails.'
const TS_SDK = '## Writing code for run_code\nProgram-only SDK bindings: `tools` exposes every registered tool.'
const PY_SDK = '## Writing code for run_code\nThe available tools: `tools.compress`.'
const NATIVE_PROMPT = 'Active Context Pruning — model-driven context management\n\nYOU decide whether and when to compress context.'

/** Built by concatenation so no literal markup appears in this source (the same
 *  reason src/ptc.ts builds it). */
const leaked = (name: string): string => `${TOOL_CALL_MARK}\n<function=${name}>\n<parameter=content>\n[]\n</parameter>\n</function>\n</tool_call>`

function fakeDecision(pct: number, emergency: boolean): NudgeDecision {
  return {
    shouldInject: true,
    reason: 'probe',
    compressibleRanges: [],
    tierTargetBlocks: [],
    contextUsage: pct / 100,
    tier: null,
    breakdown: {
      usage: pct / 100,
      growth: 0,
      growthReference: 0,
      effectiveThreshold: 0,
      nudgeGrowthTokens: 50000,
      growthFloor: 20000,
      hasPendingNudge: 0,
      overLimit: emergency ? 1 : 0,
      emergencyOverride: emergency ? 1 : 0,
      pendingT1: 0,
      pendingT2: 0,
      pendingT3: 0,
    },
  } as never
}

test('PTC 1: detection reads the host constants (TS SDK, Python SDK) and never a native prompt', () => {
  assert.equal(detectPtcMode([PTC_ONLY]), true, 'PTC_ONLY_INSTRUCTION marker')
  assert.equal(detectPtcMode([TS_SDK]), true, 'TS SDK instructions header')
  assert.equal(detectPtcMode([PY_SDK]), true, 'Python SDK variant carries the same header')
  assert.equal(detectPtcMode([NATIVE_PROMPT]), false, 'native system prompt is not PTC')
  assert.equal(detectPtcMode([]), false, 'empty surface is native')
  assert.equal(detectPtcMode(['', '', TS_SDK] as Iterable<string>), true, 'marker anywhere in the scan set wins')
  assert.equal(PTC_MARKERS.length, 2, 'exactly the two host constants')
})

test('PTC 2: native mode is byte-identical — nothing the PTC work adds leaks into it', () => {
  const rendered = renderSystemPrompt(resolvePrompts())
  assert.ok(rendered.startsWith('Active Context Pruning — model-driven context management'), 'native head unchanged')
  assert.ok(!rendered.includes('await tools.'), 'no SDK form in the native system prompt')
  assert.ok(!rendered.includes(PTC_SYSTEM_LINE.slice(0, 40)), 'no PTC convention line in the native system prompt')
  assert.ok(!DEFAULT_PROMPTS.tools.compress.includes('await tools.'), 'native compress description unchanged')
  assert.ok(DEFAULT_PROMPTS.tools.compress.includes('Single range: compress({ content'), 'native call shape intact')
  assert.ok(DEFAULT_PROMPTS.rangeTable.footer.startsWith('Compress with: compress({'), 'native footer shape intact')
  assert.equal(ptcText(rendered, 'native'), rendered, 'ptcText is identity in native mode')
})

test('PTC 3: every model-visible call shape becomes the run_code SDK form', () => {
  const ptc = toPtcPrompts(DEFAULT_PROMPTS)
  assert.ok(ptc.systemPromptTemplate.startsWith(PTC_SYSTEM_LINE), 'convention stated once at the top of the system prompt')
  assert.ok(ptc.tools.compress.includes('await tools.compress({'), 'compress description carries the SDK form')
  // The native decompress description names no call shape at all, so there is
  // nothing to rewrite there — its SDK form is stated by the convention line.
  assert.equal(ptc.tools.decompress, DEFAULT_PROMPTS.tools.decompress, 'decompress description has no call shape to rewrite')
  assert.ok(ptc.systemPromptTemplate.includes('await tools.decompress({ blockId })'), 'decompress SDK form stated in the convention line')
  assert.ok(ptc.tools.searchContext.includes('await tools.search_context('), 'search_context description carries the SDK form')
  assert.ok(ptc.rangeTable.footer.startsWith(PTC_FOOTER_NOTICE), 'range-table footer names the reachable shape first')
  // No bare direct-call shape survives anywhere in the PTC copy.
  for (const text of [
    ptc.systemPromptTemplate,
    ptc.tools.compress,
    ptc.tools.decompress,
    ptc.tools.searchContext,
    ptc.rangeTable.footer,
  ]) {
    assert.ok(!/(?<!tools\.)\b(?:compress|decompress|search_context|acp_status)\(\{/.test(text), 'no direct-call shape left')
  }
  // The nudge group keeps its object identity so the kernel renderer still runs.
  assert.equal(ptc.nudge, DEFAULT_PROMPTS.nudge, 'nudge group identity preserved (kernel path stays on)')
})

test('PTC 4: the rewrite is idempotent and placeholder-safe', () => {
  const once = toPtcCallSyntax(DEFAULT_PROMPTS.systemPromptTemplate)
  assert.equal(toPtcCallSyntax(once), once, 'double rewrite does not stack `await tools.`')
  assert.ok(!once.includes('await tools.await tools.'), 'no double prefix')
  // renderTemplate substitutes `{identifier}`; PTC copy must not mint any.
  const rendered = renderSystemPrompt(toPtcPrompts(DEFAULT_PROMPTS))
  assert.ok(rendered.includes('Compression Philosophy:'), 'kernel philosophy still injected')
  assert.ok(rendered.includes('await tools.compress({'), 'rendered PTC prompt carries the SDK form')
  assert.ok(!rendered.includes('{philosophy}'), 'no unresolved placeholder')
})

test('PTC 5: the nudge switches shape but keeps the kernel frame (rule 9)', () => {
  const session = buildTextSession(12)
  const view = wholeSurfaceRangeView(session)
  const native = buildNudgeText(fakeDecision(50, false), false, session, view)
  const ptc = buildNudgeText(fakeDecision(50, false), false, session, view, undefined, undefined, 'ptc')
  assert.ok(native.startsWith('This is an efficiency nudge to compress early and keep context lean'), 'native kernel frame')
  assert.ok(ptc.startsWith('This is an efficiency nudge to compress early and keep context lean'), 'PTC keeps the kernel frame verbatim')
  assert.ok(native.includes('compress({ content: [{ startSeq, endSeq, summary }] })'), 'native array-form note names the direct shape')
  assert.ok(!native.includes('await tools.'), 'native nudge carries no SDK form')
  assert.ok(ptc.includes('await tools.compress({'), 'PTC array-form note names the SDK shape')
  assert.ok(!/(?<!tools\.)\bcompress\(\{/.test(ptc), 'no direct shape left in the PTC nudge')
  assert.ok(!ptc.includes('PTC mode: call it inside'), 'the footer notice is not duplicated into the nudge twice')
})

test('PTC 6: a leaked direct call gets one correction line on the nudge', () => {
  assert.equal(looksLikeLeakedAcpCall(leaked('compress')), true, 'leaked markup naming compress')
  assert.equal(looksLikeLeakedAcpCall(leaked('acp_status')), true, 'leaked markup naming acp_status')
  assert.equal(looksLikeLeakedAcpCall('compressing ranges is cheap'), false, 'prose is not a leak')
  assert.equal(looksLikeLeakedAcpCall('a bash command ran'), false, 'unrelated text is not a leak')

  const session = buildTextSession(4)
  const view = wholeSurfaceRangeView(session)
  const withNote = buildNudgeText(fakeDecision(7, false), false, session, view, undefined, undefined, 'ptc', true)
  assert.ok(withNote.endsWith('inside `run_code` (`await tools.compress({ content: [...] })`).'), 'correction appended last')
  const withoutNote = buildNudgeText(fakeDecision(7, false), false, session, view, undefined, undefined, 'ptc', false)
  assert.ok(!withoutNote.includes('Correction:'), 'no correction when nothing leaked')
  // Native sessions never get it — the direct shape works there.
  const native = buildNudgeText(fakeDecision(7, false), false, session, view, undefined, undefined, 'native', true)
  assert.ok(!native.includes('Correction:'), 'native ignores the leak flag')
})

test('PTC 7: tool-result hints follow the mode', () => {
  assert.equal(callHint('native', 'compress', '{ content: [...] }'), 'compress({ content: [...] })')
  assert.equal(callHint('ptc', 'compress', '{ content: [...] }'), 'await tools.compress({ content: [...] })')
  assert.equal(callHint('ptc', 'decompress', '{ blockId }'), 'await tools.decompress({ blockId })')
  assert.equal(callHint('native', 'decompress', '{ blockId }'), 'decompress({ blockId })')
})

/** Engine wiring: a pinned config applies at mount; `auto` starts native and
 *  flips when a PTC session is seen (the flip is what re-registers the
 *  descriptions, so the SDK face the model reads cannot lag the nudge copy). */
test('PTC 8: toolCallStyle config pins the mode at mount', async () => {
  const { Context } = await import('@deepseek-ai/cordis')
  const { AcpCompactionEngine } = await import('../src/index.ts')
  const auto = new AcpCompactionEngine(new Context(), {})
  assert.equal(auto.env.toolCallStyle, 'native', 'auto starts native until a session proves otherwise')
  assert.equal(auto.env.prompts, DEFAULT_PROMPTS, 'auto mount serves the native copy')
  const pinned = new AcpCompactionEngine(new Context(), { toolCallStyle: 'ptc' })
  assert.equal(pinned.env.toolCallStyle, 'ptc', 'a pinned ptc config applies at mount')
  assert.ok(pinned.env.prompts?.tools.compress.includes('await tools.compress({'), 'pinned mount registers PTC descriptions')
  assert.equal(pinned.prompts, DEFAULT_PROMPTS, 'the composition prompts stay native; only the applied set is the variant')
})
