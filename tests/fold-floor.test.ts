/**
 * Working-set floor (gap G1, arXiv 2611.05306 Appendix D).
 *
 * The paper’s open-weight band is this machine’s biography: at the 16K/12K
 * regime Qwen 3 Coder Next fell 53.0 → 3.6, Gemma 4 31B 65.2 → 58.0, Step 3.7
 * Flash 70.0 → 56.6, and EVERY model lost — diagnosed as too little room to make
 * progress between compactions, not summary quality. So a fold that leaves a
 * shallow working set is refused (advisory in the table, hard in condense), and the
 * refusal points at the row that STILL LEAVES the floor: a fold leaves exactly as
 * much room as it does not remove, so the widest row is the one that strands it.
 * When no row can keep the floor, condense degrades instead of deadlocking.
 */

import { test } from 'node:test'
import assert from 'node:assert/strict'
import { createCore, type CompressionCore } from 'acp-kernel'
import { Session } from '@deepseek-ai/dsh-session'
import { AcpStateStore } from '../src/state.ts'
import { condenseDegradable, condenseFloorRefusal, makeTools, type ToolEnvironment } from '../src/tools.ts'
import {
  foldPricingFor,
  WORKING_SET_FLOOR_MAX_TOKENS,
  WORKING_SET_FLOOR_MIN_TOKENS,
} from '../src/config.ts'
import { buildCompressibleSeqRanges, DEFAULT_FOLD_FLOOR_PCT } from '../src/region.ts'
import { rangeTable } from '../src/nudge.ts'
import { appendAssistant, appendTurn, appendUser, longText, wholeSurfaceRangeView } from './helpers.ts'

const NOTE = ['## Goal', 'Finish the fold-floor guard.', '## Next', 'Run the suite; 0 fail is the completion condition.'].join('\n')

function envWith(limit: number, foldFloorPct?: number): ToolEnvironment {
  return {
    kernel: createCore({}) as CompressionCore,
    store: new AcpStateStore(),
    modelContextLimit: limit,
    ...(foldFloorPct === undefined ? {} : { foldFloorPct }),
    compressCallIdsToHide: new Set(),
  }
}

function execStub(session: Session): never {
  return {
    callId: 'call_1_condense',
    name: 'compress',
    arguments: {},
    signal: new AbortController().signal,
    agent: { id: session.id, session, options: { provider: 'p', model: 'm' }, ctx: {} } as never,
  }
}

/** A surface big enough to clear the character gate but small in tokens. */
function workFixture(id: string): Session {
  const session = Session.create(id)
// condense protects the recent tail (5) and the last real user turn, so the
// foldable stretch needs its own nodes AHEAD of that window, each big enough
// that the run clears the 5000-CHARACTER kernel gate.
  appendTurn(session, 1)
  appendUser(session, longText('task', 3) + longText('task2', 3))
  appendAssistant(session, longText('work', 3) + longText('work2', 3))
  appendUser(session, longText('ask', 3) + longText('ask2', 3))
  appendAssistant(session, longText('mid', 3) + longText('mid2', 3))
  appendUser(session, longText('tail1', 2))
  appendAssistant(session, longText('tail2', 2))
  appendUser(session, longText('tail3', 2))
  appendAssistant(session, longText('tail4', 2))
  return session
}

test('a surface smaller than the floor is never guarded — the floor needs room to keep', async () => {
  const session = workFixture('floor-room')
  // Real window, default pct ⇒ floor 39322 tokens, far above this 12392-token
  // surface. Guarding here would refuse every compress on a short session; the
  // floor protects a working set that could have been kept, not a wish.
  const definition = makeTools(envWith(262144)).find((candidate) => candidate.name === 'compress')
  const result = await definition!.execute({ condense: { summary: NOTE } } as never, execStub(session) as never)
  assert.match((result as { text: string }).text, /condense elected seqs/, 'no refusal when the floor exceeds the whole surface')
  assert.equal(session.snapshotEvents().filter((event) => event.type === 'compaction/summary').length, 1)
})

test('a model-named range is guarded by the same floor as an elected one', async () => {
  const session = workFixture('floor-explicit')
  const definition = makeTools(envWith(16000, 0.5)).find((candidate) => candidate.name === 'compress')
  const result = await definition!.execute({ content: [{ startSeq: 1, endSeq: 3, summary: NOTE }] } as never, execStub(session) as never)
  const text = (result as { text: string }).text
  assert.match(text, /under the working-set floor/, 'the explicit range cannot slip past the guard')
  assert.match(text, /widen blindly/i, 'refusal explains that widening is not automatically valid')
  assert.equal(session.snapshotEvents().filter((event) => event.type === 'compaction/summary').length, 0, 'nothing written')
})

test('the floor is window × foldFloorPct, clamped to 8K–16K tokens', () => {
  // 0.15 of 262144 is 39322 — above the 16K ceiling (one KV block), because a bigger forced
  // working set squeezes the foldable window against the guarded tail.
  assert.equal(foldPricingFor({ modelContextLimit: 262144 }).floorTokens, WORKING_SET_FLOOR_MAX_TOKENS)
  assert.equal(foldPricingFor({ modelContextLimit: 262144, foldFloorPct: 0.25 }).floorTokens, WORKING_SET_FLOOR_MAX_TOKENS)
  // Inside the band the percentage rules.
  assert.equal(foldPricingFor({ modelContextLimit: 100000, foldFloorPct: DEFAULT_FOLD_FLOOR_PCT })?.floorTokens, 15000)
  // A small window still keeps a workable working set (the paper’s 7–9K peak band).
  assert.equal(foldPricingFor({ modelContextLimit: 32768 }).floorTokens, WORKING_SET_FLOOR_MIN_TOKENS)
  assert.equal(foldPricingFor({ modelContextLimit: 262144, foldFloorPct: 0 }).floorTokens, undefined, 'pct 0 disables the guard')
})

test('a row that would strand the working set is marked in the table', () => {
  const session = workFixture('floor-row')
  const ranges = buildCompressibleSeqRanges(session, wholeSurfaceRangeView(session), {
    // The fixture surface totals 12392 tokens and the elected fold leaves 6197, so
    // 8000 is under the surface (guardable) and over the remainder (below floor).
    // A floor ABOVE the whole surface must NOT guard — see belowWorkingSetFloor.
    foldPricing: { floorTokens: 8000 },
  })
  assert.ok(ranges.length > 0, 'the fixture yields a range')
  assert.ok(ranges.every((range) => range.belowFloor), `every row lands under the floor: ${JSON.stringify(ranges.map((r) => r.liveAfterFoldTokens))}`)
  const table = rangeTable(session, wholeSurfaceRangeView(session), undefined, undefined, { floorTokens: 8000 })
  assert.match(table, /below floor/, 'the row warns before the model picks it')
  assert.match(table, /STILL LEAVES the floor/, 'and points at the row that is safe, not the widest one')
})

test('condense degrades when no eligible row can keep the floor, and stamps it', async () => {
  const session = workFixture('floor-refuse')
  // floor clamps to 8192: over the ~6197 that would stay, under the 12392 surface.
  // Every row of this fixture is below the floor, so there is NO compliant fold —
  // refusing here is the deadlock the round-3 reviews reported.
  const definition = makeTools(envWith(16000, 0.5)).find((candidate) => candidate.name === 'compress')
  const result = await definition!.execute(
    { condense: { summary: NOTE } } as never,
    execStub(session) as never,
  )
  const text = (result as { text: string }).text
  assert.match(text, /condense elected seqs/, 'the only available move is taken')
  assert.match(lastSummaryText(session), /\[dsh:degraded\]/, 'one machine-readable provenance line')
  assert.match(lastSummaryText(session), /no eligible row could keep the working-set floor/, 'states why it is desperate')
  assert.equal(session.snapshotEvents().filter((event) => event.type === 'compaction/summary').length, 1, 'one summary written')
})

test('force folds it anyway — overflow pressure is a legitimate override', async () => {
  const session = workFixture('floor-force')
  const definition = makeTools(envWith(16000, 0.5)).find((candidate) => candidate.name === 'compress')
  const result = await definition!.execute(
    { condense: { summary: NOTE, force: true } } as never,
    execStub(session) as never,
  )
  const text = (result as { text: string }).text
  assert.match(text, /condense elected seqs/, 'the forced fold lands')
  assert.equal(session.snapshotEvents().filter((event) => event.type === 'compaction/summary').length, 1, 'one summary written')
})

test('the refusal separates the three situations it can mean', () => {
  const pick = {
    start: 1, end: 4, count: 4, tokens: 9000, foldCostTokens: 3000, liveAfterFoldTokens: 2000,
    belowFloor: true, guardedRowCount: 3, eligibleRangeCount: 2,
  }
  const compliant = condenseFloorRefusal({ ...pick, aboveFloorAvailable: true }, 8000)
  assert.match(compliant, /AN ELIGIBLE ROW STILL LEAVES THE FLOOR/)
  assert.match(compliant, /widest row is the one that strands/, 'names the direction: wider is the danger, not the fix')
  assert.doesNotMatch(compliant, /NO ELIGIBLE ROW LEAVES THE FLOOR/)
  // Guards fragment the area: telling this session to stop compressing is wrong,
  // and telling it to widen is a loop into the guard.
  const fragmented = condenseFloorRefusal({ ...pick, aboveFloorAvailable: false }, 8000)
  assert.match(fragmented, /NO ELIGIBLE ROW LEAVES THE FLOOR/)
  assert.match(fragmented, /Do NOT widen blindly/)
  assert.match(fragmented, /force:true/, "names the only compliant escape")
  assert.doesNotMatch(fragmented, /AN ELIGIBLE ROW STILL LEAVES/)
  const nothing = condenseFloorRefusal({ ...pick, aboveFloorAvailable: false, eligibleRangeCount: 0 }, 8000)
  assert.match(nothing, /NOTHING ELSE IS ELIGIBLE AT ALL/)
  assert.match(nothing, /stop compressing and keep working/)
})

test('degrading needs the absence of a compliant row, not a fold count', () => {
  const base = {
    start: 1, end: 4, count: 4, tokens: 9000, foldCostTokens: 3000, liveAfterFoldTokens: 2000,
    belowFloor: true, aboveFloorAvailable: false, guardedRowCount: 3, eligibleRangeCount: 2, degraded: false,
  }
  assert.equal(condenseDegradable({ ...base, aboveFloorAvailable: true }, 4096), false,
    'a row that keeps the floor exists — fold it, do not strand the working set')
  assert.equal(condenseDegradable(base, 4096), true, "nothing compliant left, so the fold is the only move")
  assert.equal(condenseDegradable({ ...base, tokens: 3000 }, 4096), false,
    'under MIN_DEGRADED_FOLD_TOKENS the barrier buys almost nothing — refuse')
  assert.equal(condenseDegradable({ ...base, belowFloor: false }, 4096), false, "compliant folds never take the degraded path")
})


function lastSummaryText(session: Session): string {
  const events = session.snapshotEvents().filter((event) => event.type === 'compaction/summary')
  const data = events.at(-1)?.data as { summary?: { text?: string }[] } | undefined
  return data?.summary?.map((block) => block.text ?? '').join('\n') ?? ''
}

test('a compliant row keeps the refusal, so the stamp never rides a voluntary fold', async () => {
  const session = workFixture('floor-force')
  // force is the operator's own desperation and must NOT be labelled as the
  // engine's degraded election — a floor statistic reading the stamp would count
  // a forced fold as one the rows left no choice about.
  const definition = makeTools(envWith(16000, 0.5)).find((candidate) => candidate.name === 'compress')!
  await definition.execute({ condense: { summary: NOTE, force: true } } as never, execStub(session) as never)
  assert.doesNotMatch(lastSummaryText(session), /\[dsh:degraded\]/, 'a forced fold carries no degrade provenance')
})