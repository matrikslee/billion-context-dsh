/**
 * Superseded nudge rows are the engine's own largest per-step writer (measured
 * 2026-11-15: 37 rows = 89,790 bytes permanently resident in one session).
 * The sweep keeps the newest row and prunes the rest.
 */
import test from 'node:test'
import assert from 'node:assert/strict'
import { Session } from '@deepseek-ai/dsh-session'
import { pruneStaleNudgeRows, staleNudgeRowsOf, stripOrphanedSurfaceToolMessages } from '../src/region.ts'
import { eventAtOf } from '../src/session-events.ts'
import { isAcpNudgeRow } from '../src/messages.ts'
import { appendAssistant, appendNudge, appendToolCall, appendToolResult, appendTurn, appendUser, buildShortTextSession, longText } from './helpers.ts'
import { hostPriceEvent } from '../src/host-tokens.ts'
import { sessionEventsOf } from '../src/session-events.ts'

function visibleNudgeSeqs(session: Session): number[] {
  const out: number[] = []
  for (const seq of session.surface.nodes) {
    const event = eventAtOf(session, seq)
    if (event !== undefined && isAcpNudgeRow(event)) out.push(seq)
  }
  return out
}

test('the newest nudge survives and every older one is pruned', () => {
  const session = buildShortTextSession(6)
  appendNudge(session, longText('pressure table one', 1))
  appendAssistant(session, 'work between the rows', 2, 1)
  appendNudge(session, longText('pressure table two', 2))
  appendNudge(session, longText('pressure table three', 3))
  assert.equal(visibleNudgeSeqs(session).length, 3)
  // Only the row the model has already answered is gone: the two nudges after the
  // last assistant turn are still unread (normal + emergency of one step).
  assert.equal(pruneStaleNudgeRows(session, 0), 1)
  const left = visibleNudgeSeqs(session)
  assert.equal(left.length, 2, 'the two UNREAD rows stay — they were never answered')
})

test('a single nudge is never pruned — it is the live seqs', () => {
  const session = buildShortTextSession(6)
  appendNudge(session, longText('only table', 4))
  assert.equal(staleNudgeRowsOf(session, 0).length, 0)
  assert.equal(pruneStaleNudgeRows(session, 0), 0)
  assert.equal(visibleNudgeSeqs(session).length, 1)
})

test('rows inside the recent-protection window stay visible', () => {
  const session = buildShortTextSession(6)
  appendNudge(session, longText('one', 5))
  appendNudge(session, longText('two', 6))
  appendNudge(session, longText('three', 7))
  assert.equal(staleNudgeRowsOf(session, 20).length, 0, 'the window covers every nudge row')
  assert.equal(pruneStaleNudgeRows(session, 20), 0)
})

test('the sweep is idempotent and leaves real turns alone', () => {
  const session = buildShortTextSession(6)
  appendNudge(session, longText('one', 8))
  appendAssistant(session, 'the model answered the request that carried it', 2, 1)
  appendUser(session, 'a real user turn')
  appendNudge(session, longText('two', 9))
  assert.equal(pruneStaleNudgeRows(session, 0), 1)
  assert.equal(pruneStaleNudgeRows(session, 0), 0, 'second sweep finds nothing')
  const dumped = session.surface.nodes.map((seq) => JSON.stringify(eventAtOf(session, seq) ?? null)).join('')
  assert.ok(dumped.includes('a real user turn'), 'the real turn is untouched')
  assert.ok(dumped.includes('superseded pressure nudge'), 'the pruned row carries the tombstone')
})

test('a nudge between a tool call and its result no longer deletes the pair', () => {
  // Rule 8 used to treat any row between the call and its result as a broken pair
  // and prune BOTH ends. A nudge we injected there is our own garbage: the sweep
  // must drop the nudge and keep the real tool result (round-8 Q3, ship-blocking).
  const session = buildShortTextSession(4)
  appendToolCall(session, 'do the thing', 'c1', 2, 1)
  appendNudge(session, longText('pressure table injected mid-step', 10))
  appendToolResult(session, 'the real tool output the model still needs', 'c1', 2, 1)
  const dropped = stripOrphanedSurfaceToolMessages(session)
  const dumped = session.surface.nodes.map((seq) => JSON.stringify(eventAtOf(session, seq) ?? null)).join('')
  assert.ok(dumped.includes('the real tool output the model still needs'), 'the tool result survives')
  assert.ok(dumped.includes('do the thing'), 'the call node survives')
  assert.ok(!dumped.includes('pressure table injected mid-step'), 'the nudge between them is gone')
  assert.ok(dropped >= 1, 'the sweep reports the nudge it removed')
})

test('the prune claim speaks the host estimator (rule 12)', () => {
  const session = buildShortTextSession(4)
  appendNudge(session, longText('context pressure 上下文压力提醒 ranges', 11))
  appendAssistant(session, 'answered', 2, 1)
  appendNudge(session, 'second table')
  const before = sessionEventsOf(session)
  assert.equal(pruneStaleNudgeRows(session, 0), 1)
  const events = sessionEventsOf(session)
  const prune = events.slice(before.length).find((event) => event.type === 'compaction/prune')
  assert.ok(prune !== undefined, 'the sweep writes a durable prune claim')
  const shadowed = (prune as { data: { shadowedSeqs: number[]; shadowedTokenCount: number } }).data
  const original = eventAtOf(session, shadowed.shadowedSeqs[0]!)!
  assert.equal(shadowed.shadowedTokenCount, hostPriceEvent(original), 'claim == host price of that exact row')
})

test('a row too short to pay for its tombstone is left alone', () => {
  // Measured: a ~150-byte synthetic nudge costs MORE to prune than to keep.
  const session = buildShortTextSession(4)
  appendNudge(session, 'tiny table')
  appendAssistant(session, 'answered', 2, 1)
  appendNudge(session, 'second row')
  assert.equal(pruneStaleNudgeRows(session, 0), 0)
  assert.equal(visibleNudgeSeqs(session).length, 2)
})


test('a foreign result between a call and its result is still detected', () => {
  // The adjacency loop must check EVERY node between a call and its result. Stepping
  // over the node after a nudge stopped the scan at the nudge, so a foreign result in the
  // middle went unchecked, the pair read as adjacent, and the unserialisable shape
  // stayed on the surface (round-9 review, Gemini finding 1).
  const session = Session.create('nudge-foreign-result')
  const last = (): number => session.surface.nodes[session.surface.nodes.length - 1]!
  appendTurn(session, 1)
  appendToolCall(session, longText('call A', 0), 'call-a', 1, 1)
  const callA = last()
  appendNudge(session, longText('nudge between the pair', 1))
  const nudgeSeq = last()
  appendToolResult(session, longText('a result belonging to another call', 2), 'call-b', 1, 2)
  const foreign = last()
  appendToolResult(session, longText('result A', 3), 'call-a', 1, 3)
  const resultA = last()
  const removed = stripOrphanedSurfaceToolMessages(session)
  const visible = new Set(session.surface.nodes)
  assert.ok(removed >= 1, 'something is pruned: ' + removed)
  assert.ok(!visible.has(resultA), 'the non-adjacent result is pruned, not accepted as adjacent')
  assert.ok(!visible.has(nudgeSeq), 'the nudge row inside the pair is hidden')
  assert.ok(!visible.has(foreign), 'the orphaned foreign result is hidden')
  assert.ok(!(visible.has(callA) && visible.has(resultA)), 'a broken pair never stays visible on both ends')
})
