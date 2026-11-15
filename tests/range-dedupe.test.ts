/**
 * round-16 review Q2 — ONE seq range can arrive through several kernel ref edges (a
 * multi-call assistant node resolves through its `${seq}#${callId}` SUB-ids), so the same span
 * reaches the caller more than once. The dedupe lives INSIDE `buildCompressibleSeqRanges`, the
 * single source every caller reads, because `compactForOverflow` sums `remainingReclaim` over
 * that list: a span counted twice inflated the "still reachable" projection, the
 * unreachable-remainder early stop never fired, and the loop burned permanent barriers for a
 * number that could not move. `range !== best` cannot catch it — duplicates are distinct objects.
 */
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { Session } from '@deepseek-ai/dsh-session'
import { buildCompressibleSeqRanges } from '../src/region.ts'
import {
  appendAssistant,
  appendMultiToolCall,
  appendToolResult,
  appendTurn,
  appendUser,
  longText,
  wholeSurfaceRangeView,
} from './helpers.ts'

function workSession(id: string): Session {
  const session = Session.create(id)
  appendTurn(session, 1)
  appendUser(session, longText('u1', 1) + longText('u1b', 2))
  appendMultiToolCall(session, longText('a2', 3), ['c1', 'c2'])
  appendToolResult(session, longText('r3', 4), 'c1')
  appendToolResult(session, longText('r4', 5), 'c2')
  appendUser(session, longText('u5', 6) + longText('u5b', 7))
  appendAssistant(session, longText('a6', 8))
  appendUser(session, longText('u7', 9))
  appendAssistant(session, longText('a8', 10))
  appendUser(session, longText('u9', 11))
  appendAssistant(session, longText('a10', 12))
  appendUser(session, longText('u11', 13))
  appendAssistant(session, longText('a12', 14))
  return session
}

test('one seq range arriving through two ref edges is offered exactly once', () => {
  const session = workSession('range-dedupe')
  const base = wholeSurfaceRangeView(session)
  const edge = base.ranges[0]!
  const subRef = `${edge.startRef}#c1`
  const view = {
    ranges: [edge, { startRef: subRef, endRef: edge.endRef }],
    refs: { byRef: { ...base.refs.byRef, [subRef]: base.refs.byRef[edge.startRef]! } },
  }
  const ranges = buildCompressibleSeqRanges(session, view, { foldPricing: { floorTokens: 16_000 } })
  const spans = ranges.map((range) => `${range.start}..${range.end}`)
  assert.ok(spans.length >= 1, 'the fixture offers a real range')
  assert.equal(spans.length, new Set(spans).size, `no duplicate span survives the source: ${spans.join(', ')}`)
  // The point of deduping at the source: a reclaim sum over these rows cannot count a span
  // twice, which is exactly what `compactForOverflow`'s `remainingReclaim` does.
  const summed = ranges.reduce((total, range) => total + range.tokens, 0)
  assert.equal(summed, ranges[0]!.tokens, 'two input edges for one span contribute their tokens ONCE to a reclaim sum')
})
