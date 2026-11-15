/**
 * A condense runs INSIDE the step that called it, so its own tool call is
 * still open on the surface.
 *
 * Measured 2026-11-15 in the e2e scenario condense-handoff: the assistant node
 * carrying the compress call sits on the surface while its result has not landed
 * yet. The range table ran its orphan strip with an empty protection set, so that
 * strip pruned the node the caller had just read. The kernel range then mapped to
 * `to === undefined`, every row vanished, and condense answered
 * "nothing contiguous enough to fold" on EVERY real session — the whole
 * engine-elected path was dead in the host while unit fixtures stayed green
 * (rule 5: a fixture without an open call cannot see this).
 */

import { test } from 'node:test'
import assert from 'node:assert/strict'
import { Session } from '@deepseek-ai/dsh-session'
import { appendAssistant, appendToolCall, appendToolResult, appendTurn, appendUser, longText, wholeSurfaceRangeView } from './helpers.ts'
import { buildCompressibleSeqRanges, openToolCallIds } from '../src/region.ts'

/** A worked-over surface, then the live compress call with its result still open. */
const inflightFixture = (id: string): Session => {
  const session = Session.create(id)
  appendTurn(session, 1)
  appendUser(session, longText('u1', 1))
  appendAssistant(session, longText('a2', 2))
  appendUser(session, longText('u3', 3))
  appendAssistant(session, longText('a4', 4))
  appendUser(session, longText('u5', 5))
  appendAssistant(session, longText('a6', 6))
  appendUser(session, longText('u7', 7))
  appendAssistant(session, longText('a8', 8))
  appendToolCall(session, longText('a9', 9), 'live-call')
  return session
}

test('the range table never prunes the call it was invoked from', () => {
  const session = inflightFixture('inflight-open')
  const nodes = session.surface.nodes
  const openSeq = nodes[nodes.length - 1]!
  assert.ok(openToolCallIds(session).has('live-call'), 'the fixture really is an in-flight call')
  buildCompressibleSeqRanges(session, wholeSurfaceRangeView(session), { preserveRecent: 5 })
  const after = session.surface.nodes
  assert.ok(after.includes(openSeq), 'the open call node survived the strip: ' + JSON.stringify(after))
})

test('a condense sees an eligible run while its own call is open', () => {
  const session = inflightFixture('inflight-run')
  const ranges = buildCompressibleSeqRanges(session, wholeSurfaceRangeView(session), { preserveRecent: 5 })
  const spans = ranges.map((range) => range.start + '..' + range.end).join(' ')
  assert.ok(ranges.length >= 1, 'expected at least one eligible run, got []')
  const widest = ranges.reduce((best, range) => (range.tokens > best.tokens ? range : best), ranges[0]!)
  const nodes = session.surface.nodes
  assert.equal(nodes.indexOf(widest.start), Math.min(...ranges.map((range) => nodes.indexOf(range.start))), 'the widest run starts at the oldest offered row: ' + spans)
  assert.ok(widest.tokens > 0, 'the run carries real tokens: ' + widest.tokens)
})

test('a result with no call at all is still pruned (only OPEN calls gain protection)', () => {
  const session = Session.create('inflight-prune')
  appendTurn(session, 1)
  appendUser(session, longText('u1', 1))
  appendToolResult(session, longText('orphan', 2), 'never-called')
  const orphanSeq = session.surface.nodes[session.surface.nodes.length - 1]!
  buildCompressibleSeqRanges(session, wholeSurfaceRangeView(session), { preserveRecent: 5 })
  assert.ok(!session.surface.nodes.includes(orphanSeq), 'the orphan left the surface; an OPEN call keeps its node (test above)')
})

// Round-5 cross-review (DeepSeek) asked for the companion invariant: protecting an OPEN call
// keeps it on the surface for ref mapping, but it must never become an offered fold candidate —
// Measured here: the pairing resolver already excludes it (rows 1..2 and 4..4 with
// preserveRecent 0, the open node never covered), so this pins behaviour, not a new guard.
test('an open call is never an offered fold candidate, even with no recent protection', () => {
  const session = Session.create('inflight-not-a-row')
  appendTurn(session, 1)
  appendUser(session, longText('u1', 1))
  appendAssistant(session, longText('a2', 2))
  appendUser(session, longText('u3', 3))
  appendAssistant(session, longText('a4', 4))
  appendToolCall(session, longText('a5', 5), 'live-call')
  const nodes = session.surface.nodes
  const openIndex = nodes.length - 1
  const ranges = buildCompressibleSeqRanges(session, wholeSurfaceRangeView(session), { preserveRecent: 0 })
  const covering = ranges.filter((range) => nodes.indexOf(range.start) <= openIndex && openIndex <= nodes.indexOf(range.end))
  assert.equal(covering.length, 0, 'the in-flight node must not appear in any offered row')
  assert.ok(ranges.length >= 1, 'other rows are still offered')
})

