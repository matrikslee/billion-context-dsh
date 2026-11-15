/**
 * The range table must show what a fold COSTS, not only what it saves.
 *
 * Measured 2026-11-15 on the live host with a real `condense` call and the engine's
 * own usage table: prompt 125,665 with reused 31,378 on the first request after a
 * fold that removed 16,205 tokens, and reused ~= prompt again one request later.
 * 31,378 is the rewrite point's token offset — NOT a multiple of the 2^14 KV block,
 * so the earlier rule ('reuse collapses to a constant floor, the rewrite point is
 * not a lever') was wrong and the price it produced (33,722) under-stated the real
 * 94,287 non-reused tokens by ~2.8x.
 *
 * The honest price is therefore: everything AFTER the END edge and nothing before it —
 * the fixed request head (system prompt, tool registry) sits BEFORE every rewrite point
 * so a warm cache reuses it, while an injected row that landed AFTER the rewrite point
 * IS part of this price.
 */

import { test } from 'node:test'
import assert from 'node:assert/strict'
import { defaultCountTokens } from 'acp-kernel'
import { Session } from '@deepseek-ai/dsh-session'
import { createUserMessage } from '@deepseek-ai/dsh-llm'
import { appendAssistant, appendTurn, appendUser, longText, wholeSurfaceRangeView } from './helpers.ts'
import { buildCompressibleSeqRanges } from '../src/region.ts'
import { rangeTable } from '../src/nudge.ts'
import { eventAtOf } from '../src/session-events.ts'
import { extractEventText } from '../src/messages.ts'
import { DEFAULT_RESOLVED } from '../src/prompts.ts'

/** The host's AGENTS.md row shape — a barrier that splits the range table. */
function appendInstruction(session: Session, version: string): void {
  session.append('user/message', createUserMessage({
    content: [{ type: 'text', text: `Instructions for AGENTS.md (${version}): keep tests green.` }],
    source: {
      kind: 'agent-instructions',
      form: 'instructions',
      baseline: true,
      changes: [{ action: 'set', scope: './\u0000AGENTS.md', path: 'AGENTS.md', digest: `digest-${version}` }],
    },
  }), { surfaceOp: 'append' })
}

/** Three offered rows: two instruction barriers and the protected newest turn. */
function threeSegmentSession(id: string): Session {
  const session = Session.create(id)
  appendTurn(session, 1)
  appendUser(session, longText('q0', 0))
  appendAssistant(session, longText('a0', 1), 1, 1)
  appendInstruction(session, 'v1')
  appendUser(session, longText('q1', 2))
  appendAssistant(session, longText('a1', 3), 1, 3)
  appendInstruction(session, 'v2')
  appendUser(session, longText('q2', 4))
  appendAssistant(session, longText('a2', 5), 1, 5)
  return session
}

/** Independent recomputation: what the prompt must re-prefill after END. */
function afterEnd(session: Session, endSeq: number): number {
  const nodes = session.surface.nodes
  const index = nodes.indexOf(endSeq)
  assert.notEqual(index, -1, `the END edge ${endSeq} is a live surface node`)
  let total = 0
  for (const seq of nodes.slice(index + 1)) {
    const event = eventAtOf(session, seq)
    total += event === undefined ? 0 : defaultCountTokens(extractEventText(event))
  }
  return total
}

test('fold cost is everything AFTER the rewrite point', () => {
  const session = threeSegmentSession('fold-cost')
  const ranges = buildCompressibleSeqRanges(session, wholeSurfaceRangeView(session), { preserveRecent: 0 })
  assert.ok(ranges.length >= 3, `the fixture offers several rows (got ${ranges.length})`)
  for (const range of ranges) {
    assert.equal(
      range.foldCostTokens,
      afterEnd(session, range.end),
      `row ${range.start}..${range.end} prices the tail after its own END edge`,
    )
  }
})

test('the rewrite point IS a lever: an earlier fold is priced dearer', () => {
  const session = threeSegmentSession('fold-order')
  const ranges = buildCompressibleSeqRanges(session, wholeSurfaceRangeView(session), { preserveRecent: 0 })
  const nodes = session.surface.nodes
  const byStart = [...ranges].sort((a, b) => nodes.indexOf(a.start) - nodes.indexOf(b.start))
  for (let index = 1; index < byStart.length; index += 1) {
    assert.ok(
      byStart[index - 1]!.foldCostTokens >= byStart[index]!.foldCostTokens,
      `a rewrite point earlier in the surface is never cheaper (${byStart.map((r) => r.foldCostTokens).join(', ')})`,
    )
  }
  assert.ok(
    byStart[0]!.foldCostTokens > byStart[byStart.length - 1]!.foldCostTokens,
    'the oldest row must actually cost more than the newest one on this fixture',
  )
})

// The fixed request head (system prompt, tool registry, injected rows) sits BEFORE any
// rewrite point, so a warm cache reuses it — charging it per fold double-counted it
// and inflated the payback gate (round-10 review B1). Measured the same day: folds at
// two folds at different rewrite points billed 51,122 uncached tokens in total afterwards
// while the range table quoted ~92K for a row on that same surface.
test('the fold price is the suffix only — the fixed request head is not added', () => {
  const session = threeSegmentSession('fold-overhead')
  const view = wholeSurfaceRangeView(session)
  const plain = buildCompressibleSeqRanges(session, view, { preserveRecent: 0 })
  const priced = buildCompressibleSeqRanges(session, view, {
    preserveRecent: 0,
    foldPricing: { floorTokens: 16_000 },
  })
  assert.ok(plain.length > 0 && priced.length === plain.length, 'same rows in both builds')
  for (let index = 0; index < priced.length; index += 1) {
    assert.equal(
      priced[index]!.foldCostTokens,
      plain[index]!.foldCostTokens,
      'a pricing payload must not inflate the re-prefill estimate'
    )
    const endsAtLast = session.surface.nodes.indexOf(priced[index]!.end) === session.surface.nodes.length - 1
    if (endsAtLast) {
      assert.equal(priced[index]!.foldCostTokens, 0, `the last row has nothing after its END edge: ${JSON.stringify(priced[index])}`)
    } else {
      assert.ok(priced[index]!.foldCostTokens > 0, `a row with content after it must cost something: ${JSON.stringify(priced[index])}`)
    }
  }
})

test('a fold ending at the last node costs nothing to re-prefill', () => {
  const session = Session.create('fold-tail')
  appendTurn(session, 1)
  appendUser(session, longText('q0', 0))
  appendAssistant(session, longText('a1', 1), 1, 1)
  const ranges = buildCompressibleSeqRanges(session, wholeSurfaceRangeView(session), {
    preserveRecent: 0,
    foldPricing: { floorTokens: 16_000 },
  })
  const widest = ranges.reduce((best, range) => (range.tokens > best.tokens ? range : best), ranges[0]!)
  const nodes = session.surface.nodes
  assert.equal(nodes.indexOf(widest.end), nodes.length - 1, 'the widest row ends at the last node')
  assert.equal(widest.foldCostTokens, 0, 'nothing after the END edge, so the next request re-prefills nothing extra')
})

test('the range table prints the cost column for every row', () => {
  const session = threeSegmentSession('fold-table')
  const table = rangeTable(session, wholeSurfaceRangeView(session), DEFAULT_RESOLVED, undefined, {
    floorTokens: 16_000,
  })
  const rows = table.split('\n').filter((line) => line.includes('fold ~'))
  assert.ok(rows.length >= 1, `rows carry the cost column (${rows.length})`)
  for (const row of rows) {
    assert.match(row, /fold ~\d+/, 'a real number per row')
  }
})
