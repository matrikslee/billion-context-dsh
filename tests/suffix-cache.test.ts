import { test } from 'node:test'
import assert from 'node:assert/strict'
import { Session } from '@deepseek-ai/dsh-session'
import { createUserMessage } from '@deepseek-ai/dsh-llm'
import { AttachmentId } from '@deepseek-ai/dsh-attachment'
import { surfaceSuffixTokens } from '../src/region.ts'
import { appendTurn, appendUser, longText } from './helpers.ts'

// Rule 5: the real DSH image block shape — nodePriceOf only consults the media
// callback for nodes that actually carry media, so a text-only fixture proves
// nothing about the cache key.
const IMAGE = {
  type: 'image',
  attachment: {
    attachmentId: AttachmentId('att-cache-1'),
    mediaType: 'image/png',
    bytes: 4321,
    width: 800,
    height: 600,
    name: 'shot.png',
  },
} as const

// Two media resolvers can price the SAME unchanged surface array (the nudge table and
// the emergency path each pass their own). A cache key that only records that A
// callback was used serves the first callback's numbers to the second.
test('the suffix cache is keyed on the media price callback, not just its presence', () => {
  const session = Session.create('suffix-cache-identity')
  appendTurn(session, 1)
  appendUser(session, longText('text row', 0))
  session.append('user/message', createUserMessage({ content: [IMAGE], source: { kind: 'user' } }), { surfaceOp: 'append' })
  const expensiveMedia = (): number => 1000
  const cheapMedia = (): number => 0
  const expensive = surfaceSuffixTokens(session, expensiveMedia)
  assert.equal(expensive, surfaceSuffixTokens(session, expensiveMedia), 'the same callback on an unchanged surface is a cache hit')
  const cheap = surfaceSuffixTokens(session, cheapMedia)
  assert.notEqual(expensive, cheap, 'a DIFFERENT callback must not reuse the cached walk')
  assert.ok(expensive[0]! > cheap[0]! + 900, `the media price must not leak between callbacks: ${expensive[0]} vs ${cheap[0]}`)
  const unpriced = surfaceSuffixTokens(session)
  assert.notEqual(unpriced, cheap, 'no callback at all is a third identity')
  assert.ok(unpriced[0]! <= cheap[0]! + 1, `the structural price alone stays the lower bound: ${unpriced[0]} vs ${cheap[0]}`)
})

// Round-11 review Q2: a durable replace can keep the SAME array, the SAME length and the
// SAME last seq while rewriting an interior node — a 1-for-1 fold of one middle row, or
// hideSurfaceSeqs turning one interior row into one PRUNE_NOTE tombstone (rule 13 writes a
// VISIBLE user message, so the count survives). A key of identity+length+lastSeq reads a
// clean hit there and hands back stale suffix numbers, which is exactly what the payback
// gate and the floor guard price.
test('an in-place replace that keeps length and last seq still invalidates the suffix cache', () => {
  const session = Session.create('suffix-replace')
  appendTurn(session, 1)
  appendUser(session, longText('middle', 1))
  const middle = session.surface.nodes[session.surface.nodes.length - 1]!
  appendUser(session, longText('tail', 2))
  const lastSeq = session.surface.nodes[session.surface.nodes.length - 1]!
  const media = (): number => 0
  const before = surfaceSuffixTokens(session, media)
  assert.equal(before, surfaceSuffixTokens(session, media), 'an unchanged surface is a cache hit')
  session.append('user/message', createUserMessage({
    content: [{ type: 'text', text: longText('replaced', 3) + longText('replacedb', 4) }],
    source: { kind: 'user' },
  }), { surfaceOp: { op: 'replace', startSeq: middle, endSeq: middle }, sourceEventSeqs: [middle] })
  assert.equal(session.surface.nodes.length, 2, 'the replace kept the node count')
  assert.equal(session.surface.nodes[session.surface.nodes.length - 1]!, lastSeq, 'the replace kept the last seq')
  const after = surfaceSuffixTokens(session, media)
  assert.notEqual(after, before, 'the rewrite must not be served from the stale cache')
  assert.ok(after[0]! > before[0]!, `the bigger replacement is priced: ${before[0]} → ${after[0]}`)
})
