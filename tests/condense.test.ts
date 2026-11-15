/**
 * condense — the engine elects the stretch, the model writes the handoff note.
 *
 * Pinned contracts:
 * 1. The elected span is the LARGEST eligible continuous run. The fold price is the
 *    prompt tail AFTER the rewrite point ONLY — the fixed request head sits before
 *    every rewrite point, so a warm cache reuses it (round-10 review B1).
 * 2. The CURRENT injected instruction copy is a wall; a SUPERSEDED copy folds with
 *    the run (it cannot come back, and it is most of a long session residue).
 * 3. The note must carry ## Goal and ## Next, or nothing lands.
 * 4. condense and content are mutually exclusive.
 * 5. The result prices the fold: one-time re-prefill plus the resident note. The
 *    price no longer borrows the paired request sample (B1), so the copy carries no
 *    LOWER-bound hedge and no request-overhead claim in either state.
 */

import { test } from 'node:test'
import assert from 'node:assert/strict'
import { createCore, type CompressionCore } from 'acp-kernel'
import { Session } from '@deepseek-ai/dsh-session'
import { createUserMessage } from '@deepseek-ai/dsh-llm'
import { AcpStateStore } from '../src/state.ts'
import { kernelConfigFor } from '../src/config.ts'
import { allLogMessages } from '../src/messages.ts'
import { recordRequestPair } from '../src/usage.ts'
import { CONDENSE_PAYBACK_REQUESTS, MAX_NOTE_WEIGHT_RATIO, MIN_CONDENSE_SPAN_TOKENS, MIN_HANDOFF_NOTE_TOKENS, condenseNoteBudget, condenseNotePayback, makeTools, missingCondenseHeadings, pickCondenseSpan, reliefMinSavingTokens, reliefSpanReachable, type ToolEnvironment } from '../src/tools.ts'
import { appendAssistant, appendTurn, appendUser, longText } from './helpers.ts'

/** Two stacked longText copies — one message alone clears the kernel 5000-char floor. */
const bigText = (label: string, seed: number): string => longText(label, seed) + longText(label + 'b', seed + 100)

const NOTE = '## Goal\nBuild the token rotation in src/auth/login.ts and get the auth suite green.\n'
  + '## Next\nRun node --import tsx --test tests/auth.test.ts; done at 12/12 pass.\n'

function makeEnv(): ToolEnvironment {
  return {
    kernel: createCore({}) as CompressionCore,
    store: new AcpStateStore(),
    modelContextLimit: 128000,
    // This file tests span election, not the working-set guard; the fixtures are
    // small on purpose, so the 15% floor would refuse every fold (see
    // tests/fold-floor.test.ts for the guard itself).
    foldFloorPct: 0,
    compressCallIdsToHide: new Set(),
  }
}

function execStub(session: Session, callId: string): never {
  const agent = {
    id: session.id,
    session,
    options: { provider: 'test-provider', model: 'test-model' },
    ctx: { tokenMeter: undefined },
  }
  return {
    callId,
    name: 'compress',
    arguments: {},
    signal: new AbortController().signal,
    agent,
  } as never
}

function compressTool(env: ToolEnvironment) {
  const definition = makeTools(env).find((candidate) => candidate.name === 'compress')
  assert.ok(definition, 'compress tool registered')
  return definition
}

/** An AGENTS.md injection row in the host shape, always the same scope. */
function appendInstruction(session: Session, digest: string): void {
  session.append('user/message', createUserMessage({
    content: [{ type: 'text', text: 'Instructions v' + digest + ': ' + longText('instr' + digest, 77) }],
    source: {
      kind: 'agent-instructions',
      form: 'instructions',
      baseline: true,
      changes: [{ action: 'set', scope: './\u0000AGENTS.md', path: 'AGENTS.md', digest }],
    },
  }), { surfaceOp: 'append' })
}

/** Work nodes with two injections of the SAME scope in the middle. */
function workFixture(id: string): { session: Session; staleSeq: number; newestSeq: number } {
  const session = Session.create(id)
  appendTurn(session, 1)
  appendUser(session, bigText('u1', 1))
  appendAssistant(session, bigText('a2', 2))
  appendUser(session, bigText('u3', 3))
  appendAssistant(session, bigText('a4', 4))
  appendInstruction(session, 'v1')
  const staleSeq = session.surface.nodes[session.surface.nodes.length - 1]!
  appendUser(session, bigText('u5', 5))
  appendAssistant(session, bigText('a6', 6))
  appendInstruction(session, 'v2')
  const newestSeq = session.surface.nodes[session.surface.nodes.length - 1]!
  appendUser(session, bigText('u7', 7))
  appendAssistant(session, bigText('a8', 8))
  appendUser(session, bigText('u9', 9))
  appendAssistant(session, bigText('a10', 10))
  appendUser(session, bigText('u11', 11))
  appendAssistant(session, bigText('a12', 12))
  return { session, staleSeq, newestSeq }
}

function turnOf(env: ToolEnvironment, session: Session) {
  return env.kernel.processTurn({
    messages: allLogMessages(session),
    state: env.store.stateFor(session),
    config: kernelConfigFor({ modelContextLimit: 128000 }),
    tokenCount: 300000,
  })
}

test('condense elects the largest eligible run and prices it as one re-prefill', async () => {
  const env = makeEnv()
  const { session } = workFixture('condense-largest')
  const result = await compressTool(env).execute({ condense: { summary: NOTE } } as never, execStub(session, 'c1'))
  const text = (result as { text: string }).text
  assert.match(text, /Compressed 1 block/, 'the elected span lands as one block')
  assert.match(text, /condense elected seqs \d+\.\.\d+/, 'the result names the span the engine picked')
  assert.match(text, /re-prefills ~\d+ tokens ONCE/, 'the result prices the one-time re-prefill')
  assert.match(text, /stays resident on every request/, 'the result names the durable note cost')
})

test('condense folds the superseded injection copy and never the current one', async () => {
  const env = makeEnv()
  const { session, staleSeq, newestSeq } = workFixture('condense-walls')
  const pick = pickCondenseSpan(session, turnOf(env, session), 5)
  assert.ok('start' in pick, 'a run is eligible on this fixture: ' + JSON.stringify(pick))
  const elected = pick as { start: number; end: number }
  assert.ok(elected.start <= staleSeq && staleSeq <= elected.end,
    'the stale copy seq ' + staleSeq + ' folds inside the elected ' + elected.start + '..' + elected.end)
  assert.ok(elected.end < newestSeq || elected.start > newestSeq,
    'the current copy seq ' + newestSeq + ' is a wall, not inside ' + elected.start + '..' + elected.end)
})

test('condense without both headings lands nothing and asks for the note', async () => {
  const env = makeEnv()
  const { session } = workFixture('condense-headings')
  const result = await compressTool(env).execute({ condense: { summary: 'Explored the auth module and learned things.' } } as never, execStub(session, 'c2'))
  const text = (result as { text: string }).text
  assert.match(text, /needs a handoff note/, 'the refusal names the missing handoff shape')
  assert.match(text, /## Goal/, 'it lists the first missing heading')
  assert.match(text, /## Next/, 'it lists the second missing heading')
  assert.match(text, /re-picks the span from the CURRENT surface/, 'it tells the model the retry is cheap')
  const wrote = session.snapshotEvents().filter((event) => event.kind === 'compaction/summary')
  assert.equal(wrote.length, 0, 'nothing durable landed')
})

test('condense and content are mutually exclusive', async () => {
  const env = makeEnv()
  const { session } = workFixture('condense-exclusive')
  const result = await compressTool(env).execute({
    condense: { summary: NOTE },
    content: [{ startSeq: 1, endSeq: 2, summary: NOTE }],
  } as never, execStub(session, 'c3'))
  assert.match((result as { text: string }).text, /mutually exclusive/, 'the engine refuses the ambiguous call')
})

test('a surface with nothing contiguous enough reports a miss with its reason', async () => {
  const env = makeEnv()
  const session = Session.create('condense-miss')
  appendTurn(session, 1)
  appendUser(session, 'short question')
  appendAssistant(session, 'short answer')
  const pick = pickCondenseSpan(session, turnOf(env, session), 5)
  assert.ok('reason' in pick, 'nothing is eligible: ' + JSON.stringify(pick))
  assert.match((pick as { reason: string }).reason, /minimum range/, 'the reason names the character floor')
})

test('the note gate accepts the Chinese headings too', () => {
  assert.deepEqual(missingCondenseHeadings('## \u76ee\u6807\n\u505a\u767b\u5f55\u6001\u6ce8\u5165\n## \u4e0b\u4e00\u6b65\n\u8dd1\u6d4b\u8bd5'), [])
  const missing = missingCondenseHeadings('just a paragraph')
  assert.equal(missing.length, 2, 'both headings are reported missing')
  assert.match(missing[0]!, /## Goal/, 'first missing heading is Goal')
  assert.match(missing[1]!, /## Next/, 'second missing heading is Next')
})

test('the fold price is the rewrite-point suffix alone, with no overhead hedge', async () => {
  const env = makeEnv()
  const { session } = workFixture('condense-no-sample')
  const result = await compressTool(env).execute({ condense: { summary: NOTE } } as never, execStub(session, 'c-no-sample'))
  const text = (result as { text: string }).text
  assert.match(text, /condense elected seqs/, 'the fold lands')
  assert.match(text, /every token AFTER the rewrite point/, 'the copy names the real price basis')
  assert.doesNotMatch(text, /fixed request overhead/, 'the head is not part of the fold price (B1)')
  assert.doesNotMatch(text, /LOWER bound/, 'no paired sample is needed, so no hedge is printed')
})

test('a paired request sample changes nothing in the fold price copy', async () => {
  const env = makeEnv()
  const { session } = workFixture('condense-with-sample')
  // Same call with a paired sample recorded: the fold price never reads it, so the
  // copy must stay identical in shape (the sample only feeds the emergency check).
  recordRequestPair(session, 254_000, 250_000)
  const result = await compressTool(env).execute({ condense: { summary: NOTE } } as never, execStub(session, 'c-measured'))
  const text = (result as { text: string }).text
  assert.match(text, /condense elected seqs/, 'the fold lands')
  assert.doesNotMatch(text, /LOWER bound|fixed request overhead/, 'a sample neither adds a claim nor a hedge')
})


test('a net-negative election is refused with its break-even count', async () => {
  // Volume alone used to pick this: an eligible run that removes less than the NEXT
  // request re-prefills. Measured live 2026-11-15: 23,588 removed vs 155,927
  // re-prefilled. The engine must not auto-fold that; it hands the decision to the
  // model with the payback count instead.
  const env = makeEnv()
  const { session } = workFixture('condense-negative-net')
  // No synthetic overhead any more: the loss must come from the shape itself — a small
  // eligible row sitting in front of a big resident stretch re-prefills far more than
  // it removes, exactly the measured live failure (23,588 removed vs 155,927 re-prefilled).
  appendUser(session, longText('a large resident summary in front of every older row', 9).repeat(120))
  const pick = pickCondenseSpan(session, turnOf(env, session), 0)
  assert.ok('reason' in pick, 'the suffix-in-front shape makes every row net-negative: ' + JSON.stringify(pick))
  assert.match((pick as { reason: string }).reason, /costs MORE to re-prefill/, 'the reason names the loss')
  assert.match((pick as { reason: string }).reason, /explicit content range/, 'it points at the explicit path')
  const wrote = session.snapshotEvents().filter((event) => event.kind === 'compaction/summary')
  assert.equal(wrote.length, 0, 'nothing durable lands')
})


test('the note-aware payback counts the resident note against the re-prefill', () => {
  // span 40K, re-prefill 96K: a short note pays back inside the horizon…
  assert.equal(condenseNotePayback(40_000, 96_000, 60).worth, true, 'a slim note keeps the fold worth it')
  // …and a fat note that eats the saving does not: net saving falls to 8K, and the note is
  // part of the one-time hit too (it sits AT the rewrite point, so the next request prefills
  // it uncached — round-11 Q1), giving 128K against 8K.
  assert.equal(condenseNotePayback(40_000, 96_000, 32_000).worth, false, 'the note can turn a paying fold into a loss')
  assert.equal(condenseNotePayback(40_000, 96_000, 32_000).payback, 16, 'the refusal speaks the payback count, note included')
  assert.equal(condenseNotePayback(0, 10, 0).worth, false, 'a zero span is never worth folding')
  // Round-11 Q7: a note as big as the span saves NOTHING, and that verdict must not hinge on
  // how cheap the re-prefill happens to be. The old Math.max(1, netSaving) read "no saving" as
  // "1 token/request", so a tail fold (foldCost 0..4) was accepted with zero saving forever.
  assert.equal(condenseNotePayback(40_000, 2, 40_000).worth, false, 'note == span is refused even at the surface tail')
  assert.equal(condenseNotePayback(40_000, 0, 45_000).worth, false, 'a note bigger than the span is refused at zero fold cost')
  assert.equal(condenseNotePayback(40_000, 0, 40_000).payback, Number.POSITIVE_INFINITY, 'no saving means no finite payback')
})

test('the payback gate steps aside only under window pressure', () => {
  // Round-10 review B2: refusing a costly fold is right while there is room, but at
  // 95% of the window refusing means the NEXT request is rejected by the provider —
  // a costly fold beats a hard failure. The bypass is pressure-only; the floor rules
  // and the note gate still apply.
  const { session } = workFixture('condense-pressure-relief')
  appendUser(session, longText('a large resident summary in front of every older row', 9).repeat(120))
  const turn = turnOf(makeEnv(), session)
  const strict = pickCondenseSpan(session, turn, 0)
  assert.ok('reason' in strict, 'no pressure reading means the loss is refused: ' + JSON.stringify(strict))
  const relieved = pickCondenseSpan(session, turn, 0, undefined, { liveTokens: 90_000, limitTokens: 100_000 })
  assert.ok(!('reason' in relieved), 'at 90% of the window the fold is taken: ' + JSON.stringify(relieved))
  assert.equal((relieved as { pressureRelief: boolean }).pressureRelief, true, 'the pick records why it fired')
  const cool = pickCondenseSpan(session, turn, 0, undefined, { liveTokens: 80_000, limitTokens: 100_000 })
  assert.ok('reason' in cool, 'at 80% of the window the gate still refuses: ' + JSON.stringify(cool))
})

/**
 * Round-12 review Q5.1 (both reviewers): the election side steps aside at ≥85% of the window,
 * but `handleCompress` then ran the note-priced payback gate unconditionally — and whenever
 * relief was needed, `foldCost > tokens × 4` guarantees the note-priced horizon fails too, so
 * the fold the pick had just allowed was refused anyway and the refusal copy ("the gate steps
 * aside at ≥85%") was a lie. Under relief only a note that saves NOTHING is refused; here the
 * fold must land and still report its honest break-even.
 */
test('round-12 Q5.1: under window pressure the note gate lets the relieved fold land', async () => {
  const { session } = workFixture('condense-note-gate-relief')
  appendUser(session, longText('a large resident summary in front of every older row', 9).repeat(120))
  // inflation clamps to 2 and the absolute overhead alone puts liveTokens past 85% of 128000.
  recordRequestPair(session, 200_000, 60_000)
  const result = await compressTool(makeEnv()).execute(
    { condense: { summary: NOTE } } as never,
    execStub(session, 'call-relief'),
  ) as { text: string }
  assert.match(result.text, /condense elected/, result.text)
  const even = /breaks even after ~(\d+)/.exec(result.text)
  assert.ok(even, 'the result still prices the fold honestly: ' + result.text)
  assert.ok(Number(even![1]) > CONDENSE_PAYBACK_REQUESTS, `break-even ${even![1]} must exceed the horizon`)
})

/**
 * Round-13 review Q5: condense elected a 750-token run, the model wrote a note for it, and the
 * write-time weight cap refused it — a turn and a re-prefill burned on a span that can never
 * carry a handoff. The election must refuse such a span up front and say what to do instead.
 */
test('a span too small to carry a handoff note is never elected', () => {
  const env = makeEnv()
  const session = Session.create('condense-span-floor')
  appendTurn(session, 1)
  // The candidate run: ~5800 chars ≈ 1450 tokens, over the kernel's 5000-CHARACTER minimum and
  // under MIN_CONDENSE_SPAN_TOKENS. The assistant reply keeps the run balanced; what follows is
  // the LAST real user turn (host-protected) plus a stub too small for the character minimum, so
  // this stub run IS the largest eligible run — the shape the live session produced at 750 tokens.
  session.append('user/message', createUserMessage({
    content: [{ type: 'text', text: 'Authentication uses JWT access tokens with 15 minute expiry. '.repeat(95) }],
    source: { kind: 'user' },
  }), { surfaceOp: 'append' })
  appendAssistant(session, 'Noted. '.repeat(40))
  appendUser(session, 'The live turn that follows. '.repeat(140))
  appendAssistant(session, 'Acknowledged. '.repeat(20))
  const pick = pickCondenseSpan(session, turnOf(env, session), 0)
  assert.ok('reason' in pick, 'the small run must be refused at election time: ' + JSON.stringify(pick))
  assert.match((pick as { reason: string }).reason, new RegExp('under the ' + MIN_CONDENSE_SPAN_TOKENS + '-token floor'), (pick as { reason: string }).reason)
  assert.match((pick as { reason: string }).reason, /explicit content range/, 'the refusal names the compliant alternative')
})

test('relief demands a window-relative net reclaim, not merely a positive one', () => {
  assert.equal(reliefMinSavingTokens(false, 128_000), 0, 'outside relief nothing extra is required')
  assert.equal(reliefMinSavingTokens(true, 0), 0, 'no window reading means no relief claim')
  assert.equal(reliefMinSavingTokens(true, 128_000), 2048, '2% of a 128K window, capped by the dedicated relief cap')
  assert.equal(reliefMinSavingTokens(true, 262_144), 2048, 'clamped by RELIEF_MIN_NET_SAVING_MAX_TOKENS, not MIN_DEGRADED_FOLD_TOKENS')
  assert.equal(reliefMinSavingTokens(true, 3000), 60, 'a small window keeps a proportional floor')
})

/**
 * Round-14 review Q2: pricing the WORST-CASE 35% note at election refused spans a real handoff
 * pays back — a 40K row against a 96K re-prefill breaks even in 3 requests with an ~85-token note,
 * yet the worst-case price read 5 and refused before the model could write anything. The election
 * now solves for the largest note that still pays back and refuses only below a briefing.
 */
test('round-14 Q2: the election prices the note budget, not the worst-case note', () => {
  assert.equal(condenseNoteBudget(40_000, 96_000), 12_800, 'notes up to 12.8K stored tokens (32% of the span) still pay back in 4')
  assert.ok(condenseNoteBudget(40_000, 96_000) >= MIN_HANDOFF_NOTE_TOKENS, 'that span is electable again')
  assert.ok(condenseNoteBudget(40_000, 96_000) < 40_000 * MAX_NOTE_WEIGHT_RATIO, 'the budget stays under the write-time 35% cap')
  assert.equal(condenseNoteBudget(2000, 7800), 40, 'a thin row against a big suffix barely allows a note')
  assert.ok(condenseNoteBudget(2000, 7800) < MIN_HANDOFF_NOTE_TOKENS, 'refused at election, not after a burned turn')
  assert.equal(condenseNoteBudget(2000, 9000), -200, 'negative: no note size earns the re-prefill back')
})

/**
 * Round-14 review Q3: the relief floor shared a value with a GROSS span-size floor, and the
 * election kept offering rows that the write-time gate rejected unconditionally on a 262K window.
 */
test('round-14 Q3: the relief floor is decoupled and unreachable spans are refused at election', () => {
  assert.equal(reliefMinSavingTokens(true, 32_768), 655, 'proportional on a small window')
  assert.equal(reliefSpanReachable(2000, 262_144), false, 'even a bare note cannot reach 2048 net')
  assert.equal(reliefSpanReachable(2100, 262_144), true, 'just past reliefFloor + frame prefix is reachable')
  assert.equal(reliefSpanReachable(2000, 32_768), true, 'a 32K window demands only 655')
})
