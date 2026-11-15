import { test } from 'node:test'
import assert from 'node:assert/strict'
import { Context } from '@deepseek-ai/cordis'
import type { Agent } from '@deepseek-ai/dsh-agent'
import { Session } from '@deepseek-ai/dsh-session'
import AcpCompactionEngine, { AcpCompactionEngine as Named, resolveAcpConfig } from '../src/index.ts'
import { rebuildBlockLedger } from '../src/region.ts'
import { SUMMARY_FRAME_PREFIX, eventsToCoreMessages, surfaceEventsOf } from '../src/messages.ts'
import { sessionEventsOf } from '../src/session-events.ts'
import { buildTextSession, appendUser, appendAssistant, appendTurn, longText } from './helpers.ts'
import { createUserMessage } from '@deepseek-ai/dsh-llm'
import { recordRequestPair } from '../src/usage.ts'

/**
 * M5 — the automatic trigger half that compaction-basic has and ACP was
 * missing: `agent/request-error` context-overflow recovery
 * (docs/dsh-porting-verification.md, architecture fact 2). ACP stays
 * model-driven for pressure — the pre-step listener only nudges — but a
 * provider-confirmed CONTEXT_WINDOW_EXCEEDED is the ONE automatic action:
 * emergency-compact the largest eligible block and answer the seam's
 * `{ kind: 'retry' }` protocol instead of letting the error rethrow.
 */

/** Minimal agent handle, same shape the tools tests use. */
function fakeAgent(session: Session, ctx: Context): Agent {
  return {
    id: session.id,
    session,
    options: { provider: 'test-provider', model: 'test-model' },
    ctx,
  } as unknown as Agent
}

/** Dispatch one request-error through the same bus the loop uses. */
function requestError(
  ctx: Context,
  agent: Agent,
  failure: { code: string; message: string },
  signal: AbortSignal = new AbortController().signal,
): Promise<{ kind: 'retry' } | undefined> {
  return ctx.waterfall(
    'agent/request-error',
    { agent, turn: 1, step: 1, provider: 'test-provider', failure, retryPolicy: undefined, signal },
    () => Promise.resolve(undefined),
  )
}

const OVERFLOW = { code: 'CONTEXT_WINDOW_EXCEEDED', message: 'input too long' }

function compactionSummaryCount(session: Session): number {
  return sessionEventsOf(session).filter((event) => event.type === 'compaction/summary').length
}

test('M5: request-error listener delegates non-overflow failures (error preserved)', async () => {
  const ctx = new Context()
  const engine = new Named(ctx, { modelContextLimit: 100000 })
  const session = buildTextSession(30)
  const action = await requestError(ctx, fakeAgent(session, ctx), { code: 'RATE_LIMITED', message: 'slow down' })
  assert.equal(action, undefined, 'a non-overflow failure must pass through to the default')
  assert.equal(compactionSummaryCount(session), 0, 'no compaction for a non-overflow failure')
  void engine
})

test('M5: context overflow emergency-compacts the largest eligible block and returns retry', async () => {
  const ctx = new Context()
  const engine = new Named(ctx, { modelContextLimit: 100000 })
  const session = buildTextSession(30)
  const before = sessionEventsOf(session)
  const surfaceBefore = session.surface.nodes.length
  const action = await requestError(ctx, fakeAgent(session, ctx), OVERFLOW)
  assert.deepEqual(action, { kind: 'retry' }, 'the listener owns recovery and asks the loop to retry')
  assert.equal(compactionSummaryCount(session), 1, 'exactly one durable compaction/summary')
  // The transaction is complete: start + summary + replacement + end.
  const types = sessionEventsOf(session).map((event) => event.type)
  assert.ok(types.includes('compaction/start'), 'compaction/start recorded')
  assert.ok(types.includes('compaction/end'), 'compaction/end recorded (no dangling start)')
  // The originals are intact in the append-only log; the surface shrank.
  assert.equal(
    sessionEventsOf(session).filter((event) => event.type === 'user/message' || event.type === 'assistant/message').length,
    before.filter((event) => event.type === 'user/message' || event.type === 'assistant/message').length + 1,
    'one checkpoint summary node added, originals retained',
  )
  assert.ok(session.surface.nodes.length < surfaceBefore, 'the surface shrank')
  const ledger = rebuildBlockLedger(sessionEventsOf(session))
  assert.equal(ledger.length, 1, 'one ledger block')
  const block = ledger[0]!
  assert.equal(block.topic, 'context-overflow recovery', 'the block is labeled as an overflow recovery')
  assert.match(block.summary, /context-overflow emergency compaction/, 'the marker summary names its origin')
  assert.ok(!block.summary.includes(SUMMARY_FRAME_PREFIX), 'engine-written marker is NOT stamped as a model-written summary')
  assert.ok(block.shadowedSeqs.length > 0, 'the block shadowed a non-empty range')
  // What the MODEL sees must be the same bytes: the projection net frames every
  // checkpoint node, so the marker has to be recognized as engine-written on
  // that path too — otherwise the frame would mislabel it as model-written and
  // the decompress header (durable text) would differ from the live context.
  const projected = eventsToCoreMessages(surfaceEventsOf(session))
  assert.ok(
    projected.some((message) => /context-overflow emergency compaction/.test(message.text)),
    'the marker is visible on the surface',
  )
  assert.ok(
    projected.every((message) => !message.text.includes(SUMMARY_FRAME_PREFIX)),
    'the engine marker is never mislabeled as a model-written summary in the model-visible projection',
  )
  assert.ok(
    projected.some((message) => message.text === block.summary),
    'projected text and durable text are the same bytes (no stored/projected split)',
  )
  void engine
})

test('M5: retry budget spent without progress → original error preserved (no infinite retry)', async () => {
  const ctx = new Context()
  const engine = new Named(ctx, { modelContextLimit: 100000 }) // default maxOverflowRetries: 1
  const session = buildTextSession(30)
  const agent = fakeAgent(session, ctx)
  assert.deepEqual(await requestError(ctx, agent, OVERFLOW), { kind: 'retry' })
  const summariesAfterFirst = compactionSummaryCount(session)
  const second = await requestError(ctx, agent, OVERFLOW)
  assert.equal(second, undefined, 'second consecutive overflow without progress is not retried')
  assert.equal(compactionSummaryCount(session), summariesAfterFirst, 'no second compaction while the budget is spent')
})

test('M5: assistant message progress resets the recovery budget', async () => {
  const ctx = new Context()
  const engine = new Named(ctx, { modelContextLimit: 100000 })
  const session = buildTextSession(30)
  const agent = fakeAgent(session, ctx)
  assert.deepEqual(await requestError(ctx, agent, OVERFLOW), { kind: 'retry' })
  // The host publishes assistant/message on the same bus when the model
  // responds — that is progress, so a LATER overflow gets a fresh budget.
  ctx.emit('session/event', session, { type: 'assistant/message' })
  // New conversation content grows a new compressible span (the first
  // recovery already consumed everything eligible).
  for (let index = 0; index < 10; index += 1) {
    appendUser(session, longText('more', index))
    appendAssistant(session, longText('more-reply', index), 2, index + 1)
  }
  assert.deepEqual(await requestError(ctx, agent, OVERFLOW), { kind: 'retry' })
  assert.equal(compactionSummaryCount(session), 2, 'a fresh budget allowed a second recovery')
})

test('M5: agent idle resets the recovery budget', async () => {
  const ctx = new Context()
  const engine = new Named(ctx, { modelContextLimit: 100000 })
  const session = buildTextSession(30)
  const agent = fakeAgent(session, ctx)
  assert.deepEqual(await requestError(ctx, agent, OVERFLOW), { kind: 'retry' })
  ctx.emit('agent/status', { agent, status: 'idle' })
  for (let index = 0; index < 10; index += 1) {
    appendUser(session, longText('more', index))
    appendAssistant(session, longText('more-reply', index), 2, index + 1)
  }
  assert.deepEqual(await requestError(ctx, agent, OVERFLOW), { kind: 'retry' })
  assert.equal(compactionSummaryCount(session), 2, 'idle reset the budget for a second recovery')
})

test('M5: nothing eligible to compact → error preserved without a phantom block', async () => {
  const ctx = new Context()
  const engine = new Named(ctx, { modelContextLimit: 100000 })
  // Every message sits inside the protected recent tail, so no eligible range
  // exists — the listener must hand the error back untouched.
  const session = buildTextSession(4)
  const action = await requestError(ctx, fakeAgent(session, ctx), OVERFLOW)
  assert.equal(action, undefined, 'no eligible range → the original error is preserved')
  assert.equal(compactionSummaryCount(session), 0, 'no phantom compaction')
})

test('M5: aborted turn never compacts', async () => {
  const ctx = new Context()
  const engine = new Named(ctx, { modelContextLimit: 100000 })
  const session = buildTextSession(30)
  const controller = new AbortController()
  controller.abort()
  const action = await requestError(ctx, fakeAgent(session, ctx), OVERFLOW, controller.signal)
  assert.equal(action, undefined, 'aborted turn → no retry')
  assert.equal(compactionSummaryCount(session), 0)
})

test('M5: maxOverflowRetries 0 disables the automatic recovery', async () => {
  const ctx = new Context()
  const engine = new Named(ctx, { modelContextLimit: 100000, maxOverflowRetries: 0 })
  const session = buildTextSession(30)
  const action = await requestError(ctx, fakeAgent(session, ctx), OVERFLOW)
  assert.equal(action, undefined, 'explicit zero budget → error preserved')
  assert.equal(compactionSummaryCount(session), 0)
})

test('M5: invalid maxOverflowRetries fails engine construction loudly', () => {
  assert.throws(() => new Named(new Context(), { maxOverflowRetries: -1 }), /maxOverflowRetries must be a non-negative integer/)
  assert.throws(() => new Named(new Context(), { maxOverflowRetries: 1.5 }), /maxOverflowRetries must be a non-negative integer/)
})

test('M5: an explicitly undefined maxOverflowRetries keeps the host-parity default of 1', async () => {
  // The key survives `{ ...DEFAULT_CONFIG, ...config }` as `undefined`, which a
  // `?? 0` read would silently turn into "recovery disabled" while the config
  // reports the default. The resolver must write the validated value back.
  assert.equal(resolveAcpConfig({ maxOverflowRetries: undefined }).maxOverflowRetries, 1)
  const ctx = new Context()
  const engine = new Named(ctx, { modelContextLimit: 100000, maxOverflowRetries: undefined })
  const session = buildTextSession(30)
  assert.deepEqual(await requestError(ctx, fakeAgent(session, ctx), OVERFLOW), { kind: 'retry' }, 'default budget still recovers')
  assert.equal(compactionSummaryCount(session), 1)
  void engine
})

test('M5: overflow bookkeeping maps release their entries on both terminal paths', async () => {
  const readInternals = (engine: Named): { overflowRetries: Map<unknown, number>; overflowSessions: Map<unknown, unknown> } =>
    engine as unknown as { overflowRetries: Map<unknown, number>; overflowSessions: Map<unknown, unknown> }
  // Progress (assistant message) is one terminal path.
  const progressCtx = new Context()
  const progressEngine = new Named(progressCtx, { modelContextLimit: 100000 })
  const progressSession = buildTextSession(30)
  const progressAgent = fakeAgent(progressSession, progressCtx)
  assert.deepEqual(await requestError(progressCtx, progressAgent, OVERFLOW), { kind: 'retry' })
  const progressInternals = readInternals(progressEngine)
  assert.equal(progressInternals.overflowSessions.get(progressSession), progressAgent, 'reverse lookup is live while recovery is pending')
  progressCtx.emit('session/event', progressSession, { type: 'assistant/message' })
  assert.equal(progressInternals.overflowSessions.size, 0, 'progress releases the session→agent lookup')
  assert.equal(progressInternals.overflowRetries.size, 0, 'progress releases the retry budget')
  // Idle (unrelieved overflow, error surfaced) is the other one.
  const idleCtx = new Context()
  const idleEngine = new Named(idleCtx, { modelContextLimit: 100000 })
  const idleSession = buildTextSession(30)
  const idleAgent = fakeAgent(idleSession, idleCtx)
  assert.deepEqual(await requestError(idleCtx, idleAgent, OVERFLOW), { kind: 'retry' })
  idleCtx.emit('agent/status', { agent: idleAgent, status: 'idle' })
  const idleInternals = readInternals(idleEngine)
  assert.equal(idleInternals.overflowSessions.size, 0, 'idle releases the session→agent lookup')
  assert.equal(idleInternals.overflowRetries.size, 0, 'idle releases the retry budget')
})

test('M5: AcpCompactionEngine default export carries the overflow listener too', async () => {
  const ctx = new Context()
  ctx.plugin(AcpCompactionEngine as never)
  await new Promise((resolve) => setTimeout(resolve, 20))
  const engine = ctx.compaction as Named
  assert.ok(engine instanceof Named)
  const session = buildTextSession(30)
  const action = await requestError(ctx, fakeAgent(session, ctx), OVERFLOW)
  assert.deepEqual(action, { kind: 'retry' })
  assert.equal(compactionSummaryCount(session), 1)
})
test('M5: an unfit projection still folds and stamps FIT RISK (advisory, not a gate)', async () => {
  const ctx = new Context()
  const engine = new Named(ctx, { modelContextLimit: 100000 })
  const session = buildTextSession(30)
  // A paired sample the projection cannot satisfy: the ratio sits at the clamp ceiling and
  // the absolute overhead alone (130K) already exceeds the 100K window, so a hard gate here
  // would refuse. Refusing IS the recovery failure — the budget is one retry, then the
  // provider error stands (measured 2026-11-15: the hard gate deleted the emergency fold in
  // the e2e overflow scenario outright, five assertions red with nothing folded).
  recordRequestPair(session, 260_000, 130_000)
  const action = await requestError(ctx, fakeAgent(session, ctx), OVERFLOW)
  assert.deepEqual(action, { kind: 'retry' }, 'an unfit projection still recovers')
  assert.equal(compactionSummaryCount(session), 1, 'the fold lands even though the projection says it cannot fit')
  const block = rebuildBlockLedger(sessionEventsOf(session))[0]!
  assert.match(block.summary, /FIT RISK when folding: projected ~\d+ > window 100000/, 'the marker carries the projection it ignored')
  assert.match(block.summary, /paired sample/, 'and names where the inflation came from')
  assert.match(block.summary, /inflation 2\.00/, 'the clamped ratio is printed rather than hidden')
  void engine
})

test('M5: a projection that fits folds with no FIT RISK stamp', async () => {
  const ctx = new Context()
  const engine = new Named(ctx, { modelContextLimit: 100000 })
  const session = buildTextSession(30)
  // No paired sample: the fallback ratio prices a session this small far inside the window,
  // so the stamp must stay absent — a marker that always carried FIT RISK would make the
  // stamp worthless for judging which recoveries were desperate.
  const action = await requestError(ctx, fakeAgent(session, ctx), OVERFLOW)
  assert.deepEqual(action, { kind: 'retry' }, 'the fold lands')
  const block = rebuildBlockLedger(sessionEventsOf(session))[0]!
  assert.ok(!block.summary.includes('FIT RISK'), 'no risk claimed when the projection fits')
  void engine
})

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

/** `count` work nodes; user/assistant alternate, same open turn. */
function appendWork(session: Session, label: string, count: number, base: number): void {
  for (let index = 0; index < count; index += 1) {
    const seq = base + index
    if (index % 2 === 0) appendUser(session, longText(label + seq, seq))
    else appendAssistant(session, longText(label + seq, seq), 1, seq)
  }
}

// Round-11 review Q5: barriers (an instruction row, a checkpoint, the active turn) split the
// surface, so folding the LARGEST span can leave the projection above the window while a second,
// disjoint span would fit. The old single pass burned the ONE retry on the insufficient fold,
// committed a permanent barrier, and stalled the session. Extra passes inside the SAME retry
// cost a barrier and a re-prefill — never a retry.
test('M5: an insufficient fold keeps folding disjoint spans inside the SAME retry', async () => {
  const ctx = new Context()
  const engine = new Named(ctx, { modelContextLimit: 100000 })
  const session = Session.create('overflow-multi-span')
  appendTurn(session, 1)
  appendWork(session, 'segA', 6, 0)
  appendInstruction(session, 'v1')
  appendWork(session, 'segB', 6, 100)
  appendInstruction(session, 'v2')
  appendWork(session, 'tail', 5, 200)
  // A projection the FIRST fold cannot satisfy but the second can: the paired ratio clamps to 2
  // and the absolute overhead is 80K, BELOW the 100K window so the overhead cap does not stop
  // the loop. Folding segA leaves ~12.5K live → projected ~105K (FIT RISK, keep folding);
  // folding segB leaves ~6.5K → projected ~93K (fits, stop). (The old fixture used overhead
  // 130K > the window — a projection NO fold can satisfy, which round-12 review Q1 showed now
  // stops after one pass: folding removes surface, never overhead.)
  recordRequestPair(session, 125_000, 45_000)
  const action = await requestError(ctx, fakeAgent(session, ctx), OVERFLOW)
  assert.deepEqual(action, { kind: 'retry' }, 'recovery still answers the waterfall')
  const summaries = compactionSummaryCount(session)
  assert.ok(summaries >= 2, `one insufficient fold must not spend the retry — got ${summaries} compaction(s)`)
  const ledger = rebuildBlockLedger(sessionEventsOf(session))
  assert.equal(ledger.length, summaries, 'every pass wrote its own ledger block')
  assert.ok(ledger.every((block) => block.topic === 'context-overflow recovery'), 'both passes are labeled overflow recoveries')
  assert.ok(ledger.some((block) => /FIT RISK when folding/.test(block.summary)), 'the unfit projection is stamped, not hidden')
  // The retry budget is untouched: a second overflow after this one still sees the budget spent,
  // i.e. the extra passes did not manufacture retries.
  const second = await requestError(ctx, fakeAgent(session, ctx), OVERFLOW)
  assert.equal(second, undefined, 'multi-pass folding does not consume a second retry budget')
  void engine
})

// Round-12 review Q1 (both reviewers): when the fixed request head ALONE beats the window, no
// fold can bring the projection under it — a fold removes SURFACE tokens, never overhead. The
// old `if (!fitRisk) break` kept looping anyway and burned up to OVERFLOW_FOLD_PASSES permanent
// barriers for a projection that is mathematically unreachable.
test('M5: a fixed head that alone beats the window folds once, not three times', async () => {
  const ctx = new Context()
  const engine = new Named(ctx, { modelContextLimit: 100000 })
  const session = Session.create('overflow-head-alone')
  appendTurn(session, 1)
  appendWork(session, 'segA', 6, 0)
  appendInstruction(session, 'v1')
  appendWork(session, 'segB', 6, 100)
  appendInstruction(session, 'v2')
  appendWork(session, 'tail', 5, 200)
  // overhead = 260_000 − 130_000 = 130_000 ≥ the 100_000 window.
  recordRequestPair(session, 260_000, 130_000)
  assert.deepEqual(await requestError(ctx, fakeAgent(session, ctx), OVERFLOW), { kind: 'retry' })
  const summaries = compactionSummaryCount(session)
  assert.equal(summaries, 1, `an unreachable projection must stop after one fold, got ${summaries}`)
  const ledger = rebuildBlockLedger(sessionEventsOf(session))
  assert.ok(/FIT RISK when folding/.test(ledger[0]!.summary), 'the unreachable projection is still stamped')
  void engine
})

// Round-13 review Q3c: the fixed head is not the only way a projection can be unreachable. When
// the UN-foldable remainder (the protected recent tail, instruction rows, the newest skill
// catalog, the active turn) times the calibrated inflation plus the overhead already beats the
// window, NO combination of further folds can fit — yet `overheadBlocksFit` is false, so the old
// loop kept folding up to OVERFLOW_FOLD_PASSES times, each pass buying a permanent barrier for a
// number that cannot move. The reachability test is computed from the rows, not from the head.
test('M5: when the un-foldable remainder alone beats the window, folding stops after one pass', async () => {
  const ctx = new Context()
  const engine = new Named(ctx, { modelContextLimit: 100000 })
  const session = Session.create('overflow-remainder-unreachable')
  appendTurn(session, 1)
  appendWork(session, 'segA', 6, 0)
  appendInstruction(session, 'v1')
  appendWork(session, 'segB', 6, 100)
  appendInstruction(session, 'v2')
  // The recent tail is protected (preserveRecent) and alone is ~75K tokens, so even folding BOTH
  // eligible spans leaves the projection far above the 100K window: inflation 1.4 and overhead
  // 20K give a floor of ~125K. Overhead alone (20K) does NOT trip the head cap — the loop must
  // stop because the ROWS say so.
  for (let i = 0; i < 5; i += 1) appendUser(session, longText('protected recent tail ' + i, 900 + i).repeat(15))
  recordRequestPair(session, 70_000, 50_000)
  assert.deepEqual(await requestError(ctx, fakeAgent(session, ctx), OVERFLOW), { kind: 'retry' })
  const summaries = compactionSummaryCount(session)
  assert.equal(summaries, 1, `no fold can reach the window, so exactly one barrier should be spent, got ${summaries}`)
  const ledger = rebuildBlockLedger(sessionEventsOf(session))
  assert.ok(/FIT RISK when folding/.test(ledger[0]!.summary), 'the projection is still stamped rather than hidden')
  void engine
})

// Round-14 review Q4: `remainingReclaim` summed over the PASS's `pool` under-estimates what the
// NEXT pass can fold. `pool` is `aboveFloor` while any row clears the working-set floor; once that
// row is folded, the next pass finds `aboveFloor` empty and falls back to `ranges`, exposing a
// below-floor row the snapshot never counted — so the reachability break stopped after one pass on
// a surface a second fold could have brought inside the window.
test('M5: a below-floor row the next pass will expose still counts as reclaimable', async () => {
  const ctx = new Context()
  const engine = new Named(ctx, { modelContextLimit: 100000 })
  const session = Session.create('overflow-pool-blindness')
  appendTurn(session, 1)
  appendWork(session, 'segA', 4, 0)     // ~10K: folds and still leaves the 15K floor
  appendInstruction(session, 'v1')
  appendWork(session, 'segB', 10, 100)  // ~25K: BELOW the floor while segA was resident
  appendInstruction(session, 'v2')
  appendWork(session, 'tail', 3, 200)
  // inflation clamps to 2, overhead 80K (< the 100K window, so the fixed-head cap does not fire).
  // Pass 0 folds segA and still projects over the window; the pool snapshot read remainingReclaim 0
  // and stopped there. Summed over every eligible row the second fold is reachable and fits.
  recordRequestPair(session, 118_000, 38_000)
  const action = await requestError(ctx, fakeAgent(session, ctx), OVERFLOW)
  assert.deepEqual(action, { kind: 'retry' }, 'recovery still answers the waterfall')
  const ledger = rebuildBlockLedger(sessionEventsOf(session))
  assert.ok(ledger.length >= 2, `the exposed below-floor row must still be folded, got ${ledger.length} block(s)`)
  assert.ok(
    !ledger[ledger.length - 1]!.summary.includes('FIT RISK'),
    'the last pass fits the window, so it carries no risk stamp: ' + ledger[ledger.length - 1]!.summary,
  )
  void engine
})
