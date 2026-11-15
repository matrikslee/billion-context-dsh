/**
 * round-16 review Q4 — `setCalibrationRoute` runs on EVERY `agent/pre-step`, and a route switch
 * must drop the samples: both the tokenizer ratio and the fixed `requestOverhead` were measured
 * on the PREVIOUS route, so a stale pair can read a huge overhead on a smaller-window route and
 * trip `requestOverhead >= window.limit`, stopping emergency recovery after one pass.
 * `routeFor` shares that call site, where its contract is that it never throws.
 */
import { test } from 'node:test'
import assert from 'node:assert/strict'
import type { Session } from '@deepseek-ai/dsh-session'
import { calibrateRequestPair, pairedInflation, requestOverhead, setCalibrationRoute } from '../src/usage.ts'
import { routeFor } from '../src/window.ts'

function makeSession(startSeq: number, logTail = startSeq) {
  const state = { logTail }
  const session = {
    surface: { nodes: [1, startSeq] as number[] },
    snapshotEvents: () => [{ seq: state.logTail }],
  } as unknown as Session
  const advance = (seq: number) => {
    ;(session.surface as unknown as { nodes: number[] }).nodes = [1, seq]
    state.logTail = seq
  }
  return { session, advance }
}

/** Anchor on the OLD route, then bill a request: overhead 30K, ratio ≈ 1.16. */
function pairOn(session: Session, advance: (seq: number) => void): void {
  calibrateRequestPair(session, 0, 0, 100_000)
  advance(120)
  calibrateRequestPair(session, 130_000, 120, 112_000)
}

test('a route switch drops both derived quantities', () => {
  const { session, advance } = makeSession(100)
  setCalibrationRoute(session, 'deepseek/old-model')
  pairOn(session, advance)
  assert.equal(requestOverhead(session), 30_000, 'the old route carries its measured overhead')
  assert.ok(pairedInflation(session) !== undefined, 'and its measured ratio')
  setCalibrationRoute(session, 'deepseek/new-model')
  assert.equal(requestOverhead(session), 0, 'a different tokenizer leaves no overhead to inherit')
  assert.equal(pairedInflation(session) ?? null, null, 'nor a ratio — the conservative fallback takes over')
})

test('repeating the same route key keeps the samples', () => {
  const { session, advance } = makeSession(100)
  setCalibrationRoute(session, 'deepseek/one-model')
  pairOn(session, advance)
  for (let step = 0; step < 3; step += 1) setCalibrationRoute(session, 'deepseek/one-model')
  assert.equal(requestOverhead(session), 30_000, 'this runs every step; a no-op switch must not discard real samples')
})

test('routeFor degrades instead of throwing when the agent carries no options snapshot', () => {
  const bare = { session: { requestContext: () => undefined } } as unknown as Parameters<typeof routeFor>[0]
  assert.deepEqual(routeFor(bare), { provider: '', model: '' }, 'a host-lite or detached agent must not crash the step')
})
