import { test } from "node:test"
import assert from "node:assert/strict"
import type { Session } from "@deepseek-ai/dsh-session"
import { calibrateRequestPair, pairedInflation, requestOverhead, setCalibrationAnchor } from "../src/usage.ts"

/**
 * The anchor is dated by the newest seq the session recorded — the max of the surface tail and
 * the log tail, because a `surfaceOp: replace` splices its seq IN PLACE (the surface tail can
 * sit BELOW the log's newest event) and an `assistant/attempt` billed row is not a surface node
 * at all (round-13 review Q1/Q4). So the fixture carries both knobs: `nodes` for the surface and
 * `logTail` for the newest event, advanced independently.
 */
function makeSession(startSeq: number, logTail = startSeq) {
  const state = { logTail }
  const session = {
    surface: { nodes: [1, startSeq] as number[] },
    snapshotEvents: () => [{ seq: state.logTail }],
  } as unknown as Session
  const setSurface = (seq: number) => {
    ;(session.surface as unknown as { nodes: number[] }).nodes = [1, seq]
  }
  return {
    session,
    /** Move the surface tail and the log tail together (the normal step case). */
    advance: (seq: number) => {
      setSurface(seq)
      state.logTail = seq
    },
    /** Move them apart: an interior replace keeps the surface tail low while the log grows. */
    advanceSplit: (surfaceSeq: number, logSeq: number) => {
      setSurface(surfaceSeq)
      state.logTail = logSeq
    },
  }
}

/**
 * Round-10 review S2: the pair must describe ONE request. The newest usage row on the
 * log belongs to the PREVIOUS step, so the surface it was built from is the surface of
 * that step — pairing it with the current, larger surface would shrink both the tokenizer
 * ratio and the absolute overhead toward zero, which is the direction that lets a fold
 * that still overflows be accepted.
 */
test("the calibration pair compares the billed request with the PREVIOUS step surface", () => {
  const { session, advance } = makeSession(100)
  // First pre-step: no anchor yet, nothing recorded — the caller keeps the fallback.
  calibrateRequestPair(session, 0, 0, 100_000)
  assert.equal(pairedInflation(session), undefined)
  assert.equal(requestOverhead(session), 0)
  // Second pre-step: the usage row (130_000) belongs to the request built from 100_000,
  // not from the 112_000 this step sees. Its seq (120) is newer than the anchor seq (100).
  advance(120)
  calibrateRequestPair(session, 130_000, 120, 112_000)
  const pair = pairedInflation(session)
  assert.ok(pair !== undefined)
  assert.ok(pair! >= 1.29 && pair! <= 1.31, `ratio ${pair}`)
  assert.equal(requestOverhead(session), 30_000)
  // A step with no billed usage still refreshes the anchor for the next request.
  advance(130)
  calibrateRequestPair(session, 0, 0, 120_000)
  assert.equal(requestOverhead(session), 30_000)
})

test("a degenerate anchor is never stored, so the pair cannot be poisoned", () => {
  const { session } = makeSession(50)
  calibrateRequestPair(session, 0, 0, 0)
  calibrateRequestPair(session, 90_000, 50, 90_000)
  assert.equal(pairedInflation(session), undefined)
})

/**
 * Round-11 Q4 + round-12 Q2: a request that produced NO usage row leaves the newest billed row
 * untouched, so pairing it again with a newer anchor would forge a tokenizer ratio the engine
 * never observed. The guard dates the pair by the billed ROW'S SEQ, not only by its value.
 */
test("a stale billed row (no new usage) records nothing, even against a newer anchor", () => {
  const { session, advance } = makeSession(100)
  calibrateRequestPair(session, 0, 0, 100_000)
  advance(120)
  calibrateRequestPair(session, 130_000, 120, 112_000)
  assert.equal(requestOverhead(session), 30_000)
  // The engine folded / injected rows (anchor now 130) but the newest billed row is still
  // the OLD seq-120 request — the pair must not be re-forged from it.
  advance(130)
  calibrateRequestPair(session, 130_000, 120, 140_000)
  assert.equal(requestOverhead(session), 30_000)
})

/**
 * Round-12 Q2: the OLD guard dropped any repeat of the billed VALUE, so a genuinely new
 * request that happens to bill the same total was silently discarded. Both components
 * (billed, anchor seq) repeating is what makes a pair a repeat.
 */
test("a new request with an identical billed total is still a fresh sample", () => {
  const { session, advance } = makeSession(100)
  calibrateRequestPair(session, 0, 0, 100_000)
  advance(120)
  calibrateRequestPair(session, 130_000, 120, 112_000)
  assert.equal(requestOverhead(session), 30_000)
  // Newer usage row (seq 140) with the same billed number against the newer anchor (112_000).
  advance(140)
  calibrateRequestPair(session, 130_000, 140, 120_000)
  assert.equal(requestOverhead(session), 130_000 - 112_000)
  const pair = pairedInflation(session)
  assert.ok(pair !== undefined && pair! > 1.15 && pair! < 1.17, `ratio ${pair}`)
})

/**
 * Round-11 Q4: an emergency fold re-points the anchor at the folded surface, and the retry's
 * billed row must be dated against THAT seq — a billed row older than the folded anchor is
 * never paired with it.
 */
test("setCalibrationAnchor dates the anchor so a pre-fold row cannot pair with it", () => {
  const { session, advance } = makeSession(200)
  calibrateRequestPair(session, 0, 0, 260_000)
  // The fold lands: the anchor becomes the folded surface, dated at the current last seq.
  advance(210)
  setCalibrationAnchor(session, 130_000)
  // The pre-fold billed row (seq 200) is older than the folded anchor (seq 210) — not paired.
  calibrateRequestPair(session, 260_000, 200, 130_000)
  assert.equal(pairedInflation(session), undefined)
  assert.equal(requestOverhead(session), 0)
  // The retried request (seq 220, billed 150_000) is the first honest pair.
  advance(220)
  calibrateRequestPair(session, 150_000, 220, 140_000)
  assert.equal(requestOverhead(session), 150_000 - 130_000)
  const pair = pairedInflation(session)
  assert.ok(pair !== undefined && pair! >= 1.14 && pair! <= 1.16, `ratio ${pair}`)
})

/**
 * Round-13 review Q1/Q4, the half the surface tail could not see: a fold rewrote an interior
 * slice, so the checkpoint seq (106) landed mid-array and the surface tail stayed 104, while the
 * rejected request wrote an `assistant/attempt` with usage at seq 105. Dated by the surface tail
 * alone, the anchor read 104 and that PRE-fold attempt looked newer than the fold — pairing them
 * again forges the ~2x inflation the anchor exists to prevent. Dated by the log tail (105), the
 * attempt is not newer and is refused.
 */
test("an interior replace keeps the surface tail below the log tail; the anchor dates by the log", () => {
  const { session, advanceSplit } = makeSession(104, 104)
  calibrateRequestPair(session, 0, 0, 200_000)
  // Surface tail 104 (the replace spliced seq 106 in place), log tail 105 (the attempt row).
  advanceSplit(104, 105)
  setCalibrationAnchor(session, 120_000)
  // The pre-fold attempt row (seq 105) is NOT newer than the anchor (dated 105) — never paired.
  calibrateRequestPair(session, 210_000, 105, 120_000)
  assert.equal(pairedInflation(session), undefined, 'a pre-fold attempt must not pair with the post-fold anchor')
  assert.equal(requestOverhead(session), 0)
  // The retried request (seq 106) is the first honest pair.
  advanceSplit(106, 106)
  calibrateRequestPair(session, 140_000, 106, 120_000)
  assert.equal(requestOverhead(session), 20_000)
})
