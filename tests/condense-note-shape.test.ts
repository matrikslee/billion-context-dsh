/**
 * Note SHAPE advisories (gaps G4 + G5, arXiv 2611.05306 Appendix A/G).
 *
 * Advisory by design: the cross-reviews rejected a hard six-heading gate (resident
 * prompt text + a retry loop costs more on an open-weight model than the fields it
 * forces) and rejected a hard anti-duplication reject (false positives on vocabulary
 * a resumed session genuinely needs). Goal/Next stay enforced — a note without them
 * turns one fold into a re-run of consumed exploration.
 */

import { test } from 'node:test'
import assert from 'node:assert/strict'
import { createCore, defaultCountTokens, type CompressionCore } from 'acp-kernel'
import { Session } from '@deepseek-ai/dsh-session'
import { AcpStateStore } from '../src/state.ts'
import { condenseNoteGaps, makeTools, MAX_NOTE_WEIGHT_RATIO, noteWeightRatio, type ToolEnvironment } from '../src/tools.ts'
import { appendAssistant, appendTurn, appendUser, longText } from './helpers.ts'

const MINIMAL = ['## Goal', 'Ship the note-shape advisory.', '## Next', 'Run the suite; 0 fail is the completion condition.'].join('\n')

const FULL = [
  '## Objective', 'Ship G4/G5.',
  '## Established facts', 'foldFloorPct defaults to 0.15; spill keys are spillPath / stdoutSpillPath / stderrSpillPath.',
  '## Applied changes', 'src/tools.ts adds condenseNoteGaps and noteDuplicationWarning.',
  '## Verification', 'node --import tsx --test: 0 fail; tsc --noEmit clean.',
  '## Dead ends', 'Hard six-heading gate rejected — costs resident text plus retries.',
  '## Remaining plan', 'Commit; then cross-review the whole batch. Done when tests are green.',
].join('\n')

function env(): ToolEnvironment {
  return {
    kernel: createCore({}) as CompressionCore,
    store: new AcpStateStore(),
    modelContextLimit: 128000,
    foldFloorPct: 0,
    compressCallIdsToHide: new Set(),
  }
}

function execStub(session: Session): never {
  return {
    callId: 'call_1_note',
    name: 'compress',
    arguments: {},
    agent: { id: session.id, session, options: { provider: 'p', model: 'm' }, ctx: {} } as never,
  }
}

function fixture(id: string): Session {
  const session = Session.create(id)
  appendTurn(session, 1)
  appendUser(session, longText('task', 3) + longText('task2', 3))
  appendAssistant(session, longText('work', 3) + longText('work2', 3))
  appendUser(session, longText('ask', 3) + longText('ask2', 3))
  appendAssistant(session, longText('mid', 3) + longText('mid2', 3))
  appendUser(session, longText('tail1', 2))
  appendAssistant(session, longText('tail2', 2))
  appendUser(session, longText('tail3', 2))
  return session
}

test('a Goal/Next-only note names the missing Appendix G fields without blocking', () => {
  const gaps = condenseNoteGaps(MINIMAL)
  assert.ok(gaps.includes('Verification'), 'the field a resumed session re-derives most expensively')
  assert.ok(gaps.includes('Dead ends'), 'the field that stops a retry of a known failure')
  assert.equal(gaps.length, 4, `Objective and Remaining plan are covered by Goal/Next: ${gaps.join(', ')}`)
})

test('the six-field note reports no gaps', () => {
  assert.deepEqual(condenseNoteGaps(FULL), [])
})

test('condense folds a thin note and warns about the missing fields', async () => {
  const session = fixture('shape-warn')
  const definition = makeTools(env()).find((candidate) => candidate.name === 'compress')
  const result = await definition!.execute({ condense: { summary: MINIMAL } } as never, execStub(session) as never)
  const text = (result as { text: string }).text
  assert.match(text, /condense elected seqs/, 'the fold lands — the shape check is advisory')
  assert.match(text, /Appendix G field\(s\)/, 'and the missing fields are named in the result')
})

test('the six-field note lands without a shape warning', async () => {
  const session = fixture('shape-full')
  const definition = makeTools(env()).find((candidate) => candidate.name === 'compress')
  const result = await definition!.execute({ condense: { summary: FULL } } as never, execStub(session) as never)
  assert.doesNotMatch((result as { text: string }).text, /Appendix G field\(s\)/)
})

// The post-fold lexical duplication warning is GONE (three-reviewer verdict): it
// fires after the fold lands, so obeying it costs a second full re-prefill, and
// shingles flag the verbatim identifiers a note SHOULD carry. What replaced it is
// deterministic, pre-fold and language-independent.
test('a note that is a second copy of the stretch is refused before anything lands', async () => {
  const session = fixture('weight-heavy')
  const heavy = MINIMAL + '\n' + longText('work', 3) + longText('work2', 3) + longText('mid', 3) + longText('mid2', 3)
  const definition = makeTools(env()).find((candidate) => candidate.name === 'compress')
  const result = await definition!.execute({ condense: { summary: heavy } } as never, execStub(session) as never)
  const text = (result as { text: string }).text
  assert.match(text, /condense refused — your note is ~\d+% of the \d+ tokens it replaces/)
  assert.match(text, /second copy, not a briefing/)
  assert.equal(session.snapshotEvents().filter((event) => event.type === 'compaction/summary').length, 0, 'nothing written — rewrite and resend costs no re-prefill')
})

test('the ratio is note tokens over the tokens the fold removes', () => {
  assert.equal(noteWeightRatio('anything', 0), Number.POSITIVE_INFINITY, 'a span with nothing to remove cannot carry ANY note (round-11 Q7)')
  const note = longText('brief', 4)
  const ratio = noteWeightRatio(note, defaultCountTokens(note))
  assert.ok(ratio > 0.95 && ratio <= 1.05, `a note the size of its span reads as ~1: ${ratio}`)
  assert.equal(MAX_NOTE_WEIGHT_RATIO, 0.35)
})

test('a lean note clears the weight check and lands', async () => {
  const session = fixture('weight-lean')
  const definition = makeTools(env()).find((candidate) => candidate.name === 'compress')
  const result = await definition!.execute({ condense: { summary: MINIMAL } } as never, execStub(session) as never)
  assert.doesNotMatch((result as { text: string }).text, /second copy, not a briefing/)
  assert.equal(session.snapshotEvents().filter((event) => event.type === 'compaction/summary').length, 1)
})