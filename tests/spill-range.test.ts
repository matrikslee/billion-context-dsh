/**
 * Spill-aware range rows (paper-derived gap G3, arXiv 2611.05306 RQ2).
 *
 * The host writes an oversized tool stdout/stderr to a file, keeps those bytes
 * OUT of the surface, and names the file in the result. A row carrying that mark
 * is the safest fold on the surface — recoverable by a narrow read — so the table
 * must SAY so, otherwise the model treats a big tool result as unrecoverable and
 * keeps paying for it.
 */

import { test } from 'node:test'
import assert from 'node:assert/strict'
import { Session } from '@deepseek-ai/dsh-session'
import { appendAssistant, appendToolCall, appendToolResult, appendTurn, appendUser, longText, wholeSurfaceRangeView } from './helpers.ts'
import { buildCompressibleSeqRanges } from '../src/region.ts'
import { rangeTable } from '../src/nudge.ts'
import { spillPathsOfEvent } from '../src/messages.ts'

// What dsh-bash-local returns when the tail spilled: a JSON document whose
// stdout/stderr objects carry spillPath (single read) or stdoutSpillPath /
// stderrSpillPath (streamed command).
const SPILLED = JSON.stringify({
  kind: 'foreground',
  stdout: { text: longText('stdout', 5), truncated: true, spillPath: '/tmp/dsh-job-abc/stdout.log' },
  stderr: { text: 'warning: two lines dropped', truncated: false },
})

// A truncated document the model still has to read the file name out of.
const TRUNCATED_SPILL = '{ "kind": "foreground", "stdout": { "truncated": true, "stdoutSpillPath": "/tmp/dsh-job-xyz/out.log" } … truncated'

const PLAIN = JSON.stringify({ kind: 'foreground', stdout: { text: longText('plain', 6), truncated: false } })

function spillFixture(id: string): Session {
  const session = Session.create(id)
  appendTurn(session, 1)
  appendUser(session, longText('u1', 1) + longText('u1b', 2))
  appendToolCall(session, longText('a2', 3), 'c1')
  appendToolResult(session, SPILLED, 'c1')
  appendToolCall(session, longText('a3', 4), 'c2')
  appendToolResult(session, TRUNCATED_SPILL, 'c2')
  appendToolCall(session, longText('a4', 12), 'c3')
  appendToolResult(session, PLAIN, 'c3')
// Past the protected tail (recent 5 + the last real user turn) so BOTH
// spilled results sit inside an eligible run rather than in the guard window.
  appendUser(session, longText('u5', 7) + longText('u5b', 8))
  appendAssistant(session, longText('a6', 9))
  appendUser(session, longText('u7', 10))
  appendAssistant(session, longText('a8', 11))
  appendUser(session, longText('u9', 13))
  appendAssistant(session, longText('a10', 14))
  appendUser(session, longText('u11', 15))
  appendAssistant(session, longText('a12', 16))
  return session
}

test('a result that only DOCUMENTS the field is not counted as spilled', () => {
  const session = Session.create('spill-prose')
  appendTurn(session, 1)
  appendUser(session, longText('u1', 1))
  appendToolCall(session, longText('a2', 3), 'p1')
  appendToolResult(session, 'Reviewing the scanner: it reads "spillPath" and "stdoutSpillPath" keys out of tool output. Nothing was written to disk.', 'p1')
  const results = session.snapshotEvents().filter((event) => event.type === 'tool/result')
  assert.deepEqual(spillPathsOfEvent(results[0]!), [], 'prose about the key is not a spill — the row must not claim bytes are on disk')
})

test('a path carrying escaped quotes survives the structural parse', () => {
  const session = Session.create('spill-escaped')
  appendTurn(session, 1)
  appendUser(session, longText('u1', 1))
  appendToolCall(session, longText('a2', 3), 'e1')
  const path = '/tmp/dsh job/\"odd\" out.log'
  appendToolResult(session, JSON.stringify({ kind: 'foreground', stdout: { truncated: true, spillPath: path } }), 'e1')
  const results = session.snapshotEvents().filter((event) => event.type === 'tool/result')
  assert.deepEqual(spillPathsOfEvent(results[0]!), [path], `escaped by JSON.stringify, returned intact: ${JSON.stringify(spillPathsOfEvent(results[0]!))}`)
})

test('spill file names are read out of the tool result JSON', () => {
  const session = spillFixture('spill-detect')
  const results = session.snapshotEvents().filter((event) => event.type === 'tool/result')
  assert.equal(results.length, 3, 'three tool results in the fixture')
  assert.deepEqual(spillPathsOfEvent(results[0]!), ['/tmp/dsh-job-abc/stdout.log'], 'valid JSON document: key read structurally')
  assert.deepEqual(spillPathsOfEvent(results[1]!), ['/tmp/dsh-job-xyz/out.log'], 'truncated JSON: value still recovered by pattern')
  assert.deepEqual(spillPathsOfEvent(results[2]!), [], 'a plain tool result reports no spill')
})

test('the spill gate is keyed on the JSON shape, not on where the key sits', () => {
  const session = Session.create('spill-shapes')
  appendTurn(session, 1)
  appendUser(session, longText('u1', 1))
  // Nested one level down and pretty-printed with other keys first: both are
  // serialized results the host can legitimately emit, and the gate must catch
  // them (round-4 Q5 — a gate that requires the key right after the brace is
  // brittle). Documenting the key is still not a spill.
  appendToolCall(session, longText('a1', 3), 'e1')
  appendToolResult(session, JSON.stringify({ ok: true, result: { spillPath: '/tmp/nested.log' } }), 'e1')
  appendToolCall(session, longText('a2', 3), 'e2')
  appendToolResult(session, '{\n  "kind": "foreground",\n  "stdout": {\n    "spillPath": "/tmp/pretty.log"\n  }\n}', 'e2')
  appendToolCall(session, longText('a3', 3), 'e3')
  appendToolResult(session, '{"note": "the spillPath key documents where output went"}', 'e3')
  const results = session.snapshotEvents().filter((event) => event.type === 'tool/result')
  assert.deepEqual(spillPathsOfEvent(results[0]!), ['/tmp/nested.log'], 'nested under another key')
  assert.deepEqual(spillPathsOfEvent(results[1]!), ['/tmp/pretty.log'], 'pretty-printed across lines')
  assert.deepEqual(spillPathsOfEvent(results[2]!), [], 'prose inside a JSON object is still not a spill')
})

test('a compressible span counts the spilled rows it contains', () => {
  const session = spillFixture('spill-count')
  const ranges = buildCompressibleSeqRanges(session, wholeSurfaceRangeView(session))
  assert.ok(ranges.length > 0, 'the fixture produces at least one range')
  const total = ranges.reduce((sum, range) => sum + range.spills, 0)
  assert.equal(total, 2, `both spilled results are counted: ${JSON.stringify(ranges.map((r) => r.spills))}`)
})

test('the range table marks spilled rows and says they are re-readable', () => {
  const session = spillFixture('spill-row')
  const table = rangeTable(session, wholeSurfaceRangeView(session))
  assert.match(table, /\d+ spilled/, 'the row carries the spill mark')
  assert.match(table, /readable from disk/, 'the footer states why the mark matters')
})

// Round-11 review Q3: a decorated wrapper (the pre-0.2.0 host shape) carries an UPPERCASE S
// in stdoutSpillPath/stderrSpillPath, and its stdout value can itself contain braces — the
// delimited slice used to stop at that inner `}`, so the parse failed AND the fallback
// searched a substring that no longer held the key. Both read as "no spill" on a real spill.
test('a decorated wrapper yields its spill path despite the uppercase key and inner braces', () => {
  const session = Session.create('spill-inner-brace')
  appendTurn(session, 1)
  appendUser(session, longText('u1', 1))
  appendToolCall(session, longText('a2', 3), 'b1')
  appendToolResult(session, 'Command exited 1\n{"stdout":"fn(){return 1}","stdoutSpillPath":"/tmp/dsh-job-brace/stdout.log","exitCode":1}', 'b1')
  appendToolCall(session, longText('a3', 4), 'b2')
  // Same shape, truncated before the closing brace: the last `}` now belongs to the inner
  // field, so only a scan that runs to the END of the text can still find the key.
  appendToolResult(session, 'Command exited 1\n{"stdout":"fn(){return 1}","stdoutSpillPath":"/tmp/dsh-job-trunc/stdout.log"', 'b2')
  const results = session.snapshotEvents().filter((event) => event.type === 'tool/result')
  assert.deepEqual(spillPathsOfEvent(results[0]!), ['/tmp/dsh-job-brace/stdout.log'], `uppercase key found despite braces inside stdout: ${JSON.stringify(spillPathsOfEvent(results[0]!))}`)
  assert.deepEqual(spillPathsOfEvent(results[1]!), ['/tmp/dsh-job-trunc/stdout.log'], `truncated wrapper still named by the fallback: ${JSON.stringify(spillPathsOfEvent(results[1]!))}`)
})

test('an undelimited wrapper and a decorated header quoting the key claim nothing', () => {
  const session = Session.create('spill-conservative')
  appendTurn(session, 1)
  appendUser(session, longText('u1', 1))
  appendToolCall(session, longText('a2', 3), 'c1')
  // Truncated with NO closing brace at all: an undelimited wrapper cannot prove the bytes went
  // to a file, and a false mark makes the model fold real in-context bytes (round-10 S3). A
  // missed mark costs one row marker; a false mark costs the content.
  appendToolResult(session, 'Command exited 1\n{"stdoutSpillPath":"/tmp/dsh-job-cut/stdout.log"', 'c1')
  appendToolCall(session, longText('a3', 4), 'c2')
  // `[stderr]` is a decorated header, not an array document (`[{`): prose after it that merely
  // QUOTES the key must not mark the row (round-11 review Q3.2).
  appendToolResult(session, '[stderr]\nDocs note that "stdoutSpillPath": "/tmp/quoted-only.log" names a file; nothing was spilled here.', 'c2')
  const results = session.snapshotEvents().filter((event) => event.type === 'tool/result')
  assert.deepEqual(spillPathsOfEvent(results[0]!), [], 'an undelimited wrapper claims nothing — the conservative side is the safe side')
  assert.deepEqual(spillPathsOfEvent(results[1]!), [], 'a [word] header plus quoted prose is not an array document')
})

// ── round-15 review Q1/Q2: the host writes spills as PROSE on its hottest tools ────
// Verbatim tail of two live bash results (seq 3157 / seq 3220, each a 3 MB stdout), measured
// 2026-11-15: BOTH sentences, no JSON key at all. The key probe alone marked zero rows.
const PROSE_TAIL =
  'work line 1\nwork line 2\n\n[output truncated; full output: /tmp/dsh-subprocess-v5Resx/dsh-subprocess-882-1-69f2853ec1d3-stdout.log]\n\n' +
  '(Omitted 48373 bytes. Full formatted result stored at: /tmp/dsh-spill-DELYH2/session-bd16ec674eb0/b22bfdc180c0-bash.txt. Use read with offset/limit, or grep this path to search within it.)'

function oneResult(id: string, text: string): Session {
  const session = Session.create(id)
  appendTurn(session, 1)
  appendUser(session, longText('u1', 1))
  appendToolCall(session, longText('a2', 2), 'c1')
  appendToolResult(session, text, 'c1')
  return session
}

function toolResults(session: Session) {
  return session.snapshotEvents().filter((event) => event.type === 'tool/result')
}

test('the host prose notice marks a real spill on both of its forms', () => {
  const results = toolResults(oneResult('spill-prose', PROSE_TAIL))
  assert.deepEqual(
    spillPathsOfEvent(results[0]!),
    ['/tmp/dsh-subprocess-v5Resx/dsh-subprocess-882-1-69f2853ec1d3-stdout.log', '/tmp/dsh-spill-DELYH2/session-bd16ec674eb0/b22bfdc180c0-bash.txt'],
    `both sentences name a file, so both must be collected: ${JSON.stringify(spillPathsOfEvent(results[0]!))}`,
  )
})

test('prose that only QUOTES the notice, or names no spill directory, is not a spill', () => {
  // grep over dsh-tool-bash/lib/index.js and dsh-spill-policy/lib/index.js prints the SOURCE:
  // template placeholders and prose, never a concrete spill path (round-15 MISSED 2).
  const quoted = toolResults(oneResult('spill-quote', 'index.js:144: `\n[output truncated; full output: ${spillPath}]`\nindex.js:9: LOCATION = " Full formatted result stored at: "'))
  const outside = toolResults(oneResult('spill-outside', '(Omitted 12 bytes. Full formatted result stored at: /tmp/whatever/plain.txt. Use read)'))
  const globbed = toolResults(oneResult('spill-glob', '[output truncated; full output: /tmp/dsh-spill-…/glob-star.txt]'))
  assert.deepEqual(spillPathsOfEvent(quoted[0]!), [], 'the host SOURCE carries templates, not a spill path — 0 marks')
  assert.deepEqual(spillPathsOfEvent(outside[0]!), [], 'a path outside a dsh-spill-*/dsh-subprocess-* directory is not evidence')
  assert.deepEqual(spillPathsOfEvent(globbed[0]!), [], 'an ellipsis placeholder is not a concrete path')
})

test('the prose scan is limited to tool results', () => {
  const session = Session.create('spill-user-row')
  appendTurn(session, 1)
  appendUser(session, 'Here is what the host printed: ' + PROSE_TAIL)
  const userRows = session.snapshotEvents().filter((event) => event.type === 'user/message')
  assert.deepEqual(spillPathsOfEvent(userRows[0]!), [], 'a user row QUOTING the notice is not a spill — the scan is gated on tool/result')
})

test('the plural spillPaths array is read structurally and from truncated JSON', () => {
  const session = Session.create('spill-plural')
  appendTurn(session, 1)
  appendUser(session, longText('u1', 1))
  appendToolCall(session, longText('a2', 2), 'j1')
  appendToolResult(session, JSON.stringify({ jobId: 'j', spillPaths: ['/tmp/dsh-job-p/spill.log', '/tmp/dsh-job-p/stderr.log'] }), 'j1')
  appendToolCall(session, longText('a3', 3), 'j2')
  appendToolResult(session, '{"jobId": "j", "spillPaths": ["/tmp/dsh-job-q/out.log"', 'j2')
  appendToolCall(session, longText('a4', 4), 'j3')
  appendToolResult(session, '{"note": "spillPaths is a key in the payload, no file was written"}', 'j3')
  const results = toolResults(session)
  assert.deepEqual(spillPathsOfEvent(results[0]!), ['/tmp/dsh-job-p/spill.log', '/tmp/dsh-job-p/stderr.log'], 'array elements are paths the parent key vouches for (round-15 Q2)')
  assert.deepEqual(spillPathsOfEvent(results[1]!), ['/tmp/dsh-job-q/out.log'], 'truncated JSON: the array scan still finds the element')
  assert.deepEqual(spillPathsOfEvent(results[2]!), [], 'prose naming the plural key inside an object is still not a spill')
})

// ── round-16 review Q3: the prose gate's three edges ─────────────────────────────
test('a prose notice inside a JSON string field is still found after parsing', () => {
  // The host escapes newlines inside a string field, so the notice only becomes matchable
  // AFTER JSON.parse; the recursive call must keep the tool-result flag (round-16 Q3.1).
  const body = JSON.stringify({ kind: 'foreground', output: 'line a\nline b\n[output truncated; full output: /tmp/dsh-subprocess-QQ/dsh-subprocess-9-1-abc-stdout.log]' })
  const results = toolResults(oneResult('spill-json-prose', body))
  assert.deepEqual(
    spillPathsOfEvent(results[0]!),
    ['/tmp/dsh-subprocess-QQ/dsh-subprocess-9-1-abc-stdout.log'],
    `parsed value carries the notice: ${JSON.stringify(spillPathsOfEvent(results[0]!))}`,
  )
})

test('a path followed immediately by another bracketed marker stops at the first bracket', () => {
  const results = toolResults(oneResult('spill-bracket', '[output truncated; full output: /tmp/dsh-spill-ZZ/session-q/b22bfdc180c0-bash.txt][stderr] tail lines'))
  assert.deepEqual(spillPathsOfEvent(results[0]!), ['/tmp/dsh-spill-ZZ/session-q/b22bfdc180c0-bash.txt'], 'the greedy class must not eat the `]` of the next marker (round-16 Q3.2)')
})

test('a large tool result with no spill marker costs the scan nothing and marks nothing', () => {
  const filler = 'processing line\n'.repeat(40_000)
  assert.ok(!filler.includes('/dsh-'), 'fixture has no spill directory substring')
  const results = toolResults(oneResult('spill-noop', filler))
  assert.deepEqual(spillPathsOfEvent(results[0]!), [], 'the `/dsh-` substring gate skips the regex scans (round-16 Q3.3)')
})
