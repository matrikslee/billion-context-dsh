/**
 * acp_status must show the model what a fold costs it in the ONLY units that
 * matter on this machine: the LIVE request input (what a fold leaves behind to
 * re-prefill), the PEAK it has already paid, and the ROOM to the window.
 *
 * Paper vocabulary (arXiv 2611.05306 §2.4): total length vs peak length, and a
 * named compaction threshold. Rule 9 used to keep window semantics human-side;
 * measured fold pricing (one durable replace ≈ one near-full re-prefill) moved
 * them to the model, and all three cross-reviews asked for this row.
 */

import { test } from 'node:test'
import assert from 'node:assert/strict'
import { createCore, type CompressionCore } from 'acp-kernel'
import { Session } from '@deepseek-ai/dsh-session'
import { createAssistantMessage, createUserMessage } from '@deepseek-ai/dsh-llm'
import { AcpStateStore } from '../src/state.ts'
import { makeTools, type ToolEnvironment } from '../src/tools.ts'
import { lastRequestInput } from '../src/usage.ts'
import { longText } from './helpers.ts'

function makeEnv(limit = 128000): ToolEnvironment {
  return {
    kernel: createCore({}) as CompressionCore,
    store: new AcpStateStore(),
    modelContextLimit: limit,
    compressCallIdsToHide: new Set(),
  }
}

function execStub(session: Session): never {
  return {
    callId: 'call_1_status',
    name: 'acp_status',
    arguments: {},
    signal: new AbortController().signal,
    agent: { id: session.id, session, options: { provider: 'p', model: 'm' }, ctx: {} } as never,
  }
}

type Usage = { inputTokens: number; outputTokens: number; cacheReadTokens: number; cacheWriteTokens: number }

/** One request with provider usage attached (the real event shape). */
function request(session: Session, turn: number, text: string, usage: Usage): void {
  session.append('turn/start', { turn })
  session.append('step/start', { turn, step: 1 })
  session.append('user/message', createUserMessage({ content: [{ type: 'text', text }], source: { kind: 'user' } }), { surfaceOp: 'append' })
  session.append('assistant/message', {
    turn,
    step: 1,
    stream: [],
    message: createAssistantMessage({ content: [{ type: 'text', text: 'answer' }], provider: 'p', model: 'm' }),
    usage,
  }, { surfaceOp: 'append' })
  session.append('step/end', { turn, step: 1 })
}

test('lastRequestInput reads the newest request, not a cumulative sum', () => {
  const session = Session.create('sizes-last')
  request(session, 1, longText('u1', 1), { inputTokens: 2000, outputTokens: 100, cacheReadTokens: 60000, cacheWriteTokens: 0 })
  request(session, 2, longText('u2', 2), { inputTokens: 3000, outputTokens: 120, cacheReadTokens: 1000, cacheWriteTokens: 0 })
  assert.equal(lastRequestInput(session), 4000, 'the newest request input, not 62000 nor 66000')
})

test('acp_status shows live, peak, window and room in one row', async () => {
  const session = Session.create('sizes-status')
  request(session, 1, longText('u1', 1), { inputTokens: 2000, outputTokens: 100, cacheReadTokens: 60000, cacheWriteTokens: 0 })
  request(session, 2, longText('u2', 2), { inputTokens: 3000, outputTokens: 120, cacheReadTokens: 1000, cacheWriteTokens: 0 })
  const definition = makeTools(makeEnv()).find((candidate) => candidate.name === 'acp_status')
  assert.ok(definition, 'acp_status registered')
  const result = await definition.execute({}, execStub(session))
  const text = (result as { text: string }).text
  assert.match(text, /Request input sizes: live 4000 \| peak 62000 \| window 128000 \| room 124000/, 'the row names all four numbers')
})

test('the row degrades quietly when the session has no billed request yet', async () => {
  const session = Session.create('sizes-empty')
  session.append('user/message', createUserMessage({ content: [{ type: 'text', text: 'hello' }], source: { kind: 'user' } }), { surfaceOp: 'append' })
  const definition = makeTools(makeEnv()).find((candidate) => candidate.name === 'acp_status')
  const result = await definition.execute({}, execStub(session))
  assert.doesNotMatch((result as { text: string }).text, /Request input sizes/, 'no invented numbers')
})