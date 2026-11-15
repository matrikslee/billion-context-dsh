/**
 * Rewrite-cost plumbing (src/usage.ts): the provider-anchored billing the model
 * needs to judge a compression, and the per-compress baseline behind the nudge's
 * rewrite-cost line. Real host projections, mirroring tests/shadow-price.test.ts.
 */
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { Context } from '@deepseek-ai/cordis'
import { SessionProjectionRegistry } from '@deepseek-ai/dsh-session-projection'
import { TokenMeter } from '@deepseek-ai/dsh-token-meter'
import { Session, SessionId } from '@deepseek-ai/dsh-session'
import { createUserMessage, createAssistantMessage } from '@deepseek-ai/dsh-llm'
import {
  costSinceLastCompress,
  peakRequestInput,
  providerUsage,
  recordCompressCost,
  totalRequestInput,
} from '../src/usage.ts'
import { DEFAULT_RESOLVED, renderTemplate } from '../src/prompts.ts'

async function hosted() {
  const ctx = new Context()
  await ctx.plugin(SessionProjectionRegistry)
  await ctx.plugin(TokenMeter)
  return { ctx, registry: ctx.sessionProjections, meter: ctx.get('tokenMeter') }
}

function drive(registry: { drive: (session: unknown, event: unknown) => void }, session: Session) {
  for (const event of session.snapshotEvents()) registry.drive(session, event)
}

function request(session: Session, turn: number, text: string, usage: {
  inputTokens: number
  outputTokens: number
  cacheReadTokens: number
  cacheWriteTokens: number
}) {
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

test('providerUsage reads the host tokenUsage projection and sums to total input', async () => {
  const { ctx, registry } = await hosted()
  const session = Session.create(SessionId('usage-total'))
  request(session, 1, 'q1', { inputTokens: 5000, outputTokens: 100, cacheReadTokens: 20000, cacheWriteTokens: 5000 })
  drive(registry, session)
  const usage = providerUsage({ session, ctx } as never)
  assert.ok(usage !== undefined, 'projection must expose tokenUsage on a hosted session')
  assert.equal(usage.uncachedInputTokens, 5000)
  assert.equal(usage.cacheReadTokens, 20000)
  assert.equal(usage.cacheWriteTokens, 5000)
  assert.equal(totalRequestInput(usage), 30000)

  request(session, 2, 'q2', { inputTokens: 30000, outputTokens: 80, cacheReadTokens: 0, cacheWriteTokens: 0 })
  drive(registry, session)
  const after = providerUsage({ session, ctx } as never)
  assert.ok(after !== undefined)
  assert.equal(after.uncachedInputTokens, 35000, 'a cache-miss request adds its whole input to the uncached bucket')
  assert.equal(after.cacheReadTokens, 20000, 'cache reads accumulate, they do not reset')
})

test('providerUsage degrades to undefined when the host has no usage unit', () => {
  assert.equal(providerUsage({ session: {}, ctx: { get: () => undefined } } as never), undefined)
  assert.equal(providerUsage({ session: {}, ctx: undefined } as never), undefined)
})

test('recordCompressCost + costSinceLastCompress report the prefix re-read', async () => {
  const { ctx, registry } = await hosted()
  const session = Session.create(SessionId('usage-cost'))
  request(session, 1, 'q1', { inputTokens: 40000, outputTokens: 50, cacheReadTokens: 0, cacheWriteTokens: 0 })
  drive(registry, session)
  const agent = { session, ctx } as never
  assert.equal(costSinceLastCompress(session, providerUsage(agent)), null, 'no compress yet -> no line')
  recordCompressCost(session, providerUsage(agent))
  assert.deepEqual(costSinceLastCompress(session, providerUsage(agent)), { delta: 0, count: 1 })
  request(session, 2, 'q2', { inputTokens: 60000, outputTokens: 50, cacheReadTokens: 0, cacheWriteTokens: 0 })
  drive(registry, session)
  assert.deepEqual(costSinceLastCompress(session, providerUsage(agent)), { delta: 60000, count: 1 })
  recordCompressCost(session, providerUsage(agent))
  assert.deepEqual(costSinceLastCompress(session, providerUsage(agent)), { delta: 0, count: 2 })
})

test('peakRequestInput takes the max over the log, not the newest request', async () => {
  const session = Session.create(SessionId('usage-peak'))
  request(session, 1, 'q1', { inputTokens: 30000, outputTokens: 10, cacheReadTokens: 40000, cacheWriteTokens: 0 })
  request(session, 2, 'q2', { inputTokens: 5000, outputTokens: 10, cacheReadTokens: 1000, cacheWriteTokens: 0 })
  assert.equal(peakRequestInput(session), 70000)
})

test('the default rewrite-cost line renders and the slot passes template validation', () => {
  const line = renderTemplate(DEFAULT_RESOLVED.nudge.cacheCost, { uncached: 60240, count: 2 })
  assert.match(line, /\+60240 uncached input tokens billed \(2 compress\(es\) so far\)/)
})