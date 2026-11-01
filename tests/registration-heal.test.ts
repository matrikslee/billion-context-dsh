/**
 * Registration self-heal (src/registration.ts).
 *
 * Live signature this fixes (2026-11-01): after an interrupted agent loop was
 * resumed, the host had rebuilt its tool registry, so the four ACP tools were
 * absent from the step's tool surface and every model call answered
 * `unknown tool` — for two steps, until a fresh plugin instance mounted. The
 * activation `done` guard meant the mounted plugin never tried again.
 *
 * These tests drive the heal against registry stand-ins that speak the host's
 * actual lookup shapes (`tools.get(name, scope)` for one scope's view,
 * `tools.get(name)` for the global view, `commands.find(agent, name)`) rather
 * than a cordis harness — the heal's job is
 * exactly "ask the registry what is missing, insert what is missing", and the
 * registry-side semantics (scope-local restrictions, duplicate inserts) are what
 * the loop-safety rules depend on.
 */

import { test } from 'node:test'
import assert from 'node:assert/strict'
import { createCore, type CompressionCore } from 'acp-kernel'
import { Context } from '@deepseek-ai/cordis'
import type { Agent } from '@deepseek-ai/dsh-agent'
import { AcpStateStore } from '../src/state.ts'
import type { ToolEnvironment } from '../src/tools.ts'
import {
  ACP_COMMAND_NAME,
  ACP_TOOL_NAMES,
  healRegistrations,
  shouldRestoreTool,
  type CommandsRegistry,
  type ToolsRegistry,
} from '../src/registration.ts'

function makeEnv(): ToolEnvironment {
  return {
    kernel: createCore({}) as CompressionCore,
    store: new AcpStateStore(),
    modelContextLimit: 128000,
    compressCallIdsToHide: new Set(),
  }
}

/** Stand-in for `dsh-tools`: a global layer (`insert`) plus a per-agent mask. */
class FakeTools implements ToolsRegistry {
  global = new Map<string, unknown>()
  /** Names a scope restriction hides from THIS agent (scope-local, per the host). */
  deniedForAgent = new Set<string>()
  inserted: string[] = []
  throwOnRegister = false

  register(definition: { name: string }) {
    if (this.throwOnRegister) throw new Error(`tool "${definition.name}" is already registered`)
    this.global.set(definition.name, definition)
    this.inserted.push(definition.name)
    return () => {}
  }
  get(name: string, scope?: unknown) {
    // The host's single accessor: `get(name, scope)` resolves one scope's view
    // (restrictions apply), `get(name)` is the global view.
    if (scope !== undefined && this.deniedForAgent.has(name)) return undefined
    return this.global.get(name)
  }
}

/** Stand-in for `dsh-commands`. */
class FakeCommands implements CommandsRegistry {
  entries = new Map<string, unknown>()
  inserted: string[] = []

  register(definition: { name: string }) {
    this.entries.set(definition.name, definition)
    this.inserted.push(definition.name)
    return () => {}
  }
  find(_agent: unknown, name: string) {
    return this.entries.get(name)
  }
}

const agent = { id: 'agent-1', session: { id: 'session-1' }, ctx: new Context() } as unknown as Agent

function populate(tools: FakeTools, commands: FakeCommands) {
  for (const name of ACP_TOOL_NAMES) tools.global.set(name, { name })
  commands.entries.set(ACP_COMMAND_NAME, { name: ACP_COMMAND_NAME })
}

test('heal restores all four tools and /acp-prune when the registry came back empty', () => {
  const tools = new FakeTools()
  const commands = new FakeCommands()
  const result = healRegistrations(makeEnv(), tools, commands, agent)

  assert.deepEqual(result.tools, [...ACP_TOOL_NAMES])
  assert.equal(result.command, true)
  assert.deepEqual(tools.inserted, [...ACP_TOOL_NAMES])
  assert.deepEqual(commands.inserted, [ACP_COMMAND_NAME])
  // The restored definitions are the REAL ones — callable, schema-bearing —
  // not placeholders: a healed `compress` must still carry its parameters.
  const compress = tools.global.get('compress') as { name: string; parameters: unknown; execute: unknown }
  assert.equal(compress.name, 'compress')
  assert.ok(compress.parameters, 'restored tool keeps its schema')
  assert.equal(typeof compress.execute, 'function', 'restored tool stays executable')
  const command = commands.entries.get(ACP_COMMAND_NAME) as { name: string; handler: unknown }
  assert.equal(command.name, ACP_COMMAND_NAME)
  assert.equal(typeof command.handler, 'function', 'restored command keeps its handler')
})

test('heal is a no-op when the registry already carries the ACP surface', () => {
  const tools = new FakeTools()
  const commands = new FakeCommands()
  populate(tools, commands)
  const result = healRegistrations(makeEnv(), tools, commands, agent)

  assert.deepEqual(result.tools, [])
  assert.equal(result.command, false)
  assert.deepEqual(tools.inserted, [], 'no duplicate insert behind the guard')
  assert.deepEqual(commands.inserted, [])
})

test('heal restores only the names that are actually missing', () => {
  const tools = new FakeTools()
  const commands = new FakeCommands()
  populate(tools, commands)
  // The signature of a rebuilt registry mid-repopulation: one name present, the
  // rest gone.
  tools.global.delete('compress')
  tools.global.delete('acp_status')
  commands.entries.delete(ACP_COMMAND_NAME)

  const result = healRegistrations(makeEnv(), tools, commands, agent)
  assert.deepEqual(result.tools, ['compress', 'acp_status'])
  assert.equal(result.command, true)
  assert.deepEqual(tools.inserted, ['compress', 'acp_status'])
})

test('a scope restriction is NOT a lost registration — the heal must not fight it', () => {
  const tools = new FakeTools()
  const commands = new FakeCommands()
  populate(tools, commands)
  // `tools.restrict({ deny: ['compress'] })` is scope-local: invisible to this
  // agent, still present globally. Re-registering here would both override the
  // operator's mask and insert a fresh layer on every single step.
  tools.deniedForAgent.add('compress')

  const result = healRegistrations(makeEnv(), tools, commands, agent)
  assert.deepEqual(result.tools, [])
  assert.deepEqual(tools.inserted, [])
  assert.equal(shouldRestoreTool(tools, agent, 'compress'), false)
  // A name gone from BOTH views is still restorable.
  tools.global.delete('decompress')
  assert.equal(shouldRestoreTool(tools, agent, 'decompress'), true)
})

test('a registry that refuses the insert does not break the step', () => {
  const tools = new FakeTools()
  tools.throwOnRegister = true
  const commands = new FakeCommands()
  commands.register = () => {
    throw new Error('commands: registry is disposed')
  }
  const result = healRegistrations(makeEnv(), tools, commands, agent)
  assert.deepEqual(result.tools, [])
  assert.equal(result.command, false)
})

test('absent registries are tolerated (cold start races with the composition row)', () => {
  const result = healRegistrations(makeEnv(), undefined, undefined, agent)
  assert.deepEqual(result.tools, [])
  assert.equal(result.command, false)
})
