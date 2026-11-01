/**
 * Registration self-healing.
 *
 * `tools.register()` / `commands.register()` are `layers.effect(ctx, …)` calls
 * (host `@deepseek-ai/dsh-tools/lib/index.js:433-446`,
 * `@deepseek-ai/dsh-commands/lib/index.js:266-269`): the contribution lives in a
 * LAYER tied to the registry instance that received it. When the host tears down
 * and rebuilds an agent realm (an interrupted loop, a stopped session that is
 * resumed later), the registry service is re-instantiated with empty layers
 * while THIS plugin instance stays mounted — so its eager registration is
 * already gone, and the activation-time `done` guard prevents any second
 * attempt. The measured signature (live session, 2026-11-01): the four ACP tools
 * were absent from the tool surface for the first two steps after a resumed
 * session — every model call answered `unknown tool` — and reappeared only when
 * a fresh plugin instance mounted on step 3.
 *
 * The heal runs at `agent/pre-step`, i.e. immediately before the tool surface is
 * computed for the step, so a rebuilt registry is repopulated in the same step
 * instead of two steps later.
 */

import type { Agent } from '@deepseek-ai/dsh-agent'
import type { ToolDefinition } from '@deepseek-ai/dsh-tools'
import { makeTools, type ToolEnvironment } from './tools.ts'
import { acpCommand } from './commands.ts'

/** The ACP surface the model needs; a step missing any of them is unusable. */
export const ACP_TOOL_NAMES = ['compress', 'decompress', 'search_context', 'acp_status'] as const

/** ACP slash command name (`/acp-prune`), healed on the same pass. */
export const ACP_COMMAND_NAME = 'acp-prune'

/** Minimal structural view of the host tool registry (only what the heal calls).
 *  `get` is optional because a partial stand-in (a test stub, or a future seam
 *  that renames it) must degrade to "nothing to heal", never to a thrown step.
 *  The two lookups mirror the host's ONE public accessor: `get(name, scope)` is
 *  what the step's surface resolves, `get(name)` is the global view (`view`
 *  itself is a private member of `ToolsService`, so it is not usable here). */
export interface ToolsRegistry {
  register(definition: ToolDefinition): unknown
  get?(name: string, scope?: unknown): unknown
}

/** Minimal structural view of the host command registry. */
export interface CommandsRegistry {
  register(definition: unknown): unknown
  find?(agent: unknown, name: string): unknown
}

export interface HealResult {
  /** Tool names this pass re-registered. */
  tools: string[]
  /** Whether `/acp-prune` was re-registered. */
  command: boolean
}

/**
 * A tool is "ours to restore" only when BOTH views are empty: absent for the
 * agent about to run the step AND absent from the global view. The global leg is
 * what makes the heal loop-safe — `tools.restrict()` is scope-local, so a tool
 * an operator deliberately denied for one agent stays visible globally and is
 * never re-registered behind their back (nor re-registered every step).
 */
export function shouldRestoreTool(tools: ToolsRegistry, agent: Agent, name: string): boolean {
  if (tools.get === undefined) return false
  if (tools.get(name, agent) !== undefined) return false
  return tools.get(name) === undefined
}

/**
 * Re-register whatever the registry lost. Never throws: a registry that rejects
 * a registration (e.g. a name that appeared between the check and the insert) or
 * a stand-in host without the lookup helpers must not break the step — the next
 * pre-step pass sees what happened and does nothing. Returns what was restored
 * so the caller can log one warn per heal.
 */
export function healRegistrations(
  env: ToolEnvironment,
  tools: ToolsRegistry | undefined,
  commands: CommandsRegistry | undefined,
  agent: Agent,
): HealResult {
  const result: HealResult = { tools: [], command: false }
  if (tools !== undefined) {
    // Built lazily: a healthy registry runs this pass every step, so the tool
    // definitions are only materialized when something is actually missing.
    let definitions: Map<string, ToolDefinition> | undefined
    for (const name of ACP_TOOL_NAMES) {
      if (!shouldRestoreTool(tools, agent, name)) continue
      definitions ??= new Map(makeTools(env).map((definition) => [definition.name, definition]))
      const definition = definitions.get(name)
      if (definition === undefined) continue
      try {
        tools.register(definition)
        result.tools.push(name)
      } catch {
        // Already there (raced) or rejected by the registry — nothing to do.
      }
    }
  }
  // A registry without the lookup helper cannot be verified, so it is left
  // alone — re-registering blindly would insert a fresh layer on every step.
  if (commands !== undefined && commands.find !== undefined && commands.find(agent, ACP_COMMAND_NAME) === undefined) {
    try {
      commands.register(acpCommand(env))
      result.command = true
    } catch {
      // Same rationale as the tool branch.
    }
  }
  return result
}
