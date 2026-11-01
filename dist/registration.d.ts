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
import type { Agent } from '@deepseek-ai/dsh-agent';
import type { ToolDefinition } from '@deepseek-ai/dsh-tools';
import { type ToolEnvironment } from './tools.ts';
/** The ACP surface the model needs; a step missing any of them is unusable. */
export declare const ACP_TOOL_NAMES: readonly ['compress', 'decompress', 'search_context', 'acp_status'];
/** ACP slash command name (`/acp-prune`), healed on the same pass. */
export declare const ACP_COMMAND_NAME = "acp-prune";
/** Minimal structural view of the host tool registry (only what the heal calls).
 *  `get` is optional because a partial stand-in (a test stub, or a future seam
 *  that renames it) must degrade to "nothing to heal", never to a thrown step.
 *  The two lookups mirror the host's ONE public accessor: `get(name, scope)` is
 *  what the step's surface resolves, `get(name)` is the global view (`view`
 *  itself is a private member of `ToolsService`, so it is not usable here). */
export interface ToolsRegistry {
    register(definition: ToolDefinition): unknown;
    get?(name: string, scope?: unknown): unknown;
}
/** Minimal structural view of the host command registry. */
export interface CommandsRegistry {
    register(definition: unknown): unknown;
    find?(agent: unknown, name: string): unknown;
}
export interface HealResult {
    /** Tool names this pass re-registered. */
    tools: string[];
    /** Whether `/acp-prune` was re-registered. */
    command: boolean;
}
/**
 * A tool is "ours to restore" only when BOTH views are empty: absent for the
 * agent about to run the step AND absent from the global view. The global leg is
 * what makes the heal loop-safe — `tools.restrict()` is scope-local, so a tool
 * an operator deliberately denied for one agent stays visible globally and is
 * never re-registered behind their back (nor re-registered every step).
 */
export declare function shouldRestoreTool(tools: ToolsRegistry, agent: Agent, name: string): boolean;
/**
 * Re-register whatever the registry lost. Never throws: a registry that rejects
 * a registration (e.g. a name that appeared between the check and the insert) or
 * a stand-in host without the lookup helpers must not break the step — the next
 * pre-step pass sees what happened and does nothing. Returns what was restored
 * so the caller can log one warn per heal.
 */
export declare function healRegistrations(env: ToolEnvironment, tools: ToolsRegistry | undefined, commands: CommandsRegistry | undefined, agent: Agent): HealResult;
