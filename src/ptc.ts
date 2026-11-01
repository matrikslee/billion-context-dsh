/**
 * PTC (code-mode) awareness.
 *
 * On a host that mounts the PTC tool executor, the tool surface collapses to ONE
 * directly callable tool — `run_code` — and everything else is reachable only
 * from inside a `run_code` program (`await tools.compress({ … })`). Measured on
 * a live PTC session (host 0.2.2-alpha.2, `@deepseek-ai/dsh-tool-executor-ptc`):
 * `describe()` publishes `schemas: [run_code]` / `knownNames: [run_code]`, and
 * `reject()` answers any other direct call with `ToolNotFoundError` ("only
 * `run_code` is callable directly"). The ACP prompt copy is written in direct-call
 * shape, so the model copied it into a direct call; the engine then parsed the
 * unknown function name as plain TEXT — no call, no error result — and the
 * session dead-ended in raw markup output.
 *
 * This module owns the adapter half of that gap: detect the mode, and render the
 * model-visible call syntax in the shape the mode actually accepts. The swap is a
 * REPLACEMENT, not an addition — the PTC variants carry the same information in
 * roughly the same bytes (in-place call-shape rewrite plus one short line per
 * surface), and native mode stays byte-identical (tests/prompts-ptc.test.ts).
 * @module billion-context-dsh/ptc
 */

import type { ResolvedPrompts } from './prompts.ts'

/** Config knob: `auto` detects from the assembled system prompt; the two fixed
 *  values override the detection (an operator can pin the mode). */
export type ToolCallStyle = 'auto' | 'native' | 'ptc'
/** What the engine has actually applied for the current step. */
export type ResolvedCallStyle = 'native' | 'ptc'

/**
 * Verbatim substrings of the host's PTC executor output — both appear in the
 * ASSEMBLED system prompt of a PTC session:
 * - `PTC_ONLY_INSTRUCTION` (`dsh-tool-executor-ptc/lib/index.js:1704`),
 * - the `run_code` SDK instructions header, emitted by BOTH the TS and the
 *   Python SDK variant (`:916` / `:1639`).
 * Matching on the host's own constants (not on a config key) is what keeps this
 * correct when the mode arrives from a preset the engine cannot read.
 */
export const PTC_MARKERS: readonly string[] = [
  'is the only tool you can call directly',
  '## Writing code for run_code',
]

/** Built by concatenation so the literal markup never appears in this source. */
export const TOOL_CALL_MARK = `<` + `tool_` + `call` + `>`

/** True when any visible text carries a PTC marker. Call with system-prompt texts. */
export function detectPtcMode(texts: Iterable<string>): boolean {
  for (const text of texts) {
    if (typeof text !== 'string') continue
    for (const marker of PTC_MARKERS) {
      if (text.includes(marker)) return true
    }
  }
  return false
}

/**
 * The convention line, stated ONCE — it goes in the system prompt section (one
 * injection per session). Kept short because every model-visible byte is billed
 * on every request.
 *
 * IMPORTANT for anything injected into a `systemPromptTemplate`: the renderer
 * substitutes `{identifier}` placeholders, so example calls must keep spaces
 * inside their braces (`{ blockId }`, not `{blockId}`) or the render throws.
 */
export const PTC_SYSTEM_LINE =
  'Calling convention in this session (PTC / code mode): `run_code` is the only directly callable tool — reach the ACP tools from inside a `run_code` program (`await tools.compress({ content: [...] })`, `await tools.acp_status({})`, `await tools.decompress({ blockId })`, `await tools.search_context({ query })`). A direct call naming them produces no call and no error — it silently emits text. Build summary strings from single-quoted lines joined by a newline; template literals break on the backticks summaries contain.'

/** One short line for the range-table footer (the per-step decision point). */
export const PTC_FOOTER_NOTICE =
  'PTC mode: call it inside `run_code` — `await tools.compress({ content: [...] })`; a direct call fails silently.'

/** The compress-description clause (the SDK renders it into `ToolArgsMap`). */
export const PTC_TOOL_CLAUSE =
  'PTC mode: call it as `await tools.compress({ content: [...] })` from a `run_code` program — a direct call fails silently. Build `summary` from single-quoted strings joined by a newline (summaries contain backticks).'

/** Direct-call syntax → the `run_code` SDK form. The `(?<!tools\.)` guard keeps
 *  the rewrite idempotent (a re-scan must not double-prefix what it just wrote)
 *  and leaves prose mentions of the tool names alone — only `name({` and
 *  `search_context(query)` are call shapes. Exported for the nudge path, which
 *  rewrites the ALREADY-RENDERED kernel text rather than its templates. */
export function toPtcCallSyntax(text: string): string {
  return text
    .replace(/\bsearch_context\(query\)/g, 'await tools.search_context({ query })')
    .replace(/(?<!tools\.)\bcompress\(\{/g, 'await tools.compress({')
    .replace(/(?<!tools\.)\bdecompress\(\{/g, 'await tools.decompress({')
    .replace(/(?<!tools\.)\bsearch_context\(\{/g, 'await tools.search_context({')
    .replace(/(?<!tools\.)\bacp_status\(\{/g, 'await tools.acp_status({')
}

/** Render-time mode switch for text the kernel produced (the nudge): rewrite the
 *  call shapes when the session is in PTC mode, byte-identical otherwise. */
export function ptcText(text: string, style: ResolvedCallStyle): string {
  return style === 'ptc' ? toPtcCallSyntax(text) : text
}

/**
 * Same resolved prompts, re-rendered for PTC mode: call shapes rewritten in
 * place, plus one short notice per decision surface (system prompt, range-table
 * footer, compress description).
 *
 * `nudge` keeps its ORIGINAL OBJECT IDENTITY on purpose: the nudge path picks the
 * kernel renderer by comparing that reference against `DEFAULT_RESOLVED.nudge`
 * (rule 9 — the kernel owns the nudge copy), so rewriting the group here would
 * silently switch every PTC session to the template path. PTC nudge copy is
 * produced at render time by `ptcText` instead.
 *
 * Pure — the input is never mutated.
 */
export function toPtcPrompts(prompts: ResolvedPrompts): ResolvedPrompts {
  // Interfaces have no index signature, so the group is walked through entries
  // and rebuilt — same keys, rewritten values.
  const rewriteGroup = <T>(group: T): T => {
    const out: Record<string, string> = {}
    for (const [key, value] of Object.entries(group as unknown as Record<string, string>)) {
      out[key] = toPtcCallSyntax(value)
    }
    return out as T
  }
  return {
    nudge: prompts.nudge,
    rangeTable: {
      ...rewriteGroup(prompts.rangeTable),
      footer: `${PTC_FOOTER_NOTICE}\n${toPtcCallSyntax(prompts.rangeTable.footer)}`,
    },
    tools: {
      ...rewriteGroup(prompts.tools),
      compress: `${PTC_TOOL_CLAUSE}\n${toPtcCallSyntax(prompts.tools.compress)}`,
    },
    systemPromptTemplate: `${PTC_SYSTEM_LINE}\n\n${toPtcCallSyntax(prompts.systemPromptTemplate)}`,
  }
}

/** The call hint used in tool RESULT text (errors / recommendations). */
export function callHint(style: ResolvedCallStyle, name: 'compress' | 'decompress', shape: string): string {
  return style === 'ptc' ? `await tools.${name}(${shape})` : `${name}(${shape})`
}

/** One-shot correction appended to the next nudge after a leaked direct call was
 *  detected. The host gives the model NO error result in that case (the markup
 *  lands as assistant text), so an nudge is the only channel that can say so. */
export const PTC_LEAK_CORRECTION =
  'Correction: an earlier turn emitted a direct `compress` call as plain text — on that path nothing executes and nothing errors. Emit no XML markup; reach the ACP tools inside `run_code` (`await tools.compress({ content: [...] })`).'

/** True when an assistant text block carries leaked XML-call markup naming an
 *  ACP tool — the signature of a direct call the engine parsed as text. */
export function looksLikeLeakedAcpCall(text: string): boolean {
  return text.includes(TOOL_CALL_MARK)
    && /\b(compress|decompress|search_context|acp_status)\b/.test(text)
}
