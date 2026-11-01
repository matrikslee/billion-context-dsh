/**
 * billion-context-dsh — Active Context Pruning (ACP) for the DeepSeek Harness,
 * delivered as a `CompactionEngine` backend.
 *
 * The model decides when and what to compress (pure ACP semantics):
 *  - the `compress` tool durably shadows a surface range with the model-written
 *    summary (no second LLM summarization call — the ACP cost win);
 *  - the original events stay in the append-only session log, so `decompress`,
 *    `search_context`, and replay always work;
 *  - refs are surface seqs carried by the injected nudge's range table (DSH
 *    has no in-memory message rewrite hook — see docs/dsh-porting-verification.md);
 *  - automatic policy never summarizes by itself: it nudges the model.
 *
 * Mount it wherever a compaction backend is expected:
 *
 * ```yaml
 * - id: compaction-billion-context
 *   name: 'billion-context-dsh'
 *   config:
 *     modelContextLimit: 128000
 * ```
 *
 * The package registers `ctx.compaction` plus the four model tools and the
 * `/acp-prune` command when the hosting composition provides `ctx.tools` /
 * `ctx.commands`.
 * @module billion-context-dsh
 */
import type { Context } from '@deepseek-ai/cordis';
import { CompactionEngine, type CompactionAgentContext, type CompactionResult, type CompactionTrigger, type ManualCompactAgentContext } from '@deepseek-ai/dsh-compaction';
import { type CompressionCore } from 'acp-kernel';
import type { Agent } from '@deepseek-ai/dsh-agent';
import { AcpStateStore } from './state.ts';
import { type ToolEnvironment } from './tools.ts';
import { type AcpPrompts, type ResolvedPrompts } from './prompts.ts';
import { type AcpWindow } from './window.ts';
import { type ToolCallStyle } from './ptc.ts';
import { type SettingsCommandSurface } from './settings.ts';
import { type PresetName } from './presets.ts';
export { AcpStateStore } from './state.ts';
export { kernelConfigFor, type KernelConfigInput } from './config.ts';
export { PRESETS, PRESET_NAMES, isPresetName, resolvePreset, type NudgePreset, type PresetName, } from './presets.ts';
export { ACP_SYSTEM_PROMPT, ACP_SYSTEM_PROMPT_ORDER } from './system-prompt.ts';
export { DEFAULT_PROMPTS, DEFAULT_RESOLVED, renderSystemPrompt, renderTemplate, resolvePrompts, type AcpPrompts, type NudgePrompts, type PromptInput, type PromptOverride, type RangeTablePrompts, type ResolvedPrompts, type ToolPrompts, } from './prompts.ts';
export { makeTools, type ToolEnvironment } from './tools.ts';
export { acpCommand } from './commands.ts';
export { buildNudge, resolveTokenCount, EMERGENCY_NUDGE_MAX_PER_TURN, type NudgeEnvironment, type NudgeOutcome } from './nudge.ts';
export { DEFAULT_CONTEXT_WINDOW, detectContextWindow, projectedContextWindow, windowSourceLabel, type AcpWindow, } from './window.ts';
export { AlreadyCompressedRangeError, rebuildBlockLedger, resolveSurfaceRange, runCompactionTransaction, shadowedSeqsOf, findOpenTurn, assertNoActiveCompaction, blockRegistry, blockRefForSummarySeq, compactionIdsOfKernelBlocks, summarySeqOfKernelBlock, expandShadowedSeqs, hideCompressToolPair, stripOrphanedSurfaceToolMessages, type AcpBlockLedgerEntry, type CompactionTransactionInput, type ResolvedSurfaceRange, } from './region.ts';
export { eventsToCoreMessages, projectEvent, surfaceEventsOf, extractEventText } from './messages.ts';
export { ACP_SETTINGS_NAMESPACE, AcpPluginConfigSchema, AcpSettingsSchema, describeSettingsChange, filterSettingsEntry, isVolatileRef, liveSettingsFromRefs, makeSettingsCommandSurface, parseSettingValue, resolveAcpSettings, SETTINGS_KEYS, SETTING_DEFAULTS, unwrapVolatile, VOLATILE_WRITE, type AcpSettings, type AcpSettingsDescriptor, type AcpSettingsInput, type AcpSettingsService, type LegacySettingsService, type SettingsChangeEffect, type SettingsCommandSurface, type SettingsKey, } from './settings.ts';
export interface AcpConfig {
    /**
     * The context window used for pressure decisions, in tokens. When omitted,
     * `autoModelContextLimit` (default true) resolves it automatically: the live
     * host session projection (`contextPressure.contextWindow`) is preferred,
     * then the model's real window is probed via
     * `agent.ctx.llm.resolveModelInfo(provider, model)`; an explicit value
     * always wins and disables both.
     */
    readonly modelContextLimit?: number;
    /** Auto-resolve the real context window: host session projection first, then the LLM runtime probe. Default true. */
    readonly autoModelContextLimit: boolean;
    /** Nudge window lower bound (usage fraction; validation only — the growth-driven trigger has no percentage floor). Kernel default 0.45 — same as billion-context-pi. */
    readonly nudgeMinContextLimitPct?: number;
    /**
     * Nudge window upper bound — over-limit guarantee line: above this the
     * kernel injects a nudge regardless of growth or cadence. Engine default
     * 0.70 (deliberately BELOW the kernel/billion-context-pi default 0.75 and
     * the host compaction-basic auto-compaction line 0.80, so the forced nudge
     * always fires first); an explicit value wins over this default — a
     * same-name key in `coreOverrides.nudge` wins over both (it merges last).
     */
    readonly nudgeMaxContextLimitPct?: number;
    /**
     * Emergency nudge threshold (bypasses the per-turn dedup, but is capped at
     * EMERGENCY_NUDGE_MAX_PER_TURN = 3 injections per user turn — issue #108).
     * Engine default 0.85 (down from the kernel/billion-context-pi default 0.95:
     * 95% leaves the model no room to act before the API rejects, and the host's
     * 80% compaction-basic line shadows it in standard/code/cordis modes).
     */
    readonly nudgeEmergencyThresholdPct?: number;
    /**
     * Named bundle for the three nudge thresholds — how eagerly the model is
     * asked to compress, in one word. One of 'preserve' | 'relaxed' | 'balanced'
     * | 'efficient' | 'aggressive' (see src/presets.ts). It fills ONLY the nudge
     * thresholds you did not set explicitly, so precedence is explicit value >
     * preset > engine default and a partial override on top of a preset still
     * wins. An unknown name fails engine construction (fail-fast). No effect on
     * any other knob (`modelContextLimit`, `autoNudge`, prompts, coreOverrides).
     * An unknown name fails construction, and so does a merged window that ends up
     * inverted (min > max, max > emergency or min > emergency — the kernel itself
     * only warns about that, see `assertNudgeThresholdOrder`).
     */
    readonly preset?: PresetName;
    /**
     * Any other acp-kernel Config override (billion-context-pi's `coreOverrides`
     * escape hatch). Merge order per section: kernel defaults → the engine pct
     * knobs above → these keys land LAST, so a same-name key here wins.
     */
    readonly coreOverrides?: Partial<import('acp-kernel').Config>;
    /**
     * Custom token-count function for the kernel's internal estimation.
     * Defaults to the kernel's `defaultCountTokens` (CJK: 1 char = 1 token,
     * other: 4 chars = 1 token — aligns with billion-context-pi).
     * Can be overridden for provider-specific tokenization, e.g. DeepSeek's
     * official coefficient: 1 CJK char ≈ 0.6 tokens, 1 other char ≈ 0.3 tokens.
     * Only affects the kernel's internal estimation (compressible range sizing,
     * nudge text, growth branch pending); the `projectedTokens` reading from
     * `sessionProjections` (used for nudge pressure decisions and acp_status)
     * is provider-anchored and unaffected by this function.
     */
    readonly countTokens?: (text: string) => number;
    /** Register the four model tools on `ctx.tools`. Default true. */
    readonly autoTools: boolean;
    /** Register the `/acp-prune` command on `ctx.commands`. Default true. */
    readonly autoCommand: boolean;
    /** Inject the nudge into `agent/pre-step` when the kernel recommends it. Default true. */
    readonly autoNudge: boolean;
    /**
     * How many consecutive provider-confirmed context-overflow failures one
     * agent may answer with an emergency compaction + retry before the original
     * error is preserved (mirrors compaction-basic's `maxOverflowRetries`;
     * default 1, the host's own default). The budget resets when the agent
     * makes progress (an assistant message lands) or returns to idle, so a
     * request that cannot be repaired cannot retry forever.
     */
    readonly maxOverflowRetries?: number;
    /**
     * Escape hatch: disable the runtime-settings integration entirely
     * (composition-layer ONLY — deliberately not exposed through the settings
     * layer itself: a switch that turns off its own plumbing could not be
     * reached if the plumbing broke). Default: enabled.
     */
    readonly settingsEnabled?: boolean;
    /** Per-stage prompt template overrides (nudge / range table / system prompt / tool descriptions). See docs/configurable-prompts-design.md. */
    readonly prompts?: AcpPrompts;
    /**
     * How the model reaches the ACP tools. `auto` (default) detects PTC / code mode
     * from the assembled system prompt of each session every step; `native` or
     * `ptc` pin it (an operator override, or a host whose PTC copy changes).
     * Composition-layer only — deliberately NOT a settings knob, so the six-key
     * settings surface (AcpPluginConfigSchema) is untouched. See src/ptc.ts.
     */
    readonly toolCallStyle?: ToolCallStyle;
}
export declare function resolveAcpConfig(config?: Partial<AcpConfig>): AcpConfig;
/**
 * The ACP compaction backend. Subclasses the seam exactly like
 * `dsh-compaction-basic`; swaps summarization-driven compaction for
 * model-driven block compression without touching the agent loop.
 */
export declare class AcpCompactionEngine extends CompactionEngine {
    /**
     * Static plugin config schema — cordis reads `plugin.Config` off the raw
     * plugin value at fiber start and validates the composition row against it
     * (`resolveConfig`). Class-shaped mounts carry this (the bundle/composition
     * row path); function-shaped mounts skip validation and receive plain
     * values. On dsh-settings ≥ 0.1.7 hosts this SAME schema is what SettingsForms
     * builds its form from — the volatile fields are exactly what `/acp-prune
     * config` exposes (see AcpPluginConfigSchema for why fields are volatile and
     * default-free).
     */
    static readonly Config: import("@deepseek-ai/schemastery").default<Schemastery.ObjectS<NoInfer<{
        modelContextLimit: import("@deepseek-ai/schemastery").default<number, number, "plain">;
        autoModelContextLimit: import("@deepseek-ai/schemastery").default<boolean, boolean, "plain">;
        nudgeMinContextLimitPct: import("@deepseek-ai/schemastery").default<number, number, "plain">;
        nudgeMaxContextLimitPct: import("@deepseek-ai/schemastery").default<number, number, "plain">;
        nudgeEmergencyThresholdPct: import("@deepseek-ai/schemastery").default<number, number, "plain">;
        autoNudge: import("@deepseek-ai/schemastery").default<boolean, boolean, "plain">;
    }>>, Schemastery.ObjectT<NoInfer<{
        modelContextLimit: import("@deepseek-ai/schemastery").default<number, number, "plain">;
        autoModelContextLimit: import("@deepseek-ai/schemastery").default<boolean, boolean, "plain">;
        nudgeMinContextLimitPct: import("@deepseek-ai/schemastery").default<number, number, "plain">;
        nudgeMaxContextLimitPct: import("@deepseek-ai/schemastery").default<number, number, "plain">;
        nudgeEmergencyThresholdPct: import("@deepseek-ai/schemastery").default<number, number, "plain">;
        autoNudge: import("@deepseek-ai/schemastery").default<boolean, boolean, "plain">;
    }>>, "plain">;
    /** The framework-agnostic ACP compression core, reused verbatim. */
    readonly kernel: CompressionCore;
    /** Per-session kernel state. */
    readonly store: AcpStateStore;
    /** Resolved engine configuration. */
    readonly config: AcpConfig;
    /** Resolved prompt templates (validated at construction — fail-fast on template typos). */
    readonly prompts: ResolvedPrompts;
    /** The prompt set ACTUALLY applied to the current step: `prompts` in native
     *  mode, its PTC variant in code mode (src/ptc.ts). `prompts` keeps the
     *  composition value so introspection and tests see what was configured. */
    private currentPrompts;
    /** The calling convention the currently registered surface was built for. */
    private appliedStyle;
    /** Disposers of the tool registrations this instance made — `tools.register()`
     *  returns the exact disposer, which is what lets a mode flip REPLACE the
     *  descriptions instead of stacking a second layer per flip. */
    private toolDisposers;
    /** Same, for the ACP system-prompt section (`systemPrompt.section()`). */
    private systemPromptDisposer;
    /** Per session: the newest leaked direct call already corrected, plus how many
     *  corrections were spent (the leak is silent on the host, so the nudge is the
     *  only channel — bounded to avoid a per-step nag). */
    private readonly ptcLeakNotes;
    /**
     * The environment wired into tools / command / nudge. Exposed so tests (and
     * introspection) can assert the forwarding actually happened: the config
     * chain user config → this.config → env → kernelConfigFor is all OPTIONAL
     * fields, so a dropped forwarding line fails typecheck silently and would
     * revive lost-config bugs with every unit test green.
     */
    readonly env: ToolEnvironment;
    private readonly lastNudgeTurn;
    /** Per-session emergency-nudge injection budget for the current user turn (issue #108). */
    private readonly emergencyNudges;
    /** Successful compress call ids awaiting their tool/result so the pair can be hidden. */
    private readonly compressCallIdsToHide;
    /** Per provider/model route the resolved window (probe failures cached too). */
    private readonly windowCache;
    /** Live settings snapshot thunk (composition → user settings layer); swapped when the settings provider attaches (installSection on ≤0.1.6 lines, volatile-ref re-read on the ≥0.1.7 forms line). */
    private readSettingsSource;
    /** Hot-apply driver: re-reads the live source and runs the change diff; called once per agent step, a no-op unless something changed. */
    private resyncSettings;
    /** The settings service, captured lazily for /acp-prune config (undefined in provider-less processes). Structurally typed — both host lines' real services satisfy it. */
    private settingsService;
    /** /acp-prune config read/write surface. */
    readonly settingsCommand: SettingsCommandSurface;
    /** Per route the adapter's per-request output cap (the output reservation); null = undisclosed. */
    private readonly outputReservationCache;
    /** Per-agent context-overflow recovery budget (mirrors the host's `overflowRetries`). */
    private readonly overflowRetries;
    /** Per-session overflow agents, so session progress can reset the recovery budget. */
    private readonly overflowSessions;
    constructor(ctx: Context, config?: Partial<AcpConfig>);
    /**
     * Register (and remember the disposer of) the four ACP tools for the CURRENT
     * prompt set. `tools.register()` returns the exact disposer, which
     * {@link applyToolCallStyle} uses to replace the surface on a mode flip.
     */
    private registerToolSurface;
    /** Register (and remember the disposer of) the ACP system-prompt section. */
    private registerPromptSurface;
    /**
     * Swap the model-visible calling convention (src/ptc.ts). Runs from
     * `agent/pre-step`, i.e. before the step builds its request, so the very next
     * request carries the new descriptions and section text.
     *
     * Dispose-then-register: `register()` on a name that is already registered
     * would stack a second layer contribution, so the old contributions go first.
     * A failed re-registration is not fatal — the registration self-heal
     * (`src/registration.ts`) restores whatever is missing on the next pre-step.
     */
    private applyToolCallStyle;
    /**
     * Resolve the calling convention for this step: a pinned config always wins,
     * otherwise the assembled system prompt of the session decides (`auto`).
     */
    private resolveToolCallStyle;
    /** Per-session cap on direct-call corrections (a nag on every step would be
     *  noise; three covers a model that keeps trying the direct shape). */
    private static readonly PTC_LEAK_NOTE_MAX;
    /**
     * True when the model just emitted a direct ACP call as TEXT — in PTC mode the
     * host rejects the name and the markup lands in the assistant text with no call
     * and no error result, so nothing tells the model it failed. The next nudge
     * carries the correction. Scans only the recent tail (a leak is a fresh event),
     * and reports once per distinct leaked seq.
     */
    private takePtcLeakNote;
    /**
     * Resolve the effective context window for an agent. An explicitly
     * configured `modelContextLimit` always wins (no probe). Otherwise the live
     * session projection (`contextPressure.contextWindow`) is preferred when it
     * discloses one — it tracks the session's CURRENT route, so a mid-session
     * model switch repairs itself without a restart or config (see
     * projectedContextWindow). Falls back to probing the model's real window
     * via `agent.ctx.llm.resolveModelInfo` (cached per provider/model route,
     * probe failures cached too) and finally to DEFAULT_CONTEXT_WINDOW when
     * auto-detection is disabled or unavailable. On the auto-detected paths the
     * adapter's per-request output cap is then SUBTRACTED from the window
     * (applyReservation): every downstream usage computation must run against
     * the SUSTAINABLE input budget (window minus output reservation), not the
     * raw window — a 96K window with a 16K cap carries at most 80K of input,
     * so the raw denominator understates usage by cap/window (≈17% there, and
     * far worse on short-window models). An explicit limit keeps the operator's
     * exact value (they own the denominator); a failed probe keeps the raw
     * fallback.
     */
    windowFor(agent: Agent): Promise<AcpWindow>;
    /**
     * Diff handler for runtime settings changes: drop the window cache when a
     * window-related key changed (probe FAILURES are cached too — clearing is
     * what lets the next pre-step re-probe after a fix), clear the per-turn
     * nudge dedup when nudges come back on, and warn on order anomalies
     * (accepted, never rejected — rejecting a write cannot fix an externally
     * edited settings.yaml, and an invalid stored section would fail the next
     * boot loud anyway).
     */
    private onSettingsChanged;
    /**
     * The adapter's per-request output cap for a route, from one
     * probeModelWindow call (a local catalog lookup — no request is sent),
     * cached per route like the window itself.
     */
    private outputCapFor;
    /**
     * Subtract the output reservation from a resolved window: `limit` becomes
     * the SUSTAINABLE input budget (`rawLimit - outputReserved`) that every
     * downstream usage computation (nudge tiers, truncate, growth) measures
     * against. No-op when the cap is unknown or not smaller than the window
     * (degenerate config) — the raw-window behavior is preserved.
     */
    private applyReservation;
    /**
     * Best-effort emergency compaction for one provider-confirmed context
     * overflow: pick the largest eligible (guarded, tool-pairing-balanced)
     * surface range and land the normal durable transaction with a fixed
     * engine-written marker summary. No LLM call — the provider just rejected
     * the request for being too large, so there is no model turn available to
     * write a summary; the originals stay in the append-only log, so
     * search_context still indexes them, decompress restores them, and the
     * model can re-run the compress tool over the marker later to write a real
     * summary. Returns null when nothing eligible exists (nothing to reclaim).
     */
    private compactForOverflow;
    /** ACP is model-driven: automatic pressure policy never summarizes by itself. */
    compactIfNeeded(_agent: CompactionAgentContext, _trigger: CompactionTrigger, signal: AbortSignal): Promise<CompactionResult | null>;
    /** Explicit idle-session compaction: ACP leaves the decision to the model. */
    compactNow(_agent: ManualCompactAgentContext, signal: AbortSignal): Promise<CompactionResult | null>;
    /**
     * The model-driven path lands through the `compress` tool, which runs the
     * full durable transaction directly. This seam method rejects with guidance:
     * automatic summarization is exactly what ACP replaces.
     */
    compactRegion(_start: number, _end: number, _agent: CompactionAgentContext, signal?: AbortSignal): Promise<CompactionResult>;
}
export default AcpCompactionEngine;
