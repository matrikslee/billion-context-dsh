/**
 * Kernel configuration assembly — the DSH counterpart of billion-context-pi's
 * `resolveConfig`: build acp-kernel's `Config` from adapter-level knobs.
 *
 * Defaults are deliberately the acp-kernel `defaultConfig` values (the same
 * defaults billion-context-pi ships: nudge window 45%–75%, emergency 95%,
 * growth ratio 5%, protected last messages 5). Every knob is optional — an
 * omitted value keeps the kernel default, so the behavior matches the Pi
 * adapter exactly unless a deployment opts out.
 *
 * NOTE: `AcpCompactionEngine` (src/index.ts) ships its own engine-level
 * defaults 0.70/0.85 for the two nudge thresholds on top of this layer, so an
 * engine with no explicit config lands on 0.70/0.85, not 0.75/0.95.
 * @module billion-context-dsh/config
 */

import { defaultConfig, type Config } from 'acp-kernel'
import { DEFAULT_FOLD_FLOOR_PCT, KV_BLOCK_TOKENS, type FoldPricing } from './region.ts'

/** The kernel-facing knobs shared by the nudge path and the compress tool. */
export interface KernelConfigInput {
  readonly modelContextLimit: number
  /** Nudge window lower bound (usage fraction; validation only — the growth-driven trigger has no percentage floor). Kernel default: 0.45. */
  readonly nudgeMinContextLimitPct?: number
  /** Nudge window upper bound — over-limit guarantee line. Kernel default: 0.75. */
  readonly nudgeMaxContextLimitPct?: number
  /** Emergency nudge threshold (bypasses per-turn dedup; capped at 3 injections per user turn — issue #108). Kernel default: 0.95. */
  readonly nudgeEmergencyThresholdPct?: number
  /**
   * Working-set floor as a fraction of the effective window: a fold that leaves
   * less than this stays flagged (range row) and is refused by `condense` unless
   * forced. Default DEFAULT_FOLD_FLOOR_PCT (0.15); 0 disables the guard.
   * Appendix D of arXiv 2611.05306 — see src/region.ts.
   */
  readonly foldFloorPct?: number
  /** Any other acp-kernel Config override (the billion-context-pi escape hatch). */
  readonly coreOverrides?: Partial<Config>
}

/**
 * Assemble the kernel config: `defaultConfig(limit)` merged with the optional
 * nudge thresholds (merged into the defaults, never replacing them wholesale)
 * and any additional `coreOverrides`.
 */
/**
 * Fold pricing for a composition: the working-set floor (window × foldFloorPct).
 * There is NO reuse-floor term: measured 2026-11-15, post-fold `reused` is the
 * rewrite point's token offset, not a block constant. An unset pct lands on
 * DEFAULT_FOLD_FLOOR_PCT; pct 0 disables the working-set guard entirely.
 */
/**
 * NOT one of the six hot-editable settings keys, deliberately. The six hot keys are
 * pressure thresholds — changing them mid-session only changes when we ask. These
 * two are PRICING inputs that appear in numbers already shown to the model (range-row
 * fold costs, the `below floor` marks, the elected-span refusal); editing them
 * mid-session would make the tool contradict its own earlier quotes, and adding them
 * to the settings layer means declaring them volatile in AcpPluginConfigSchema,
 * filtering the composition base, and extending the legacy-seam tests — a change in
 * its own right, not a follow-up to this batch. Until then they are composition-row
 * keys set per deployment, which is also where they belong: `foldFloorPct` is
 * the working set that provider can hold. There is NO reuse-floor knob any more:
 * measured 2026-11-15 the post-fold `reused` was 31,378 — the rewrite point's
 * token offset, not a block-aligned constant — so a subtracted floor had no
 * measurement behind it.
 */
/**
 * The working-set floor is window x foldFloorPct, CLAMPED to 8K-16K tokens.
 * A bare percentage is wrong in both directions: on a 262144 window 0.15 demands
 * 39K resident, which together with the guarded rows at the tail squeezes the
 * foldable window until the tool mostly refuses; on a 32K window it demands
 * 4.8K, thinner than any workable working set (arXiv 2611.05306 Appendix D).
 * The lower bound 8K is the paper peak band (7-9K). foldFloorPct 0 disables it.
 */
export const WORKING_SET_FLOOR_MIN_TOKENS = 8192

/**
 * The ceiling is ONE KV BLOCK — literally KV_BLOCK_TOKENS, the granularity this
 * engine's slot reports (measured 2026-11-14 as a common reused value; the 2026-11-15
 * fold proved reused is NOT block-aligned, so it stays a floor choice and not a
 * price constant). Keep it that way: an operator
 * knob here would let someone tune the UNMEASURED half of the trade (fold
 * frequency) with no comparator, which is exactly what the caveat below warns about.
 * A bigger floor is paid on EVERY later request (deeper resident
 * context slows decode: measured 69.9 tok/s at 16K, 58.1 at 127K, ≈ 0.0008 s per
 * token over that range), while a smaller floor is paid once per EXTRA fold
 * (16K vs 32K is ≈ N/32K extra folds at ~45 s each ≈ 0.0014 s per generated
 * token). In a long session the per-request side wins, so the ceiling is 16K —
 * twice the paper peak band. The fold-frequency term is DERIVED, NOT measured:
 * the first long session should count folds per 100K tokens and revisit this.
 * The counter-argument for a larger floor is that every fold also leaves a
 * permanent un-foldable summary barrier, equally unquantified. That term is NOT
 * in the break-even above: a fold counted as ~45 s really costs ~45 s plus one
 * surface split that can never be folded again, so this arithmetic is biased
 * TOWARD the larger floor and the 16K choice is the conservative reading.
 */
export const WORKING_SET_FLOOR_MAX_TOKENS = KV_BLOCK_TOKENS

/**
 * Smallest reclaim worth a permanent barrier once the election degrades below the
 * floor (cross-review Q5): a 1K fold buys 1K of relief for one surface split that
 * can never be folded again — the worst barrier-per-byte ratio possible.
 */
export const MIN_DEGRADED_FOLD_TOKENS = 4096

// Round-3 added a fold-count gate here (ceil(limit / (2 × floor))) and round 4
// deleted it: a barrier is a summary NOTE, not a floor-sized hole, so no fold
// count measures the debt that formula claimed, and on a 32K window with a 16K
// clamp the gate could never be reached at all — the deadlock it was meant to
// open stayed closed. condenseDegradable (src/tools.ts) reads the live rows
// instead: degrade only when NO eligible row can keep the floor, which needs no
// measurement and can never fire early.
export function foldPricingFor(input: KernelConfigInput): FoldPricing {
  const pct = input.foldFloorPct ?? DEFAULT_FOLD_FLOOR_PCT
  return {
    ...(pct <= 0 ? {} : {
      floorTokens: Math.min(
        WORKING_SET_FLOOR_MAX_TOKENS,
        Math.max(WORKING_SET_FLOOR_MIN_TOKENS, Math.round(input.modelContextLimit * pct)),
      ),
    }),
  }
}

export function kernelConfigFor(input: KernelConfigInput): Config {
  const nudgePatch: Partial<Config['nudge']> = {}
  if (input.nudgeMinContextLimitPct !== undefined) nudgePatch.minContextLimitPct = input.nudgeMinContextLimitPct
  if (input.nudgeMaxContextLimitPct !== undefined) nudgePatch.maxContextLimitPct = input.nudgeMaxContextLimitPct
  if (input.nudgeEmergencyThresholdPct !== undefined) nudgePatch.emergencyThresholdPct = input.nudgeEmergencyThresholdPct

  const overrides: Partial<Config> = { ...input.coreOverrides }
  if (Object.keys(nudgePatch).length > 0 || input.coreOverrides?.nudge) {
    // The engine always ships explicit pct defaults (0.70/0.85, see
    // DEFAULT_CONFIG), so nudgePatch is never empty and the plain replace
    // below used to discard coreOverrides.nudge entirely — the documented
    // escape hatch was unreachable whenever the pct knobs were set. User
    // overrides must land LAST so they win over both kernel defaults and
    // the engine pct values.
    overrides.nudge = {
      ...defaultConfig(input.modelContextLimit).nudge,
      ...nudgePatch,
      ...input.coreOverrides?.nudge,
    }
  }
  return defaultConfig(input.modelContextLimit, overrides)
}
