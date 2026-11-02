// Fix12: the full combat board in the shape sts2-rs's state_from_dict reads (through
// spire-mentor's sts2_hint adapter): every pile in order, layered card costs, enemy move
// states, power vars, per-turn play counters and the run's RNG cursors. Read-only; the
// "combat_wire" command returns it, combat_play replies are unchanged.
//
// Ported from tfriedel/spire-mentor (commit 52bfee16), sts2-pycli/src/Sts2Headless/
// RunSimulator.cs: CombatPlayState and its helpers, v0.111.0 branches only.
// Copyright (c) 2025-2026 Thomas Friedel, MIT License (text in bot/vendor/LICENSE-spire-mentor). Renamed where our
// RunSimulator already has a member of that name (CombatPlayState -> WireCombatBoard,
// PlayerSummary -> WirePlayerSummary, RunContext -> WireRunContext).
using System.Reflection;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Combat.History;
using MegaCrit.Sts2.Core.Combat.History.Entries;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Models.Cards;
using MegaCrit.Sts2.Core.Models.Powers;
using MegaCrit.Sts2.Core.MonsterMoves.Intents;
using MegaCrit.Sts2.Core.Random;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;

namespace Sts2Headless;

internal static class WireRngCompat
{
    public static int Counter(Rng rng) => rng.ToSerializable().counter;
}

public partial class RunSimulator
{

    /// <summary>Fix12 "combat_wire": the board of the fight in progress (WireCombatBoard), or an
    /// error outside one. Read-only: the engine state and its RNG are not touched.</summary>
    public Dictionary<string, object?> CombatWire()
    {
        try
        {
            if (_runState == null || !CombatManager.Instance.IsInProgress)
                return Error("combat_wire: no fight in progress");
            var board = WireCombatBoard(_runState.Players[0]);
            board["type"] = "combat_wire";
            return board;
        }
        catch (Exception ex)
        {
            return Error($"combat_wire: {ex.GetType().Name}: {ex.Message}");
        }
    }
    // empty dict if no run is active.
    private Dictionary<string, object?> RngCounters()
    {
        var d = new Dictionary<string, object?>();
        var rng = _runState?.Rng;
        if (rng == null) return d;
        try
        {
            d["up_front"] = WireRngCompat.Counter(rng.UpFront);
            d["shuffle"] = WireRngCompat.Counter(rng.Shuffle);
            d["combat_card_generation"] = WireRngCompat.Counter(rng.CombatCardGeneration);
            d["combat_potion_generation"] = WireRngCompat.Counter(rng.CombatPotionGeneration);
            d["combat_card_selection"] = WireRngCompat.Counter(rng.CombatCardSelection);
            d["combat_energy_costs"] = WireRngCompat.Counter(rng.CombatEnergyCosts);
            d["combat_targets"] = WireRngCompat.Counter(rng.CombatTargets);
            d["monster_ai"] = WireRngCompat.Counter(rng.MonsterAi);
            d["niche"] = WireRngCompat.Counter(rng.Niche);
            d["combat_orbs"] = WireRngCompat.Counter(rng.CombatOrbGeneration);
            var prng = (_runState?.Players != null && _runState.Players.Count > 0)
                ? _runState.Players[0].PlayerRng
                : null;
            if (prng != null)
            {
                d["rewards"] = WireRngCompat.Counter(prng.Rewards);
                d["shops"] = WireRngCompat.Counter(prng.Shops);
                d["transformations"] = WireRngCompat.Counter(prng.Transformations);
            }
        }
        catch { /* best-effort diagnostic */ }
        return d;
    }

    /// <summary>
    /// The mirror's `$.rng` block: `run_seed` and `counters`, together or not at
    /// all.
    ///
    /// <see cref="RngCounters"/> is best-effort by design — it also feeds a
    /// purely diagnostic field where a partial answer beats none. This consumer
    /// cannot tolerate that: `combat_mirror.reconcile` reads the presence of
    /// `run_seed` alone as "the world owns the dice" and stops carrying its own
    /// valid cursors, and a `state_from_dict` handed a seed with no counters
    /// rebuilds every stream at count 0. So seed-without-counters does not
    /// degrade to the pre-field behaviour — it destroys lineage that was intact.
    /// Emit the pair or emit an empty block, which every consumer already reads
    /// as "this producer does not carry the run's dice".
    /// </summary>
    private Dictionary<string, object?> MirrorRngBlock()
    {
        var d = new Dictionary<string, object?>();
        var seed = _runState?.Rng?.Seed;
        if (seed == null) return d;
        var counters = RngCounters();
        if (counters.Count == 0) return d;
        d["run_seed"] = seed;
        d["counters"] = counters;
        return d;
    }

    private static Dictionary<string, object?>? PowerVars(PowerModel pw)
    {
        try
        {
            var vars = pw.DynamicVars?.Values;
            if (vars == null) return null;
            var d = new Dictionary<string, object?>();
            foreach (var dv in vars)
                d[dv.Name] = (int)dv.BaseValue;
            // Skittish blocks once a turn and keeps whether it did in private
            // data, behind a public getter; a rebuilt board let it block a
            // second time (GH #989). Not a DynamicVar, but carried as one.
            if (pw.GetType().Name == "SkittishPower"
                && pw.GetType().GetProperty("HasGainedBlockThisTurn")?.GetValue(pw) is bool gained)
                d["HasGainedBlockThisTurn"] = gained ? 1 : 0;
            // Surrounded's Facing (Right 0, Left 1) is a public getter, not a
            // DynamicVar: which crab arm is behind a player. Ours can be read
            // off the arms' intents; a teammate's cannot (GH #989).
            if (pw.GetType().Name == "SurroundedPower"
                && pw.GetType().GetProperty("Facing")?.GetValue(pw) is System.Enum facing)
                d["Facing"] = Convert.ToInt32(facing, System.Globalization.CultureInfo.InvariantCulture);
            return d.Count > 0 ? d : null;
        }
        catch { return null; }
    }

    /// <summary>
    /// Monsters whose cross-decision AI state lives in a PRIVATE field on the
    /// mutable <see cref="MonsterModel"/> — the moveset analogue of
    /// <see cref="RelicLatchFields"/>, and the same evidence-gated ALLOWLIST
    /// discipline: an entry is added only when the engine demonstrably tracks
    /// the value (on <c>MonsterAi.counter</c>) and the wire never carries it.
    ///
    /// - <c>Fabricator._lastSpawned</c> (Fabricator.cs:37): excluded from the
    ///   next <c>SpawnBot</c> pool. The engine infers it as the kind of the
    ///   highest-uid living bot, which is unrecoverable once that bot died —
    ///   the mirror then re-widens a forced pick back to a coin flip
    ///   (Zapbot-vs-Stabbot summon flips, v35 seed 1001).
    /// - <c>KnowledgeDemon._curseOfKnowledgeCounter</c> (KnowledgeDemon.cs:66):
    ///   indexes both the curse card set and the Disintegration damage table
    ///   (6/7/8) and gates CURSE_OF_KNOWLEDGE re-entry at 3. Not derivable
    ///   from the wire mid-fight — the DISINTEGRATION off-by-one.
    /// </summary>
    private static readonly System.Reflection.FieldInfo? FabricatorLastSpawnedField =
        MonsterField("Fabricator", "_lastSpawned");

    private static readonly System.Reflection.FieldInfo? KnowledgeDemonCurseCounterField =
        MonsterField("KnowledgeDemon", "_curseOfKnowledgeCounter");

    private static System.Reflection.FieldInfo? MonsterField(string typeName, string fieldName)
    {
        try
        {
            var t = typeof(MonsterModel).Assembly.GetType(
                "MegaCrit.Sts2.Core.Models.Monsters." + typeName);
            return t?.GetField(
                fieldName,
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
        }
        catch { return null; }
    }

    /// <summary>
    /// Appends the allowlisted private-AI-state fields for this monster to an
    /// already-built enemy dict. Quiet for every monster not on the list, and
    /// quiet for a Fabricator that has not spawned yet (absent field ==
    /// "nothing to disclose", mirroring the relic-latch convention).
    /// Wire shape: <c>fabricator_last_spawned</c> = the spawned bot's
    /// canonical <c>MonsterModel.Id.Entry</c> (e.g. <c>"ZAPBOT"</c>);
    /// <c>curse_of_knowledge_counter</c> = the raw int (0..3).
    /// </summary>
    private static void AppendMonsterHiddenState(Dictionary<string, object?> d, MonsterModel? m)
    {
        if (m == null) return;
        try
        {
            if (FabricatorLastSpawnedField?.DeclaringType?.IsInstanceOfType(m) == true
                && FabricatorLastSpawnedField.GetValue(m) is MonsterModel last)
                d["fabricator_last_spawned"] = last.Id.Entry;
            if (KnowledgeDemonCurseCounterField?.DeclaringType?.IsInstanceOfType(m) == true
                && KnowledgeDemonCurseCounterField.GetValue(m) is int curse)
                d["curse_of_knowledge_counter"] = curse;
        }
        catch { }
    }

    // Null when the cost model cannot be read at all — NOT -1, which the
    // consumer decodes as X-cost/unplayable. Guarded like its bridge twin
    // `SimStateExtractor.SafeBaseCost`: `GetWithModifiers` walks the modifier
    // list and a serializer must not take the whole message down over one card.
    internal static int? BaseEnergyCost(CardModel c)
    {
        try
        {
            var ec = c.EnergyCost;
            if (ec == null) return null;
            if (ec.CostsX) return -1;
            var v = ec.GetWithModifiers(CostModifiers.Local);
            return v < 0 ? -1 : v;
        }
        catch { return null; }
    }

    // CardEnergyCost._localModifiers. The list itself is the one piece of the
    // cost model with no public accessor; LocalCostModifier and every member
    // we read off it (Amount / Type / Expiration / IsReduceOnly / Modify) are
    // public, so this is a single field read, not a reimplementation.
    private static readonly System.Reflection.FieldInfo? LocalModifiersField =
        typeof(CardEnergyCost).GetField(
            "_localModifiers",
            System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);

    /// <summary>
    /// What <see cref="BaseEnergyCost"/> will be once this turn's cost
    /// modifiers wear off — the card's cost after C# <c>EndOfTurnCleanup()</c>.
    ///
    /// A local modifier carries a LIFETIME (EndOfCombat / EndOfTurn /
    /// WhenPlayed) that a single number cannot express, and collapsing them
    /// loses exactly the distinction a state reconstruction needs: Stomp's
    /// "1 less per Attack played THIS TURN" and Enlightenment's "costs 1 for
    /// the rest of COMBAT" both arrive as a cheaper card, but only the first
    /// springs back at turn start. Rebuilt from <c>base_cost</c> alone, a
    /// mirror keeps every transient discount forever: on the v8 sweep a
    /// shaved STOMP stayed at 0-2 while the game reset it to 3, and 163 of
    /// its 175 cost divergences landed on the turn boundary that should have
    /// restored it.
    ///
    /// Computed by re-folding the local modifiers while skipping the ones
    /// EndOfTurnCleanup would remove — the same loop as
    /// <c>GetWithModifiers(Local)</c>, with the same guards. Null when the
    /// value cannot differ from <c>base_cost</c> (no local modifiers, X-cost,
    /// canonical/negative base) or when the field read fails, which keeps the
    /// wire quiet for the overwhelmingly common unmodified card.
    /// </summary>
    internal static int? PersistentEnergyCost(CardModel c) =>
        FoldLocalModifiers(c, LocalCostModifierExpiration.EndOfTurn);

    /// <summary>
    /// The cost this card will have once it has been PLAYED — the local
    /// modifiers re-folded with the <see cref="LocalCostModifierExpiration.WhenPlayed"/>
    /// ones dropped, mirroring <c>CardEnergyCost.AfterCardPlayedCleanup()</c>.
    /// </summary>
    /// <remarks>
    /// Distinct from <see cref="PersistentEnergyCost"/>, and the difference is
    /// not academic: the game's own comments say the "costs N this turn"
    /// helpers are <c>SetThisTurnOrUntilPlayed</c> / <c>AddThisTurnOrUntilPlayed</c>
    /// (<c>EndOfTurn | WhenPlayed</c>, whichever comes first), and flag the
    /// pure-EndOfTurn variants "BE CAREFUL USING THIS!". A consumer that
    /// restores such a discount only at the turn boundary keeps it alive on a
    /// card the game already retired it on — 169 of the 1569 divergences on
    /// the v15 sweep were a discarded card whose cost the mirror still had
    /// shaved to 0.
    /// </remarks>
    internal static int? PlayedEnergyCost(CardModel c) =>
        FoldLocalModifiers(c, LocalCostModifierExpiration.WhenPlayed);

    /// <summary>
    /// The cost once BOTH transient edges have passed — the card played and
    /// the turn ended — leaving only the modifiers that last all combat.
    /// </summary>
    /// <remarks>
    /// The third number is load-bearing, not belt-and-braces — though not
    /// because the other two collide. One combined <c>EndOfTurn | WhenPlayed</c>
    /// modifier folds to <c>(persistent, played) = (base, base)</c>; two
    /// separate modifiers fold to <c>(M2(base), M1(base))</c>, and those differ
    /// unless both modifiers are no-ops.
    ///
    /// What the pair cannot say is how much of the shave SURVIVES the first
    /// edge. With one combined modifier the whole shave retires at whichever
    /// edge fires first and there is nothing left to re-bank; with two, each
    /// edge retires only its own half. The cost once BOTH cleanups have run is
    /// the fixed point that tells the consumer's
    /// <c>rebank_wire_cost_after_edge</c> which of the two it is holding.
    /// </remarks>
    internal static int? StableEnergyCost(CardModel c) =>
        FoldLocalModifiers(
            c, LocalCostModifierExpiration.EndOfTurn | LocalCostModifierExpiration.WhenPlayed);

    /// Re-fold a card's local cost modifiers from its permanent base, skipping
    /// every modifier carrying ANY of the <paramref name="expiring"/> flags.
    /// Null when the card has no local modifiers to fold, or the fold cannot
    /// be trusted.
    private static int? FoldLocalModifiers(CardModel c, LocalCostModifierExpiration expiring)
    {
        try
        {
            // Guards INSIDE the try, like the bridge twin: `c.EnergyCost` and
            // `c.IsCanonical` are the reads most likely to throw here, and
            // outside the guard they took the whole message with them.
            var ec = c.EnergyCost;
            if (ec == null || ec.CostsX || !ec.HasLocalModifiers || c.IsCanonical) return null;
            // GetWithModifiers(None) returns the private `_base` untouched by
            // any modifier — the upgrade-adjusted permanent cost to fold from.
            var baseCost = ec.GetWithModifiers(CostModifiers.None);
            if (baseCost < 0) return null;
            if (LocalModifiersField?.GetValue(ec) is not IEnumerable<LocalCostModifier> mods) return null;
            var cost = baseCost;
            foreach (var m in mods)
            {
                if ((m.Expiration & expiring) != 0) continue;
                cost = m.Modify(cost);
            }
            return Math.Max(0, cost);
        }
        catch { return null; }
    }

    /// <summary>
    /// Serialize a creature's telegraphed intents. Shared by the alive and the
    /// dead enemy branches — a corpse the game keeps in combat (Decimillipede
    /// segments awaiting REATTACH) telegraphs its revive one turn ahead, and
    /// that is exactly the turn a consumer most needs to know about it.
    /// Returns an empty list when the move carries no intents; C# builds plenty
    /// of moves that way (`new MoveState("DEAD_MOVE", DeadMove)` passes no
    /// `params AbstractIntent[]`), and "no intents" is a real, observable state
    /// distinct from a `HiddenIntent`.
    /// </summary>
    private static List<Dictionary<string, object?>> SerializeIntents(
        Creature e, List<Creature>? playerCreatures)
    {
        var intents = new List<Dictionary<string, object?>>();
        try
        {
            if (e.Monster?.NextMove?.Intents == null) return intents;
            foreach (var intent in e.Monster.NextMove.Intents)
            {
                var intentInfo = new Dictionary<string, object?>
                {
                    ["type"] = intent.IntentType.ToString(),
                };
                // Get damage for attack intents. Emit PER-HIT damage
                // (GetSingleDamage) and the repeat count separately so
                // downstream consumers (notably sts2-rs's
                // state_dict::parse_intent under bead sts2-rs-2rp7) can
                // reconstruct Intent::MultiAttack { damage, hits } without
                // double-multiplying. The previous GetTotalDamage emission was
                // overstating multi-hit damage by `hits×` once consumers
                // learned to honor the `hits` field. Bead sts2-rs-x4nn.
                if (intent is MegaCrit.Sts2.Core.MonsterMoves.Intents.AttackIntent atk && playerCreatures != null)
                {
                    try
                    {
                        intentInfo["damage"] = atk.GetSingleDamage(playerCreatures, e);
                        intentInfo["hits"] = atk.Repeats;
                    }
                    catch { }
                }
                intents.Add(intentInfo);
            }
        }
        catch { }
        return intents;
    }

    private Dictionary<string, object?> SerializeCombatCard(
        CardModel c, int index, PlayerCombatState? pcs, bool includePlayability)
    {
        var stats = new Dictionary<string, object?>();
        try
        {
            foreach (var dv in c.DynamicVars.Values)
                stats[dv.Name.ToLowerInvariant()] = (int)dv.BaseValue;
        }
        catch { }

        var starCost = c.BaseStarCost;
        var cardInfo = new Dictionary<string, object?>
        {
            ["index"] = index,
            ["id"] = c.Id.ToString(),
            ["name"] = _loc.Card(c.Id.Entry),
            ["cost"] = c.EnergyCost?.GetResolved() ?? 0,
            // The card instance's OWN cost: upgrade-adjusted base with its
            // LOCAL modifiers applied (Stomp's per-attack shave, Eidolon's
            // until-played zero) but WITHOUT the global power hooks
            // (FreeAttack, Tangled, Enthralled). That is exactly what the
            // engine stores in `Card.cost` — the globals are re-derived from
            // the powers we already ship, so a consumer that rebuilds from
            // `cost` alone bakes a transient discount in FOREVER (a spent
            // FreeAttack left every Strike in hand at 0 for the rest of the
            // fight). X-cost cards report -1, the engine's X sentinel;
            // `GetWithModifiers` would give 0 for them, which is a real cost.
            ["base_cost"] = BaseEnergyCost(c),
            // What `base_cost` becomes once this turn's cost modifiers wear
            // off. Only present when it can differ — see PersistentEnergyCost.
            ["persistent_cost"] = PersistentEnergyCost(c),
            // What this card's cost becomes the moment it is PLAYED. The
            // "this turn" discounts are really "this turn OR until played",
            // so a consumer that only expires them at the turn boundary keeps
            // a spent discount alive. See PlayedEnergyCost.
            ["played_cost"] = PlayedEnergyCost(c),
            // And once BOTH edges have passed. Needed to tell a single
            // "this turn or until played" modifier apart from two separate
            // ones — see StableEnergyCost.
            ["stable_cost"] = StableEnergyCost(c),
            ["type"] = c.Type.ToString(),
            // Upgrade identity for state reconstruction: `stats` only carries
            // damage/block, so without this flag every effect-var upgrade
            // (magic numbers, draw counts, status amounts) reverts to the
            // unupgraded registry value in the dict-reconstructed search
            // state. The run-scope deck serializer already emits it.
            ["upgraded"] = c.IsUpgraded,
            ["stats"] = stats.Count > 0 ? stats : null,
            ["description"] = _loc.Bilingual("cards", c.Id.Entry + ".description"),
            ["tinker_time_rider"] = MadScienceRider(c),
            // GH #370. The SINGLE-TURN halves of Sly and Retain live in two
            // private `CardModel` bools (`_hasSingleTurnSly`,
            // `_hasSingleTurnRetain`, cleared in `EndOfTurnCleanup`) that the
            // `keywords` list below cannot show: `Keywords` carries only the
            // intrinsic set plus `LocalKeywords` (what `AddKeyword` adds).
            // Without these two the reconstruction cannot tell a Well-Laid
            // Plans retain-for-this-turn from an intrinsic Retain, so it had
            // to drop both — and a rebuilt mirror then let the card flush at
            // end of turn while the world kept it.
            //
            // Emitted as the PUBLIC properties rather than the private bools,
            // so the consumer subtracts the keyword itself:
            //   single-turn retain == will_retain && !keywords.Retain
            //   single-turn sly    == sly_this_turn && !keywords.Sly
            // (`CardModel.cs:591` / `:620` — each is `keyword || singleTurn`.)
            ["will_retain"] = SafeCardBool(() => c.ShouldRetainThisTurn),
            ["sly_this_turn"] = SafeCardBool(() => c.IsSlyThisTurn),
        };
        if (includePlayability)
        {
            cardInfo["can_play"] = c.CanPlay(out _, out _);
            cardInfo["target_type"] = c.TargetType.ToString();
        }
        if (starCost > 0)
        {
            // The CURRENT star cost (`CurrentStarCost`), not the printed one,
            // plus what it becomes once this turn's temporary costs expire
            // (`GetStarCostThisCombat()`). `SetToFreeThisTurn` (Splash,
            // Discovery, Infernal Blade, the generative potions) zeroes the
            // star cost with the energy cost; shipping `BaseStarCost` here told
            // the consumer a free Splash'd Falling Star still cost 2 stars, so
            // the rebuild listed it unplayable. Same fields, same meaning, as
            // the mod's `SimStateExtractor.SafeStarCost` /
            // `SafePersistentStarCost` (PROTOCOL.md "Card object").
            cardInfo["star_cost"] = Math.Max(0, c.CurrentStarCost);
            cardInfo["persistent_star_cost"] = Math.Max(0, c.GetStarCostThisCombat());
            // BUG-007: Override can_play for star-cost cards when player lacks
            // stars. Judged on the cost the play would actually spend
            // (`GetStarCostWithModifiers`, the number `SpendResources` uses),
            // so a card made free -- or zeroed by VoidForm / Brilliant Scarf --
            // is not vetoed over a star price it no longer has.
            if (includePlayability && pcs != null && pcs.Stars < Math.Max(0, c.GetStarCostWithModifiers()))
                cardInfo["can_play"] = false;
        }
        var kws = c.Keywords?.Where(k => k != CardKeyword.None).Select(k => k.ToString()).ToList();
        if (kws?.Count > 0) cardInfo["keywords"] = kws;
        if (c.Enchantment != null)
        {
            cardInfo["enchantment"] = _loc.Bilingual("enchantments", c.Enchantment.Id.Entry + ".title");
            try { if (c.Enchantment.Amount != 0) cardInfo["enchantment_amount"] = c.Enchantment.Amount; } catch { }
        }
        if (c.Affliction != null)
        {
            cardInfo["affliction"] = _loc.Bilingual("afflictions", c.Affliction.Id.Entry + ".title");
            try { if (c.Affliction.Amount != 0) cardInfo["affliction_amount"] = c.Affliction.Amount; } catch { }
        }
        return cardInfo;
    }


    private Dictionary<string, object?> WireCombatBoard(Player player)
    {
        var pcs = player.PlayerCombatState;
        var combatState = CombatManager.Instance.DebugOnlyGetState();

        // (read-only: the game-over HP tracking of combat_play stays in combat_play)

        var hand = pcs?.Hand?.Cards?.Select((c, i) =>
            SerializeCombatCard(c, i, pcs, includePlayability: true)).ToList() ?? new();

        var playerCreatures = combatState?.PlayerCreatures?.ToList();

        var enemies = combatState?.Enemies?
            .Where(e => e != null && e.IsAlive)
            .Select((e, i) =>
            {
                var intents = SerializeIntents(e, playerCreatures);

                // Enemy powers
                var ePowers = e.Powers?.Select(pw => new Dictionary<string, object?>
                {
                    ["name"] = _loc.Power(pw.Id.Entry),
                    // Canonical power id (PowerModel.Id.Entry, e.g. "MANGLE_POWER"),
                    // the same value the canonical/wire shape emits as "kind". The
                    // hint adapter reconstructs the power from this id unambiguously
                    // — the localized title alone cannot (e.g. "Mangle" maps to both
                    // the real ManglePower and its legacy Mangle alias, and powers
                    // with no loc title resolve to None and were silently dropped,
                    // defanging the enemy in the dict-reconstructed search).
                    ["id"] = pw.Id.Entry,
                    ["description"] = _loc.Bilingual("powers", pw.Id.Entry + ".description"),
                    ["amount"] = pw.Amount,
                    // Counter powers (e.g. Slow) override DisplayAmount: Amount is
                    // the stack count (constant) while DisplayAmount is the value
                    // shown to the player and the one that actually scales. Emit
                    // both so the UI can render the meaningful number.
                    ["display_amount"] = pw.DisplayAmount,
                    // Same contract as the player-power block below: a power's
                    // per-instance accumulator lives in a DynamicVar that
                    // neither Amount nor DisplayAmount describes. Withholding
                    // it from ENEMY powers only (the historical shape) left
                    // Thievery's stolen-Gold tally — and every other
                    // enemy-side var — structurally undisclosed to the
                    // mirror, while the SpireMentor twin (SerializePower)
                    // already emitted vars for all creatures.
                    ["vars"] = PowerVars(pw),
                }).ToList();

                var enemyDict = new Dictionary<string, object?>
                {
                    ["index"] = i,
                    ["name"] = _loc.Monster(e.Monster?.Id.Entry ?? "UNKNOWN"),
                    ["hp"] = e.CurrentHp,
                    ["max_hp"] = e.MaxHp,
                    ["block"] = e.Block,
                    ["intents"] = intents.Count > 0 ? intents : null,
                    ["intends_attack"] = e.Monster?.IntendsToAttack ?? false,
                    ["powers"] = ePowers?.Count > 0 ? ePowers : null,
                    // The telegraphed move-state id (MoveState.StateId), same
                    // field the parity exporter emits. The hint adapter pins
                    // the rebuilt monster AI to it (sts2_rs force_current_
                    // state_by_name_relaxed); without it every reconstructed
                    // enemy is AI-less after its first move in rollouts — the
                    // exact planning blindness fixed live in PRs #263/#264.
                    ["move_state"] = e.Monster?.NextMove?.Id,
                };
                // Allowlisted private AI state (Fabricator._lastSpawned,
                // KnowledgeDemon curse counter) — see AppendMonsterHiddenState.
                AppendMonsterHiddenState(enemyDict, e.Monster);
                return enemyDict;
            }).ToList() ?? new();

        // Dead-but-still-in-combat enemies (revivers: Decimillipede segments
        // awaiting REATTACH, Eye with Teeth illusions, Test Subject phases...).
        // The game removes ordinary corpses from Enemies; whatever remains
        // with IsAlive == false WILL come back, and a search model that never
        // sees the corpse treats every kill as permanent. Kept in a SEPARATE
        // list so the alive list's index semantics (play_card targeting) are
        // untouched.
        var deadEnemies = combatState?.Enemies?
            .Where(e => e != null && !e.IsAlive)
            .Select(e =>
            {
                var dIntents = SerializeIntents(e, playerCreatures);
                var dPowers = e.Powers?.Select(pw => new Dictionary<string, object?>
                {
                    ["name"] = _loc.Power(pw.Id.Entry),
                    ["id"] = pw.Id.Entry,
                    ["amount"] = pw.Amount,
                    ["display_amount"] = pw.DisplayAmount,
                    // See the alive-enemy block: same vars contract, and a
                    // reviving corpse keeps its accumulators through death
                    // (ReattachPower and friends), so the corpse wire must
                    // not silently zero them either.
                    ["vars"] = PowerVars(pw),
                }).ToList();
                var corpseDict = new Dictionary<string, object?>
                {
                    ["name"] = _loc.Monster(e.Monster?.Id.Entry ?? "UNKNOWN"),
                    ["hp"] = e.CurrentHp,
                    ["max_hp"] = e.MaxHp,
                    ["block"] = e.Block,
                    ["is_dead"] = true,
                    ["powers"] = dPowers?.Count > 0 ? dPowers : null,
                    // A corpse that is coming back telegraphs what it will do.
                    // `ReattachPower.AfterDeath` arms the segment onto
                    // DEAD_MOVE (no intents), whose FollowUpState is
                    // REATTACH_MOVE (a HealIntent) — so the turn before a
                    // Decimillipede segment revives, the wire is the only
                    // place that says so. Omitting this field made every
                    // reviving corpse read as an intent divergence against a
                    // mirror that models the revive correctly, and left the
                    // search blind to a 25-HP enemy about to re-enter the
                    // fight. Same shape as the alive branch above.
                    ["intents"] = dIntents.Count > 0 ? dIntents : null,
                    ["move_state"] = e.Monster?.NextMove?.Id,
                };
                // Same allowlisted private AI state as the alive branch — a
                // reviving monster's counters survive its death.
                AppendMonsterHiddenState(corpseDict, e.Monster);
                return corpseDict;
            }).ToList() ?? new();

        // Player powers/buffs
        var playerPowers = player.Creature?.Powers?.Select(pw => new Dictionary<string, object?>
        {
            ["name"] = _loc.Power(pw.Id.Entry),
            // Canonical power id; see the enemy-power note above (lets the hint
            // adapter reconstruct the power from the id, not the ambiguous title).
            ["id"] = pw.Id.Entry,
            ["description"] = _loc.Bilingual("powers", pw.Id.Entry + ".description"),
            ["amount"] = pw.Amount,
            // See enemy-power note above: DisplayAmount is the player-facing
            // (and scaling) value for Counter powers; Amount is the stack count.
            ["display_amount"] = pw.DisplayAmount,
            ["vars"] = PowerVars(pw),
        }).ToList();

        // Serialize draw/discard/exhaust piles. Cards in piles use the same
        // shape as hand cards but without can_play/target_type (not applicable).
        // Listing draw pile in actual order matches what set_draw_order expects;
        // it's a deliberate design choice for testing/AI use cases.
        var drawPile = pcs?.DrawPile?.Cards?.Select((c, i) =>
            SerializeCombatCard(c, i, pcs, includePlayability: false)).ToList() ?? new();
        var discardPile = pcs?.DiscardPile?.Cards?.Select((c, i) =>
            SerializeCombatCard(c, i, pcs, includePlayability: false)).ToList() ?? new();
        // The card that is MID-PLAY. C# parks it in its own `PlayPile` "so
        // that it isn't counted towards your hand or your discard pile"
        // (PileType.Play), and moves it to the discard only once the play
        // finishes — which, for a card whose OnPlay awaits a selection screen,
        // is after the player picks.
        //
        // A `card_select` board is therefore ALWAYS serialized mid-play, and
        // without this pile the played card appears in none of them. A state
        // rebuilt from that board has nothing to flush when the choice
        // resolves, so the card silently fails to reach the discard: every
        // Headbutt / Snap / Photon Cut that opened a screen showed up as one
        // missing discard card on the v10 sweep (173 world-only HEADBUTT
        // entries alone). The engine has carried a `play_pile` on the wire
        // since the oracle found the same gap on its own side.
        List<Dictionary<string, object?>> playPile = new();
        try
        {
            var inPlay = pcs?.PlayPile?.Cards;
            if (inPlay != null)
                playPile = inPlay.Select((c, i) =>
                    SerializeCombatCard(c, i, pcs, includePlayability: false)).ToList();
        }
        catch { }
        List<Dictionary<string, object?>> exhaustPile = new();
        try
        {
            var exhaust = pcs?.ExhaustPile?.Cards;
            if (exhaust != null)
                exhaustPile = exhaust.Select((c, i) =>
                    SerializeCombatCard(c, i, pcs, includePlayability: false)).ToList();
        }
        catch { }

        // Per-turn play state the mid-turn ShouldPlay gates need (Ringing /
        // Normality / Sloth buckets + ChainsOfBinding's once-per-turn Bound
        // gate). Same fields the SpireMentor mod emits; without them a
        // dict-reconstructed search state believes zero cards were played
        // this turn and re-offers plays the engine vetoes.
        var playerSummary = WirePlayerSummary(player);
        try
        {
            playerSummary["cards_played_this_turn"] = CountCardsPlayedThisTurn(combatState, player.Creature);
            if (BoundCardPlayedThisTurn(combatState, player.Creature))
                playerSummary["bound_card_played_this_turn"] = true;
        }
        catch { }

        var result = new Dictionary<string, object?>
        {
            ["type"] = "decision",
            ["decision"] = "combat_play",
            ["context"] = WireRunContext(),
            ["round"] = combatState?.RoundNumber ?? 0,
            ["energy"] = pcs?.Energy ?? 0,
            ["max_energy"] = pcs?.MaxEnergy ?? 0,
            ["hand"] = hand,
            ["enemies"] = enemies,
            ["dead_enemies"] = deadEnemies.Count > 0 ? deadEnemies : null,
            ["player"] = playerSummary,
            ["player_powers"] = playerPowers?.Count > 0 ? playerPowers : null,
            ["draw_pile"] = drawPile,
            ["discard_pile"] = discardPile,
            ["exhaust_pile"] = exhaustPile,
            // Only present when a card is actually mid-play — see above.
            ["play_pile"] = playPile.Count > 0 ? playPile : null,
            ["draw_pile_count"] = drawPile.Count,
            ["discard_pile_count"] = discardPile.Count,
            ["exhaust_pile_count"] = exhaustPile.Count,
            // The run's RNG position. `run_seed` is RunRngSet.Seed — the
            // once-hashed master seed every per-stream salt mixes into, which
            // is exactly what sts2-rs's `RunRngSet::new` takes — and
            // `counters` is each stream's read count.
            //
            // Without them a state reconstruction has no RNG lineage at all:
            // it seeds from the SEARCH seed, so its monster_ai and shuffle
            // streams have nothing to do with the game's. Every enemy intent
            // and every reshuffle is then a guess, and the resulting
            // divergence says nothing about fidelity because the value was
            // never knowable. With them the mirror rolls the same dice the
            // game does, and an intent mismatch becomes what it should be: a
            // real disagreement about what the engine does with a given roll.
            //
            // This is ground truth for the MIRROR, not for the search — the
            // search clone is blindfolded at its root and is provably
            // invariant to the cursor (sts2-mcts
            // `root_scramble_invariant_to_pile_order_and_rng_cursor`).
            // Published as a PAIR or not at all — see `MirrorRngBlock`.
            ["rng"] = MirrorRngBlock(),
        };

        // Character-specific mechanics
        try
        {
            // Defect: Orbs. Emit whenever the queue has slots (Defect) so the
            // workbench shows the orb belt — including its empty slots — from
            // turn 1, not only once an orb is channelled. ``orbs`` is the list
            // of channelled orbs (empty when none); ``orb_slots`` is the belt
            // capacity the frontend pads with empty slots.
            var orbQueue = pcs?.OrbQueue;
            if (orbQueue != null && orbQueue.Capacity > 0)
            {
                result["orbs"] = orbQueue.Orbs.Select((orb, i) => new Dictionary<string, object?>
                {
                    ["index"] = i,
                    ["name"] = _loc.Bilingual("orbs", orb.Id.Entry + ".title"),
                    ["type"] = orb.GetType().Name.Replace("Orb", ""),
                    ["passive"] = (int)orb.PassiveVal,
                    ["evoke"] = (int)orb.EvokeVal,
                }).ToList();
                result["orb_slots"] = orbQueue.Capacity;
            }

            // Regent: Stars
            if (pcs != null && pcs.Stars >= 0 && player.Character?.Id.Entry == "REGENT")
            {
                result["stars"] = pcs.Stars;
            }

            // Necrobinder: Osty (minion)
            var osty = player.Osty;
            if (osty != null)
            {
                result["osty"] = new Dictionary<string, object?>
                {
                    ["name"] = _loc.Monster(osty.Monster?.Id.Entry ?? "OSTY"),
                    ["hp"] = osty.CurrentHp,
                    ["max_hp"] = osty.MaxHp,
                    ["block"] = osty.Block,
                    ["alive"] = osty.IsAlive,
                };
            }
            else if (player.Character?.Id.Entry == "NECROBINDER")
            {
                result["osty"] = new Dictionary<string, object?> { ["alive"] = false };
            }
        }
        catch (Exception ex)
        {
            LogException("Character-specific data", ex);
        }

        return result;
    }


    private static int[] CountCardsPlayedThisTurn(CombatState? combatState, Creature? playerCreature)
    {
        var buckets = new int[5];
        if (combatState == null || playerCreature == null)
            return buckets;
        foreach (var entry in CombatManager.Instance.History.CardPlaysStarted)
        {
            if (!entry.HappenedThisTurn(combatState))
                continue;
            if (entry.CardPlay.Card.Owner.Creature != playerCreature)
                continue;
            int idx = entry.CardPlay.Card.Type switch
            {
                CardType.Attack => 0,
                CardType.Skill => 1,
                CardType.Power => 2,
                CardType.Curse => 3,
                CardType.Status => 4,
                _ => -1,
            };
            if (idx >= 0)
                buckets[idx]++;
        }
        return buckets;
    }

    /// <summary>
    /// ChainsOfBinding's once-per-turn gate, recomputed the way the power
    /// tracks it (ChainsOfBindingPower.BeforeCardPlayed sets a private
    /// flag when a non-dupe Bound card is played; afflictions clear only
    /// at turn end, so a history scan reproduces it exactly).
    /// </summary>
    private static bool BoundCardPlayedThisTurn(CombatState? combatState, Creature? playerCreature)
    {
        if (combatState == null || playerCreature == null)
            return false;
        foreach (var entry in CombatManager.Instance.History.CardPlaysStarted)
        {
            if (!entry.HappenedThisTurn(combatState))
                continue;
            if (entry.CardPlay.Card.Owner.Creature != playerCreature)
                continue;
            if (entry.CardPlay.Card.IsDupe)
                continue;
            if (entry.CardPlay.Card.Affliction is MegaCrit.Sts2.Core.Models.Afflictions.Bound)
                return true;
        }
        return false;
    }

    private Dictionary<string, object?> WirePlayerSummary(Player player)
    {
        return new Dictionary<string, object?>
        {
            ["name"] = _loc.Bilingual("characters", (player.Character?.Id.Entry ?? "IRONCLAD") + ".title"),
            ["hp"] = player.Creature?.CurrentHp ?? 0,
            ["max_hp"] = player.Creature?.MaxHp ?? 0,
            ["block"] = player.Creature?.Block ?? 0,
            ["gold"] = player.Gold,
            // Total belt capacity (filled + empty). Without it a driver cannot
            // tell a full belt from a partially-filled one, and shop buy_potion
            // retries loop forever on TooFull (same failure the live bridge
            // fixed via !Player.HasOpenPotionSlots).
            ["potion_slots"] = player.MaxPotionCount,
            ["relics"] = player.Relics?.Select(r =>
            {
                var vars = new Dictionary<string, object?>();
                try { foreach (var dv in r.DynamicVars.Values) vars[dv.Name] = (int)dv.BaseValue; } catch { }
                return new Dictionary<string, object?>
                {
                    // Entry id alongside the display name: the parity harness
                    // (run_parity._relic_set) compares relic IDENTITY across
                    // engines; the Rust backend emits Id.Entry, so without this
                    // the C# side fell back to the loc TITLE and any relic
                    // whose title differs from its entry words (e.g.
                    // CHOSEN_CHEESE → "The Chosen Cheese") false-diverged.
                    ["id"] = r.Id.Entry,
                    ["name"] = _loc.Relic(r.Id.Entry),
                    ["description"] = _loc.Bilingual("relics", r.Id.Entry + ".description"),
                    ["vars"] = vars.Count > 0 ? vars : null,
                    // The relic's LIVE counter, for the relics that have one.
                    // `vars` above are the canonical thresholds (Nunchaku's
                    // "every 10 attacks"); this is the progress toward them.
                    // `ShowCounter` is the game's own declaration that the
                    // relic tracks something, and `DisplayAmount` is what it
                    // tracks — the private tally is not otherwise reachable.
                    //
                    // Without it every reconstruction restarted the tally at
                    // zero and the mirror skipped whatever it was about to
                    // trigger. On the v10 sweep the counter relics were the
                    // top lift over baseline in the single-field
                    // `$.player.energy` class (188 records where the mirror
                    // sat exactly 1 energy low): Nunchaku 3.0x, Kusarigama
                    // 3.3x, HappyFlower 2.3x.
                    ["counter"] = SafeRelicCounter(r),
                    // ToyBox wax/melt state. A melted relic STAYS in
                    // Player.Relics (so it serializes like any other) but is
                    // filtered out of every hook dispatch
                    // (CombatState.cs:133, RunState.cs:148). Without these
                    // flags the mirror keeps applying relics the world has
                    // disabled: StrikeDummy +3, RuinedHelmet doubling and
                    // IntimidatingHelmet block were all one melt bug across
                    // the ToyBox seeds (1001/1018/1019). Field names match
                    // the live bridge (SimStateExtractor.cs).
                    ["is_wax"] = r.IsWax,
                    ["is_melted"] = r.IsMelted,
                };
            }).ToList(),
            ["potions"] = player.Potions?.Select((p, i) =>
            {
                if (p == null) return null;
                var pvars = new Dictionary<string, object?>();
                try { foreach (var dv in p.DynamicVars.Values) pvars[dv.Name] = (int)dv.BaseValue; } catch { }
                return new Dictionary<string, object?>
                {
                    ["index"] = i,
                    // Registry entry alongside the display name, for the same
                    // reason relics carry `id`: a trace that compares potion
                    // IDENTITY across worlds cannot use the localized title
                    // (the engine names the same potion `AttackPotion`, this
                    // side renders "Attack Potion", and titles that differ
                    // from their entry words diverge outright).
                    //
                    // Deliberately NOT called `id`: `_potion_obs_id` prefers
                    // an `id` key, so adding one there would swap the token
                    // the PPO vocab sees for every potion in the real-DLL arm
                    // and silently re-base every measured sweep. This key is
                    // read only by the run tracer.
                    ["entry"] = p.Id.Entry,
                    ["name"] = _loc.Potion(p.Id.Entry),
                    ["description"] = _loc.Bilingual("potions", p.Id.Entry + ".description"),
                    ["vars"] = pvars.Count > 0 ? pvars : null,
                    ["target_type"] = p.TargetType.ToString(),
                };
            }).Where(x => x != null).ToList(),
            ["deck_size"] = player.Deck?.Cards?.Count(c => c != null) ?? 0,
            ["deck"] = player.Deck?.Cards?.Where(c => c != null).Select(c =>
            {
                var dstats = new Dictionary<string, object?>();
                try { foreach (var dv in c.DynamicVars.Values) dstats[dv.Name.ToLowerInvariant()] = (int)dv.BaseValue; } catch { }
                var dkws = c.Keywords?.Where(k => k != CardKeyword.None).Select(k => k.ToString()).ToList();
                return new Dictionary<string, object?>
                {
                    ["id"] = c.Id.ToString(),
                    ["name"] = _loc.Card(c.Id.Entry),
                    ["cost"] = c.EnergyCost?.GetResolved() ?? 0,
                    ["type"] = c.Type.ToString(),
                    ["upgraded"] = c.IsUpgraded,
                    ["description"] = _loc.Bilingual("cards", c.Id.Entry + ".description"),
                    ["stats"] = dstats.Count > 0 ? dstats : null,
                    ["keywords"] = dkws?.Count > 0 ? dkws : null,
                    ["tinker_time_rider"] = MadScienceRider(c),
                    ["after_upgrade"] = GetUpgradedInfo(c),
                };
            }).ToList(),
        };
    }

    /// <summary>Common context added to every decision point.</summary>
    private Dictionary<string, object?> WireRunContext()
    {
        if (_runState == null) return new();
        var ctx = new Dictionary<string, object?>
        {
            ["act"] = _runState.CurrentActIndex + 1,
            ["act_name"] = _loc.Act(_runState.Act?.Id.Entry ?? "OVERGROWTH"),
            ["floor"] = _runState.ActFloor,
            // Run-wide floor: `floor` above is act-LOCAL (resets each act),
            // but the AI bridge envelope needs the live mod's TotalFloor
            // semantics (SimStateExtractor.cs) for the net's total_floor
            // feature.
            ["total_floor"] = _runState.TotalFloor,
            ["room_type"] = _runState.CurrentRoom?.RoomType.ToString(),
        };

        // Boss encounter info — use BossEncounter?.Id?.Entry
        try
        {
            var bossIdEntry = _runState.Act?.BossEncounter?.Id?.Entry;
            if (!string.IsNullOrEmpty(bossIdEntry))
            {
                var monsterKey = bossIdEntry.EndsWith("_BOSS") ? bossIdEntry[..^5] : bossIdEntry;
                // Handle special mappings
                if (monsterKey == "THE_KIN") monsterKey = "KIN_PRIEST";
                ctx["boss"] = new Dictionary<string, object?>
                {
                    ["id"] = bossIdEntry,
                    ["name"] = _loc.Monster(monsterKey),
                };
            }
        }
        catch { }

        return ctx;
    }

    private static string? MadScienceRider(CardModel c) =>
        c is MadScience ms ? ms.TinkerTimeRider.ToString() : null;

    private static readonly Dictionary<string, System.Reflection.FieldInfo?> RelicLatchFields =
        new()
        {
            ["VAMBRACE"] = LatchField("Vambrace", "_blockGainedThisCombat"),
            ["RUINED_HELMET"] = LatchField("RuinedHelmet", "_usedThisCombat"),
            // First-Power-card-this-combat 7-block grant; the engine latches it
            // on Relic.counter. 51 of the 140 on-play $.player.block records on
            // the v27wr sweep were a re-armed Permafrost paying its unpowered 7
            // again on every Power card (delta exactly -7, invariant across
            // which card was played).
            ["PERMAFROST"] = LatchField("Permafrost", "_activatedThisCombat"),
            // First-HP-loss-this-combat draw 3; the engine latches it on
            // Relic.counter too. Rebuilt unspent, a co-op replay drew the 3
            // again on the next hit.
            ["CENTENNIAL_PUZZLE"] = LatchField("CentennialPuzzle", "_usedThisCombat"),
            // First debuffing card of a combat doubles its debuffs; spent, it
            // is the engine's UNSETTLING_LAMP_FINISHED (RelicLatchSpent).
            // Rebuilt unspent, a co-op replay doubled a later Bash+ (GH #989).
            ["UNSETTLING_LAMP"] = LatchField("UnsettlingLamp", "_isFinishedTriggering"),
        };

    /// <summary>The counter a spent latch reads as, where the engine's is not
    /// 1: UnsettlingLamp's 1 is "a card captured, mid-play", its 2 "finished".</summary>
    private static readonly Dictionary<string, int> RelicLatchSpent = new()
    {
        ["UNSETTLING_LAMP"] = 2,
    };

    private static System.Reflection.FieldInfo? LatchField(string typeName, string fieldName)
    {
        try
        {
            var t = typeof(RelicModel).Assembly.GetType(
                "MegaCrit.Sts2.Core.Models.Relics." + typeName);
            return t?.GetField(
                fieldName,
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
        }
        catch { return null; }
    }

    /// <summary>
    /// Read a `CardModel` bool that walks the keyword set, defaulting to false
    /// if the model throws. `Keywords` is computed (intrinsic + `LocalKeywords`)
    /// and a card mid-`RemoveFromState` can fault on it; a missing flag costs
    /// the reconstruction one turn of Sly/Retain, an exception costs the whole
    /// decision record. GH #370.
    /// </summary>
    private static bool SafeCardBool(Func<bool> read)
    {
        try { return read(); }
        catch { return false; }
    }

    private static int? SafeRelicCounter(RelicModel r)
    {
        try
        {
            if (r.ShowCounter) return r.DisplayAmount;
            // A latch relic declares no counter, but its spent/unspent bit is
            // exactly the state a reconstruction must not lose.
            if (RelicLatchFields.TryGetValue(r.Id.Entry, out var f) && f != null)
            {
                return (f.GetValue(r) is bool b && b)
                    ? RelicLatchSpent.GetValueOrDefault(r.Id.Entry, 1)
                    : 0;
            }
            return null;
        }
        catch { return null; }
    }


    // (ours: the stack-trace frames of spire-mentor's version are left out)
    private static void LogException(string context, Exception ex)
    {
        var inner = ex;
        while (inner.InnerException != null) inner = inner.InnerException;
        Console.Error.WriteLine($"[SIM] {context}: {inner.GetType().Name}: {inner.Message}");
    }
}
