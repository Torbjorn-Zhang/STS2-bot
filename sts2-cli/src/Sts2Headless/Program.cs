using System.Reflection;
using System.Runtime.Loader;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace Sts2Headless;

class Program
{
    private static readonly JsonSerializerOptions JsonOpts = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
        WriteIndented = false,
    };

    /// <summary>
    /// Locate the directory containing sts2.dll: STS2_LIB env, walk up from BaseDirectory, then BaseDirectory/lib.
    /// </summary>
    private static string ResolveLibDirectory()
    {
        var envLib = Environment.GetEnvironmentVariable("STS2_LIB");
        if (!string.IsNullOrWhiteSpace(envLib))
        {
            var p = Path.GetFullPath(envLib.Trim());
            if (Directory.Exists(p) && File.Exists(Path.Combine(p, "sts2.dll")))
                return p;
        }

        var dir = AppContext.BaseDirectory;
        for (var depth = 0; depth < 16 && !string.IsNullOrEmpty(dir); depth++)
        {
            var candidate = Path.Combine(dir, "lib");
            if (Directory.Exists(candidate) && File.Exists(Path.Combine(candidate, "sts2.dll")))
                return Path.GetFullPath(candidate);
            dir = Directory.GetParent(dir)?.FullName ?? "";
        }

        return Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "lib"));
    }

    static void Main(string[] args)
    {
        // Prevent unhandled exceptions from crashing the process
        AppDomain.CurrentDomain.UnhandledException += (_, e) =>
        {
            Console.Error.WriteLine($"[FATAL] Unhandled: {e.ExceptionObject}");
        };
        TaskScheduler.UnobservedTaskException += (_, e) =>
        {
            Console.Error.WriteLine($"[WARN] Unobserved task exception: {e.Exception}");
            e.SetObserved();
        };

        var libDir = ResolveLibDirectory();

        AssemblyLoadContext.Default.Resolving += (ctx, name) =>
        {
            var path = Path.Combine(libDir, name.Name + ".dll");
            if (File.Exists(path))
                return ctx.LoadFromAssemblyPath(Path.GetFullPath(path));

            // Also check game directory (via STS2_GAME_DIR env var)
            var gameDir = Environment.GetEnvironmentVariable("STS2_GAME_DIR") ?? "";
            if (!string.IsNullOrEmpty(gameDir))
            {
                path = Path.Combine(gameDir, name.Name + ".dll");
                if (File.Exists(path))
                    return ctx.LoadFromAssemblyPath(path);
            }

            return null;
        };

        var sim = new RunSimulator();
        sim.CommandHandler = c => HandleCommand(sim, c);
        WriteLine(new Dictionary<string, object?> { ["type"] = "ready", ["version"] = "0.2.0" });

        string? line;
        while ((line = Console.ReadLine()) != null)
        {
            line = line.Trim();
            if (string.IsNullOrEmpty(line)) continue;

            Dictionary<string, object?>? result;
            try
            {
                var cmd = JsonSerializer.Deserialize<JsonElement>(line);
                // Commands that change game state are logged for snapshot/restore replay.
                if (cmd.TryGetProperty("cmd", out var c) && c.GetString() is "action" or "set_player" or "enter_room" or "set_draw_order" or "determinize" or "determinize_run")
                    sim.RecordCommand(line);
                var hsw = System.Diagnostics.Stopwatch.StartNew();
                result = HandleCommand(sim, cmd);
                if (RunSimulator.ProfOn)
                {
                    var nm = cmd.TryGetProperty("action", out var an) ? an.GetString() : c.ValueKind == JsonValueKind.String ? c.GetString() : "?";
                    sim.ProfAdd("live." + nm, hsw);
                }
            }
            catch (JsonException ex)
            {
                result = new Dictionary<string, object?> { ["type"] = "error", ["message"] = $"Invalid JSON: {ex.Message}" };
            }
            catch (Exception ex)
            {
                result = new Dictionary<string, object?> { ["type"] = "error", ["message"] = $"{ex.GetType().Name}: {ex.Message}", ["stack_trace"] = ex.ToString() };
            }

            if (result != null)
            {
                var wsw = System.Diagnostics.Stopwatch.StartNew();
                WriteLine(result);
                if (RunSimulator.ProfOn) sim.ProfAdd("write", wsw);
                if (result.TryGetValue("type", out var resultTypeObj) &&
                    string.Equals(resultTypeObj as string, "quit_result", StringComparison.Ordinal))
                {
                    break;
                }
            }
        }
    }

    static Dictionary<string, object?>? HandleCommand(RunSimulator sim, JsonElement cmd)
    {
        var cmdType = cmd.GetProperty("cmd").GetString() ?? "";
        switch (cmdType)
        {
            case "start_run":
            {
                var started = sim.StartRun(
                    cmd.TryGetProperty("character", out var ch) ? ch.GetString() ?? "Ironclad" : "Ironclad",
                    cmd.TryGetProperty("ascension", out var asc) ? asc.GetInt32() : 0,
                    cmd.TryGetProperty("seed", out var s) ? s.GetString() : null,
                    cmd.TryGetProperty("lang", out var lang) ? lang.GetString() ?? "en" : "en",
                    cmd.TryGetProperty("unlock_from_save", out var ufs) ? ufs.GetString() : null
                );
                if (!(started.TryGetValue("type", out var st) && st as string == "error"))
                    sim.OnStartRunCommand(cmd.GetRawText());
                return started;
            }

            case "action":
            {
                var action = cmd.GetProperty("action").GetString() ?? "";
                Dictionary<string, object?>? actionArgs = null;
                if (cmd.TryGetProperty("args", out var argsElem))
                {
                    actionArgs = new Dictionary<string, object?>();
                    foreach (var prop in argsElem.EnumerateObject())
                    {
                        actionArgs[prop.Name] = prop.Value.ValueKind switch
                        {
                            JsonValueKind.Number => prop.Value.GetInt32(),
                            JsonValueKind.String => prop.Value.GetString(),
                            JsonValueKind.True => true,
                            JsonValueKind.False => false,
                            _ => prop.Value.ToString(),
                        };
                    }
                }
                return sim.ExecuteAction(action, actionArgs);
            }

            case "load_save":
            {
                var savePath = cmd.TryGetProperty("path", out var sp) ? sp.GetString() : null;
                var saveJson = cmd.TryGetProperty("json", out var sj) ? sj.GetString() : null;
                if (saveJson == null && savePath != null)
                {
                    if (!File.Exists(savePath))
                        return new Dictionary<string, object?> { ["type"] = "error", ["message"] = $"Save file not found: {savePath}" };
                    saveJson = File.ReadAllText(savePath);
                }
                if (saveJson == null)
                    return new Dictionary<string, object?> { ["type"] = "error", ["message"] = "Provide 'path' or 'json' for load_save" };
                var loadLang = cmd.TryGetProperty("lang", out var le) ? (le.GetString() ?? "en") : "en";
                var loaded = sim.LoadSave(saveJson, loadLang);
                if (!(loaded.TryGetValue("type", out var lt) && lt as string == "error"))
                    sim.OnLoadSaveCommand(saveJson);
                return loaded;
            }
            case "get_map":
                return sim.GetFullMap();

            case "set_player":
            {
                var args = new Dictionary<string, JsonElement>();
                foreach (var prop in cmd.EnumerateObject())
                    if (prop.Name != "cmd") args[prop.Name] = prop.Value;
                return sim.SetPlayer(args);
            }

            case "enter_room":
            {
                var roomType = cmd.TryGetProperty("type", out var rt) ? rt.GetString() ?? "" : "";
                var encounter = cmd.TryGetProperty("encounter", out var enc) ? enc.GetString() : null;
                var eventId = cmd.TryGetProperty("event", out var ev) ? ev.GetString() : null;
                return sim.EnterRoom(roomType, encounter, eventId);
            }

            case "set_draw_order":
            {
                var cards = new List<string>();
                if (cmd.TryGetProperty("cards", out var cardsArr))
                    foreach (var c in cardsArr.EnumerateArray())
                        cards.Add(c.GetString() ?? "");
                return sim.SetDrawOrder(cards);
            }

            case "executor_state":   // debug: is the game's action executor paused / busy?
            {
                var ex = MegaCrit.Sts2.Core.Runs.RunManager.Instance.ActionExecutor;
                return new Dictionary<string, object?> { ["type"] = "executor", ["paused"] = ex.IsPaused,
                    ["running"] = ex.IsRunning, ["current"] = ex.CurrentlyRunningAction?.GetType().Name,
                    ["room"] = sim.CurrentRoomName(), ["bg_task"] = sim.BackgroundTaskStatus() };
            }

            case "determinize":
                return sim.Determinize(cmd.TryGetProperty("seed", out var ds) ? ds.GetUInt64() : 0UL);

            case "determinize_run":
                return sim.DeterminizeRun(cmd.TryGetProperty("seed", out var drs) ? drs.GetUInt64() : 0UL);

            case "write_continue_save":
            {
                var outputPath = cmd.TryGetProperty("path", out var op) ? op.GetString() : null;
                return sim.SaveCheckpoint(outputPath);
            }

            case "snapshot":
                return sim.Snapshot();

            case "restore":
                // "quiet": true builds no state for the reply (the caller already has it): an ack only
                return cmd.TryGetProperty("id", out var sid)
                    ? sim.Restore(sid.GetInt32(), cmd.TryGetProperty("quiet", out var rq) && rq.ValueKind == JsonValueKind.True)
                    : new Dictionary<string, object?> { ["type"] = "error", ["message"] = "restore requires 'id'" };

            case "drop_snapshots":
                return sim.DropSnapshots(cmd.TryGetProperty("from", out var df) ? df.GetInt32() : null);

            case "restore_prof":
                return sim.RestoreProf(cmd.TryGetProperty("reset", out var rp) && rp.GetBoolean());

            case "gc_stats":
            {
                // Debug: managed heap before / after a full blocking collection, process RSS.
                long before = GC.GetTotalMemory(false);
                GC.Collect(2, GCCollectionMode.Forced, blocking: true, compacting: true);
                GC.WaitForPendingFinalizers();
                GC.Collect(2, GCCollectionMode.Forced, blocking: true, compacting: true);
                long after = GC.GetTotalMemory(true);
                return new Dictionary<string, object?>
                {
                    ["type"] = "gc_stats",
                    ["heap_mb_before"] = before / 1048576,
                    ["heap_mb_after_full_gc"] = after / 1048576,
                    ["working_set_mb"] = Environment.WorkingSet / 1048576,
                    ["gen2_collections"] = GC.CollectionCount(2),
                    ["snapshot_store"] = sim.SnapshotStoreStats(),
                };
            }

            case "heap_census":
            {
                // Debug: breadth-first walk of everything reachable from static fields of all
                // loaded assemblies. Counts instances per type and shows how the first few
                // instances of `focus` (default RunState) are reached.
                var focus = cmd.TryGetProperty("focus", out var fe) ? fe.GetString() : "MegaCrit.Sts2.Core.Runs.RunState";
                var maxObjs = cmd.TryGetProperty("max", out var me) ? me.GetInt32() : 4_000_000;
                var sflags = BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.DeclaredOnly;
                var iflags = BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.DeclaredOnly;
                var parent = new Dictionary<object, (object? from, string via)>(ReferenceEqualityComparer.Instance);
                var counts = new Dictionary<string, int>();
                var bytes = new Dictionary<string, long>();
                var bigStrings = new List<(int len, string path)>();
                var seenStrings = new HashSet<object>(ReferenceEqualityComparer.Instance);
                var queue = new Queue<object>();
                var fieldCache = new Dictionary<Type, FieldInfo[]>();
                var focusObjs = new List<object>();
                static bool Leaf(Type t) => t.IsPrimitive || t.IsEnum || t.IsPointer || t == typeof(string) || t == typeof(decimal)
                    || typeof(MemberInfo).IsAssignableFrom(t) || t.Namespace?.StartsWith("System.Reflection") == true;
                FieldInfo[] Fields(Type t)
                {
                    if (fieldCache.TryGetValue(t, out var fs)) return fs;
                    var list = new List<FieldInfo>();
                    for (var b = t; b != null; b = b.BaseType)
                        list.AddRange(b.GetFields(iflags).Where(f => !Leaf(f.FieldType) || f.FieldType == typeof(object)));
                    return fieldCache[t] = list.ToArray();
                }
                void Push(object? v, object? from, string via)
                {
                    if (v is string str)
                    {
                        if (seenStrings.Add(str))
                        {
                            bytes["System.String"] = bytes.GetValueOrDefault("System.String") + 2L * str.Length + 20;
                            if (str.Length > 20000 && bigStrings.Count < 20) bigStrings.Add((str.Length, $"{via} (in {from?.GetType().Name})"));
                        }
                        return;
                    }
                    if (v == null || Leaf(v.GetType()) || parent.ContainsKey(v)) return;
                    parent[v] = (from, via);
                    queue.Enqueue(v);
                }
                foreach (var asm in AppDomain.CurrentDomain.GetAssemblies())
                {
                    var an = asm.GetName().Name ?? "";
                    if (an.StartsWith("System.Net") || an.StartsWith("System.Security") || an.StartsWith("Microsoft.Win32")) continue;
                    Type[] ts;
                    try { ts = asm.GetTypes(); } catch (ReflectionTypeLoadException e) { ts = e.Types.Where(t => t != null).ToArray()!; } catch { continue; }
                    foreach (var t in ts)
                    {
                        if (t.ContainsGenericParameters) continue;
                        FieldInfo[] sf;
                        try { sf = t.GetFields(sflags); } catch { continue; }
                        foreach (var f in sf)
                        {
                            if (f.IsLiteral || Leaf(f.FieldType) && f.FieldType != typeof(object)) continue;
                            object? v;
                            try { v = f.GetValue(null); } catch { continue; }
                            Push(v, null, $"{t.FullName}.{f.Name}");
                        }
                    }
                }
                Push(sim, null, "<the RunSimulator instance>");
                Push(Thread.CurrentThread, null, "<main thread>");
                Push(ExecutionContext.Capture(), null, "<main ExecutionContext>");
                Push(SynchronizationContext.Current, null, "<main SynchronizationContext>");
                while (queue.Count > 0 && parent.Count < maxObjs)
                {
                    var o = queue.Dequeue();
                    var t = o.GetType();
                    var tn = t.FullName ?? t.Name;
                    counts[tn] = counts.GetValueOrDefault(tn) + 1;
                    if (tn == focus && focusObjs.Count < 6) focusObjs.Add(o);
                    if (o is Array arr0)
                    {
                        var et0 = t.GetElementType();
                        long per = et0 == null ? 8 : et0.IsValueType ? Math.Max(1, System.Runtime.InteropServices.Marshal.SizeOf(et0.IsEnum ? Enum.GetUnderlyingType(et0) : (et0.IsPrimitive ? et0 : typeof(IntPtr)))) : 8;
                        if (et0 != null && et0.IsValueType && !et0.IsPrimitive && !et0.IsEnum) per = 24;
                        bytes[tn] = bytes.GetValueOrDefault(tn) + per * arr0.LongLength + 24;
                    }
                    else bytes[tn] = bytes.GetValueOrDefault(tn) + 24 + 8L * Fields(t).Length;
                    if (o is Array arr)
                    {
                        var et = t.GetElementType();
                        if (et != null && Leaf(et) && et != typeof(object)) continue;
                        int i = 0;
                        foreach (var e in arr) { Push(e, o, $"[{i}]"); if (++i > 100000) break; }
                        continue;
                    }
                    foreach (var f in Fields(t))
                    {
                        object? v;
                        try { v = f.GetValue(o); } catch { continue; }
                        Push(v, o, f.Name);
                    }
                }
                string PathOf(object o)
                {
                    var parts = new List<string>();
                    for (var cur = o; cur != null && parts.Count < 40;)
                    {
                        var (from, via) = parent[cur];
                        parts.Add(from == null ? via : $"{via} ({cur.GetType().Name})");
                        cur = from;
                    }
                    parts.Reverse();
                    return string.Join(" -> ", parts);
                }
                return new Dictionary<string, object?>
                {
                    ["type"] = "heap_census",
                    ["objects"] = parent.Count,
                    ["truncated"] = queue.Count > 0,
                    ["focus_count"] = counts.GetValueOrDefault(focus ?? ""),
                    ["approx_mb"] = bytes.Values.Sum() / 1048576.0,
                    ["top_bytes"] = bytes.OrderByDescending(kv => kv.Value).Take(25).ToDictionary(kv => kv.Key, kv => (object?)(kv.Value / 1024)),
                    ["big_strings"] = bigStrings.Select(b => $"{b.len} chars: {b.path}").ToList(),
                    ["focus_paths"] = focusObjs.Select(PathOf).ToList(),
                    ["top"] = counts.OrderByDescending(kv => kv.Value).Take(cmd.TryGetProperty("top", out var tp) ? tp.GetInt32() : 40).ToDictionary(kv => kv.Key, kv => (object?)kv.Value),
                };
            }

            case "static_audit":
            {
                // Debug: sizes of collections / delegate lists reachable from static fields of
                // game types (and one level into singleton instances). Diff two audits to find leaks.
                var flags = BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.DeclaredOnly;
                var iflags = BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic;
                var sizes = new Dictionary<string, object?>();
                static int Size(object? v) => v switch
                {
                    null => 0,
                    Delegate d => d.GetInvocationList().Length,
                    System.Collections.ICollection c => c.Count,
                    _ => -1,
                };
                var asm = typeof(MegaCrit.Sts2.Core.Combat.CombatManager).Assembly;
                Type[] types;
                try { types = asm.GetTypes(); } catch (ReflectionTypeLoadException e) { types = e.Types.Where(t => t != null).ToArray()!; }
                foreach (var t in types)
                {
                    if (t.ContainsGenericParameters) continue;
                    foreach (var f in t.GetFields(flags))
                    {
                        object? v;
                        try { v = f.GetValue(null); } catch { continue; }
                        var n = Size(v);
                        if (n > 0) sizes[$"{t.FullName}.{f.Name}"] = n;
                        if (v == null || n >= 0 || v.GetType().Assembly != asm) continue;
                        foreach (var g in v.GetType().GetFields(iflags))
                        {
                            object? w;
                            try { w = g.GetValue(v); } catch { continue; }
                            var m = Size(w);
                            if (m > 0) sizes[$"{t.FullName}.{f.Name}->{g.Name}"] = m;
                        }
                    }
                }
                var handlers = new List<string>();
                try
                {
                    var ev = typeof(MegaCrit.Sts2.Core.Combat.CombatManager).GetField("CombatEnded", iflags)?.GetValue(MegaCrit.Sts2.Core.Combat.CombatManager.Instance) as Delegate;
                    if (ev != null)
                        foreach (var d in ev.GetInvocationList())
                            handlers.Add($"{d.Method.DeclaringType?.FullName}.{d.Method.Name} target={d.Target?.GetType().FullName}");
                }
                catch (Exception ex) { handlers.Add("error: " + ex.Message); }
                return new Dictionary<string, object?> { ["type"] = "static_audit", ["sizes"] = sizes, ["combat_ended_handlers"] = handlers };
            }

            case "checksum":
                return sim.Checksum(cmd.TryGetProperty("dump", out var dmp) && dmp.GetBoolean());

            case "quit":
            {
                var outputPath = cmd.TryGetProperty("path", out var op) ? op.GetString() : null;
                if (!string.IsNullOrEmpty(outputPath))
                {
                    var saveResult = sim.SaveCheckpoint(outputPath);
                    bool saveOk = saveResult.TryGetValue("success", out var sObj) && sObj is bool b && b;
                    if (!saveOk)
                    {
                        // Save failed — do NOT clean up so the caller can retry with a different path.
                        return new Dictionary<string, object?>
                        {
                            ["type"] = "save_error",
                            ["save"] = saveResult,
                        };
                    }
                    sim.CleanUp();
                    return new Dictionary<string, object?>
                    {
                        ["type"] = "quit_result",
                        ["success"] = true,
                        ["save"] = saveResult,
                    };
                }
                sim.CleanUp();
                return new Dictionary<string, object?>
                {
                    ["type"] = "quit_result",
                    ["success"] = true,
                    ["save"] = null,
                };
            }

            default:
                return new Dictionary<string, object?> { ["type"] = "error", ["message"] = $"Unknown command: {cmdType}" };
        }
    }

    static void WriteLine(Dictionary<string, object?> data)
    {
        Console.Out.WriteLine(JsonSerializer.Serialize(data, JsonOpts));
        Console.Out.Flush();
    }
}
