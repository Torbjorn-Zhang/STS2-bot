using System; using System.Linq; using System.Reflection; using System.Runtime.Loader;
var dirs = args[0].Split(':'); var dir = dirs[0];
var ctx = new AssemblyLoadContext("r", true);
ctx.Resolving += (c, n) => { foreach (var d in dirs) { var p = System.IO.Path.Combine(d, n.Name + ".dll"); if (System.IO.File.Exists(p)) return c.LoadFromAssemblyPath(p); } return null; };
var asm = ctx.LoadFromAssemblyPath(System.IO.Path.Combine(dir, "sts2.dll"));
Type[] types; try { types = asm.GetTypes(); } catch (ReflectionTypeLoadException e) { types = e.Types.Where(t => t != null).ToArray(); }
if (args[1] == "TYPES_MODE") { foreach (var t in types) foreach (var pat in args.Skip(2)) if (t.FullName != null && t.FullName.Contains(pat, StringComparison.OrdinalIgnoreCase)) { Console.WriteLine(t.FullName); break; } return; }
if (args[1] == "GREP_MODE") { foreach (var t in types) { MethodInfo[] ms; try { ms = t.GetMethods(BindingFlags.Public|BindingFlags.NonPublic|BindingFlags.Instance|BindingFlags.Static|BindingFlags.DeclaredOnly); } catch { continue; } foreach (var m in ms) foreach (var pat in args.Skip(2)) if (m.Name.Contains(pat, StringComparison.OrdinalIgnoreCase)) Console.WriteLine(t.FullName + "::" + m.Name); } return; }
foreach (var name in args.Skip(1)) foreach (var t in types.Where(t => t.Name == name)) {
  Console.WriteLine("== " + t.FullName);
  foreach (var m in t.GetMembers(BindingFlags.Public|BindingFlags.NonPublic|BindingFlags.Instance|BindingFlags.Static|BindingFlags.DeclaredOnly))
    if (m.MemberType is MemberTypes.Property or MemberTypes.Field or MemberTypes.Method or MemberTypes.Constructor && !m.Name.StartsWith("<")) { string s; try { s = m.ToString(); } catch (Exception e) { s = m.Name + " (sig unavailable: " + e.GetType().Name + ")"; } Console.WriteLine("  " + m.MemberType + " " + s); }
}
