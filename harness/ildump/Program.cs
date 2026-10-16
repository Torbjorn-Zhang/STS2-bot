using System; using System.Linq; using Mono.Cecil; using Mono.Cecil.Cil;
// usage: ildump <sts2.dll> <TypeName> <MethodName>   (also dumps compiler-generated state machines)
var module = ModuleDefinition.ReadModule(args[0]);
if (args[1] == "GENERIC_STATICS") {  // static fields declared in generic types
  foreach (var t in module.GetTypes().Where(t => t.HasGenericParameters || (t.DeclaringType?.HasGenericParameters ?? false)))
    foreach (var f in t.Fields.Where(f => f.IsStatic && !f.IsLiteral))
      Console.WriteLine($"{t.FullName}::{f.Name} : {f.FieldType.FullName}");
  return;
}
if (args[1] == "BODY_MODE") {   // every instruction of Type::Method (exact names, nested types with /)
    foreach (var t in module.GetTypes().Where(t => t.FullName == args[2] || t.Name == args[2]))
        foreach (var m in t.Methods.Where(m => m.Name == args[3] && m.HasBody))
        {
            Console.WriteLine($"== {t.FullName}::{m.Name}");
            foreach (var ins in m.Body.Instructions) Console.WriteLine($"   IL_{ins.Offset:x4} {ins.OpCode} {ins.Operand}");
        }
    return;
}
if (args[1] == "SIGS_MODE") {   // method signatures with parameter names: SIGS_MODE Type Method
    foreach (var t in module.GetTypes().Where(t => t.Name == args[2]))
        foreach (var m in t.Methods.Where(m => args[3] == "*" || m.Name == args[3]))
            Console.WriteLine($"{t.FullName}::{m.Name}(" + string.Join(", ", m.Parameters.Select(p => $"{p.ParameterType.Name} {p.Name}{(p.HasDefault ? " = default" : "")}")) + ")");
    return;
}
if (args[1] == "FIELDS_OF_TYPE") {   // list fields and call sites whose type mentions args[2]
  foreach (var t in module.GetTypes()) {
    foreach (var f in t.Fields) if (f.FieldType.FullName.Contains(args[2])) Console.WriteLine($"FIELD {(f.IsStatic ? "static " : "")}{t.FullName}::{f.Name} : {f.FieldType.FullName}");
    foreach (var m in t.Methods.Where(m => m.HasBody))
      foreach (var ins in m.Body.Instructions)
        if (ins.Operand is MemberReference mr && mr.FullName.Contains(args[2])) { Console.WriteLine($"USE {t.FullName}::{m.Name} -> {mr.FullName}"); break; }
  }
  return;
}
var types = module.GetTypes().Where(t => t.Name == args[1] || (t.DeclaringType?.Name == args[1] && t.Name.Contains("<" + args[2] + ">"))).ToList();
foreach (var t in types)
  foreach (var m in t.Methods.Where(m => m.HasBody && (m.Name == args[2] || (t.Name.Contains("<" + args[2] + ">") && m.Name == "MoveNext"))))
  {
    Console.WriteLine($"== {t.FullName}::{m.Name}");
    foreach (var ins in m.Body.Instructions)
      if (ins.OpCode.FlowControl == FlowControl.Call || ins.OpCode.Code is Code.Ldfld or Code.Ldsfld or Code.Callvirt or Code.Newobj or Code.Ldstr || Environment.GetEnvironmentVariable("ILDUMP_ALL") == "1")
        Console.WriteLine($"   IL_{ins.Offset:x4} {ins.OpCode} {ins.Operand}");
  }
