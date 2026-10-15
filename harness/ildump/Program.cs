using System; using System.Linq; using Mono.Cecil; using Mono.Cecil.Cil;
// usage: ildump <sts2.dll> <TypeName> <MethodName>   (also dumps compiler-generated state machines)
var module = ModuleDefinition.ReadModule(args[0]);
var types = module.GetTypes().Where(t => t.Name == args[1] || (t.DeclaringType?.Name == args[1] && t.Name.Contains("<" + args[2] + ">"))).ToList();
foreach (var t in types)
  foreach (var m in t.Methods.Where(m => m.HasBody && (m.Name == args[2] || (t.Name.Contains("<" + args[2] + ">") && m.Name == "MoveNext"))))
  {
    Console.WriteLine($"== {t.FullName}::{m.Name}");
    foreach (var ins in m.Body.Instructions)
      if (ins.OpCode.FlowControl == FlowControl.Call || ins.OpCode.Code is Code.Ldfld or Code.Ldsfld or Code.Callvirt or Code.Newobj)
        Console.WriteLine($"   IL_{ins.Offset:x4} {ins.OpCode} {ins.Operand}");
  }
