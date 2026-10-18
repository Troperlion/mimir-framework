// Minimal C# example for --csharp runtime.
System.Console.WriteLine("[csharp] mimir script started");
System.Console.WriteLine($"[csharp] argv count: {arg.Count}");
System.Console.WriteLine($"[csharp] conf path: {CONF_PATH}");
System.Console.WriteLine($"[csharp] conf dir: {CONF_DIR}");
System.Console.WriteLine($"[csharp] has Mimir: {Mimir != null}");
System.Console.WriteLine($"[csharp] has model alias: {model != null}");
System.Console.WriteLine("[csharp] done");
