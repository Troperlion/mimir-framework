var avail = (System.Collections.IEnumerable)Architectures.available();
int ac = 0; foreach (var _ in avail) ac++;
System.Console.WriteLine($"[bridge-cs] available count: {ac}");

var dtypes = (System.Collections.IEnumerable)Architectures.dtypes();
int dc = 0; foreach (var _ in dtypes) dc++;
System.Console.WriteLine($"[bridge-cs] dtypes count: {dc}");

// Normalisation: aliases est un array de strings
var dtypesList = (System.Collections.Generic.List<System.Collections.Generic.Dictionary<string, object>>)Architectures.dtypes();
var f32 = dtypesList.Find(d => d.ContainsKey("name") && d["name"]?.ToString() == "float32");
if (f32 != null) {
    var aliases = f32["aliases"] as System.Text.Json.JsonElement?;
    var aliasCount = aliases?.GetArrayLength() ?? 0;
    System.Console.WriteLine($"[bridge-cs] f32 aliases is array: true | count: {aliasCount}");
}

Model.create("basic_mlp", "");
Model.allocate_params();
Model.init_weights("he", 42);
long p = (long)Model.total_params();
System.Console.WriteLine($"[bridge-cs] total_params (cached): {p}");
System.Console.WriteLine("[bridge-cs] ok");
