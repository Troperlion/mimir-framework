-- Explicit help for MPK tools; evaluated before Args and runtime side effects.
local Help = dofile(ROOTWORK.."/scripts/modules/help_cli.lua")
local specs = {
  build_mpk = {
    description = "Créer un MPK depuis un checkpoint, le registre ou un template. Architecture et configuration uniquement, sans poids.",
    options = {
      "--name <string>               Package name (required for templates; exported_arch for source exports)",
      "--type <architecture>         Architecture type (required for templates; inferred for checkpoints)",
      "--author <string>             Header author (default: unknown)",
      "--created-at <iso8601>        Header created_at (default: current UTC)",
      "--modifiable / --no-modifiable Header modifiable flag (default: true)",
      "--viz / --no-viz              Header viz_specified flag (default: false)",
      "--config-json <path>          Base config JSON source",
      "--checkpoint <path>          Export checkpoint architecture (rawfolder, safetensor, debugjson)",
      "--format <format>            auto|rawfolder|safetensor|debugjson",
      "--register <architecture>    Export full registry graph (alias: --arch)",
      "--from-registry               Export full registry graph selected by --type",
      "--structure-json <path>       Model structure JSON source",
      "--template <name>             model_structure template: vae_conv|unet|auto",
      "--description <text>          Human description text",
      "--description-file <path>     Description text file",
      "--out <path.mpk>              Output MPK path (required, .mpk)",
      "--compile [path.mpk.bin]      Also emit opaque optimized binary-v4",
      "--arch <name>                Alias de --register <name>",
    },
    examples = {
      "./bin/mimir --lua scripts/tools/build_mpk.lua -- --checkpoint checkpoints/run --out exports/run.mpk --compile",
      "./bin/mimir --lua scripts/tools/build_mpk.lua -- --checkpoint model.safetensors --format safetensor --out exports/model.mpk",
      "./bin/mimir --lua scripts/tools/build_mpk.lua -- --checkpoint debug.json --format debugjson --out exports/debug.mpk",
      "./bin/mimir --lua scripts/tools/build_mpk.lua -- --register vae_conv --out exports/vae.mpk",
      "./bin/mimir --lua scripts/tools/build_mpk.lua -- --name custom --type vae_conv --template auto --out exports/custom.mpk",
    },
  },
  export_arch_mpk = {
    description = "Exporter le graphe sérialisé d’un checkpoint ou le graphe réel du registre/modèle courant, sans poids.",
    options = {
      "--out <file.mpk>             sortie MPK (required)",
      "--name <string>              nom package (default: exported_arch)",
      "--author <string>            auteur (default: unknown)",
      "--description <text>         description",
      "--checkpoint <path>          checkpoint rawfolder, safetensor ou debugjson (architecture sans poids)",
      "--format <format>            auto|rawfolder|safetensor|debugjson",
      "--arch <registry_name>       architecture du registre à instancier avant export",
      "--config-json <path>         overrides config pour --arch",
      "--from-current-model         exporter le modèle actuellement chargé (standalone ou registre)",
      "--type <string>              type à écrire dans le header (utile hors registre)",
      "--compile [path.mpk.bin]     produit aussi le binaire v4 opaque",
      "--viz                        header.viz_specified=true",
      "--register <name>            Alias de --arch <name>",
      "--created-at <iso8601>       Date du package (défaut : UTC actuelle)",
      "--modifiable / --no-modifiable  Modification autorisée (défaut : true)",
      "--description-file <path>    Description depuis un fichier (prioritaire sur --description)",
    },
    examples = {
      "./bin/mimir --lua scripts/tools/export_arch_mpk.lua -- --checkpoint checkpoints/run --out exports/run.mpk",
      "./bin/mimir --lua scripts/tools/export_arch_mpk.lua -- --arch transformer --out exports/transformer.mpk",
    },
  },
  compile_mpk = {
    description = "Compiler une source MPK pseudocode moderne en binaire v4. JSON historique et binaires sont refusés en entrée.",
    options = {
      "--in <source.mpk>            Source pseudocode moderne (obligatoire)",
      "--out <compiled.mpk.bin>     Binaire v4 de sortie (obligatoire)",
    },
    examples = {
      "./bin/mimir --lua scripts/tools/compile_mpk.lua -- --in exports/model.mpk --out exports/model.mpk.bin",
    },
  },
  load_mpk = {
    description = "Lire, vérifier et éventuellement instancier un MPK. --verify-only quitte sans créer de modèle.",
    options = {
      "--in <path.mpk>      Fichier .mpk ou .mpk.bin (obligatoire)",
      "--create             Créer le modèle (défaut : true)",
      "--no-create          Only inspect/decode MPK",
      "--show-config        Print decoded base config JSON",
      "--apply-graph        Apply model_structure.graph nodes dynamically (push_layer + set_layer_io)",
      "--replace-layers     Clear existing model layers before graph apply (default: true)",
      "--no-replace-layers  Append graph layers without clearing",
      "--allocate           Allocate graph parameters after apply (default: true)",
      "--no-allocate        Keep the reconstructed graph unallocated",
      "--init <method>      Initialize new graph weights (default: xavier; none disables)",
      "--seed <integer>     Initialization seed (default: 0)",
      "--allow-non-registry If registry creation fails, fallback to create_empty + graph import (default: true)",
      "--no-allow-non-registry Disable fallback mode",
      "--verify-only        Validate checksum/header and exit",
    },
    examples = {
      "./bin/mimir --lua scripts/tools/load_mpk.lua -- --in exports/model.mpk.bin --verify-only",
      "./bin/mimir --lua scripts/tools/load_mpk.lua -- --in exports/model.mpk --no-create --show-config",
    },
  },
  mpk_node_wizard = {
    description = "Créer un graphe MPK avec un assistant interactif. --list-layer-types affiche les types puis quitte.",
    options = {
      "--out <file.mpk>      chemin de sortie (sinon prompt interactif)",
      "--compile [file]       produit aussi un .mpk.bin v4 opaque",
      "--list-layer-types     affiche les types acceptés puis quitte",
    },
    examples = {
      "./bin/mimir --lua scripts/tools/mpk_node_wizard.lua -- --out exports/custom.mpk --compile",
      "./bin/mimir --lua scripts/tools/mpk_node_wizard.lua -- --list-layer-types",
    },
  },
  add_vision_mpk_architectures = {
    description = "Générer les prototypes de vision dans _archi/ (remplace les fichiers correspondants). Aucune option de configuration.",
    options = {
    },
    examples = {
      "./bin/mimir --lua scripts/tools/add_vision_mpk_architectures.lua",
    },
  },
}
local M = {}
function M.show(name)
  local spec = assert(specs[name], "unknown MPK tool: "..tostring(name))
  spec.script_path = "scripts/tools/"..name..".lua"
  Help.auto_exit_help(spec)
end
return M
