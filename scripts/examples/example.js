// Minimal JavaScript example for --js runtime.
console.log("[js] mimir script started");
console.log("[js] argv:", Array.isArray(arg) ? arg.join(" | ") : "<missing>");
console.log("[js] conf path:", CONF_PATH || "<none>");
console.log("[js] conf dir:", CONF_DIR || "<none>");
console.log("[js] globals:", {
  hasMimir: typeof Mimir !== "undefined",
  hasModelAlias: typeof model !== "undefined",
  hasArchitecturesAlias: typeof architectures !== "undefined"
});
console.log("[js] done");
