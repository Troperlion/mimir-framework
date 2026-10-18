#pragma once

#include <filesystem>
#include <string>

#include "scriptings/ScriptingContext.hpp"

namespace ScriptingBridgeCommon {

// Variables d'environnement injectées dans chaque runtime.
inline constexpr const char* kEnvBridgeCmdFile         = "MIMIR_BRIDGE_CMD_FILE";
inline constexpr const char* kEnvBridgeArchInfoJson    = "MIMIR_BRIDGE_ARCH_INFO_JSON";
inline constexpr const char* kEnvBridgeDtypesJson      = "MIMIR_BRIDGE_DTYPES_JSON";
inline constexpr const char* kEnvBridgeArchAvailJson   = "MIMIR_BRIDGE_ARCH_AVAILABLE_JSON";
inline constexpr const char* kEnvBridgeArchCacheJson   = "MIMIR_BRIDGE_ARCH_CACHE_JSON";
inline constexpr const char* kEnvBridgeDtypesCount     = "MIMIR_BRIDGE_DTYPES_COUNT";
inline constexpr const char* kEnvBridgeArchAvailCount  = "MIMIR_BRIDGE_ARCH_AVAIL_COUNT";
inline constexpr const char* kEnvRootWork              = "ROOTWORK";

// Racine stable utilisee par les scripts pour resoudre leurs dependances.
std::string ensureRootWorkEnv();

// Chemin du fichier cache inter-exécutions (dans $TMPDIR).
std::string archCacheFilePath();

// Données pré-injectées en lecture seule.
std::string buildArchitecturesInfoJson();
std::string buildAvailableJson();
std::string buildDtypesJson();
std::string loadArchCacheJson();

// Traitement des commandes bridge après exécution du script.
bool processBridgeCommands(ScriptingContext& ctx,
                           const std::filesystem::path& cmdFile,
                           const std::string& logPrefix);

}  // namespace ScriptingBridgeCommon
