#pragma once

// Only the legacy backend sees SFML headers. The scene uses this neutral name;
// its layout and control code is shared by all backends.
#ifdef MIMIR_VIZ_BACKEND_SFML
#include <SFML/Graphics.hpp>
namespace vizgfx = sf;
#else
#include "SoftwareGraphics.hpp"
namespace vizgfx = mimir::viz::graphics;
#endif
