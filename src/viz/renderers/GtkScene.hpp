#pragma once
#include "CairoScene.hpp"
#include <gtk/gtk.h>

namespace mimir::viz::gtk {
// GTK3 exposes Cairo directly in the drawing area's native draw signal.
inline void render(cairo_t* context,const Scene& scene,unsigned width,unsigned height) {
    cairo_scene::render(context,scene,width,height);
}
}
