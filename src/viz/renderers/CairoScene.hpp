#pragma once
#include "../Scene.hpp"
#include <cairo/cairo.h>
#include <cstring>
#include <cmath>

namespace mimir::viz::cairo_scene {
inline void color(cairo_t* cr,const Scene& v) {cairo_set_source_rgba(cr,v[0].get<double>()/255.,v[1].get<double>()/255.,v[2].get<double>()/255.,v[3].get<double>()/255.);}
inline void render(cairo_t* cr,const Scene& scene,unsigned width,unsigned height) {
    cairo_save(cr);cairo_set_operator(cr,CAIRO_OPERATOR_SOURCE);
    color(cr,scene.at("background"));cairo_paint(cr);cairo_restore(cr);
    std::vector<cairo_surface_t*> images;
    for(const auto& encoded:scene.at("images")) {
        const auto bytes=unbase64(encoded.get<std::string>());
        struct Reader {const std::vector<uint8_t>& bytes;size_t offset=0;} reader{bytes};
        auto* surface=cairo_image_surface_create_from_png_stream(+[](void* closure,unsigned char* data,unsigned length){
            auto& source=*static_cast<Reader*>(closure);
            if(length>source.bytes.size()-source.offset)return CAIRO_STATUS_READ_ERROR;
            std::memcpy(data,source.bytes.data()+source.offset,length);source.offset+=length;return CAIRO_STATUS_SUCCESS;
        },&reader);
        images.push_back(surface);
    }
    for(const auto& cmd:scene.at("commands")) {
        cairo_save(cr);const auto& v=cmd.at("view");
        const double x=v[4].get<double>()*width,y=v[5].get<double>()*height,w=v[6].get<double>()*width,h=v[7].get<double>()*height;
        cairo_rectangle(cr,x,y,w,h);cairo_clip(cr);cairo_translate(cr,x,y);cairo_scale(cr,w/std::max(1.,v[2].get<double>()),h/std::max(1.,v[3].get<double>()));cairo_translate(cr,-v[0].get<double>(),-v[1].get<double>());
        const auto kind=cmd.at("kind").get<std::string>();
        if(kind=="rect") {
            const auto& r=cmd.at("rect");const double x=r[0],y=r[1],w=r[2],h=r[3];
            cairo_rectangle(cr,x,y,w,h);color(cr,cmd.at("fill"));cairo_fill(cr);
            const double t=std::abs(cmd.at("thickness").get<double>());
            if(t){cairo_set_line_width(cr,t);cairo_rectangle(cr,x+t/2,y+t/2,std::max(0.,w-t),std::max(0.,h-t));color(cr,cmd.at("outline"));cairo_stroke(cr);}
        } else if(kind=="text") {
            cairo_select_font_face(cr,"Open Sans",CAIRO_FONT_SLANT_NORMAL,cmd.at("bold").get<bool>()?CAIRO_FONT_WEIGHT_BOLD:CAIRO_FONT_WEIGHT_NORMAL);
            const double size=cmd.at("size");cairo_set_font_size(cr,size);color(cr,cmd.at("fill"));const auto& p=cmd.at("position");
            cairo_move_to(cr,p[0].get<double>(),p[1].get<double>()+size*.82);cairo_show_text(cr,cmd.at("text").get_ref<const std::string&>().c_str());
        } else if(kind=="line") {
            cairo_set_line_width(cr,1);const auto& vertices=cmd.at("vertices");
            for(size_t i=1;i<vertices.size();++i){const auto& a=vertices[i-1];const auto& b=vertices[i];
                cairo_move_to(cr,a[0].get<double>(),a[1].get<double>());cairo_line_to(cr,b[0].get<double>(),b[1].get<double>());
                cairo_set_source_rgba(cr,b[2].get<double>()/255.,b[3].get<double>()/255.,b[4].get<double>()/255.,b[5].get<double>()/255.);cairo_stroke(cr);}
        } else if(kind=="image") {
            const auto asset=cmd.at("asset").get<size_t>();if(asset<images.size() && cairo_surface_status(images[asset])==CAIRO_STATUS_SUCCESS) {
                const auto& p=cmd.at("position");const auto& s=cmd.at("scale");cairo_translate(cr,p[0].get<double>(),p[1].get<double>());cairo_scale(cr,s[0].get<double>(),s[1].get<double>());
                cairo_set_source_surface(cr,images[asset],0,0);cairo_pattern_set_filter(cairo_get_source(cr),cmd.at("smooth").get<bool>()?CAIRO_FILTER_BILINEAR:CAIRO_FILTER_NEAREST);cairo_paint_with_alpha(cr,cmd.at("alpha").get<double>());
            }
        }
        cairo_restore(cr);
    }
    for(auto* image:images)cairo_surface_destroy(image);
}
}
