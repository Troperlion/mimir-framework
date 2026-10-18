#include "SoftwareGraphics.hpp"
#include "Scene.hpp"
#include "renderers/CairoScene.hpp"

#include <cairo/cairo.h>
#include <cmath>
#include <cstring>
#include <unordered_map>

namespace mimir::viz::graphics {
namespace {
struct SurfaceDeleter { void operator()(cairo_surface_t* p) const { if (p) cairo_surface_destroy(p); } };
struct ContextDeleter { void operator()(cairo_t* p) const { if (p) cairo_destroy(p); } };
using Surface = std::unique_ptr<cairo_surface_t, SurfaceDeleter>;
using Context = std::unique_ptr<cairo_t, ContextDeleter>;

void source(cairo_t* cr, Color c) {
    cairo_set_source_rgba(cr, c.r / 255., c.g / 255., c.b / 255., c.a / 255.);
}

std::string utf8(const String& string) {
    std::string result;
    for (char32_t cp : string.codepoints()) {
        if (cp <= 0x7f) result.push_back(char(cp));
        else if (cp <= 0x7ff) {
            result.push_back(char(0xc0 | (cp >> 6)));
            result.push_back(char(0x80 | (cp & 0x3f)));
        } else if (cp <= 0xffff) {
            result.push_back(char(0xe0 | (cp >> 12)));
            result.push_back(char(0x80 | ((cp >> 6) & 0x3f)));
            result.push_back(char(0x80 | (cp & 0x3f)));
        } else {
            result.push_back(char(0xf0 | (cp >> 18)));
            result.push_back(char(0x80 | ((cp >> 12) & 0x3f)));
            result.push_back(char(0x80 | ((cp >> 6) & 0x3f)));
            result.push_back(char(0x80 | (cp & 0x3f)));
        }
    }
    return result;
}

void selectFont(cairo_t* cr, const Text& text) {
    cairo_select_font_face(cr, "Open Sans", CAIRO_FONT_SLANT_NORMAL,
        (text.style() & Text::Bold) ? CAIRO_FONT_WEIGHT_BOLD : CAIRO_FONT_WEIGHT_NORMAL);
    cairo_set_font_size(cr, text.characterSize());
}
} // namespace

struct FontData { std::filesystem::path path; };
struct RenderTexture::Impl {
    Surface surface;
    Context context;
    Scene scene={{"background", {0,0,0,255}}, {"commands", Scene::array()}, {"images", Scene::array()}};
    std::unordered_map<uint64_t, size_t> assets;
    std::unordered_map<uint64_t, std::string> encodedAssets, previousAssets;
};

bool FloatRect::contains(Vector2f p) const {
    return p.x >= position.x && p.y >= position.y && p.x < position.x + size.x && p.y < position.y + size.y;
}
std::optional<FloatRect> FloatRect::findIntersection(const FloatRect& other) const {
    const float l=std::max(position.x,other.position.x), t=std::max(position.y,other.position.y);
    const float r=std::min(position.x+size.x,other.position.x+other.size.x);
    const float b=std::min(position.y+size.y,other.position.y+other.size.y);
    if (r <= l || b <= t) return {};
    return FloatRect{{l,t},{r-l,b-t}};
}

void Image::resize(Vector2u size, Color color) {
    size_=size; pixels_.resize(size_t(size.x)*size.y*4);
    for (size_t i=0;i<pixels_.size();i+=4) {
        pixels_[i]=color.r;pixels_[i+1]=color.g;pixels_[i+2]=color.b;pixels_[i+3]=color.a;
    }
}
Color Image::getPixel(Vector2u p) const {
    if (p.x>=size_.x || p.y>=size_.y) return {};
    const size_t i=(size_t(p.y)*size_.x+p.x)*4;
    return {pixels_[i],pixels_[i+1],pixels_[i+2],pixels_[i+3]};
}
void Image::setPixel(Vector2u p, Color c) {
    if (p.x>=size_.x || p.y>=size_.y) return;
    const size_t i=(size_t(p.y)*size_.x+p.x)*4;
    pixels_[i]=c.r;pixels_[i+1]=c.g;pixels_[i+2]=c.b;pixels_[i+3]=c.a;
}
bool Image::loadFromFile(const std::filesystem::path& file) {
    Surface surface(cairo_image_surface_create_from_png(file.c_str()));
    if (cairo_surface_status(surface.get()) != CAIRO_STATUS_SUCCESS) return false;
    cairo_surface_flush(surface.get());
    const int w=cairo_image_surface_get_width(surface.get()), h=cairo_image_surface_get_height(surface.get());
    resize({unsigned(w),unsigned(h)},Color::Transparent);
    const auto* src=cairo_image_surface_get_data(surface.get());
    const int stride=cairo_image_surface_get_stride(surface.get());
    for(int y=0;y<h;++y) for(int x=0;x<w;++x) {
        const auto* p=src+y*stride+x*4; setPixel({unsigned(x),unsigned(y)},{p[2],p[1],p[0],p[3]});
    }
    return true;
}
bool Image::saveToFile(const std::filesystem::path& file) const {
    Surface surface(cairo_image_surface_create(CAIRO_FORMAT_ARGB32,int(size_.x),int(size_.y)));
    if (cairo_surface_status(surface.get()) != CAIRO_STATUS_SUCCESS) return false;
    auto* dst=cairo_image_surface_get_data(surface.get()); const int stride=cairo_image_surface_get_stride(surface.get());
    for(unsigned y=0;y<size_.y;++y) for(unsigned x=0;x<size_.x;++x) {
        const auto c=getPixel({x,y}); auto* p=dst+y*stride+x*4;
        p[0]=uint8_t(c.b*c.a/255);p[1]=uint8_t(c.g*c.a/255);p[2]=uint8_t(c.r*c.a/255);p[3]=c.a;
    }
    cairo_surface_mark_dirty(surface.get());
    return cairo_surface_write_to_png(surface.get(),file.c_str()) == CAIRO_STATUS_SUCCESS;
}
bool Image::copy(const Image& image, Vector2u at) {
    if (at.x+image.size_.x>size_.x || at.y+image.size_.y>size_.y) return false;
    for(unsigned y=0;y<image.size_.y;++y)
        std::memcpy(pixels_.data()+(size_t(at.y+y)*size_.x+at.x)*4,
                    image.pixels_.data()+size_t(y)*image.size_.x*4,size_t(image.size_.x)*4);
    return true;
}

String::String(const char* text) : String(std::string(text ? text : "")) {}
String::String(const std::string& text) {
    for(size_t i=0;i<text.size();) {
        const auto c=uint8_t(text[i++]); char32_t cp=0; int extra=0;
        if(c<0x80)cp=c; else if((c&0xe0)==0xc0){cp=c&0x1f;extra=1;}
        else if((c&0xf0)==0xe0){cp=c&0x0f;extra=2;} else {cp=c&7;extra=3;}
        while(extra-- && i<text.size())cp=(cp<<6)|(uint8_t(text[i++])&0x3f);
        text_.push_back(cp);
    }
}
bool Font::openFromFile(const std::filesystem::path& file) {
    if (!std::filesystem::exists(file)) return false;
    data_=std::make_shared<FontData>();data_->path=file;return true;
}
FloatRect Text::getLocalBounds() const {
    Surface s(cairo_image_surface_create(CAIRO_FORMAT_ARGB32,1,1));Context cr(cairo_create(s.get()));
    selectFont(cr.get(),*this);cairo_text_extents_t ext{};const auto value=utf8(string_);
    cairo_text_extents(cr.get(),value.c_str(),&ext);
    return {{float(ext.x_bearing),float(ext.y_bearing)},{float(ext.width),float(ext.height)}};
}

RenderTexture::RenderTexture(Vector2u size) { resize(size); }
RenderTexture::~RenderTexture()=default;
bool RenderTexture::resize(Vector2u size) {
    if(!size.x||!size.y)return false;
    impl_=std::make_unique<Impl>();
    impl_->surface.reset(cairo_image_surface_create(CAIRO_FORMAT_ARGB32,int(size.x),int(size.y)));
    if(cairo_surface_status(impl_->surface.get())!=CAIRO_STATUS_SUCCESS)return false;
    impl_->context.reset(cairo_create(impl_->surface.get()));
    texture_=Texture(size);view_=View(FloatRect{{0,0},{float(size.x),float(size.y)}});return true;
}
Vector2u RenderTexture::getSize() const { return texture_.getSize(); }
Vector2f RenderTexture::mapPixelToCoords(Vector2i p) const {
    const auto s=getSize(); const auto& vp=view_.viewport();const auto& r=view_.rectangle();
    const float vx=vp.position.x*s.x,vy=vp.position.y*s.y,vw=vp.size.x*s.x,vh=vp.size.y*s.y;
    return {r.position.x+(p.x-vx)*r.size.x/std::max(1.f,vw),r.position.y+(p.y-vy)*r.size.y/std::max(1.f,vh)};
}
static void setup(cairo_t* cr,const View& v,Vector2u size) {
    cairo_reset_clip(cr);cairo_identity_matrix(cr);
    const auto& p=v.viewport();const auto& r=v.rectangle();
    const double x=p.position.x*size.x,y=p.position.y*size.y,w=p.size.x*size.x,h=p.size.y*size.y;
    cairo_rectangle(cr,x,y,w,h);cairo_clip(cr);cairo_translate(cr,x,y);
    cairo_scale(cr,w/std::max(1.f,r.size.x),h/std::max(1.f,r.size.y));cairo_translate(cr,-r.position.x,-r.position.y);
}
static Scene colorValue(Color c) { return {c.r,c.g,c.b,c.a}; }
static Scene command(const char* kind, const View& v) {
    const auto r=v.rectangle(), p=v.viewport();
    return {{"kind",kind},{"view",{r.position.x,r.position.y,r.size.x,r.size.y,p.position.x,p.position.y,p.size.x,p.size.y}}};
}
std::string RenderTexture::scene() const { return impl_->scene.dump(); }
void RenderTexture::clear(Color c) {
    // Two generations keep textures reused in consecutive frames while releasing
    // revisions that disappeared. Never retain a growing training history.
    impl_->previousAssets=std::move(impl_->encodedAssets);
    impl_->encodedAssets.clear();
    impl_->scene["background"]=colorValue(c);impl_->scene["commands"].clear();impl_->scene["images"].clear();impl_->assets.clear();
    if(recordCommands_)return;
    auto* cr=impl_->context.get();cairo_save(cr);cairo_reset_clip(cr);cairo_identity_matrix(cr);source(cr,c);cairo_paint(cr);cairo_restore(cr); }
void RenderTexture::draw(const RectangleShape& s) {
    if(recordCommands_) {
        auto cmd=command("rect",view_);const auto pos=s.getPosition(),sz=s.getSize();
        cmd["rect"]={pos.x,pos.y,sz.x,sz.y};cmd["fill"]=colorValue(s.fill());cmd["outline"]=colorValue(s.outline());cmd["thickness"]=s.thickness();
        impl_->scene["commands"].push_back(std::move(cmd));
        return;
    }
    auto* cr=impl_->context.get();cairo_save(cr);setup(cr,view_,getSize());
    const auto p=s.getPosition(),z=s.getSize();cairo_rectangle(cr,p.x,p.y,z.x,z.y);source(cr,s.fill());cairo_fill(cr);
    if(s.thickness()!=0){const double t=std::abs(s.thickness());cairo_set_line_width(cr,t);cairo_rectangle(cr,p.x+t/2,p.y+t/2,std::max(0.f,z.x-float(t)),std::max(0.f,z.y-float(t)));source(cr,s.outline());cairo_stroke(cr);}cairo_restore(cr);
}
void RenderTexture::draw(const Sprite& s) {
    const auto& image=s.texture().image();if(!image.getSize().x)return;
    if(recordCommands_) {
        auto found=impl_->assets.find(s.texture().revision());size_t asset;
        if(found==impl_->assets.end()) {
            const auto revision=s.texture().revision();
            auto cached=impl_->previousAssets.find(revision);
            std::string encoded;
            if(cached!=impl_->previousAssets.end()) encoded=std::move(cached->second);
            else {
                Surface src(cairo_image_surface_create(CAIRO_FORMAT_ARGB32,int(image.getSize().x),int(image.getSize().y)));
                auto* data=cairo_image_surface_get_data(src.get());const int stride=cairo_image_surface_get_stride(src.get());
                for(unsigned y=0;y<image.getSize().y;++y)for(unsigned x=0;x<image.getSize().x;++x){auto c=image.getPixel({x,y});auto* p=data+y*stride+x*4;p[0]=uint8_t(c.b*c.a/255);p[1]=uint8_t(c.g*c.a/255);p[2]=uint8_t(c.r*c.a/255);p[3]=c.a;}
                cairo_surface_mark_dirty(src.get());
                std::vector<uint8_t> png;
                cairo_surface_write_to_png_stream(src.get(),+[](void* closure,const unsigned char* data,unsigned length){
                    auto& bytes=*static_cast<std::vector<uint8_t>*>(closure);bytes.insert(bytes.end(),data,data+length);return CAIRO_STATUS_SUCCESS;
                },&png);
                encoded=base64(png.data(),png.size());
            }
            asset=impl_->scene["images"].size();impl_->assets[revision]=asset;
            impl_->encodedAssets.emplace(revision,encoded);
            impl_->scene["images"].push_back(std::move(encoded));
        } else asset=found->second;
        auto cmd=command("image",view_);auto pos=s.getPosition(),origin=s.getOrigin(),scale=s.getScale();
        cmd["position"]={pos.x-origin.x*scale.x,pos.y-origin.y*scale.y};cmd["scale"]={scale.x,scale.y};
        cmd["asset"]=asset;cmd["alpha"]=s.color().a/255.;cmd["smooth"]=s.texture().isSmooth();impl_->scene["commands"].push_back(std::move(cmd));
        return;
    }
    Surface src(cairo_image_surface_create(CAIRO_FORMAT_ARGB32,int(image.getSize().x),int(image.getSize().y)));
    auto* data=cairo_image_surface_get_data(src.get());const int stride=cairo_image_surface_get_stride(src.get());
    for(unsigned y=0;y<image.getSize().y;++y)for(unsigned x=0;x<image.getSize().x;++x){auto c=image.getPixel({x,y});auto* p=data+y*stride+x*4;p[0]=uint8_t(c.b*c.a/255);p[1]=uint8_t(c.g*c.a/255);p[2]=uint8_t(c.r*c.a/255);p[3]=c.a;}
    cairo_surface_mark_dirty(src.get());
    auto* cr=impl_->context.get();cairo_save(cr);setup(cr,view_,getSize());
    const auto p=s.getPosition(),o=s.getOrigin(),sc=s.getScale();cairo_translate(cr,p.x-o.x*sc.x,p.y-o.y*sc.y);cairo_scale(cr,sc.x,sc.y);
    cairo_set_source_surface(cr,src.get(),0,0);cairo_pattern_set_filter(cairo_get_source(cr),s.texture().isSmooth()?CAIRO_FILTER_BILINEAR:CAIRO_FILTER_NEAREST);cairo_paint_with_alpha(cr,s.color().a/255.);cairo_restore(cr);
}
void RenderTexture::draw(const Text& t) {
    if(recordCommands_) {
        auto cmd=command("text",view_);auto pos=t.getPosition();cmd["position"]={pos.x,pos.y};cmd["text"]=utf8(t.string());
        cmd["size"]=t.characterSize();cmd["bold"]=bool(t.style()&Text::Bold);cmd["fill"]=colorValue(t.fill());impl_->scene["commands"].push_back(std::move(cmd));
        return;
    }
    auto* cr=impl_->context.get();cairo_save(cr);setup(cr,view_,getSize());selectFont(cr,t);source(cr,t.fill_);
    const auto p=t.getPosition();cairo_move_to(cr,p.x,p.y+t.characterSize()*.82);const auto value=utf8(t.string());cairo_show_text(cr,value.c_str());cairo_restore(cr);
}
void RenderTexture::draw(const VertexArray& a) {
    if (a.vertices().size() < 2) return;
    if(recordCommands_) {
        auto cmd=command("line",view_);cmd["vertices"]=Scene::array();
        for(const auto& v:a.vertices())cmd["vertices"].push_back({v.position.x,v.position.y,v.color.r,v.color.g,v.color.b,v.color.a});
        impl_->scene["commands"].push_back(std::move(cmd));
        return;
    }
    auto* cr=impl_->context.get();cairo_save(cr);setup(cr,view_,getSize());
    cairo_set_line_width(cr,1);
    for(size_t i=1;i<a.vertices().size();++i) {
        const auto& previous=a.vertices()[i-1];
        const auto& current=a.vertices()[i];
        cairo_move_to(cr,previous.position.x,previous.position.y);
        cairo_line_to(cr,current.position.x,current.position.y);
        source(cr,current.color);
        cairo_stroke(cr);
    }
    cairo_restore(cr);
}
void RenderTexture::display() {
    // Window drawing is command-only. Rasterize only for an explicit capture.
    if(recordCommands_) {
            auto* cr=impl_->context.get();cairo_save(cr);cairo_reset_clip(cr);cairo_identity_matrix(cr);
            cairo_scene::render(cr,impl_->scene,getSize().x,getSize().y);cairo_restore(cr);
    }
    cairo_surface_flush(impl_->surface.get());Image image(getSize(),Color::Transparent);const auto* src=cairo_image_surface_get_data(impl_->surface.get());const int stride=cairo_image_surface_get_stride(impl_->surface.get());
    for(unsigned y=0;y<getSize().y;++y)for(unsigned x=0;x<getSize().x;++x){
        const auto* p=src+y*stride+x*4;const uint8_t a=p[3];
        auto un=[a](uint8_t c){return a?uint8_t(std::min(255,int(c)*255/int(a))):uint8_t(0);};
        image.setPixel({x,y},{un(p[2]),un(p[1]),un(p[0]),a});
    }
    texture_.loadFromImage(image);
}
} // namespace mimir::viz::graphics
