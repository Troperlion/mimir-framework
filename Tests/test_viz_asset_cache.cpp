#include "viz/SoftwareGraphics.hpp"
#include "viz/renderers/QtScene.hpp"
#include "viz/renderers/CairoScene.hpp"
#include <chrono>
#include <iostream>
#include <stdexcept>
using namespace mimir::viz;
using namespace mimir::viz::graphics;
int main() {
 Image pixels({512,512});
 unsigned value=17;
 for(unsigned y=0;y<512;++y)for(unsigned x=0;x<512;++x){value=value*1664525+1013904223;pixels.setPixel({x,y},Color(value>>24,value>>16,value>>8));}
 Texture texture;texture.loadFromImage(pixels);Sprite sprite(texture);
 RenderTexture target({512,512});target.recordCommands(true);
 auto start=std::chrono::steady_clock::now();
 std::string first;
 for(int i=0;i<30;++i){target.clear();target.draw(sprite);auto scene=Scene::parse(target.scene());auto encoded=scene["images"][0].get<std::string>();if(i==0)first=encoded;else if(first!=encoded)throw std::runtime_error("changed stable asset");}
 std::cout<<"stable texture 30 frames ms "<<std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()<<'\n';
 cairo_scene::ImageCache cairoCache;
 qt::ImageCache cache;
 Scene assets=Scene::array({first});auto a=cache.load(assets);auto b=cache.load(assets);
 auto ca=cairoCache.load(assets);auto cb=cairoCache.load(assets);
 if(ca[0].get()!=cb[0].get())throw std::runtime_error("Cairo image decoded again");
 if(a[0].cacheKey()!=b[0].cacheKey())throw std::runtime_error("image decoded again");
 pixels.setPixel({0,0},Color(255,0,0));texture.loadFromImage(pixels);
 target.clear();target.draw(sprite);auto changed=Scene::parse(target.scene());
 if(changed["images"][0]==first)throw std::runtime_error("stale updated texture");
 auto cc=cairoCache.load(changed["images"]);
 if(cc[0].get()==ca[0].get())throw std::runtime_error("stale Cairo image");
 auto c=cache.load(changed["images"]);if(c[0].pixelColor(0,0)!=QColor(255,0,0))throw std::runtime_error("stale decoded image");
 target.display();if(target.getTexture().image().getPixel({0,0})!=Color(255,0,0))throw std::runtime_error("capture mismatch");
 cache.load(Scene::array());auto d=cache.load(assets);if(d[0].cacheKey()==a[0].cacheKey())throw std::runtime_error("unused asset retained");
 cairoCache.load(Scene::array());auto cd=cairoCache.load(assets);
 if(cd[0].get()==ca[0].get())throw std::runtime_error("unused Cairo asset retained");
 std::cout<<"ASSET_CACHE_OK stable, update, capture, eviction\n";
}
