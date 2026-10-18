#include "viz/ModelGraphLanes.hpp"
#include <iostream>
#include <stdexcept>
using namespace Mimir::Viz;
static void require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
int main(int argc,char** argv) {
    using Json=nlohmann::json;
    Json graph={{"nodes",Json::array()},{"edges",Json::array()}};
    const char* labels[]={"input / image", "encoder / stem", "encoder / residual", "skip / projection",
        "residual / add", "latent / mean", "latent / logvar", "latent / sample", "decoder / output"};
    for(const auto* name:labels)graph["nodes"].push_back({{"name",name}});
    for(const auto pair:std::vector<std::pair<size_t,size_t>>{{0,1},{1,2},{1,3},{1,4},{2,4},{3,4},{4,5},{4,6},{5,7},{6,7},{7,8}})
        graph["edges"].push_back({{"from",pair.first},{"to",pair.second}});
    const auto layout=modelGraphLanes(graph);
    require(layout.edges.size()==11,"all dependencies remain visible");
    require(layout.columns>=3,"fork and long skip need independent rails");
    require(layout.lanes[0]==layout.lanes[1],"linear chain must keep its rail");
    require(layout.lanes==modelGraphLanes(graph).lanes,"layout must be deterministic");
    for(const auto& e:layout.edges) {
        const auto path=graphLaneCurve(layout,e);
        require(path.front()==layout.position(e.from) && path.back()==layout.position(e.to),"curve endpoints");
        for(size_t i=1;i<path.size();++i)require(path[i].y>=path[i-1].y,"curves follow execution direction");
        for(size_t i=e.from+1;i<e.to;++i)require(layout.lanes[i]!=e.rail,"reserved skip rail crosses a node");
    }
    require(modelGraphLanes(Json{{"nodes",Json::array()},{"edges",Json::array()}}).lanes.empty(),"empty model");
    vizgfx::RenderTexture canvas({1000,760});canvas.clear(vizgfx::Color(16,23,32));
    vizgfx::Font font;require(font.openFromFile("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),"preview font");
    auto text=[&](float x,float y,const std::string& s,vizgfx::Color color,unsigned size=14){
        vizgfx::Text t(font);t.setCharacterSize(size);t.setPosition({x,y});t.setFillColor(color);t.setString(s);canvas.draw(t);
    };
    text(20,12,"MODEL GRAPH / PLANNER",vizgfx::Color(87,204,190),20);
    text(20,43,"cost  |  inference  |  fusion on  |  buffer reuse on",vizgfx::Color(190,203,220));
    const auto old=canvas.getView();vizgfx::View view(vizgfx::FloatRect({0,0},{1000,650}));
    view.setViewport(vizgfx::FloatRect({0,85.f/760},{1,650.f/760}));canvas.setView(view);
    drawModelGraphLanes(canvas,layout,4,0,650);
    for(size_t i=0;i<layout.lanes.size();++i) {
        const auto p=layout.position(i);
        text(layout.labelsX(),p.y-15,std::to_string(i)+"   "+labels[i],vizgfx::Color(230,240,246));
        text(layout.labelsX(),p.y+7,i==4?"Add  |  CUDA  |  fusion":"Layer  |  CUDA",graphLaneColor(layout.lanes[i]),12);
    }
    canvas.setView(old);canvas.display();
    const auto image=canvas.getTexture().copyToImage();
    size_t colorful=0;
    for(unsigned y=90;y<710;++y)for(unsigned x=10;x<unsigned(layout.labelsX()-10);++x){const auto c=image.getPixel({x,y});if(c.g>70 || c.r>70)++colorful;}
    require(colorful>1000,"offscreen rails and junctions must be rendered");
    if(argc>1)require(image.saveToFile(argv[1]),"save preview");
    std::cout<<"MODEL_GRAPH_LANES_OK: fork, merge, reserved skip rails and offscreen render\n";
}
