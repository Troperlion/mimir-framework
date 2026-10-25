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
    const size_t depths[]={0,1,2,2,3,4,4,5,6};
    const size_t lanes[]={0,0,0,1,0,0,1,0,0};
    for(size_t i=0;i<9;++i)
        graph["nodes"].push_back({{"name",labels[i]},{"depth",depths[i]},{"lane",lanes[i]}});
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
    const auto tree=modelGraphTree(graph);
    require(tree.edges.size()==11,"nodal tree retains every dependency");
    require(tree.position(1).x>tree.position(0).x,"tree nodes advance by execution depth");
    require(tree.position(2).y!=tree.position(3).y,"parallel branches occupy separate rows");
    require(tree.position(2).x==tree.position(3).x,"parallel branches share their depth column");
    require(modelGraphTree(Json{{"nodes",Json::array()},{"edges",Json::array()}}).edges.empty(),
        "empty nodal graph");
    for(const auto& edge:tree.edges) {
        const auto path=modelGraphTreeEdgePath(tree,edge);
        for(size_t segment=1;segment<path.size();++segment) {
            for(int sample=0;sample<=32;++sample) {
                const float t=float(sample)/32.f;
                const float x=path[segment-1].x+(path[segment].x-path[segment-1].x)*t;
                const float y=path[segment-1].y+(path[segment].y-path[segment-1].y)*t;
                for(size_t node=0;node<tree.depths.size();++node) {
                    const auto center=tree.position(node);
                    const float left=center.x-tree.card_width*.5f;
                    const float top=center.y-tree.card_height*.5f;
                    const bool inside=x>left+0.5f && x<left+tree.card_width-0.5f &&
                        y>top+0.5f && y<top+tree.card_height-0.5f;
                    require(!inside,"connections must never pass underneath a node card");
                }
            }
        }
    }
    auto intersects=[](vizgfx::Vector2f a,vizgfx::Vector2f b,vizgfx::Vector2f c,vizgfx::Vector2f d) {
        const bool first_horizontal=std::abs(a.y-b.y)<0.01f;
        const bool second_horizontal=std::abs(c.y-d.y)<0.01f;
        if(first_horizontal && second_horizontal) {
            if(std::abs(a.y-c.y)>0.01f)return false;
            return std::max(std::min(a.x,b.x),std::min(c.x,d.x))<
                std::min(std::max(a.x,b.x),std::max(c.x,d.x))-0.01f;
        }
        if(!first_horizontal && !second_horizontal) {
            if(std::abs(a.x-c.x)>0.01f)return false;
            return std::max(std::min(a.y,b.y),std::min(c.y,d.y))<
                std::min(std::max(a.y,b.y),std::max(c.y,d.y))-0.01f;
        }
        const auto h1=first_horizontal?a:c;
        const auto h2=first_horizontal?b:d;
        const auto v1=first_horizontal?c:a;
        const auto v2=first_horizontal?d:b;
        return v1.x>std::min(h1.x,h2.x)+0.01f && v1.x<std::max(h1.x,h2.x)-0.01f &&
            h1.y>std::min(v1.y,v2.y)+0.01f && h1.y<std::max(v1.y,v2.y)-0.01f;
    };
    for(size_t a=0;a<tree.edges.size();++a)for(size_t b=a+1;b<tree.edges.size();++b) {
        const auto first=modelGraphTreeEdgePath(tree,tree.edges[a]);
        const auto second=modelGraphTreeEdgePath(tree,tree.edges[b]);
        for(size_t i=1;i<first.size();++i)for(size_t j=1;j<second.size();++j)
            require(!intersects(first[i-1],first[i],second[j-1],second[j]),
                "different connections must not cross or overlap");
    }
    Json simple={{"nodes",Json::array({Json{{"depth",0},{"lane",0}},
        Json{{"depth",1},{"lane",0}}})},
        {"edges",Json::array({Json{{"from",0},{"to",1}}})}};
    const auto simple_layout=modelGraphTree(simple);
    const auto simple_path=modelGraphTreeEdgePath(simple_layout,simple_layout.edges.front());
    require(simple_path.size()==2 && simple_path[0].y==simple_path[1].y,
        "a simple aligned connection is one horizontal segment");
    Json fork={{"nodes",Json::array({Json{{"depth",0},{"lane",1}},
        Json{{"depth",1},{"lane",0}},Json{{"depth",1},{"lane",2}}})},
        {"edges",Json::array({Json{{"from",0},{"to",1}},Json{{"from",0},{"to",2}}})}};
    const auto fork_layout=modelGraphTree(fork);
    const auto upper_branch=modelGraphTreeEdgePath(fork_layout,fork_layout.edges[0]);
    const auto lower_branch=modelGraphTreeEdgePath(fork_layout,fork_layout.edges[1]);
    require(upper_branch.size()==4 && lower_branch.size()==4,
        "forked outputs use separate orthogonal routes");
    require(upper_branch[0].y!=lower_branch[0].y &&
        upper_branch[1].x!=lower_branch[1].x,
        "each fork uses a distinct output port and routing line");
    for(size_t i=1;i<upper_branch.size();++i)for(size_t j=1;j<lower_branch.size();++j)
        require(!intersects(upper_branch[i-1],upper_branch[i],lower_branch[j-1],lower_branch[j]),
            "sibling branches remain visually distinct");
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
    drawModelGraphTreeEdges(canvas,tree,4);
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
