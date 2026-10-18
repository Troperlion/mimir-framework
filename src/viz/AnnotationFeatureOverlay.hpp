#pragma once
#include "training/BBoxSupervision.hpp"
#include <algorithm>
#include <array>
#include <cctype>
#include <sstream>
#include <unordered_map>

namespace AnnotationFeatureOverlay {
using json=nlohmann::json;
using Color=std::array<uint8_t,3>;
inline std::string lower(std::string s){for(char& c:s)c=static_cast<char>(std::tolower(static_cast<unsigned char>(c)));return s;}
inline bool spatialOutput(const std::string& label){
    const auto s=lower(label);
    if(s.find("out_concat")!=std::string::npos || s.find("out_pack")!=std::string::npos || s.find("/vec")!=std::string::npos)return false;
    return s.find("resdiff_abs")!=std::string::npos || s.find("recon")!=std::string::npos || s.find("/dec/")!=std::string::npos || s.find("unpatchify")!=std::string::npos || s.find("/out/")!=std::string::npos || s.find("/output/")!=std::string::npos || s.find("[st]")!=std::string::npos;
}
inline void rgb(std::vector<uint8_t>& pixels,int w,int h,int channels){
    if(channels==3)return;
    if(channels!=1 && channels!=4){pixels.clear();return;}
    if(pixels.size()!=size_t(w)*h*channels){pixels.clear();return;}
    std::vector<uint8_t> out(size_t(w)*h*3);
    for(size_t p=0;p<size_t(w)*h;++p)for(int c=0;c<3;++c)out[p*3+c]=pixels[p*channels+(channels==1?0:c)];
    pixels=std::move(out);
}
inline void line(std::vector<uint8_t>& pixels,int w,int h,double ax,double ay,double bx,double by,Color color){
    int x=std::clamp(int(std::lround(ax*(w-1))),0,w-1),y=std::clamp(int(std::lround(ay*(h-1))),0,h-1);
    const int tx=std::clamp(int(std::lround(bx*(w-1))),0,w-1),ty=std::clamp(int(std::lround(by*(h-1))),0,h-1);
    const int dx=std::abs(tx-x),dy=-std::abs(ty-y),sx=x<tx?1:-1,sy=y<ty?1:-1;int error=dx+dy;
    while(true){for(int c=0;c<3;++c)pixels[(size_t(y)*w+x)*3+c]=color[c];if(x==tx && y==ty)break;const int e=2*error;if(e>=dy){error+=dy;x+=sx;}if(e<=dx){error+=dx;y+=sy;}}
}
inline Color color(size_t i){static const Color palette[]={{70,220,255},{255,170,65},{165,255,100},{235,120,245},{255,245,90},{120,155,255}};return palette[i%6];}
inline void geometry(std::vector<uint8_t>& pixels,int w,int h,const json& annotation){
    if(pixels.size()!=size_t(w)*h*3 || !annotation.contains("entities"))return;
    std::unordered_map<std::string,std::array<double,2>> centers;
    size_t i=0;
    for(const auto& e:annotation["entities"]){
        const auto tint=color(i++);
        if(e.contains("bbox")){const auto& b=e["bbox"];const double x=b[0].get<double>(),y=b[1].get<double>(),xx=b[2].get<double>(),yy=b[3].get<double>();
            line(pixels,w,h,x,y,xx,y,tint);line(pixels,w,h,xx,y,xx,yy,tint);line(pixels,w,h,xx,yy,x,yy,tint);line(pixels,w,h,x,yy,x,y,tint);centers[e["id"].get<std::string>()]={(x+xx)*.5,(y+yy)*.5};}
    }
    if(annotation.contains("relations"))for(const auto& r:annotation["relations"]){
        const auto a=centers.find(r.value("subject",std::string())),b=centers.find(r.value("object",std::string()));
        if(a!=centers.end() && b!=centers.end())line(pixels,w,h,a->second[0],a->second[1],b->second[0],b->second[1],{255,245,90});
    }
    i=0;
    for(const auto& e:annotation["entities"]){const auto tint=color(i++);if(!e.contains("skeleton"))continue;
        const auto& s=e["skeleton"];std::unordered_map<std::string,std::array<double,2>> points;
        for(const auto& p:s["points"])if(p.contains("position") && !p["position"].is_null())points[p["name"].get<std::string>()]={p["position"][0].get<double>(),p["position"][1].get<double>()};
        auto connect=[&](const std::string& a,const std::string& b){if(points.count(a) && points.count(b))line(pixels,w,h,points.at(a)[0],points.at(a)[1],points.at(b)[0],points.at(b)[1],tint);};
        for(const auto& p:s["points"]){const auto name=p["name"].get<std::string>();if(!points.count(name))continue;const auto v=points.at(name);
            line(pixels,w,h,v[0]-1./w,v[1],v[0]+1./w,v[1],tint);line(pixels,w,h,v[0],v[1]-1./h,v[0],v[1]+1./h,tint);
            if(p.contains("parent") && p["parent"].is_string())connect(name,p["parent"].get<std::string>());
            if(p.contains("entities"))for(const auto& id:p["entities"])if(centers.count(id.get<std::string>())){const auto c=centers.at(id.get<std::string>());line(pixels,w,h,v[0],v[1],c[0],c[1],{255,120,170});}
        }
        if(s.contains("edges"))for(const auto& edge:s["edges"])connect(edge["from"].get<std::string>(),edge["to"].get<std::string>());
    }
}
inline std::string legend(const json& annotation){
    std::ostringstream out;out<<"annotations: ";size_t i=0;
    for(const auto& e:annotation["entities"]){const auto tint=color(i++);out<<e.value("id",std::string())<<" RGB("<<int(tint[0])<<","<<int(tint[1])<<","<<int(tint[2])<<") ";
        if(e.contains("skeleton")) {
            out<<"skeleton="<<e["skeleton"].value("name",std::string())<<" ";
            for(const auto& p:e["skeleton"]["points"]){out<<p.value("name",std::string());if(p.contains("parent") && p["parent"].is_string())out<<"->"<<p["parent"].get<std::string>();if(p.contains("entities"))for(const auto& id:p["entities"])out<<"=>"<<id.get<std::string>();out<<"; ";}
        }}
    if(annotation.contains("relations"))for(const auto& r:annotation["relations"])out<<r.value("subject",std::string())<<" "<<r.value("predicate",std::string())<<" "<<r.value("object",std::string())<<"; ";
    return out.str();
}

template<class Frame>
void decorate(std::vector<Frame>& frames,const json& annotation,const json& config,int image_w,int image_h){
    if(!annotation.is_object() || !annotation.contains("entities") || annotation["entities"].empty())return;
    SkeletonAnnotation::validate(annotation);
    // Native spatial taps may be stretched to square for presentation. Their
    // sampled vectors are resized with the pixels, so normalized coordinates
    // stay aligned even when the display aspect differs from the image.
    (void)image_w;(void)image_h;
    const Frame* source=nullptr;
    for(const auto& f:frames)if(spatialOutput(f.label) && lower(f.label).find("resdiff_abs")==std::string::npos && f.w>0 && f.h>0 && f.tensor_sample_channels>=2 && f.tensor_values.size()==size_t(f.w)*f.h*f.tensor_sample_channels){if(!source || f.tensor_sample_channels>=source->tensor_sample_channels)source=&f;}
    Frame correspondence, impact;
    bool have_diff=false;
    for(const auto& f:frames)if(f.label.find("resdiff_abs")!=std::string::npos){correspondence.w=f.w;correspondence.h=f.h;correspondence.label=f.label.substr(0,f.label.find("resdiff_abs"))+"annotation_features";have_diff=true;break;}
    if(have_diff){
        correspondence.channels=3;correspondence.pixels.assign(size_t(correspondence.w)*correspondence.h*3,0);
        correspondence.tensor_info="Feature correspondence (cosine region prototype; not causal attribution). "+legend(annotation);
        if(source){
            const int channels=source->tensor_sample_channels;
            std::vector<std::vector<double>> prototypes;std::vector<size_t> entity_colors;size_t entity_index=0;
            for(const auto& e:annotation["entities"]){const size_t index=entity_index++;if(!e.contains("bbox"))continue;const auto& b=e["bbox"];std::vector<double> prototype(channels,0);double count=0;
                for(int y=0;y<source->h;++y)for(int x=0;x<source->w;++x){const double px=(x+.5)/source->w,py=(y+.5)/source->h;if(px<b[0].get<double>() || px>b[2].get<double>() || py<b[1].get<double>() || py>b[3].get<double>())continue;
                    for(int c=0;c<channels;++c){prototype[c]+=source->tensor_values[(size_t(y)*source->w+x)*channels+c];}
                    ++count;}
                if(count==0){const int x=std::min(source->w-1,int((b[0].get<double>()+b[2].get<double>())*.5*source->w)),y=std::min(source->h-1,int((b[1].get<double>()+b[3].get<double>())*.5*source->h));for(int c=0;c<channels;++c)prototype[c]=source->tensor_values[(size_t(y)*source->w+x)*channels+c];count=1;}
                double norm=0;for(double& v:prototype){v/=count;norm+=v*v;}if(norm<=1e-20 || !std::isfinite(norm))continue;norm=std::sqrt(norm);for(double& v:prototype)v/=norm;
                prototypes.push_back(std::move(prototype));entity_colors.push_back(index);
            }
            std::unordered_map<std::string,size_t> prototype_by_id;
            for(size_t i=0;i<entity_colors.size();++i)prototype_by_id[annotation["entities"][entity_colors[i]]["id"].template get<std::string>()]=i;
            std::ostringstream scores;scores<<std::fixed<<std::setprecision(3);
            if(annotation.contains("relations"))for(const auto& r:annotation["relations"]){const auto a=r.value("subject",std::string()),b=r.value("object",std::string());if(!prototype_by_id.count(a) || !prototype_by_id.count(b))continue;double score=0;for(int c=0;c<channels;++c)score+=prototypes[prototype_by_id.at(a)][c]*prototypes[prototype_by_id.at(b)][c];scores<<" region_cos("<<a<<","<<b<<")="<<score;}
            for(const auto& e:annotation["entities"])if(e.contains("skeleton"))for(const auto& p:e["skeleton"]["points"]){
                if(!p.contains("position") || p["position"].is_null())continue;
                std::vector<std::string> targets={e["id"].template get<std::string>()};if(p.contains("entities"))for(const auto& id:p["entities"])targets.push_back(id.template get<std::string>());
                const int x=std::min(source->w-1,int(p["position"][0].template get<double>()*source->w)),y=std::min(source->h-1,int(p["position"][1].template get<double>()*source->h));
                double norm=0;for(int c=0;c<channels;++c){const double v=source->tensor_values[(size_t(y)*source->w+x)*channels+c];norm+=v*v;}if(norm<=1e-20 || !std::isfinite(norm))continue;norm=std::sqrt(norm);
                for(const auto& id:targets)if(prototype_by_id.count(id)){double score=0;for(int c=0;c<channels;++c)score+=prototypes[prototype_by_id.at(id)][c]*source->tensor_values[(size_t(y)*source->w+x)*channels+c]/norm;scores<<" point_cos("<<p["name"].template get<std::string>()<<"->"<<id<<")="<<score;}
            }
            correspondence.tensor_info+=scores.str();
            correspondence.tensor_sample_channels=1;correspondence.tensor_values.assign(size_t(correspondence.w)*correspondence.h,0);
            correspondence.tensor_info+=" source="+source->label+" sampled_channels="+std::to_string(channels)+" valid_regions="+std::to_string(prototypes.size());
            for(int y=0;y<correspondence.h;++y)for(int x=0;x<correspondence.w;++x){const int sx=std::min(source->w-1,x*source->w/correspondence.w),sy=std::min(source->h-1,y*source->h/correspondence.h);double norm=0;for(int c=0;c<channels;++c){const double v=source->tensor_values[(size_t(sy)*source->w+sx)*channels+c];norm+=v*v;}if(norm<=1e-20 || !std::isfinite(norm) || prototypes.empty())continue;
                norm=std::sqrt(norm);double best=-2;size_t winner=0;
                for(size_t i=0;i<prototypes.size();++i){double score=0;for(int c=0;c<channels;++c)score+=prototypes[i][c]*source->tensor_values[(size_t(sy)*source->w+sx)*channels+c]/norm;if(score>best){best=score;winner=i;}}
                const size_t p=size_t(y)*correspondence.w+x;correspondence.tensor_values[p]=float(std::clamp(best,-1.,1.));const auto tint=color(entity_colors[winner]);const double alpha=(best+1)*.5;
                for(int c=0;c<3;++c)correspondence.pixels[p*3+c]=uint8_t(std::clamp(alpha,0.,1.)*tint[c]);
            }
        }else correspondence.tensor_info+=" No aligned raw spatial features available; geometry only.";
        geometry(correspondence.pixels,correspondence.w,correspondence.h,annotation);
        impact.w=correspondence.w;impact.h=correspondence.h;impact.channels=3;
        impact.label=correspondence.label.substr(0,correspondence.label.find("annotation_features"))+"annotation_impact";
        const auto weights=BBoxSupervision::makeWeights(&annotation,impact.w,impact.h,1,BBoxSupervision::Layout::HWC,config.value("bbox_loss_weight",1.f),&config);
        impact.tensor_values=weights.values.empty()?std::vector<float>(size_t(impact.w)*impact.h,1.f):weights.values;
        impact.tensor_sample_channels=1;impact.pixels.resize(size_t(impact.w)*impact.h*3);
        for(size_t p=0;p<impact.tensor_values.size();++p){const double weight=impact.tensor_values[p];const double alpha=std::min(1.,std::abs(weight-1.)/(1.+std::abs(weight-1.)));const Color tint=weight>=1?Color{255,130,40}:Color{60,155,255};for(int c=0;c<3;++c)impact.pixels[p*3+c]=uint8_t((1-alpha)*80+alpha*tint[c]);}
        impact.tensor_info="Regional loss weights projected on this tap grid (neutral=1; orange>1; blue<1); strength="+std::to_string(weights.strength)+". Not causal attribution. "+legend(annotation);
        geometry(impact.pixels,impact.w,impact.h,annotation);

    }
    for(auto& f:frames){if(!spatialOutput(f.label) || f.w<=0 || f.h<=0)continue;
        const auto weights=BBoxSupervision::makeWeights(&annotation,f.w,f.h,1,BBoxSupervision::Layout::HWC,config.value("bbox_loss_weight",1.f),&config);
        auto paint=[&](std::vector<uint8_t>& pixels,int c){if(pixels.empty())return;rgb(pixels,f.w,f.h,c);if(pixels.empty())return;
            for(size_t p=0;p<weights.values.size();++p){const double alpha=.2*std::max(0.,std::min(1.,double(weights.values[p])-1.));for(int k=0;k<3;++k)pixels[p*3+k]=uint8_t((1-alpha)*pixels[p*3+k]+alpha*Color{255,130,40}[k]);}
            geometry(pixels,f.w,f.h,annotation);};
        const int real_channels=f.pixels_real.empty()?0:int(f.pixels_real.size()/(size_t(f.w)*f.h));paint(f.pixels,f.channels);paint(f.pixels_real,real_channels);f.channels=3;
        f.tensor_info+=" "+legend(annotation)+" orange=regional loss emphasis; bbox_loss_weight="+std::to_string(weights.strength)+" (not causal attribution)";
    }
    if(have_diff){const auto it=std::find_if(frames.begin(),frames.end(),[](const Frame& f){return f.label.find("resdiff_abs")!=std::string::npos;});const auto index=static_cast<size_t>(it-frames.begin());frames.insert(frames.begin()+index+1,std::move(correspondence));frames.insert(frames.begin()+index+2,std::move(impact));}
}
}
