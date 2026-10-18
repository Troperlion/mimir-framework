#pragma once
#include "../Scene.hpp"
#include <QPainter>
#include <QImage>
#include <QByteArray>
#include <unordered_map>

namespace mimir::viz::qt {
inline QColor color(const Scene& v) { return QColor(v[0].get<int>(),v[1].get<int>(),v[2].get<int>(),v[3].get<int>()); }
// Retain only assets used by the latest frame; reused textures and expose events
// do not need another PNG decode. QImage copies share their pixel storage.
class ImageCache {
    std::unordered_map<std::string,QImage> images_;
public:
    std::vector<QImage> load(const Scene& assets) {
        std::unordered_map<std::string,QImage> next;
        std::vector<QImage> result;
        result.reserve(assets.size());
        for(const auto& asset:assets) {
            const auto& encoded=asset.get_ref<const std::string&>();
            auto found=images_.find(encoded);
            QImage image;
            if(found!=images_.end()) image=found->second;
            else {
                const auto bytes=unbase64(encoded);
                image=QImage::fromData(bytes.data(),int(bytes.size()),"PNG");
            }
            next.emplace(encoded,image);
            result.push_back(image);
        }
        images_=std::move(next);
        return result;
    }
};
inline void render(QPainter& painter, const Scene& scene, unsigned width, unsigned height, ImageCache* cache=nullptr) {
    painter.fillRect(QRectF(0,0,width,height),color(scene.at("background")));
    ImageCache temporary;
    const auto images=(cache?cache:&temporary)->load(scene.at("images"));
    painter.setRenderHint(QPainter::TextAntialiasing);
    for(const auto& cmd:scene.at("commands")) {
        painter.save();const auto& v=cmd.at("view");
        const double x=v[4].get<double>()*width,y=v[5].get<double>()*height,w=v[6].get<double>()*width,h=v[7].get<double>()*height;
        painter.setClipRect(QRectF(x,y,w,h));painter.translate(x,y);
        painter.scale(w/std::max(1.,v[2].get<double>()),h/std::max(1.,v[3].get<double>()));painter.translate(-v[0].get<double>(),-v[1].get<double>());
        const auto kind=cmd.at("kind").get<std::string>();
        if(kind=="rect") {
            const auto& r=cmd.at("rect");QRectF rect(r[0].get<double>(),r[1].get<double>(),r[2].get<double>(),r[3].get<double>());
            painter.fillRect(rect,color(cmd.at("fill")));
            const double t=std::abs(cmd.at("thickness").get<double>());
            if(t){painter.setPen(QPen(color(cmd.at("outline")),t));painter.setBrush(Qt::NoBrush);painter.drawRect(rect.adjusted(t/2,t/2,-t/2,-t/2));}
        } else if(kind=="text") {
            QFont font("Open Sans");font.setPixelSize(cmd.at("size").get<int>());font.setBold(cmd.at("bold").get<bool>());painter.setFont(font);
            painter.setPen(color(cmd.at("fill")));const auto& p=cmd.at("position");
            painter.drawText(QPointF(p[0].get<double>(),p[1].get<double>()+cmd.at("size").get<double>()*.82),QString::fromStdString(cmd.at("text").get<std::string>()));
        } else if(kind=="line") {
            const auto& vertices=cmd.at("vertices");
            for(size_t i=1;i<vertices.size();++i){const auto& a=vertices[i-1];const auto& b=vertices[i];
                painter.setPen(QPen(QColor(b[2].get<int>(),b[3].get<int>(),b[4].get<int>(),b[5].get<int>()),1));
                painter.drawLine(QPointF(a[0].get<double>(),a[1].get<double>()),QPointF(b[0].get<double>(),b[1].get<double>()));}
        } else if(kind=="image") {
            const auto asset=cmd.at("asset").get<size_t>();if(asset<images.size()) {
                const auto& p=cmd.at("position");const auto& s=cmd.at("scale");painter.translate(p[0].get<double>(),p[1].get<double>());painter.scale(s[0].get<double>(),s[1].get<double>());
                painter.setOpacity(cmd.at("alpha").get<double>());painter.setRenderHint(QPainter::SmoothPixmapTransform,cmd.at("smooth").get<bool>());painter.drawImage(QPointF(0,0),images[asset]);
            }
        }
        painter.restore();
    }
}
}
