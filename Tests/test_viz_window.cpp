// Explicit graphical smoke target, never part of headless CTest runs.
#include "VizWindow.hpp"
#include <chrono>
#include <iostream>
#include <stdexcept>
int main(int argc, char** argv) {
    try {
        auto window = createVizWindow({320,240}, "Mimir Viz backend smoke", 30);
        const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(argc>1?std::stoi(argv[1]):5);
        vizgfx::Image pixels({4,4},vizgfx::Color(45,210,90));
        vizgfx::Texture texture;if(!texture.loadFromImage(pixels))throw std::runtime_error("Cannot create test texture");
        vizgfx::Font font;
        const bool hasFont=font.openFromFile("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf");
        bool checked=false;
        while(window->isOpen() && std::chrono::steady_clock::now()<deadline) {
            while(auto e=window->pollEvent()) {
                if(e->is<vizgfx::Event::Closed>()) {std::cout<<"CLOSE"<<std::endl;window->close();}
                if(auto* k=e->getIf<vizgfx::Event::KeyPressed>()) {
                    std::cout<<"KEY "<<int(k->code)<<" "<<k->control<<" "<<k->shift<<std::endl;
                    if(k->code==vizgfx::Keyboard::Key::R)window->setSize({400,280});
                }
                if(auto* t=e->getIf<vizgfx::Event::TextEntered>())std::cout<<"TEXT "<<uint32_t(t->unicode)<<std::endl;
                if(auto* m=e->getIf<vizgfx::Event::MouseButtonPressed>())std::cout<<"DOWN "<<m->position.x<<" "<<m->position.y<<std::endl;
                if(auto* m=e->getIf<vizgfx::Event::MouseButtonReleased>())std::cout<<"UP "<<m->position.x<<" "<<m->position.y<<std::endl;
                if(auto* w=e->getIf<vizgfx::Event::MouseWheelScrolled>())std::cout<<"WHEEL "<<w->delta<<std::endl;
                if(auto* r=e->getIf<vizgfx::Event::Resized>())window->setView(vizgfx::View(vizgfx::FloatRect({0,0},{float(r->size.x),float(r->size.y)})));
            }
            if(!window->isOpen())break;
            window->clear(vizgfx::Color(17,34,51));
            vizgfx::RectangleShape rect({70,50});rect.setPosition({10,20});rect.setFillColor(vizgfx::Color(231,42,63));window->draw(rect);
            vizgfx::Sprite sprite(texture);sprite.setPosition({100,20});sprite.setScale({8,8});window->draw(sprite);
            vizgfx::VertexArray line(vizgfx::PrimitiveType::LineStrip,2);
            line[0]={{10,100},vizgfx::Color::White};line[1]={{180,100},vizgfx::Color::White};window->draw(line);
            if(hasFont){vizgfx::Text text(font);text.setString("Native renderer");text.setCharacterSize(16);text.setPosition({10,120});window->draw(text);}
            auto view=window->getView();
            vizgfx::View clipped(vizgfx::FloatRect({0,0},{40,40}));clipped.setViewport(vizgfx::FloatRect({.75f,.75f},{.125f,.125f}));window->setView(clipped);
            vizgfx::RectangleShape clipRect({80,80});clipRect.setFillColor(vizgfx::Color(90,100,240));window->draw(clipRect);window->setView(view);
            window->display();
            if(!checked) {
                auto image=window->captureImage();
                if(image.getPixel({0,0})!=vizgfx::Color(17,34,51) || image.getPixel({20,30})!=vizgfx::Color(231,42,63))throw std::runtime_error("Framebuffer pixel mismatch");
                if(image.getPixel({110,30})!=vizgfx::Color(45,210,90) || image.getPixel({250,190})!=vizgfx::Color(90,100,240) || image.getPixel({290,220})!=vizgfx::Color(17,34,51))throw std::runtime_error("Texture/viewport pixel mismatch");
                if(argc>2 && !image.saveToFile(argv[2]))throw std::runtime_error("Cannot save smoke PNG");
                std::cout<<"PIXELS_OK"<<std::endl;checked=true;
            }
        }
        return checked?0:1;
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
