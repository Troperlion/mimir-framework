#include "test_utils.hpp"
#include "viz/DatasetBBoxBadge.hpp"
#include "viz/DrawDatasetBBoxOverlay.hpp"

int main(int argc,char** argv) {
    vizgfx::RenderTexture canvas({128,128});
    const vizgfx::Color background(50,50,60);
    canvas.clear(background);
    mimir::viz::drawDatasetBBoxBadge(canvas,20.f,20.f,80.f,true);
    canvas.display();
    const auto visible=canvas.getTexture().copyToImage();
    const auto corner=visible.getPixel({77,31});
    TASSERT_TRUE(corner.g>200 && corner.b>200);
    int changed=0;
    for(unsigned y=0;y<128;++y) for(unsigned x=0;x<128;++x) {
        const auto pixel=visible.getPixel({x,y});
        if(pixel.r!=background.r || pixel.g!=background.g || pixel.b!=background.b) {
            ++changed;
            TASSERT_TRUE(x>=70 && x<=96 && y>=24 && y<=50);
        }
    }
    TASSERT_TRUE(changed>100);
    if(argc>1) TASSERT_TRUE(visible.saveToFile(argv[1]));
    // Next sample has no active bbox: the icon must disappear entirely.
    canvas.clear(background);
    mimir::viz::drawDatasetBBoxBadge(canvas,20.f,20.f,80.f,false);
    canvas.display();
    const auto absent=canvas.getTexture().copyToImage();
    for(unsigned y=0;y<128;++y) for(unsigned x=0;x<128;++x) {
        const auto pixel=absent.getPixel({x,y});
        TASSERT_TRUE(pixel.r==background.r && pixel.g==background.g && pixel.b==background.b);
    }
    canvas.clear(background);
    DatasetBBoxOverlay::draw(canvas, {{0.25f,0.25f,0.75f,0.75f}}, 10,20,80,40);
    canvas.display();
    const auto overlay=canvas.getTexture().copyToImage();
    TASSERT_TRUE(overlay.getPixel({31,31}).g>200);
    TASSERT_TRUE(overlay.getPixel({50,40}).g==background.g);
    TASSERT_TRUE(overlay.getPixel({20,30}).g==background.g);
    canvas.clear(background);
    DatasetBBoxOverlay::draw(canvas, {}, 10,20,80,40);
    canvas.display();
    TASSERT_TRUE(canvas.getTexture().copyToImage().getPixel({31,31}).g==background.g);
    std::cout << "PASS dataset bbox icon: top-right placement, visible and cleared on next sample\n";
}
