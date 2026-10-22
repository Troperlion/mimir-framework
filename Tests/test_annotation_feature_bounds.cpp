#include "test_utils.hpp"
#include "viz/AnnotationFeatureOverlay.hpp"

struct Frame {
    int w=2,h=2,channels=3,tensor_sample_channels=0;
    std::string label="[ST] output",tensor_info;
    std::vector<uint8_t> pixels,pixels_real;
    std::vector<float> tensor_values;
};

int main() {
    using json=nlohmann::json;
    const json annotation={{"entities",json::array({{{"id","box"},{"bbox",{0.,0.,1.,1.}}}})}};
    // Reproduce the training preview path with truncated RGB and real-image
    // payloads. Previously the annotation painter read/wrote past these buffers.
    for(size_t count:{size_t(1),size_t(3),size_t(11),size_t(13)}) {
        Frame frame;frame.pixels.assign(count,127);frame.pixels_real.assign(count,127);
        std::vector<Frame> frames{frame};
        AnnotationFeatureOverlay::decorate(frames,annotation,json::object(),2,2);
        TASSERT_TRUE(frames[0].pixels.empty());
        TASSERT_TRUE(frames[0].pixels_real.empty());
    }
    for(int channels:{1,3,4}) {
        Frame frame;frame.channels=channels;frame.pixels.assign(4*channels,127);
        std::vector<Frame> frames{frame};
        AnnotationFeatureOverlay::decorate(frames,annotation,json::object(),2,2);
        TASSERT_TRUE(frames[0].pixels.size()==12);
        TASSERT_TRUE(frames[0].channels==3);
    }
    std::vector<uint8_t> invalid(3,1);
    AnnotationFeatureOverlay::rgb(invalid,0,2,3);
    TASSERT_TRUE(invalid.empty());
    std::cout<<"ANNOTATION_FEATURE_BOUNDS_OK\n";
}
