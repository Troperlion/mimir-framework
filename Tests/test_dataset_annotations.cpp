#include "test_utils.hpp"
#include "Helpers.hpp"
#include "training/BBoxSupervision.hpp"
#include "training/TagAnnotation.hpp"
#include "viz/DatasetBBoxOverlay.hpp"
#include "viz/AnnotationFeatureOverlay.hpp"
#include "runtimes/ops_loss_and_grad.hpp"
#include "Models/Diffusion/LumenLatentDiffusionModel.hpp"
#include "Models/Vision/VAEConvModel.hpp"
#include "scriptings/Lua/luaScripting/LuaScripting.hpp"
#ifdef MIMIR_ENABLE_SCRIPTING_JS
#include "scriptings/JavaScript/jsScripting/JSScripting.hpp"
#endif
#include <chrono>
#include <fstream>

static void writeText(const fs::path& path, const std::string& value) {
    std::ofstream f(path, std::ios::binary); f << value;
}
static void writeImage(const fs::path& path) {
    std::ofstream f(path, std::ios::binary); f << "P6\n8 4\n255\n";
    for (int i = 0; i < 8 * 4; ++i) { const char rgb[] = {char(180), char(90), char(30)}; f.write(rgb, 3); }
}
static DatasetItem& named(std::vector<DatasetItem>& items, const std::string& name) {
    for (auto& item : items) if (item.name == name) return item;
    throw std::runtime_error("Missing dataset item: " + name);
}
static bool rejects(const json& document, const fs::path& path) {
    try { (void)datasetAnnotationCaption(document, path.string()); return false; }
    catch (const std::runtime_error&) { return true; }
}

int main() {
    const json boxes={{"boxes",json::array({{{"label","Mass"}},{{"label","Mass"}},{{"label","Calcification"}}})}};
    TASSERT_TRUE((TagAnnotation::labels(boxes.dump())==std::vector<std::string>{"mass","calcification"}));
    TASSERT_TRUE(TagAnnotation::labels("{\"boxes\":[]}").empty());
    const json annotated={{"captions",{{"primary","A cat. Red."}}},{"entities",json::array({{{"phrase","Black cat"}}})}};
    const auto annotation_labels=TagAnnotation::labels("",&annotated);
    TASSERT_TRUE(annotation_labels.front()=="black cat");
    TASSERT_TRUE(std::find(annotation_labels.begin(),annotation_labels.end(),"cat")!=annotation_labels.end());
    TASSERT_TRUE(TagAnnotation::labels(boxes.dump(),nullptr,false).front()=="Mass");

    setTestEnvironment("OMP_NUM_THREADS", "1");
    setTestEnvironment("MIMIR_RUNTIME", "cpu");
    const auto root = fs::temp_directory_path() / ("mimir_annotations_" +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directories(root / "portraits"); fs::create_directories(root / "legacy");
    struct Cleanup { fs::path root; ~Cleanup() { std::error_code ec; fs::remove_all(root, ec); } } cleanup{root};
    const std::string primary = "Une femme près d’une fenêtre.";
    json a = {
        {"image", "photo.ppm"}, {"captions", {{"primary", primary}, {"alternatives", {"Un portrait lumineux."}}}},
        {"entities", {{{"id","person_1"},{"phrase","une femme"},{"role","primary_subject"},{"bbox",{0.1,0.2,0.5,0.9}}},
                       {{"id","window_1"},{"phrase","une fenêtre"},{"role","object"},{"bbox",{0.6,0.0,1.0,1.0}}}}},
        {"attributes", {{{"entity","person_1"},{"property","hair_color"},{"value","black"}}}},
        {"relations", {{{"subject","person_1"},{"predicate","near"},{"object","window_1"}}}},
        {"lighting", {{"description","chaude"},{"source","soleil"},{"target","visage"}}},
        {"composition", {{"description","portrait"}}},
        {"quality", {{"caption_verified",false},{"relations_verified",false}}},
        {"custom_metadata", {{"keep",true}}}
    };
    a["entities"][0]["skeleton"] = {
        {"name","custom"}, {"points",json::array({
            {{"name","head"},{"position",{0.2,0.3}},{"parent",nullptr},{"entities",{"window_1"}}},
            {{"name","hand"},{"position",{0.4,0.6}},{"parent","head"}},
            {{"name","unplaced"},{"position",nullptr},{"parent",nullptr},{"entities",json::array()}}
        })}, {"edges",json::array({{{"from","hand"},{"to","head"},{"label","custom joint"}}})}
    };
    TASSERT_TRUE(datasetAnnotationCaption(a,"photo.ppm")==primary);
    const auto skeleton_prompt=BBoxSupervision::groundedPrompt(primary,&a);
    TASSERT_TRUE(skeleton_prompt.find("head at 0.200,0.300 linked to une fenêtre")!=std::string::npos);
    TASSERT_TRUE(skeleton_prompt.find("custom joint connected to head")!=std::string::npos);
    for(int kind=0;kind<5;++kind) {
        json bad=a;
        if(kind==0)bad["entities"][0]["skeleton"]["points"][0]["position"]={-0.1,0.3};
        if(kind==1)bad["entities"][0]["skeleton"]["points"][1]["name"]="head";
        if(kind==2)bad["entities"][0]["skeleton"]["points"][0]["entities"]={"missing"};
        if(kind==3)bad["entities"][0]["skeleton"]["edges"][0]["to"]="missing";
        if(kind==4)bad["entities"][0]["skeleton"]["points"][1]["parent"]="hand";
        TASSERT_TRUE(rejects(bad,"photo.ppm"));
    }
    const auto overlay_boxes=DatasetBBoxOverlay::fromAnnotation(&a);
    TASSERT_TRUE(overlay_boxes.size()==2);
    // Non-square zoom and actual-size rendering use the sprite dimensions, not its square container.
    const auto fit=DatasetBBoxOverlay::displayed(overlay_boxes[0],30,50,400,200);
    TASSERT_NEAR(fit[0],70,1e-5f); TASSERT_NEAR(fit[1],90,1e-5f);
    TASSERT_NEAR(fit[2],160,1e-5f); TASSERT_NEAR(fit[3],140,1e-5f);
    const auto actual=DatasetBBoxOverlay::displayed(overlay_boxes[0],-10,20,8,4);
    TASSERT_NEAR(actual[0],-9.2f,1e-5f); TASSERT_NEAR(actual[3],2.8f,1e-5f);
    TASSERT_TRUE(DatasetBBoxOverlay::fromAnnotation(nullptr).empty());
    json invalid_overlay={{"entities", {{{"bbox",{0.8,0.2,0.1,0.5}}},{{"bbox",{"bad",0,1,1}}}}}};
    TASSERT_TRUE(DatasetBBoxOverlay::fromAnnotation(&invalid_overlay).empty());
    const auto annotation_path = root / "portraits/photo.json";
    writeImage(root / "portraits/photo.ppm"); writeText(annotation_path, a.dump(2));
    writeText(root / "portraits/photo.txt", "Ancienne description à remplacer.");
    writeImage(root / "legacy/photo.ppm"); writeText(root / "legacy/photo.txt", "Description historique.");
    writeImage(root / "generic.ppm"); writeText(root / "generic.json", "{\"text\":\"legacy JSON\"}");
    writeImage(root / "image_only.ppm");
    writeText(root / ".annotation_state.json", "{\"annotated\":[]}");
    writeText(root / ".annotation_state.lock", ""); writeText(root / ".annotation-tmp-test.json", "bad JSON");
    const auto cache_path = root / "local-index.json";

    auto& memory = DatasetMemoryManager::instance(); memory.setMaxRAM(64 * 1024 * 1024);
    const auto baseline = memory.getCurrentRAM();
    auto items = loadDataset(root.string(), 8, 4, 2);
    TASSERT_TRUE(items.size() == 3);
    auto& portrait = named(items, "portraits/photo");
    TASSERT_TRUE(fs::path(portrait.annotation_file) == annotation_path);
    TASSERT_TRUE(portrait.text_file == annotation_path.string());
    TASSERT_TRUE(!portrait.text && !portrait.annotation && !portrait.img_loaded);
    TASSERT_TRUE(portrait.countModalities() == 2);
    TASSERT_TRUE(portrait.loadText() && *portrait.text == primary && *portrait.annotation == a);
    TASSERT_TRUE(portrait.loadImageRGB(8, 4));
    TASSERT_TRUE(portrait.img.size() == 8 * 4 * 3 && portrait.img_c == 3);
    TASSERT_TRUE(memory.getCurrentRAM() > baseline);
    portrait.unload(); TASSERT_TRUE(!portrait.text && !portrait.annotation);
    TASSERT_TRUE(memory.getCurrentRAM() == baseline);
    TASSERT_TRUE(named(items, "legacy/photo").loadText());
    TASSERT_TRUE(*named(items, "legacy/photo").text == "Description historique.");
    named(items, "legacy/photo").unload();
    TASSERT_TRUE(named(items, "generic").loadText());
    TASSERT_TRUE(*named(items, "generic").text == "{\"text\":\"legacy JSON\"}");
    named(items, "generic").unload();

    // Cache hits preserve the sidecar path but keep the document lazy.
    auto cached = loadDatasetCached(root.string(), 8, 4, 2, cache_path.string());
    TASSERT_TRUE(cached.size() == 3);
    auto hit = loadDatasetCached(root.string(), 8, 4, 2, cache_path.string());
    TASSERT_TRUE(hit.size() == 3 && !named(hit,"portraits/photo").annotation);
    TASSERT_TRUE(named(hit,"portraits/photo").loadText());
    TASSERT_TRUE(*named(hit,"portraits/photo").text == primary);
    named(hit,"portraits/photo").unload();
    json cache_document;
    { std::ifstream f(cache_path); f >> cache_document; }
    TASSERT_TRUE(cache_document["version"]=="1.2");
    cache_document["version"]="1.1"; writeText(cache_path,cache_document.dump());
    TASSERT_TRUE(!DatasetCache::load(cache_path,cache_document["dataset_hash"].get<std::string>()));
    auto any = loadDatasetCached(root.string(), 4, 4, 1, cache_path.string());
    TASSERT_TRUE(any.size() == 4 && named(any,"portraits/photo").w == 4);
    const auto hash_before = datasetSHA256Hash(root.string(), cache_path.string());
    writeText(root / ".annotation_state.json", "{\"annotated\":[\"portraits/photo.ppm\"],\"last_image\":\"image_only.ppm\"}");
    TASSERT_TRUE(datasetSHA256Hash(root.string(), cache_path.string()) == hash_before);
    const auto old_time = fs::last_write_time(annotation_path);
    a["captions"]["primary"] = "Une fille près d’une fenêtre.";
    const auto old_size = fs::file_size(annotation_path); writeText(annotation_path, a.dump(2));
    fs::last_write_time(annotation_path, old_time);
    TASSERT_TRUE(fs::file_size(annotation_path) == old_size);
    TASSERT_TRUE(datasetSHA256Hash(root.string(), cache_path.string()) != hash_before);
    auto refreshed = loadDatasetCached(root.string(), 8, 4, 2, cache_path.string());
    TASSERT_TRUE(named(refreshed,"portraits/photo").loadText());
    TASSERT_TRUE(*named(refreshed,"portraits/photo").text == a["captions"]["primary"]);
    named(refreshed,"portraits/photo").unload();

    json invalid = a; invalid["entities"][0]["bbox"][0] = -0.1;
    TASSERT_TRUE(rejects(invalid,root/"portraits/photo.ppm"));
    invalid = a; invalid["relations"][0]["object"] = "missing";
    TASSERT_TRUE(rejects(invalid,root/"portraits/photo.ppm"));
    invalid = a; invalid["image"] = "another.ppm";
    TASSERT_TRUE(rejects(invalid,root/"portraits/photo.ppm"));
    invalid = a; invalid["captions"]["primary"] = "  ";
    TASSERT_TRUE(rejects(invalid,root/"portraits/photo.ppm"));
    writeText(annotation_path, invalid.dump(2));
    bool failed = false;
    try { (void)loadDataset(root.string(),8,4,2); } catch (const std::runtime_error&) { failed = true; }
    TASSERT_TRUE(failed); // Never fall back to stale .txt when annotation is invalid.
    TASSERT_TRUE(!portrait.loadText() && !portrait.annotation && !portrait.text && !portrait.annotation_error.empty());
    writeText(annotation_path,a.dump(2));

    LuaScripting lua; lua.setString("ANNOTATION_ROOT",root.string());
    lua.setString("ANNOTATION_CONTEXT",BBoxSupervision::groundedPrompt(a["captions"]["primary"].get<std::string>(),&a));
    lua.setString("ANNOTATION_PRIMARY",a["captions"]["primary"].get<std::string>());
    TASSERT_TRUE(lua.executeScript(R"(
        local ok, n = Mimir.Dataset.load(ANNOTATION_ROOT, 8, 4, 2)
        assert(ok and n == 3)
        ANNOTATION_INDEX = nil
        for i = 1, n do
            local item, err = Mimir.Dataset.get(i, false)
            assert(item, err)
            if item.annotation then
                ANNOTATION_INDEX = i
                assert(item.text == ANNOTATION_PRIMARY)
                assert(item.annotation.captions.primary == item.text)
                assert(Mimir.Model.conditioning_prompt(item.text,item.annotation)==ANNOTATION_CONTEXT)
                assert(Mimir.Model.conditioning_prompt(item.text)==item.text)
                assert(item.annotation.entities[1].bbox[1] == 0.1)
                assert(item.annotation.relations[1].object == 'window_1')
                assert(item.annotation.lighting.source == 'soleil')
                assert(item.annotation.custom_metadata.keep == true)
                assert(item.annotation.entities[1].skeleton.points[1].entities[1] == 'window_1')
                assert(item.annotation.entities[1].skeleton.edges[1].label == 'custom joint')
                assert(item.image == nil)
                assert(Mimir.Dataset.release(i))
                item = assert(Mimir.Dataset.get(i))
                assert(#item.image == 8 * 4 * 3 and item.text == ANNOTATION_PRIMARY)
                assert(Mimir.Dataset.release(i))
            end
        end
        assert(ANNOTATION_INDEX)
    )"));
#ifdef MIMIR_ENABLE_SCRIPTING_JS
    JSScripting js;
    TASSERT_TRUE(js.executeScript(
        "const [ok,n] = Mimir.Dataset.load(" + json(root.string()).dump() + R"(,8,4,2);
        if (!ok || n !== 3) throw new Error('JS dataset load');
        let found = false;
        for (let i=1; i<=n; ++i) {
            const item = Mimir.Dataset.get(i,false);
            if (item.annotation) {
                found = true;
                if (item.text !== item.annotation.captions.primary ||
                    !Array.isArray(item.annotation.entities) ||
                    item.annotation.entities[0].bbox[0] !== 0.1 ||
                    item.annotation.relations[0].object !== 'window_1') throw new Error('JS annotation mismatch');
            }
            Mimir.Dataset.release(i);
        }
        if (!found) throw new Error('JS annotation missing');
    )"));
#endif

    // Annotation changed after loading: get reports an error, never raw JSON.
    writeText(annotation_path, invalid.dump(2));
    TASSERT_TRUE(lua.executeScript(R"(
        local item, err = Mimir.Dataset.get(ANNOTATION_INDEX, false)
        assert(item == nil and type(err) == 'string' and err:find('captions.primary'))
    )"));
    writeText(annotation_path,a.dump(2));

    // Real native VAE and DiT steps consume the loader's pixels and primary caption.
    TASSERT_TRUE(portrait.loadText() && portrait.loadImageRGB(8,4));
    VAEConvModel vae; VAEConvModel::Config vc;
    vc.image_w=8; vc.image_h=4; vc.image_c=3; vc.latent_w=4; vc.latent_h=2; vc.latent_c=2;
    vc.base_channels=4; vc.resnet=false; vc.enc_norm="none"; vc.dec_norm="none";
    vc.stochastic_latent=false; vc.text_cond=true; vc.vocab_size=64; vc.seq_len=8; vc.text_d_model=8; vc.proj_dim=8;
    vae.buildFromConfig(vc); vae.allocateParams(); vae.initializeWeights("xavier",123U);
    Tokenizer tokenizer(64); auto ids = tokenizer.tokenizeEnsure(*portrait.text); ids.resize(8,tokenizer.getPadId());
    std::vector<float> pixels; for (auto value : portrait.img) pixels.push_back(value/127.5f-1.f);
    // Distinct regional appearance is necessary to exercise the pair objective.
    for(size_t i=0;i<pixels.size();++i) pixels[i]+=0.2f*std::sin(float(i)*0.3f);
    Optimizer vo; vo.type=OptimizerType::SGD; vo.decay_strategy=LRDecayStrategy::NONE;
    Model::TrainStepRequest request; request.float_inputs["__input__"]=&pixels; request.int_inputs["text_ids"]=&ids;
    request.target=&pixels; request.optimizer=&vo; request.learning_rate=1e-4f;
    request.mode=Model::TrainStepMode::Accumulate;
    auto vresult=vae.trainStep(request); TASSERT_TRUE(vresult && std::isfinite(vresult->loss));
    std::vector<float> old_grad;
    for (const auto& layer:vae.getLayers()) old_grad.insert(old_grad.end(),layer.grad_weights.begin(),layer.grad_weights.end());
    vae.zeroGradients();
    request.annotation=&*portrait.annotation;
    const auto weighted_vae=vae.trainStep(request);
    TASSERT_TRUE(weighted_vae && std::isfinite(weighted_vae->loss));
    TASSERT_TRUE(weighted_vae->metrics.at("bbox_count")==2 && weighted_vae->metrics.at("bbox_loss_weight")==1);
    size_t gi=0; bool gradients_changed=false;
    for (const auto& layer:vae.getLayers()) for(float g:layer.grad_weights) {
        gradients_changed=gradients_changed || std::abs(g-old_grad[gi])>1e-8f; ++gi;
    }
    TASSERT_TRUE(gradients_changed);
    // A UI request must take effect before weights are computed on this very step.
    vae.publishRuntimeConfiguration(&vo);
    std::string live_bbox_error;
    TASSERT_TRUE(vae.runtimeConfiguration()->request("relation_primary_weight","7",live_bbox_error));
    const auto ranked_live=vae.trainStep(request);
    TASSERT_TRUE(ranked_live && ranked_live->metrics.at("relation_loss")>0);
    TASSERT_NEAR(ranked_live->metrics.at("relation_loss"),weighted_vae->metrics.at("relation_loss")*2.f,1e-6f);
    TASSERT_TRUE(vae.runtimeConfiguration()->request("bbox_loss_weight","0",live_bbox_error));
    const auto disabled_bbox=vae.trainStep(request);
    TASSERT_TRUE(disabled_bbox && disabled_bbox->metrics.at("bbox_loss_weight")==0);
    TASSERT_TRUE(vae.runtimeConfiguration()->request("bbox_loss_weight","2",live_bbox_error));
    const auto doubled_bbox=vae.trainStep(request);
    TASSERT_TRUE(doubled_bbox && doubled_bbox->metrics.at("bbox_loss_weight")==2);
    // Isolate pair coupling from the pixel weights and verify native encoder/decoder gradients.
    const auto relation_weights=BBoxSupervision::makeWeights(&*portrait.annotation,8,4,3,BBoxSupervision::Layout::HWC);
    TASSERT_TRUE(relation_weights.relations.size()==1);
    request.annotation=nullptr;request.loss_weights=&relation_weights.values;
    vae.zeroGradients();const auto pixel_only=vae.trainStep(request);
    std::vector<float> pixel_gradients;
    for(const auto& layer:vae.getLayers()) pixel_gradients.insert(pixel_gradients.end(),layer.grad_weights.begin(),layer.grad_weights.end());
    request.relations=&relation_weights.relations;
    vae.zeroGradients();const auto pair_step=vae.trainStep(request);
    TASSERT_TRUE(pair_step && pair_step->metrics.at("relation_loss")>0 && pair_step->loss>pixel_only->loss);
    gi=0;bool pair_gradients_changed=false;
    for(const auto& layer:vae.getLayers()) for(float gradient:layer.grad_weights) {
        pair_gradients_changed|=std::abs(gradient-pixel_gradients[gi++])>1e-9f;
    }
    TASSERT_TRUE(pair_gradients_changed);
    // Real native decoder features (not RGB bytes or residual colours) feed the
    // annotation correspondence next to its real resdiff capture.
    vae.setVizTapsEnabled(true);vae.setVizTapsLimits(256,32);
    (void)vae.forwardPassNamedView({{"__input__",pixels}},{{"text_ids",ids}},false);
    auto native_taps=vae.consumeVizTaps();
    AnnotationFeatureOverlay::decorate(native_taps,*portrait.annotation,vae.modelConfig,8,4);
    bool native_correspondence=false;
    for(size_t i=0;i<native_taps.size();++i)if(native_taps[i].label.find("annotation_features")!=std::string::npos) {
        native_correspondence=true;
        TASSERT_TRUE(i>0 && native_taps[i-1].label.find("resdiff_abs")!=std::string::npos);
        TASSERT_TRUE(native_taps[i].tensor_info.find("No aligned raw spatial features")==std::string::npos);
        TASSERT_TRUE(!native_taps[i].tensor_values.empty());
        TASSERT_TRUE(i+1<native_taps.size() && native_taps[i+1].label.find("annotation_impact")!=std::string::npos);
    }
    TASSERT_TRUE(native_correspondence);
    vae.setVizTapsEnabled(false);
    LumenLatentDiffusionModel lumen; LumenLatentDiffusionModel::Config lc;
    lc.image_w=8; lc.image_h=4; lc.image_c=3; lc.latent_w=8; lc.latent_h=4; lc.latent_c=3;
    lc.patch_size=4; lc.hidden_size=8; lc.depth=1; lc.mlp_ratio=2; lc.vocab_size=256;
    lc.text_seq_len=256; lc.text_layers=1; lc.num_heads=2;
    json alternate_context=*portrait.annotation;
    alternate_context["entities"][0]["bbox"][0]=0.25;
    tokenizer.ensureVocabFromText(BBoxSupervision::groundedPrompt(*portrait.text,&*portrait.annotation));
    tokenizer.ensureVocabFromText(BBoxSupervision::groundedPrompt(*portrait.text,&alternate_context));
    ids.resize(256,tokenizer.getPadId());
    lumen.buildFromConfig(lc); lumen.allocateParams(); lumen.initializeWeights("xavier",123U); lumen.setTokenizer(tokenizer);
    Optimizer lo; lo.type=OptimizerType::SGD; lo.decay_strategy=LRDecayStrategy::NONE;
    const auto lresult=lumen.trainDiffusionStep(portrait.img,BBoxSupervision::groundedPrompt(*portrait.text,&*portrait.annotation),123U,lo,1e-4f);
    TASSERT_TRUE(std::isfinite(lresult.loss) && std::isfinite(lresult.grad_norm));
    std::vector<float> raw_lumen_grad;
    for(const auto& layer:lumen.getLayers()) raw_lumen_grad.insert(raw_lumen_grad.end(),layer.grad_weights.begin(),layer.grad_weights.end());
    const auto raw_validation=lumen.validateDiffusionStep(portrait.img,*portrait.text,124U);
    const auto bbox_validation=lumen.validateDiffusionStep(portrait.img,*portrait.text,124U,&*portrait.annotation);
    json no_skeleton=*portrait.annotation;no_skeleton["entities"][0].erase("skeleton");
    // Keep text, input and target identical to isolate skeleton-driven gradients.
    std::vector<float> timestep(8,0.1f),diffusion_target(pixels.size(),0.3f);
    Model::TrainStepRequest skeleton_request;
    skeleton_request.float_inputs={{"__input__",&pixels},{"timestep",&timestep}};
    skeleton_request.int_inputs={{"text_ids",&ids}};skeleton_request.target=&diffusion_target;
    skeleton_request.optimizer=&lo;skeleton_request.mode=Model::TrainStepMode::Accumulate;
    skeleton_request.annotation=&no_skeleton;lumen.zeroGradients();
    const auto without_pose=lumen.trainStep(skeleton_request);
    std::vector<float> without_pose_grad;
    for(const auto& layer:lumen.getLayers())without_pose_grad.insert(without_pose_grad.end(),layer.grad_weights.begin(),layer.grad_weights.end());
    skeleton_request.annotation=&*portrait.annotation;lumen.zeroGradients();
    const auto with_pose=lumen.trainStep(skeleton_request);
    TASSERT_TRUE(with_pose && without_pose && with_pose->metrics.at("skeleton_region_count")>0);
    TASSERT_TRUE(std::abs(with_pose->loss-without_pose->loss)>1e-8f);
    gi=0;bool pose_gradient_changed=false;
    for(const auto& layer:lumen.getLayers())for(float g:layer.grad_weights)pose_gradient_changed|=std::abs(g-without_pose_grad[gi++])>1e-9f;
    TASSERT_TRUE(pose_gradient_changed);
    const auto disabled_validation=lumen.validateDiffusionStep(portrait.img,*portrait.text,124U,&*portrait.annotation,0.f);
    TASSERT_TRUE(bbox_validation.skeleton_region_count>0);
    TASSERT_TRUE(bbox_validation.bbox_count==2 && bbox_validation.bbox_loss_weight==1.f);
    const auto grounded_validation=lumen.validateDiffusionStep(portrait.img,BBoxSupervision::groundedPrompt(*portrait.text,&*portrait.annotation),124U);
    TASSERT_NEAR(grounded_validation.loss,bbox_validation.mse_unweighted,1e-6f);
    TASSERT_NEAR(grounded_validation.loss,disabled_validation.loss,1e-6f);
    TASSERT_TRUE(std::abs(raw_validation.loss-disabled_validation.loss)>1e-8f);
    const auto alternate_validation=lumen.validateDiffusionStep(portrait.img,*portrait.text,124U,&alternate_context,0.f);
    TASSERT_TRUE(std::abs(alternate_validation.loss-disabled_validation.loss)>1e-8f);
    LumenLatentDiffusionModel short_context;auto short_lc=lc;short_lc.text_seq_len=8;short_lc.dynamic_text_seq_len=false;
    short_context.buildFromConfig(short_lc);short_context.setTokenizer(tokenizer);
    bool rejected_truncation=false;
    try {(void)short_context.validateDiffusionStep(portrait.img,*portrait.text,124U,&*portrait.annotation,0.f);}
    catch(const std::invalid_argument& e) {rejected_truncation=std::string(e.what()).find("text_seq_len")!=std::string::npos;}
    TASSERT_TRUE(rejected_truncation);
    short_lc.dynamic_text_seq_len=true;
    short_context.buildFromConfig(short_lc);short_context.setTokenizer(tokenizer);
    short_context.allocateParams();short_context.initializeWeights("xavier",321U);
    const auto dynamic_validation=short_context.validateDiffusionStep(portrait.img,*portrait.text,124U,&*portrait.annotation,0.f);
    TASSERT_TRUE(std::isfinite(dynamic_validation.loss));
    TASSERT_TRUE(short_context.getTensor("lumen/text/context").size()==
        tokenizer.tokenize(BBoxSupervision::groundedPrompt(*portrait.text,&*portrait.annotation)).size()*size_t(lc.hidden_size));
    TASSERT_TRUE(short_context.getLayerByName("lumen/text/position")->getWeightsSize()==size_t(8*lc.hidden_size));

    TASSERT_TRUE(std::abs(raw_validation.loss-bbox_validation.loss)>1e-7f);
    lumen.initializeWeights("xavier",123U);
    const auto weighted_train=lumen.trainDiffusionStep(portrait.img,*portrait.text,123U,lo,1e-4f,&*portrait.annotation);
    TASSERT_TRUE(weighted_train.bbox_count==2 && weighted_train.relation_count==1 && weighted_train.relation_loss>0 && weighted_train.mse_unweighted>0 && std::isfinite(weighted_train.grad_norm));
    TASSERT_NEAR(weighted_train.mse_unweighted,lresult.loss,1e-6f);
    gi=0; gradients_changed=false;
    for(const auto& layer:lumen.getLayers()) for(float g:layer.grad_weights) {
        gradients_changed=gradients_changed || std::abs(g-raw_lumen_grad[gi])>1e-8f; ++gi;
    }
    TASSERT_TRUE(gradients_changed);
    TASSERT_TRUE(lua.executeScript(R"(
        assert(Mimir.Model.create('lumen_diffusion', {
            image_w=8,image_h=4,image_c=3,latent_w=8,latent_h=4,latent_c=3,
            patch_size=4,hidden_size=8,depth=1,mlp_ratio=2,vocab_size=256,
            text_seq_len=256,text_layers=1,num_heads=2
        }))
        assert(Mimir.Model.allocate_params())
        assert(Mimir.Model.init_weights('xavier',123))
        local item=assert(Mimir.Dataset.get(ANNOTATION_INDEX))
        local stats,err=Mimir.Model.lumen_train_step(item.image,item.text,123,0.0001,'sgd',{
            annotation=item.annotation,bbox_loss_weight=1
        })
        assert(stats,err)
        assert(stats.bbox_count==2 and stats.bbox_loss_weight==1 and stats.mse_unweighted>0)
        assert(stats.skeleton_region_count>0)
        assert(stats.relation_count==1 and stats.relation_loss>0)
        local val,err=Mimir.Model.lumen_validate_step(item.image,item.text,124,{annotation=item.annotation})
        assert(val,err)
        assert(val.bbox_count==2 and val.bbox_loss_weight==1 and val.skeleton_region_count>0)
    )"));
    // Actual conditional VAE loop consumes the same context even with loss weight zero.
    TASSERT_TRUE(lua.executeScript(R"(
        assert(Mimir.Dataset.load(ANNOTATION_ROOT .. '/portraits',8,4,1))
        assert(Mimir.Model.create('vae_conv', {
            image_w=8,image_h=4,image_c=3,latent_w=4,latent_h=2,latent_c=2,
            base_channels=4,resnet=false,enc_norm='none',dec_norm='none',
            text_cond=true,seq_len=256,vocab_size=256,text_d_model=8,proj_dim=8,
            bbox_loss_weight=0,stochastic_latent=false,max_items=1,autosave_every_epochs=0,
            validate_every_steps=1,validate_items=1,validate_holdout=false,
            csv_file=ANNOTATION_ROOT .. '/vae_context_metrics.csv'
        }))
        assert(Mimir.Model.allocate_params())
        assert(Mimir.Model.init_weights('xavier',123))
        assert(Mimir.Tokenizer.create(256))
        local ok,err=Mimir.Model.train(1,0.0001)
        assert(ok,err)
        local context = assert(Mimir.Model.conditioning_prompt(ANNOTATION_PRIMARY, assert(Mimir.Dataset.get(1)).annotation))
        local ids = assert(Mimir.Tokenizer.tokenize(context))
        for _, id in ipairs(ids) do assert(id ~= Mimir.Tokenizer.unk_id(), 'Context token not hydrated') end
        local merged = assert(Mimir.Tokenizer.tokenize('connectedto'))
        for _, id in ipairs(merged) do assert(id ~= Mimir.Tokenizer.unk_id(), 'BPE pair not learned') end
    )"));
    // Oversized absolute holdout must preserve the configured training fraction.
    fs::create_directories(root / "holdout");
    for (int i=0; i<10; ++i) {
        const std::string stem="sample_"+std::to_string(i);
        writeImage(root / "holdout" / (stem+".ppm"));
        json annotation=a; annotation["image"]=stem+".ppm";
        writeText(root / "holdout" / (stem+".json"),annotation.dump());
    }
    TASSERT_TRUE(lua.executeScript(R"(
        assert(Mimir.Dataset.load(ANNOTATION_ROOT .. '/holdout',8,4,1))
        assert(Mimir.Model.create('vae_conv', {
            image_w=8,image_h=4,image_c=3,latent_w=4,latent_h=2,latent_c=2,
            base_channels=4,resnet=false,enc_norm='none',dec_norm='none',
            text_cond=false,stochastic_latent=false,autosave_every_epochs=0,
            validate_every_steps=1000,validate_items=100,validate_holdout=true,
            validate_holdout_items=120,validate_holdout_frac=0.2,
            csv_file=ANNOTATION_ROOT .. '/holdout_metrics.csv'
        }))
        assert(Mimir.Model.allocate_params())
        assert(Mimir.Model.init_weights('xavier',123))
        local ok,steps=Mimir.Model.train(2,0.0001)
        assert(ok,steps)
        assert(steps==16,'expected eight training images per epoch, got '..tostring(steps))
    )"));
    // An undecodable image must not produce an empty epoch checkpoint.
    fs::create_directories(root / "invalid");
    writeText(root / "invalid/broken.ppm", "invalid image");
    TASSERT_TRUE(lua.executeScript(R"(
        assert(Mimir.Dataset.load(ANNOTATION_ROOT .. '/invalid',8,4,1))
        assert(Mimir.Model.create('vae_conv', {
            image_w=8,image_h=4,image_c=3,latent_w=4,latent_h=2,latent_c=2,
            base_channels=4,resnet=false,enc_norm='none',dec_norm='none',
            text_cond=false,autosave_every_epochs=1,validate_every_steps=0,
            checkpoint_dir=ANNOTATION_ROOT .. '/empty_checkpoints',
            csv_file=ANNOTATION_ROOT .. '/invalid_metrics.csv'
        }))
        assert(Mimir.Model.allocate_params())
        assert(Mimir.Model.init_weights('xavier',123))
        local ok,err=Mimir.Model.train(1,0.0001)
        assert(not ok and tostring(err):find('0 step'),tostring(err))
    )"));
    TASSERT_TRUE(!fs::exists(root / "empty_checkpoints/epoch_0001"));
    // Actual Lua VAE loop must discover boxes even without text conditioning.
    TASSERT_TRUE(lua.executeScript(R"(
        assert(Mimir.Dataset.load(ANNOTATION_ROOT .. '/portraits',8,4,1))
        assert(Mimir.Model.create('vae_conv', {
            image_w=8,image_h=4,image_c=3,latent_w=4,latent_h=2,latent_c=2,
            base_channels=4,resnet=false,enc_norm='none',dec_norm='none',
            text_cond=false,stochastic_latent=false,max_items=1,autosave_every_epochs=0,
            validate_every_steps=1,validate_items=1,validate_holdout=false,
            csv_file=ANNOTATION_ROOT .. '/vae_metrics.csv'
        }))
        assert(Mimir.Model.allocate_params())
        assert(Mimir.Model.init_weights('xavier',123))
        local ok,err=Mimir.Model.train(1,0.0001)
        assert(ok,err)
        local item=assert(Mimir.Dataset.get(1))
        local input={}
        for i=1,#item.image do input[i]=item.image[i]/127.5-1 end
        local stats,err=Mimir.Model.vision_train_step(input,input,{
            annotation=item.annotation,optimizer='sgd',learning_rate=0.0001
        })
        assert(stats,err)
        assert(stats.bbox_count==2 and stats.bbox_loss_weight==1)
    )"));
    TASSERT_TRUE(lua.executeScript(R"(
        assert(Mimir.Model.create('unet', {
            image_w=8,image_h=4,image_c=3,base_channels=4,depth=1
        }))
        assert(Mimir.Model.allocate_params())
        assert(Mimir.Model.init_weights('xavier',123))
        local item=assert(Mimir.Dataset.get(1))
        local input,target={},{}
        for c=1,3 do for p=0,31 do
            input[#input+1]=item.image[p*3+c]/127.5-1
            target[#target+1]=0
        end end
        local stats,err=Mimir.Model.vision_train_step(input,target,{
            annotation=item.annotation,optimizer='sgd',learning_rate=0.0001
        })
        assert(stats,err)
        assert(stats.bbox_count==2 and stats.bbox_loss_weight==1)
    )"));
    bool bbox_logged=false;
    for(const auto& message:LuaContext::getInstance().logs)
        bbox_logged=bbox_logged || message.find("BBox reconstruction supervision: boxes=2 weight=1")!=std::string::npos;
    TASSERT_TRUE(bbox_logged);
    portrait.unload();
    fs::remove(annotation_path);
    auto fallback=loadDataset(root.string(),8,4,2);
    auto& old=named(fallback,"portraits/photo"); TASSERT_TRUE(old.annotation_file.empty() && old.loadText());
    TASSERT_TRUE(*old.text == "Ancienne description à remplacer."); old.unload();
    // Lexical sidecar pairing also works when the image is a symlink.
    fs::create_directories(root / "links");
    fs::create_symlink(root / "portraits/photo.ppm", root / "links/photo.ppm");
    writeText(root / "links/photo.json", a.dump(2));
    auto links = loadDataset((root/"links").string(),8,4,2);
    TASSERT_TRUE(links.size()==1 && links[0].name=="photo" && links[0].loadText());
    TASSERT_TRUE(*links[0].text==a["captions"]["primary"] && links[0].loadImageRGB(8,4));
    links[0].unload();

    // COCO remains compatible, and a local annotation overrides its captions.
    const auto coco_root = root / "coco";
    fs::create_directories(coco_root/"annotations"); fs::create_directories(coco_root/"train2017");
    writeImage(coco_root/"train2017/sample.ppm");
    const json coco_doc={{"images",{{{"id",1},{"file_name","sample.ppm"}}}},
                         {"annotations",{{{"image_id",1},{"caption","Historical COCO caption."}}}}};
    writeText(coco_root/"annotations/captions_train2017.json",coco_doc.dump());
    json coco_annotation=a; coco_annotation["image"]="sample.ppm";
    writeText(coco_root/"train2017/sample.json",coco_annotation.dump());
    auto coco=loadDataset(coco_root.string(),8,4,2);
    TASSERT_TRUE(coco.size()==1 && coco[0].loadText() && *coco[0].text==a["captions"]["primary"]);
    TASSERT_TRUE(*coco[0].annotation==coco_annotation); coco[0].unload();
    fs::remove(coco_root/"train2017/sample.json");
    auto coco_old=loadDataset(coco_root.string(),8,4,2);
    TASSERT_TRUE(coco_old.size()==1 && coco_old[0].annotation_file.empty() && coco_old[0].loadText());
    TASSERT_TRUE(json::parse(*coco_old[0].text)["text"]=="Historical COCO caption.");
    coco_old[0].unload();

    // Release context-owned tensors before static tensor registries are destroyed.
    LuaContext::getInstance().currentModel.reset();
    std::cout << "PASS structured annotations: native/Lua, JSON priority, historical fallback, cache, validation, VAE/DiT training\n";
    return 0;
}
