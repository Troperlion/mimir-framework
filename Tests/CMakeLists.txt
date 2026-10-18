cmake_minimum_required(VERSION 3.15)

function(mimir_add_test test_name src_file)
    add_executable(${test_name} ${src_file} test_log_stub.cpp)
    target_compile_features(${test_name} PRIVATE cxx_std_17)
    target_link_libraries(${test_name} PRIVATE mimir_core)
    add_test(NAME ${test_name} COMMAND ${test_name})
    set_tests_properties(${test_name} PROPERTIES TIMEOUT 15)
endfunction()

mimir_add_test(LRDecayTest.Warmup               test_lr_decay_warmup.cpp)
mimir_add_test(LRDecayTest.Boundaries           test_lr_decay_boundaries.cpp)
mimir_add_test(SerializationTest.RoundTrip      test_serialization_roundtrip.cpp)
mimir_add_test(SerializationTest.Deterministic  test_serialization_deterministic.cpp)
mimir_add_test(SerializationTest.CorruptedInput test_serialization_corrupted.cpp)

mimir_add_test(RegistryTest.CfgFromConfig        test_registry_cfg_from_config.cpp)
mimir_add_test(RegistryTest.CreateFromConfig     test_registry_create_from_config.cpp)
mimir_add_test(LuaTest.CreateFromConfig          test_lua_create_from_config.cpp)

mimir_add_test(FrameworkTest.TokenizerBasics      test_tokenizer_basics.cpp)
mimir_add_test(FrameworkTest.TokenizerBPE         test_tokenizer_bpe.cpp)
mimir_add_test(FrameworkTest.EncoderBasics        test_encoder_basics.cpp)
mimir_add_test(FrameworkTest.MemoryGuard          test_memory_guard.cpp)
mimir_add_test(FrameworkTest.OverrideParsing      test_override_parsing.cpp)
mimir_add_test(RegistryTest.Aliases               test_registry_aliases.cpp)
mimir_add_test(LuaTest.ApiBasics                  test_lua_api_basics.cpp)
set_tests_properties(LuaTest.ApiBasics PROPERTIES
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
)
mimir_add_test(LuaTest.ModelDType                 test_lua_dtype_api.cpp)
mimir_add_test(LuaTest.CreateFromConfigErrors     test_lua_create_from_config_errors.cpp)
add_test(
    NAME LuaTest.CausalLMTokenizer
    COMMAND $<TARGET_FILE:mimir> --lua scripts/tests/test_causal_lm_tokenizer_bootstrap.lua
)
set_tests_properties(LuaTest.CausalLMTokenizer PROPERTIES
    TIMEOUT 15
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
)

mimir_add_test(SerializationTest.SafeTensorsModel test_serialization_safetensors_model.cpp)
mimir_add_test(SerializationTest.SafeTensorsCompliance test_safetensors_compliance.cpp)
mimir_add_test(SerializationTest.RawFolderModel   test_serialization_rawfolder_model.cpp)
mimir_add_test(SerializationTest.DebugJsonModel    test_serialization_debug_json_model.cpp)
mimir_add_test(SerializationTest.StrictMissingAssets test_serialization_strict_missing_assets.cpp)

# Bridge API multi-runtime
mimir_add_test(BridgeTest.API test_bridge_api.cpp)
set_tests_properties(BridgeTest.API PROPERTIES TIMEOUT 120)
if(ENABLE_SCRIPTING_JS)
    find_program(MIMIR_JS_TEST_NODE node)
    if(MIMIR_JS_TEST_NODE)
        add_test(NAME BridgeTest.JSConf
            COMMAND ${CMAKE_COMMAND} -DMIMIR_BIN=$<TARGET_FILE:mimir>
                -P ${CMAKE_CURRENT_SOURCE_DIR}/test_js_conf.cmake)
        add_test(NAME BridgeTest.JSSmoke
            COMMAND $<TARGET_FILE:mimir> --js scripts/examples/bridge_smoke.js)
        set_tests_properties(BridgeTest.JSConf BridgeTest.JSSmoke PROPERTIES
            TIMEOUT 15 WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
    endif()
endif()


mimir_add_test(FrameworkTest.TokenizerJsonControlChars test_tokenizer_json_control_chars.cpp)
mimir_add_test(FrameworkTest.OverrideErrors           test_override_errors.cpp)

mimir_add_test(RuntimeTest.NamedRouting              test_layers_named_routing.cpp)
mimir_add_test(RuntimeTest.EmbeddingIntInputs         test_runtime_embedding_int_inputs.cpp)
mimir_add_test(RuntimeTest.ResidualMerge              test_runtime_residual_merge.cpp)
mimir_add_test(RuntimeTest.AllLayerTypes              test_layers_all_types.cpp)
mimir_add_test(ModelTest.WeightBlocks                test_model_weight_blocks.cpp)
mimir_add_test(ModelTest.TrainStepFreeze             test_model_train_step_freeze.cpp)
mimir_add_test(ModelTest.TrainStepDispatch           test_model_train_step_dispatch.cpp)
mimir_add_test(RuntimeTest.ReconstructionLoss test_reconstruction_runtime.cpp)
set_tests_properties(RuntimeTest.ReconstructionLoss PROPERTIES TIMEOUT 180)
foreach(backend OPENCL VULKAN CUDA ROCM)
    if(ENABLE_${backend})
        add_test(NAME RuntimeTest.${backend}ReconstructionLoss COMMAND RuntimeTest.ReconstructionLoss ${backend})
        set_tests_properties(RuntimeTest.${backend}ReconstructionLoss PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 180 LABELS "GPU;${backend}")
    endif()
endforeach()
add_test(NAME RuntimeTest.TPUReconstructionLoss COMMAND RuntimeTest.ReconstructionLoss TPU)
set_tests_properties(RuntimeTest.TPUReconstructionLoss PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 1800 LABELS "TPU")
mimir_add_test(RuntimeTest.OptimizerDispatch test_optimizer_runtime.cpp)
set_tests_properties(RuntimeTest.OptimizerDispatch PROPERTIES TIMEOUT 60)
foreach(backend OPENCL VULKAN CUDA ROCM)
    if(ENABLE_${backend})
        add_test(NAME RuntimeTest.${backend}Optimizers COMMAND RuntimeTest.OptimizerDispatch ${backend})
        set_tests_properties(RuntimeTest.${backend}Optimizers PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 180 LABELS "GPU;${backend}")
    endif()
endforeach()
mimir_add_test(ModelTest.ExtendedOptimizers           test_optimizers_extended.cpp)

mimir_add_test(FrameworkTest.DType                   test_dtype.cpp)
mimir_add_test(FrameworkTest.PlannerLifetimes        test_planner_lifetimes.cpp)
mimir_add_test(FrameworkTest.PlannerFusionAndScratch test_planner_fusion_and_scratch.cpp)
mimir_add_test(FrameworkTest.RuntimeCapabilities    test_runtime_capabilities.cpp)
mimir_add_test(FrameworkTest.ExecutionPlan          test_execution_plan.cpp)
mimir_add_test(FrameworkTest.PlannerEquivalence     test_planner_equivalence.cpp)
mimir_add_test(ModelTest.VAEValidationFeedback     test_vae_validation_feedback.cpp)

mimir_add_test(RuntimeTest.MathLinear      test_runtime_math_linear.cpp)
mimir_add_test(RuntimeTest.MathConv2d     test_runtime_math_conv2d.cpp)
mimir_add_test(RuntimeTest.MathNorms      test_runtime_math_norms.cpp)
mimir_add_test(RuntimeTest.MathAttention  test_runtime_math_attention.cpp)
mimir_add_test(RuntimeTest.LayerOps       test_runtime_layer_ops.cpp)
mimir_add_test(RuntimeTest.PixelShuffle   test_runtime_pixel_shuffle.cpp)
mimir_add_test(RuntimeTest.NMS            test_runtime_nms.cpp)
mimir_add_test(AutogradTest.Numerical     test_autograd_numerical.cpp)
mimir_add_test(AutogradTest.RuntimeDispatch test_runtime_autograd_dispatch.cpp)
mimir_add_test(ModelTest.ReconLosses      test_recon_losses.cpp)
mimir_add_test(ModelTest.VAEConvContract  test_vae_conv_contract.cpp)
mimir_add_test(ModelTest.VAEConvLatentContext test_vae_conv_latent.cpp)
mimir_add_test(ModelTest.LumenDiffusionStructure test_lumen_diffusion_structure.cpp)
mimir_add_test(VizTest.TextPayload             test_viz_text_payload.cpp)
mimir_add_test(MetricsTest.AsyncHistory test_async_monitor_history.cpp)
mimir_add_test(MetricsTest.HtopCsv             test_htop_metrics_csv.cpp)
if(NOT MIMIR_VIZ_BACKEND STREQUAL "NONE")
    mimir_add_test(VizTest.MetricsCsv          test_viz_metrics_csv.cpp)
endif()
mimir_add_test(ModelTest.VGG16FeatEncNorm test_vgg16_feat_enc_norm.cpp)
mimir_add_test(ModelTest.MedicalDetector test_medical_detector_model.cpp)

# Le runtime FPGA n'est pas encore composé dans le routeur global. Ces tests
# compilent néanmoins ses composants isolés afin d'éviter que le prototype
# matériel dérive silencieusement.
add_executable(RuntimeTest.FpgaDetection
    test_runtime_fpga_detection.cpp
    test_log_stub.cpp
    ../src/runtimes/fpga/FpgaRuntime.cpp
)
target_compile_features(RuntimeTest.FpgaDetection PRIVATE cxx_std_17)
target_link_libraries(RuntimeTest.FpgaDetection PRIVATE mimir_core)
add_test(NAME RuntimeTest.FpgaDetection COMMAND RuntimeTest.FpgaDetection)
set_tests_properties(RuntimeTest.FpgaDetection PROPERTIES TIMEOUT 15)

add_executable(RuntimeTest.FpgaValidationRunner
    test_fpga_validation_runner.cpp
    test_log_stub.cpp
    ../src/runtimes/fpga/FpgaRuntime.cpp
    ../src/runtimes/fpga/FpgaValidationRunner.cpp
)
target_compile_features(RuntimeTest.FpgaValidationRunner PRIVATE cxx_std_17)
target_link_libraries(RuntimeTest.FpgaValidationRunner PRIVATE mimir_core)
add_test(NAME RuntimeTest.FpgaValidationRunner COMMAND RuntimeTest.FpgaValidationRunner)
set_tests_properties(RuntimeTest.FpgaValidationRunner PROPERTIES TIMEOUT 15)

# Spawn-only regression: no display, SFML, Qt or GTK needed.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_executable(VizTest.HostEnvironment test_viz_host_environment.cpp ../src/viz/ProcessHost.cpp)
    target_include_directories(VizTest.HostEnvironment PRIVATE ../src)
    target_link_libraries(VizTest.HostEnvironment PRIVATE Threads::Threads)
    add_test(NAME VizTest.HostEnvironment COMMAND VizTest.HostEnvironment)
    set_tests_properties(VizTest.HostEnvironment PROPERTIES TIMEOUT 15)
endif()

# Native GPU gradient parity. Missing devices are explicit CTest skips.
if(ENABLE_OPENCL OR ENABLE_VULKAN)
    add_executable(GpuBackwardParity test_gpu_backward_parity.cpp test_log_stub.cpp)
    target_link_libraries(GpuBackwardParity PRIVATE mimir_core)
    foreach(backend OPENCL VULKAN)
        if(ENABLE_${backend})
            string(TOLOWER "${backend}" backend_arg)
            add_test(NAME RuntimeTest.${backend}BackwardParity COMMAND GpuBackwardParity ${backend_arg})
            set_tests_properties(RuntimeTest.${backend}BackwardParity PROPERTIES
                SKIP_RETURN_CODE 77 TIMEOUT 120 WORKING_DIRECTORY "${CMAKE_BINARY_DIR}")
        endif()
    endforeach()
endif()

mimir_add_test(AutogradTest.ModelRuntime test_model_runtime_autograd.cpp)

mimir_add_test(AutogradTest.Convolution test_conv_autograd.cpp)

mimir_add_test(FrameworkTest.DatasetRGB test_dataset_rgb.cpp)
mimir_add_test(FrameworkTest.DatasetAnnotations test_dataset_annotations.cpp)
set_tests_properties(FrameworkTest.DatasetAnnotations PROPERTIES TIMEOUT 60 ENVIRONMENT "OMP_NUM_THREADS=1")

mimir_add_test(SerializationTest.ActiveState test_checkpoint_active_state.cpp)
set_tests_properties(SerializationTest.ActiveState PROPERTIES TIMEOUT 60 ENVIRONMENT "OMP_NUM_THREADS=1")

mimir_add_test(ModelTest.SkipConnections test_skip_connections.cpp)
add_test(NAME ModelTest.SkipConnectionsUnplanned COMMAND ModelTest.SkipConnections)
set_tests_properties(ModelTest.SkipConnections ModelTest.SkipConnectionsUnplanned PROPERTIES TIMEOUT 30)
set_tests_properties(ModelTest.SkipConnections PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1;MIMIR_ENABLE_PLANNER=1")
set_tests_properties(ModelTest.SkipConnectionsUnplanned PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1;MIMIR_ENABLE_PLANNER=0")

mimir_add_test(ModelTest.LiveConfiguration test_live_model_config.cpp)
mimir_add_test(VizTest.ModelGraphSnapshot test_model_graph_snapshot.cpp)
mimir_add_test(VizTest.HtopModelGraph test_htop_model_graph.cpp)
if(UNIX AND NOT APPLE)
    target_link_libraries(VizTest.HtopModelGraph PRIVATE util)
endif()
set_tests_properties(VizTest.HtopModelGraph PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 30)

# Metadata regression runs on CPU-only builders too.
mimir_add_test(RuntimeTest.CudaLinearCapability test_cuda_linear_capability.cpp)
if(NOT ENABLE_CUDA)
    target_sources(RuntimeTest.CudaLinearCapability PRIVATE ../src/runtimes/cuda/CudaRuntime.cpp)
endif()

set_tests_properties(ModelTest.LiveConfiguration PROPERTIES TIMEOUT 30 ENVIRONMENT "OMP_NUM_THREADS=1")

mimir_add_test(FrameworkTest.BBoxSupervision test_bbox_supervision.cpp)
set_tests_properties(FrameworkTest.BBoxSupervision PROPERTIES TIMEOUT 60 ENVIRONMENT "OMP_NUM_THREADS=1")

mimir_add_test(ModelTest.VisionBBox test_vision_bbox.cpp)
set_tests_properties(ModelTest.VisionBBox PROPERTIES TIMEOUT 120 ENVIRONMENT "OMP_NUM_THREADS=1")

# Exercise the shared vector icon offscreen even in a headless framework build.
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(BBOX_BADGE_CAIRO QUIET IMPORTED_TARGET cairo)
    if(BBOX_BADGE_CAIRO_FOUND)
        add_executable(VizTest.ModelGraphLanes test_model_graph_lanes.cpp ../src/viz/SoftwareGraphics.cpp)
        target_compile_features(VizTest.ModelGraphLanes PRIVATE cxx_std_17)
        target_include_directories(VizTest.ModelGraphLanes PRIVATE ../src)
        target_link_libraries(VizTest.ModelGraphLanes PRIVATE PkgConfig::BBOX_BADGE_CAIRO)
        add_test(NAME VizTest.ModelGraphLanes COMMAND VizTest.ModelGraphLanes)
        set_tests_properties(VizTest.ModelGraphLanes PROPERTIES TIMEOUT 15)
        add_executable(VizTest.DatasetBBoxBadge test_viz_dataset_bbox_badge.cpp ../src/viz/SoftwareGraphics.cpp)
        target_compile_features(VizTest.DatasetBBoxBadge PRIVATE cxx_std_17)
        target_include_directories(VizTest.DatasetBBoxBadge PRIVATE ../src)
        target_link_libraries(VizTest.DatasetBBoxBadge PRIVATE PkgConfig::BBOX_BADGE_CAIRO)
        add_test(NAME VizTest.DatasetBBoxBadge COMMAND VizTest.DatasetBBoxBadge)
        set_tests_properties(VizTest.DatasetBBoxBadge PROPERTIES TIMEOUT 15)
    endif()
endif()

# Real event dispatcher with an offscreen window; skip when SFML has no display.
mimir_add_test(VizTest.ButtonEvents test_viz_events.cpp)
set_tests_properties(VizTest.ButtonEvents PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 30)

# Native CUDA parity must execute on a CUDA device; CPU fallback is not a pass.
if(ENABLE_CUDA AND CUDAToolkit_FOUND)
    mimir_add_test(RuntimeTest.CudaNativeOps test_cuda_native_ops.cpp)
    set_tests_properties(RuntimeTest.CudaNativeOps PROPERTIES TIMEOUT 120 LABELS "CUDA;GPU")
endif()

if(ENABLE_OPENCL OR ENABLE_VULKAN)
    mimir_add_test(RuntimeTest.PortableWorkBudget test_portable_work_budget.cpp)
    add_executable(PortableNativeOps test_portable_native_ops.cpp test_log_stub.cpp)
    target_compile_features(PortableNativeOps PRIVATE cxx_std_17)
    target_link_libraries(PortableNativeOps PRIVATE mimir_core)
    foreach(backend OPENCL VULKAN)
        if(ENABLE_${backend})
            add_test(NAME RuntimeTest.${backend}NativeOps COMMAND PortableNativeOps ${backend})
            set_tests_properties(RuntimeTest.${backend}NativeOps PROPERTIES
                SKIP_RETURN_CODE 77 TIMEOUT 180 LABELS "GPU;${backend}" WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
        endif()
    endforeach()
endif()

mimir_add_test(ModelTest.GanStyle test_gan_style_model.cpp)
set_tests_properties(ModelTest.GanStyle PROPERTIES TIMEOUT 60 ENVIRONMENT "OMP_NUM_THREADS=1")

# Exercise real loader/serializer entrypoints against a serial firmware emulator.
find_package(Python3 QUIET COMPONENTS Interpreter)
if(Python3_Interpreter_FOUND AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
    foreach(mode host legacy corrupt-crc)
        if(mode STREQUAL "host")
            set(fpga_test_args "")
        else()
            set(fpga_test_args --${mode})
        endif()
        add_test(NAME RuntimeTest.FpgaIo.${mode}
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/tests/test_fpga_io.py ${fpga_test_args})
        set_tests_properties(RuntimeTest.FpgaIo.${mode} PROPERTIES TIMEOUT 45)
    endforeach()
    find_program(MIMIR_IVERILOG iverilog)
    find_program(MIMIR_VVP vvp)
    if(MIMIR_IVERILOG AND MIMIR_VVP)
        add_test(NAME RuntimeTest.FpgaIo.RTL
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/tests/test_fpga_io_rtl.py)
        set_tests_properties(RuntimeTest.FpgaIo.RTL PROPERTIES TIMEOUT 45)
    endif()
endif()

# Reference fixture has no dependency on JAX. Its output is also consumed by the
# optional Python differential test and the real-TPU runtime test in Colab.
mimir_add_test(RuntimeTest.TpuCpuFixture test_tpu_cpu_fixture.cpp)
set_tests_properties(RuntimeTest.TpuCpuFixture PROPERTIES TIMEOUT 60)
set(MIMIR_TPU_TEST_PYTHON "" CACHE FILEPATH "Python with JAX for TPU reference differential tests")
if(MIMIR_TPU_TEST_PYTHON)
    add_test(NAME RuntimeTest.ReconstructionJaxReference
        COMMAND "${MIMIR_TPU_TEST_PYTHON}" "${CMAKE_CURRENT_SOURCE_DIR}/test_reconstruction_tpu_reference.py"
                $<TARGET_FILE:RuntimeTest.ReconstructionLoss>)
    set_tests_properties(RuntimeTest.ReconstructionJaxReference PROPERTIES TIMEOUT 180 ENVIRONMENT "OMP_NUM_THREADS=1;JAX_PLATFORMS=cpu")
    add_test(NAME RuntimeTest.OptimizerJaxReference
        COMMAND "${MIMIR_TPU_TEST_PYTHON}" "${CMAKE_CURRENT_SOURCE_DIR}/test_optimizer_tpu_reference.py"
                $<TARGET_FILE:RuntimeTest.OptimizerDispatch>)
    set_tests_properties(RuntimeTest.OptimizerJaxReference PROPERTIES TIMEOUT 120 ENVIRONMENT "OMP_NUM_THREADS=1;JAX_PLATFORMS=cpu")
    add_test(NAME RuntimeTest.TpuReference
        COMMAND "${MIMIR_TPU_TEST_PYTHON}" "${CMAKE_CURRENT_SOURCE_DIR}/test_tpu_reference.py"
                $<TARGET_FILE:RuntimeTest.TpuCpuFixture>)
    set_tests_properties(RuntimeTest.TpuReference PROPERTIES TIMEOUT 300 ENVIRONMENT "OMP_NUM_THREADS=1")
endif()
mimir_add_test(RuntimeTest.TpuLifecycle test_tpu_lifecycle.cpp)
mimir_add_test(RuntimeTest.TpuModelTrain test_tpu_model_train.cpp)
set_tests_properties(RuntimeTest.TpuModelTrain PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 300)
