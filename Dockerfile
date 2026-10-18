# syntax=docker/dockerfile:1.7

ARG UBUNTU_VERSION=24.04

FROM ubuntu:${UBUNTU_VERSION} AS builder

ARG DEBIAN_FRONTEND=noninteractive
ARG BUILD_JOBS=2

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        ninja-build \
        pkg-config \
        liblua5.3-dev \
        liblz4-dev \
        libavcodec-dev \
        libavformat-dev \
        libavutil-dev \
        libswresample-dev \
        libswscale-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY . .

# Image générique et reproductible : CPU/headless, sans instructions générées
# pour le CPU de la machine qui construit l'image. Les backends GPU, SFML et
# bridges externes ont chacun leurs dépendances propres et restent des builds
# spécialisés hors de cette image de base.
RUN cmake -S . -B build-container -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_FLAGS_RELEASE="-O3 -DNDEBUG" \
        -DMIMIR_RUNTIME_OUTPUT_DIRECTORY=/src/container-bin \
        -DMIMIR_ENABLE_TESTS=OFF \
        -DENABLE_SIMD=ON \
        -DENABLE_OPENMP=ON \
        -DENABLE_VULKAN=OFF \
        -DENABLE_OPENCL=OFF \
        -DENABLE_CUDA=OFF \
        -DENABLE_ROCM=OFF \
        -DENABLE_FFMPEG=ON \
        -DENABLE_SFML=OFF \
        -DENABLE_LZ4=ON \
        -DENABLE_SCRIPTING_REST=OFF \
        -DENABLE_SCRIPTING_RUST=OFF \
        -DENABLE_SCRIPTING_JS=OFF \
        -DENABLE_SCRIPTING_CSHARP=OFF \
    && cmake --build build-container --target mimir --parallel "${BUILD_JOBS}" \
    && /src/container-bin/mimir --version

FROM ubuntu:${UBUNTU_VERSION} AS runtime

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        ffmpeg \
        libgomp1 \
        liblua5.3-0 \
        liblz4-1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 10001 mimir \
    && useradd --system --uid 10001 --gid mimir --create-home mimir

WORKDIR /workspace

COPY --from=builder /src/container-bin/mimir /usr/local/bin/mimir
COPY --from=builder --chown=mimir:mimir /src/scripts ./scripts
COPY --from=builder --chown=mimir:mimir /src/configs ./configs
COPY --from=builder --chown=mimir:mimir /src/_archi ./_archi
COPY --from=builder --chown=mimir:mimir /src/assets ./assets
COPY --from=builder --chown=mimir:mimir /src/mimir-api.lua ./mimir-api.lua
COPY --from=builder --chown=mimir:mimir /src/VERSION ./VERSION

RUN mkdir -p datasets checkpoint checkpoints logs .mimir-spill \
    && chown -R mimir:mimir /workspace

USER mimir

ENV OMP_NUM_THREADS=2 \
    MIMIR_DISABLE_CUDA=1 \
    MIMIR_DISABLE_ROCM=1 \
    MIMIR_DISABLE_VULKAN=1 \
    MIMIR_DISABLE_OPENCL=1

VOLUME ["/workspace/datasets", "/workspace/checkpoint", "/workspace/checkpoints"]

ENTRYPOINT ["mimir"]
CMD ["--help"]
