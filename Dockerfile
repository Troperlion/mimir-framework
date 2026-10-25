ARG ALPINE_VERSION=3.22

FROM alpine:${ALPINE_VERSION} AS builder

ARG BUILD_JOBS=2

RUN apk add --no-cache \
        build-base \
        ca-certificates \
        cmake \
        ninja-build \
        ninja-is-really-ninja \
        pkgconf \
        lua5.3-dev \
        lz4-dev \
        ffmpeg-dev

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
        -DLUA_LIBRARY=/usr/lib/lua5.3/liblua.so \
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

FROM alpine:${ALPINE_VERSION} AS runtime

RUN apk add --no-cache \
        bash \
        ca-certificates \
        coreutils \
        ffmpeg-libs \
        libgomp \
        libstdc++ \
        lua5.3-libs \
        lz4-libs \
    && addgroup -S -g 10001 mimir \
    && adduser -S -D -H -u 10001 -G mimir mimir

WORKDIR /workspace

COPY --from=builder /src/container-bin/mimir /usr/local/bin/mimir
COPY run_mimir.sh /usr/local/bin/run_mimir.sh
RUN chmod 755 /usr/local/bin/run_mimir.sh && chown mimir:mimir /workspace

USER mimir

ENV MIMIR_EXECUTABLE=/usr/local/bin/mimir \
    OMP_NUM_THREADS=2 \
    MIMIR_DISABLE_CUDA=1 \
    MIMIR_DISABLE_ROCM=1 \
    MIMIR_DISABLE_VULKAN=1 \
    MIMIR_DISABLE_OPENCL=1

ENTRYPOINT ["/bin/sh", "-c", "if [ -n \"${SCRIPT:-}\" ]; then exec /usr/local/bin/run_mimir.sh --lua \"$SCRIPT\" \"$@\"; else exec /usr/local/bin/run_mimir.sh \"$@\"; fi", "mimir"]
