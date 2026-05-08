# Build a minimal Mímir runtime image able to run Lua scripts (incl. API server).
# Multi-stage build to keep runtime smaller.

FROM debian:bookworm-slim AS build

RUN apt-get update \
  && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    pkg-config \
    lua5.3 \
    liblua5.3-dev \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY . /src

# Build headless by default (no SFML/Vulkan/OpenCL/LZ4).
RUN cmake -S . -B build \
      -DCMAKE_BUILD_TYPE=Release \
      -DENABLE_SFML=OFF \
      -DENABLE_VULKAN=OFF \
      -DENABLE_OPENCL=OFF \
      -DENABLE_LZ4=OFF \
      -DMIMIR_ENABLE_TESTS=OFF \
      -DBUILD_MIMIR_STATIC=OFF \
  && cmake --build build -j


FROM debian:bookworm-slim AS runtime

RUN apt-get update \
  && apt-get install -y --no-install-recommends \
    ca-certificates \
    libstdc++6 \
    libgcc-s1 \
    libgomp1 \
    liblua5.3-0 \
    lua5.3 \
    lua-socket \
    lua-sec \
  && rm -rf /var/lib/apt/lists/*

# The compiled binary is copied to /opt/mimir.
WORKDIR /opt/mimir
RUN mkdir -p /opt/mimir/bin
COPY --from=build /src/bin/mimir /opt/mimir/bin/mimir

# Default workdir for bind-mounted repo in docker-compose.
WORKDIR /workspace

CMD ["/opt/mimir/bin/mimir", "--help"]
