# Pin the validated llama.cpp release so CUDA, multimodal, and speculative
# decoding behavior remain reproducible across model profiles.
FROM nvidia/cuda:13.0.0-devel-ubuntu24.04

ARG LLAMA_CPP_COMMIT=1464c62d88f699ec9700c8010bbfdbc603a9efd6

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential ca-certificates cmake curl git libssl-dev \
    && rm -rf /var/lib/apt/lists/*

RUN git init /opt/llama.cpp \
    && cd /opt/llama.cpp \
    && git remote add origin https://github.com/ggml-org/llama.cpp.git \
    && git fetch --depth 1 origin "${LLAMA_CPP_COMMIT}" \
    && git checkout --detach FETCH_HEAD \
    && test "$(git rev-parse HEAD)" = "${LLAMA_CPP_COMMIT}"

RUN ln -sf libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1

RUN cmake -S /opt/llama.cpp -B /opt/llama.cpp/build \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CUDA=ON \
        -DGGML_NATIVE=OFF \
        -DLLAMA_OPENSSL=ON \
        -DCMAKE_CUDA_ARCHITECTURES=121 \
        -DCMAKE_EXE_LINKER_FLAGS=-Wl,-rpath-link,/usr/local/cuda/lib64/stubs \
    && cmake --build /opt/llama.cpp/build --target llama-server --parallel "$(nproc)" \
    && install -D /opt/llama.cpp/build/bin/llama-server /app/llama-server
