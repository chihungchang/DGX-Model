# Use Unsloth's pinned ARM64 CUDA 13 build because it includes the Qwen4Exp
# MTP graph and shared-tensor borrowing required by the Flash Next sidecar.
FROM ubuntu:24.04 AS downloader

ARG LLAMA_RELEASE=b10715-mix-86bd2d3
ARG LLAMA_ARCHIVE_SHA256=ff83d9c33d300ddd21b73ae8b52e37ddf77d5faa5a7c4b264b76fa33efd45e21

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

RUN archive="app-${LLAMA_RELEASE}-linux-arm64-cuda13-portable.tar.gz" \
    && url="https://github.com/unslothai/llama.cpp/releases/download/${LLAMA_RELEASE}/${archive}" \
    && curl --fail --location --retry 3 --output "/tmp/${archive}" "${url}" \
    && printf '%s  %s\n' "${LLAMA_ARCHIVE_SHA256}" "/tmp/${archive}" | sha256sum --check - \
    && mkdir -p /opt/llama.cpp \
    && tar -xzf "/tmp/${archive}" -C /opt/llama.cpp \
    && test -x /opt/llama.cpp/llama-server

FROM nvidia/cuda:13.0.0-runtime-ubuntu24.04

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates curl libcurl4t64 libgomp1 libssl3t64 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=downloader /opt/llama.cpp /opt/llama.cpp

ENV LD_LIBRARY_PATH=/opt/llama.cpp

RUN mkdir -p /app \
    && ln -s /opt/llama.cpp/llama-server /app/llama-server \
    && /app/llama-server --version
