ARG SGLANG_BASE_IMAGE=lmsysorg/sglang@sha256:616a3e97f45191af975896cfa644279096cb31bd408a071c2e99ca7209c3cafe
FROM ${SGLANG_BASE_IMAGE}

ARG SGLANG_BASE_IMAGE

COPY patches/patch_qwen3_5_mrope.py /tmp/patch_qwen3_5_mrope.py
RUN python3 /tmp/patch_qwen3_5_mrope.py && rm /tmp/patch_qwen3_5_mrope.py

LABEL org.opencontainers.image.title="DGX SGLang Qwen3.8 mRoPE fix"
LABEL org.opencontainers.image.description="Pinned SGLang Qwen3.8 runtime with multidimensional mRoPE routed to the native implementation"
LABEL org.opencontainers.image.base.name="${SGLANG_BASE_IMAGE}"
LABEL org.opencontainers.image.source="https://github.com/sgl-project/sglang"
