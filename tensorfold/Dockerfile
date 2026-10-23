ARG BASE_IMAGE=nvcr.io/nvidia/pytorch@sha256:2140e699b3beaf7f96a0081fd9c9406bc3832b435cdb60dfa2d261f7d2f34a1c
FROM ${BASE_IMAGE}

ARG TENSORFOLD_COMMIT=07aa02585f64cfb04c60a96de6554c0d4708ed46
RUN git init /opt/tensorfold \
 && git -C /opt/tensorfold remote add origin https://github.com/Elevatormusic/TensorFold.git \
 && git -C /opt/tensorfold fetch --depth 1 origin "$TENSORFOLD_COMMIT" \
 && git -C /opt/tensorfold checkout --detach FETCH_HEAD \
 && test "$(git -C /opt/tensorfold rev-parse HEAD)" = "$TENSORFOLD_COMMIT"
RUN pip freeze | grep -iE "^(torch|torchvision|triton|numpy)==" > /tmp/ngc-pins.txt \
 && pip install --no-cache-dir -c /tmp/ngc-pins.txt -e "/opt/tensorfold[grammar]" "xgrammar==0.2.8" \
 && python -c "import tensorfold.families.deepseek_v41, torch, triton; print(torch.__version__, triton.__version__)"
COPY patches/rdma_active_mtu.py /tmp/rdma_active_mtu.py
RUN python /tmp/rdma_active_mtu.py
COPY patches/shared_context.py /tmp/shared_context.py
RUN python /tmp/shared_context.py
COPY patches/dsml_single_tool.py /tmp/dsml_single_tool.py
RUN python /tmp/dsml_single_tool.py
COPY patches/deepseek_image_context.py /tmp/deepseek_image_context.py
RUN python /tmp/deepseek_image_context.py
COPY patches/output_context_ceiling.py /tmp/output_context_ceiling.py
RUN python /tmp/output_context_ceiling.py
ENV TORCH_EXTENSIONS_DIR=/cache/torch_extensions
WORKDIR /opt/tensorfold
LABEL org.opencontainers.image.source="https://github.com/Elevatormusic/TensorFold" \
      org.opencontainers.image.revision="${TENSORFOLD_COMMIT}" \
      ai.dgx.tensorfold.patch="rdma-active-mtu-v1" \
      ai.dgx.tensorfold.pool-context="independent-v2"

LABEL ai.dgx.tensorfold.dsml-single-tool="v1"

LABEL ai.dgx.tensorfold.image-context="v1"

LABEL ai.dgx.tensorfold.output-ceiling="v1"
