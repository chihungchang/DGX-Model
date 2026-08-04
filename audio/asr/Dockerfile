ARG VLLM_BASE_IMAGE=vllm/vllm-openai:v0.14.0-cu130@sha256:bb3c2948540aecfe5f65950ec06b3c080139ba37fd4d236c32ab11872039ebe2
FROM ${VLLM_BASE_IMAGE}

WORKDIR /opt/mazi

RUN apt-get update \
    && apt-get install -y --no-install-recommends libsndfile1 sox \
    && apt-get remove -y python3-blinker \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m pip install --no-cache-dir --upgrade-strategy only-if-needed \
        blinker==1.9.0 \
        flask==3.1.3 \
        qwen-asr==0.0.6 \
        gradio==6.10.0 \
        fastapi==0.128.0 \
        starlette==0.50.0 \
        prometheus-fastapi-instrumentator==7.1.0 \
    && python3 -c "import importlib.metadata as m; from fastapi import FastAPI; from qwen_asr import Qwen3ASRModel; from starlette.concurrency import run_in_threadpool; assert m.version('blinker') == '1.9.0'; assert m.version('flask') == '3.1.3'; assert m.version('qwen-asr') == '0.0.6'; assert m.version('gradio') == '6.10.0'; assert m.version('vllm') == '0.14.1.dev1+gd68209402.cu130'; assert m.version('fastapi') == '0.128.0'; assert m.version('starlette') == '0.50.0'; assert m.version('prometheus-fastapi-instrumentator') == '7.1.0'"

COPY runtime/profile.py /opt/mazi/audio_profile_runtime.py
COPY asr/server.py /opt/mazi/asr_server.py

ENTRYPOINT ["python3", "/opt/mazi/audio_profile_runtime.py", "launch", "asr", "/opt/mazi/profile.args"]
