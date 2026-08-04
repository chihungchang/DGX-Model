ARG VLLM_OMNI_IMAGE=vllm/vllm-omni:v0.24.0@sha256:566d89d7eaba08dd313dab2464e67fe984825d281c77437582e3008a917bbf7c
FROM ${VLLM_OMNI_IMAGE}

COPY runtime/profile.py /opt/mazi/audio_profile_runtime.py

ENTRYPOINT ["python3", "/opt/mazi/audio_profile_runtime.py", "launch", "tts", "/opt/mazi/profile.args"]
