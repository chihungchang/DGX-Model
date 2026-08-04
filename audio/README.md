# DGX Spark Audio Runtime

This directory deploys the fixed Mazi Live Mode audio stack directly on a
Linux ARM64 DGX Spark. It contains no generic audio gateway and no model
fallback:

- ASR: `Qwen/Qwen3-ASR-1.7B`, `qwen-asr==0.0.6`, vLLM `0.14.0`
- TTS: `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`, vLLM-Omni `0.24.0`
- ASR API: model-specific streaming state at `:8101/api/start`,
  `:8101/api/chunk`, and `:8101/api/finish`
- TTS API: OpenAI-compatible streaming PCM at `:8102/v1/audio/speech`

The images are pinned by multi-architecture manifest digest and Compose selects
`linux/arm64`. `profiles/*.args` is the only model/runtime configuration
source. The parser does not source a shell file, run `eval`, or expand
environment variables.

## Memory budget

`qwen-live.args` is intentionally conservative for a DGX that is already
serving the main LLM:

- ASR vLLM: `gpu_memory_utilization=0.08`
- TTS talker stage: `gpu_memory_utilization=0.08`
- TTS code2wav stage: `gpu_memory_utilization=0.06`
- Combined audio ceiling enforced by the parser: `0.30`

These are initial safety limits, not a claim that the three runtimes fit until
the concurrent smoke test succeeds on the target DGX. Do not raise them without
measuring the existing LLM reservation and live unified-memory headroom. Global
TTS `--gpu-memory-utilization` is deliberately absent because vLLM-Omni would
apply it to both stages sharing the same GPU.

## Static validation (safe on a development Mac)

```bash
cd infra/dgx-audio
bash -n audio-model
python3 -m py_compile runtime/profile.py asr/server.py scripts/smoke.py
./audio-model config qwen-live
./audio-model list
```

This does not start, pull, or deploy a model.

## DGX lifecycle

Copy this directory to the DGX, then use the manager as the lifecycle boundary:

```bash
./audio-model pull qwen-live
./audio-model use qwen-live
./audio-model status
./audio-model logs asr
./audio-model logs tts
./audio-model smoke
```

`use` clears deployment validation. Container health only proves each audio
runtime is responding. The manager reports `healthy/unvalidated` until `smoke`
runs ASR, streaming TTS, and the existing LLM chat request concurrently. Only a
successful three-runtime smoke records the exact profile fingerprint and makes
the manager report `ready`. `smoke --audio-only` is diagnostic and never marks
the deployment ready.

Model weights persist in the Docker volume `mazi-audio-hf-cache`. Ports use host
networking so the selected profile is the only port configuration source.

## Network and Cloud configuration

Allow DGX TCP `8101` and `8102` only from Mazi Cloud nodes on the private LAN;
do not publish them to the Internet. Cloud should use:

```env
QWEN_ASR_URL=http://<dgx-private-ip>:8101
QWEN_TTS_URL=http://<dgx-private-ip>:8102
```

Desktop clients never connect to these ports directly. Cloud retains user and
workspace authentication and calls the model runtimes. The ASR shim owns only
Qwen streaming state, bounded session lifecycle, and liveness/readiness; it is
not an audio policy or routing service.

## Rollback and degradation

There is no CosyVoice, alternate ASR, or alternate TTS fallback. If ASR is not
ready, Live Mode degrades to text input. If TTS is not ready, it keeps visual
output without speech. Stop this deployment with:

```bash
./audio-model stop
```
