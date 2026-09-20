#!/usr/bin/env python3

import argparse
import contextlib
import importlib.util
import io
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


HELPER_PATH = Path(__file__).with_name("dgx_model_hf.py")
RENDERER_PATH = Path(__file__).with_name("dgx_model_markdown.py")
SPEC = importlib.util.spec_from_file_location("dgx_model_hf", HELPER_PATH)
assert SPEC and SPEC.loader
hf_helper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(hf_helper)
RENDERER_SPEC = importlib.util.spec_from_file_location("dgx_model_markdown", RENDERER_PATH)
assert RENDERER_SPEC and RENDERER_SPEC.loader
markdown_renderer = importlib.util.module_from_spec(RENDERER_SPEC)
RENDERER_SPEC.loader.exec_module(markdown_renderer)


class HuggingFaceProfileTests(unittest.TestCase):
    def test_vllm_profile_can_select_a_pinned_image(self):
        with tempfile.TemporaryDirectory() as directory:
            fake_docker = Path(directory, "docker")
            fake_docker.write_text(
                "#!/usr/bin/env bash\nprintf '%s\\n' \"${VLLM_IMAGE-}\"\n",
                encoding="utf-8",
            )
            fake_docker.chmod(0o755)
            environment = os.environ.copy()
            environment["PATH"] = f"{directory}:{environment['PATH']}"
            environment["VLLM_IMAGE"] = "untrusted/ambient:latest"
            repository = HELPER_PATH.parent.parent
            manager = Path(directory, "dgx-model")
            shutil.copy2(repository / "dgx-model", manager)
            profiles = Path(directory, "vllm", "profiles")
            profiles.mkdir(parents=True)
            shutil.copy2(repository / "vllm" / "compose.yaml", profiles.parent)
            pinned_image = "vllm/vllm-openai@sha256:" + "a" * 64
            (profiles / "muse-glimmer.args").write_text(
                f"# vllm-image: {pinned_image}\n--model\n/models/test\n",
                encoding="utf-8",
            )
            (profiles / "qwen.args").write_text(
                "--model\n/models/test\n", encoding="utf-8"
            )
            muse_result = subprocess.run(
                [str(manager), "compose-config", "vllm", "muse-glimmer"],
                check=True,
                text=True,
                capture_output=True,
                env=environment,
            )
            qwen_result = subprocess.run(
                [str(manager), "compose-config", "vllm", "qwen"],
                check=True,
                text=True,
                capture_output=True,
                env=environment,
            )
            default_result = subprocess.run(
                [str(manager), "image", "pull", "vllm"],
                check=True,
                text=True,
                capture_output=True,
                env=environment,
            )
        self.assertEqual(muse_result.stdout.strip(), pinned_image)
        self.assertEqual(qwen_result.stdout, "\n")
        self.assertEqual(default_result.stdout, "\n")

    def test_markdown_preview_sanitizes_controls_and_honors_terminal_width(self):
        self.assertEqual(
            markdown_renderer.sanitize_terminal_text("safe\x1b[31m\ntext\x7f"),
            "safe[31m\ntext",
        )
        environment = os.environ.copy()
        environment["COLUMNS"] = "48"
        result = subprocess.run(
            ["python3", str(RENDERER_PATH)],
            input="# Preview\n\n- first item\n- second item\n",
            check=True,
            text=True,
            capture_output=True,
            env=environment,
        )
        self.assertIn("Preview", result.stdout)
        self.assertNotIn("# Preview", result.stdout)
        self.assertNotIn("┏", result.stdout)
        self.assertLessEqual(max(map(len, result.stdout.splitlines())), 48)

    def test_search_reports_every_supported_engine(self):
        fixture = [
            {
                "id": "Acme/Hybrid",
                "downloads": 10,
                "likes": 2,
                "pipeline_tag": "text-generation",
                "sha": "a" * 40,
                "tags": ["transformers", "safetensors", "gguf"],
                "siblings": [
                    {"rfilename": "config.json"},
                    {"rfilename": "model.safetensors"},
                    {"rfilename": "model-Q4_K_M.gguf"},
                ],
            },
            {
                "id": "Acme/MLX",
                "downloads": 5,
                "likes": 1,
                "pipeline_tag": "text-generation",
                "sha": "b" * 40,
                "tags": ["transformers", "safetensors", "mlx"],
                "siblings": [
                    {"rfilename": "config.json"},
                    {"rfilename": "model.safetensors"},
                ],
            },
            {
                "id": "Acme/VideoGenerator",
                "downloads": 3,
                "likes": 1,
                "pipeline_tag": "image-text-to-video",
                "sha": "c" * 40,
                "tags": ["transformers", "safetensors"],
                "siblings": [
                    {"rfilename": "config.json"},
                    {"rfilename": "model.safetensors"},
                ],
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            fake_hf = Path(directory, "hf")
            fake_hf.write_text(
                "#!/usr/bin/env python3\nimport json\nprint(json.dumps(" + repr(fixture) + "))\n",
                encoding="utf-8",
            )
            fake_hf.chmod(0o755)
            environment = os.environ.copy()
            environment["PATH"] = f"{directory}:{environment['PATH']}"
            result = subprocess.run(
                [str(HELPER_PATH.parent.parent / "dgx-model"), "search", "--json"],
                check=True,
                text=True,
                capture_output=True,
                env=environment,
            )
            human_result = subprocess.run(
                [str(HELPER_PATH.parent.parent / "dgx-model"), "search"],
                check=True,
                text=True,
                capture_output=True,
                env=environment,
            )
        models = {item["id"]: item["supported_engines"] for item in json.loads(result.stdout)}
        self.assertEqual(models["Acme/Hybrid"], ["vllm", "sglang", "llamacpp"])
        self.assertEqual(models["Acme/MLX"], [])
        self.assertEqual(models["Acme/VideoGenerator"], [])
        self.assertIn("[1] Acme/Hybrid\n", human_result.stdout)
        self.assertIn("    engines: vllm,sglang,llamacpp", human_result.stdout)
        self.assertIn("\n    downloads: 10", human_result.stdout)

    def test_shell_artifact_parser_includes_generated_gguf_shards(self):
        profile = """\
# model-url: https://example.invalid/model.gguf
# model-sha256: {main_sha}
# artifact: shard /models/Acme/Model/model-00002-of-00002.gguf https://example.invalid/shard.gguf {shard_sha}
--model
/models/Acme/Model/model-00001-of-00002.gguf
""".format(main_sha="1" * 64, shard_sha="2" * 64)
        with tempfile.TemporaryDirectory() as directory:
            profile_path = Path(directory, "profile.args")
            profile_path.write_text(profile, encoding="utf-8")
            result = subprocess.run(
                [
                    "bash",
                    "-c",
                    'manager="$1"; profile="$2"; set -- help; source "$manager" >/dev/null; '
                    'profile_artifacts llamacpp "$profile"',
                    "bash",
                    str(HELPER_PATH.parent.parent / "dgx-model"),
                    str(profile_path),
                ],
                check=True,
                text=True,
                capture_output=True,
            )
        records = result.stdout.splitlines()
        self.assertEqual(len(records), 2)
        self.assertTrue(records[1].startswith("shard\t/models/Acme/Model/"))

    def test_vllm_uses_standard_card_command_and_local_network_settings(self):
        card = """
## vLLM
### Standard Version
```shell
vllm serve Acme/Vision-9B --port 8000 --tensor-parallel-size 8 \\
  --max-model-len 131072 --reasoning-parser qwen3
```
### Multi-Token Prediction
```shell
vllm serve Acme/Vision-9B --speculative-config '{"method":"mtp"}'
```
"""
        plan = hf_helper.vllm_plan("Acme/Vision-9B", "a" * 40, card)
        self.assertIn("--max-model-len", plan["args"])
        self.assertEqual(plan["args"][plan["args"].index("--port") + 1], "8080")
        self.assertEqual(plan["args"][plan["args"].index("--tensor-parallel-size") + 1], "1")
        self.assertNotIn("--speculative-config", plan["args"])
        self.assertTrue(plan["card_command"].startswith("vllm serve Acme/Vision-9B"))

    def test_vllm_reads_legacy_api_server_command(self):
        card = """
```bash
python3 -m vllm.entrypoints.openai.api_server --model Acme/Legacy-7B \\
  --max-model-len 8192 --trust-remote-code
```
"""
        plan = hf_helper.vllm_plan("Acme/Legacy-7B", "e" * 40, card)
        self.assertEqual(plan["args"][plan["args"].index("--max-model-len") + 1], "8192")
        self.assertIn("--trust-remote-code", plan["args"])

    def test_sglang_uses_card_settings_and_fixed_local_topology(self):
        card = """
## SGLang
```shell
python3 -m sglang.launch_server --model-path Acme/Vision-9B \\
  --host 127.0.0.1 --port 30000 --tp-size 8 --context-length 131072 \\
  --mem-fraction-static 0.7 --reasoning-parser qwen3 \\
  --trust-remote-code --api-key do-not-import
```
"""
        info = {
            "config": {
                "quantization_config": {
                    "quant_method": "modelopt",
                    "config_groups": {
                        "group_0": {"weights": {"num_bits": 8}},
                        "group_1": {"weights": {"num_bits": 4}},
                    },
                }
            }
        }
        plan = hf_helper.sglang_plan("Acme/Vision-9B", "f" * 40, card, info)
        args = plan["args"]
        self.assertEqual(args[args.index("--model-path") + 1], "/models/Acme/Vision-9B")
        self.assertEqual(args[args.index("--host") + 1], "0.0.0.0")
        self.assertEqual(args[args.index("--port") + 1], "8080")
        self.assertEqual(args[args.index("--tp-size") + 1], "1")
        self.assertEqual(args[args.index("--context-length") + 1], "131072")
        self.assertEqual(args[args.index("--mem-fraction-static") + 1], "0.7")
        self.assertEqual(args[args.index("--quantization") + 1], "modelopt_mixed")
        self.assertIn("--trust-remote-code", args)
        self.assertNotIn("--api-key", args)
        self.assertNotIn("do-not-import", args)
        self.assertTrue(plan["card_command"].startswith("python3 -m sglang.launch_server"))

    def test_sglang_shell_artifacts_include_a_pinned_draft(self):
        profile = """\
# hf-revision: /models/Acme/Main {main_revision}
# hf-revision: /models/Acme/Draft {draft_revision}
--model-path
/models/Acme/Main
--speculative-draft-model-path
/models/Acme/Draft
""".format(main_revision="1" * 40, draft_revision="2" * 40)
        with tempfile.TemporaryDirectory() as directory:
            profile_path = Path(directory, "profile.args")
            profile_path.write_text(profile, encoding="utf-8")
            result = subprocess.run(
                [
                    "bash",
                    "-c",
                    'manager="$1"; profile="$2"; set -- help; source "$manager" >/dev/null; '
                    'profile_artifacts sglang "$profile"',
                    "bash",
                    str(HELPER_PATH.parent.parent / "dgx-model"),
                    str(profile_path),
                ],
                check=True,
                text=True,
                capture_output=True,
            )
        records = result.stdout.splitlines()
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0].split("\t")[2:], ["Acme/Main", "1" * 40])
        self.assertEqual(records[1].split("\t")[2:], ["Acme/Draft", "2" * 40])

    def test_llama_prefers_card_quantization_and_includes_shards_and_mmproj(self):
        model_id = "Acme/Model-GGUF"
        info = {
            "siblings": [
                {"rfilename": "Model-Q4_K_M.gguf"},
                {"rfilename": "Model-Q5_K_M-00002-of-00002.gguf"},
                {"rfilename": "Model-Q5_K_M-00001-of-00002.gguf"},
                {"rfilename": "mmproj-F16.gguf"},
                {"rfilename": "nested/../unsafe.gguf"},
            ]
        }
        card = f"""
## Run
Recommended command:
```shell
llama-server -hf {model_id}:Q5_K_M -c 65536 -ngl 99
```
"""
        plan = hf_helper.llama_plan(model_id, "b" * 40, card, info, "")
        filenames = [artifact["filename"] for artifact in plan["artifacts"]]
        self.assertEqual(
            filenames,
            [
                "Model-Q5_K_M-00001-of-00002.gguf",
                "Model-Q5_K_M-00002-of-00002.gguf",
                "mmproj-F16.gguf",
            ],
        )
        self.assertEqual(plan["args"][plan["args"].index("--ctx-size") + 1], "65536")

    def test_incomplete_gguf_shards_are_rejected(self):
        info = {"siblings": [{"rfilename": "Model-Q4_K_M-00001-of-00002.gguf"}]}
        with self.assertRaises(SystemExit):
            hf_helper.llama_plan("Acme/Model-GGUF", "c" * 40, "", info, "")

    def test_render_emits_pinned_urls_and_extra_artifact_metadata(self):
        plan = {
            "engine": "llamacpp",
            "model_id": "Acme/Model-GGUF",
            "revision": "d" * 40,
            "profile": "model-gguf",
            "card_command": "",
            "artifacts": [
                {
                    "kind": "model",
                    "filename": "Model-Q4_K_M-00001-of-00002.gguf",
                    "container_path": "/models/Acme/Model-GGUF/Model-Q4_K_M-00001-of-00002.gguf",
                },
                {
                    "kind": "shard",
                    "filename": "Model-Q4_K_M-00002-of-00002.gguf",
                    "container_path": "/models/Acme/Model-GGUF/Model-Q4_K_M-00002-of-00002.gguf",
                },
            ],
            "args": ["--model", "/models/Acme/Model-GGUF/Model-Q4_K_M-00001-of-00002.gguf"],
        }
        with tempfile.TemporaryDirectory() as directory:
            plan_path = Path(directory, "plan.json")
            plan_path.write_text(json.dumps(plan), encoding="utf-8")
            args = argparse.Namespace(
                plan=str(plan_path),
                checksums_json=json.dumps(
                    {
                        "Model-Q4_K_M-00001-of-00002.gguf": "1" * 64,
                        "Model-Q4_K_M-00002-of-00002.gguf": "2" * 64,
                    }
                ),
            )
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                hf_helper.render_profile(args)
        rendered = output.getvalue()
        self.assertIn(f"/resolve/{'d' * 40}/Model-Q4_K_M-00001-of-00002.gguf", rendered)
        self.assertIn("# artifact: shard /models/Acme/Model-GGUF/", rendered)


if __name__ == "__main__":
    unittest.main()
