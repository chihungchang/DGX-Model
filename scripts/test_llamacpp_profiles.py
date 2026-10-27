#!/usr/bin/env python3
"""Exercise profile routing and launch preflight without GPUs or model downloads."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


REPOSITORY = Path(__file__).resolve().parent.parent


class LlamaCppProfileTests(unittest.TestCase):
    def test_compose_routing_keeps_worker_opt_in_and_old_profiles_on_default(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copy2(REPOSITORY / "dgx-model", root / "dgx-model")
            profiles = root / "llama.cpp" / "profiles"
            profiles.mkdir(parents=True)
            (profiles / "local.args").write_text("--model\n/models/local.gguf\n")
            (profiles / "distributed.args").write_text(
                "# llama-compose: compose.tp2.yaml\n--model\n/models/main.gguf\n"
            )
            (profiles / "invalid.args").write_text(
                "# llama-compose: ../../unrelated.yaml\n"
            )
            docker = root / "docker"
            docker.write_text(
                "#!/usr/bin/env python3\n"
                "import json, os, sys\n"
                "print(json.dumps({'args': sys.argv[1:], "
                "'profiles': os.environ.get('COMPOSE_PROFILES')}))\n"
            )
            docker.chmod(0o755)
            env = dict(os.environ, PATH=f"{root}:{os.environ['PATH']}", COMPOSE_PROFILES="rpc")
            for profile, filename in (("local", "compose.yaml"), ("distributed", "compose.tp2.yaml")):
                with self.subTest(profile=profile):
                    result = subprocess.run(
                        [str(root / "dgx-model"), "compose-config", "llamacpp", profile],
                        env=env, text=True, capture_output=True, check=True,
                    )
                    actual = json.loads(result.stdout)
                    self.assertEqual(actual["profiles"], "")
                    args = actual["args"]
                    self.assertEqual(args[args.index("-f") + 1], str(profiles.parent / filename))
            rejected = subprocess.run(
                [str(root / "dgx-model"), "compose-config", "llamacpp", "invalid"],
                env=env, text=True, capture_output=True,
            )
            self.assertEqual(rejected.returncode, 64)
            self.assertEqual(rejected.stdout, "")

    def test_split_sibling_is_required_before_server_exec(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profiles = root / "config" / "profiles"
            profiles.mkdir(parents=True)
            models = root / "models"
            models.mkdir()
            for name in ("main.gguf", "draft.gguf"):
                (models / name).write_bytes(b"fixture")
            (profiles / "split.args").write_text(
                "# artifact: gguf /models/sibling.gguf https://example.invalid/sibling " + "a" * 64 + "\n"
                "--model\n/models/main.gguf\n"
                "--spec-draft-model\n/models/draft.gguf\n"
                "--rpc\n192.168.100.11:50052\n"
                "--device\nCUDA0,RPC0\n--split-mode\ntensor\n"
            )
            # Rewrite only the fixed container roots in an isolated script copy.
            script = (REPOSITORY / "llama.cpp" / "launch.sh").read_text()
            for name in ("config", "models", "app"):
                script = script.replace(f"/{name}/", f"{root}/{name}/")
            launcher = root / "launch.sh"
            launcher.write_text(script)
            app = root / "app"
            app.mkdir()
            server = app / "llama-server"
            server.write_text("#!/bin/sh\nprintf 'SERVER_ARG=%s\\n' \"$@\"\n")
            server.chmod(0o755)
            # Match the temporary model root in the fixture's args as well.
            profile = profiles / "split.args"
            profile.write_text(profile.read_text().replace("/models/", f"{models}/"))

            missing = subprocess.run(["bash", str(launcher), "split"], text=True, capture_output=True)
            self.assertEqual(missing.returncode, 66)
            self.assertIn("sibling.gguf", missing.stderr)
            self.assertNotIn("SERVER_ARG=", missing.stdout)

            (models / "sibling.gguf").write_bytes(b"fixture")
            complete = subprocess.run(
                ["bash", str(launcher), "split"], text=True, capture_output=True, check=True,
            )
            self.assertIn("SERVER_ARG=CUDA0,RPC0\n", complete.stdout)
            self.assertIn("SERVER_ARG=--split-mode\nSERVER_ARG=tensor\n", complete.stdout)
            self.assertNotIn("SERVER_ARG=#", complete.stdout)


if __name__ == "__main__":
    unittest.main()
