#!/usr/bin/env python3
"""Check two-node switching contracts without touching Docker or live services."""

import importlib.util
import json
from pathlib import Path
import subprocess
import struct
import tempfile
import unittest
from unittest import mock

SPEC = importlib.util.spec_from_file_location(
    "dgx_model_cluster", Path(__file__).with_name("dgx_model_cluster.py")
)
cluster = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cluster)


class ClusterSwitchTests(unittest.TestCase):
    def test_blackfold_uses_native_with_separate_weights_and_cache(self):
        profile = cluster.TF_BLACKFOLD_PROFILE
        self.assertTrue(cluster.native_profile(profile))
        self.assertNotEqual(cluster.model_view(profile), cluster.MODEL_VIEW)
        self.assertNotEqual(cluster.cache_dir(profile), cluster.CACHE)
        with mock.patch.object(cluster, "run") as run, mock.patch.object(cluster, "wait_ready"):
            cluster.start_tensorfold(profile)
        self.assertEqual([c.args[1] for c in run.call_args_list], [1, 0])
        for call in run.call_args_list:
            self.assertIn(cluster.ROOT / "tensorfold/compose.native.yaml", call.args[0])
            self.assertTrue(any(str(x).endswith("-blackfold.env") for x in call.args[0]))

    def test_blackfold_refuses_stock_or_unbound_cache_receipts(self):
        import hashlib
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = root / "tensorfold/native/runtime"
            runtime.mkdir(parents=True)
            files = []
            for i in range(370):
                name = f"kit-{i}"
                (runtime / name).write_bytes(b"fixture")
                files.append({"path": name, "size": 7,
                              "sha256": hashlib.sha256(b"fixture").hexdigest()})
            (runtime / "build.json").write_text(json.dumps({
                "engine_commit": cluster.TF_NATIVE_COMMIT, "patches": {}, "files": files}))
            cache = root / "cache"
            (cache / "tf-ds-rank").mkdir(parents=True)
            name = "rank0of2-fixture.bin"
            (cache / "tf-ds-rank" / name).write_bytes(struct.pack("<Q", 0) + b"TFDSRK01")
            (cache / "dsv41-token-map.json").write_text("{}")
            marker = cache / "blackfold-model.json"
            receipt = {"repo_id": cluster.BLACKFOLD_ID, "revision": cluster.BLACKFOLD_REVISION,
                       "overlay_sha256": cluster.BLACKFOLD_OVERLAY_SHA, "rank": 0,
                       "cache_file": name, "cache_size": 16, "cache_index_offset": 0}
            with mock.patch.object(cluster, "ROOT", root), \
                 mock.patch.object(cluster, "BLACKFOLD_CACHE", cache):
                self.assertIn("BLACKFOLD cache provenance mismatch",
                              cluster.native_artifact_errors(0, cluster.TF_BLACKFOLD_PROFILE))
                marker.write_text(json.dumps(receipt))
                self.assertEqual(cluster.native_artifact_errors(0, cluster.TF_BLACKFOLD_PROFILE), [])
                receipt["cache_file"] = "old-stock-cache.bin"
                marker.write_text(json.dumps(receipt))
                self.assertIn("BLACKFOLD cache receipt does not match file",
                              cluster.native_artifact_errors(0, cluster.TF_BLACKFOLD_PROFILE))

    def test_native_starts_both_ranks_with_its_own_compose(self):
        with mock.patch.object(cluster, "run") as run, mock.patch.object(cluster, "wait_ready"):
            cluster.start_tensorfold(cluster.TF_NATIVE_PROFILE)
        commands = [c.args for c in run.call_args_list]
        self.assertEqual(commands[0][1], 1)
        self.assertEqual(commands[1][1], 0)
        for command in commands:
            self.assertIn(cluster.ROOT / "tensorfold/compose.native.yaml", command[0])

    def test_switching_runtime_does_not_reuse_the_previous_live_server(self):
        previous = {"State": {"Status": "running"}, "Config": {"Labels": {
            "ai.dgx.profile": cluster.TF_PROFILE}}}
        with tempfile.TemporaryDirectory() as directory, \
             mock.patch.object(cluster, "STATE", Path(directory)), \
             mock.patch.object(cluster, "validate"), \
             mock.patch.object(cluster, "sync_config"), \
             mock.patch.object(cluster, "active", side_effect=[True, False]), \
             mock.patch.object(cluster, "inspect", return_value=previous), \
             mock.patch.object(cluster, "memory", return_value=115), \
             mock.patch.object(cluster, "stop_other_models"), \
             mock.patch.object(cluster, "stop") as stop, \
             mock.patch.object(cluster, "run"), \
             mock.patch.object(cluster, "start_tensorfold") as start:
            cluster.use("tensorfold", cluster.TF_NATIVE_PROFILE)
        stop.assert_called_once_with("tensorfold")
        start.assert_called_once_with(cluster.TF_NATIVE_PROFILE)

    def test_failed_native_switch_restores_the_previous_python_runtime(self):
        previous = {"State": {"Status": "running"}, "Config": {"Labels": {
            "ai.dgx.profile": cluster.TF_PROFILE}}}
        with tempfile.TemporaryDirectory() as directory, \
             mock.patch.object(cluster, "STATE", Path(directory)), \
             mock.patch.object(cluster, "validate"), \
             mock.patch.object(cluster, "sync_config"), \
             mock.patch.object(cluster, "active", side_effect=[True, False]), \
             mock.patch.object(cluster, "inspect", return_value=previous), \
             mock.patch.object(cluster, "memory", return_value=115), \
             mock.patch.object(cluster, "stop_other_models"), \
             mock.patch.object(cluster, "stop"), \
             mock.patch.object(cluster, "run"), \
             mock.patch.object(cluster, "start_tensorfold", side_effect=[RuntimeError("startup"), None]) as start:
            with self.assertRaisesRegex(RuntimeError, "startup"):
                cluster.use("tensorfold", cluster.TF_NATIVE_PROFILE)
        self.assertEqual(start.call_args_list, [mock.call(cluster.TF_NATIVE_PROFILE), mock.call(cluster.TF_PROFILE)])

    def test_image_requires_the_single_call_dsml_filter(self):
        labels = {"org.opencontainers.image.revision": cluster.TF_COMMIT,
                  "ai.dgx.tensorfold.patch": "rdma-active-mtu-v1",
                  "ai.dgx.tensorfold.pool-context": "independent-v2"}
        image = {"Id": "image-test", "Config": {"Labels": labels}}
        def inspect_image(*args, **kwargs):
            return subprocess.CompletedProcess([], 0, json.dumps([image]), "")
        with mock.patch.object(cluster, "run", side_effect=inspect_image):
            with self.assertRaisesRegex(ValueError, "DSML filter"):
                cluster.image_info(0)
            labels["ai.dgx.tensorfold.dsml-single-tool"] = "v1"
            with self.assertRaisesRegex(ValueError, "context-based image"):
                cluster.image_info(0)
            labels["ai.dgx.tensorfold.image-context"] = "v1"
            with self.assertRaisesRegex(ValueError, "remaining-context output"):
                cluster.image_info(0)
            labels["ai.dgx.tensorfold.output-ceiling"] = "v1"
            self.assertEqual(cluster.image_info(0), "image-test")

    def test_extension_cache_must_match_image_and_have_all_binaries(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            receipt = cache / "extension-build.json"
            with mock.patch.object(cluster, "CACHE", cache):
                self.assertTrue(cluster.extension_errors(0, "image-A"))
                names = [f"extension-{i}.so" for i in range(4)]
                receipt.write_text(json.dumps({"image_id": "image-A", "files": names}))
                for name in names:
                    (cache / name).write_bytes(b"compiled fixture")
                self.assertEqual(cluster.extension_errors(0, "image-A"), [])
                self.assertTrue(cluster.extension_errors(0, "image-B"))
                (cache / names[0]).write_bytes(b"")
                self.assertTrue(cluster.extension_errors(0, "image-A"))
                (cache / names[1]).unlink()
                self.assertTrue(cluster.extension_errors(0, "image-A"))

    def test_missing_target_never_stops_the_running_server(self):
        with mock.patch.object(cluster, "validate", side_effect=ValueError("missing Engram")), \
             mock.patch.object(cluster, "stop") as stop, \
             mock.patch.object(cluster, "stop_other_models") as stop_other:
            with self.assertRaisesRegex(ValueError, "Engram"):
                cluster.use("tensorfold", cluster.TF_PROFILE)
            stop.assert_not_called()
            stop_other.assert_not_called()

    def test_unusable_rollback_never_stops_sglang(self):
        with mock.patch.object(cluster, "validate", side_effect=[None, None, ValueError("rollback missing")]), \
             mock.patch.object(cluster, "active", return_value=False), \
             mock.patch.object(cluster, "sync_config"), \
             mock.patch.object(cluster, "stop") as stop:
            with self.assertRaisesRegex(ValueError, "rollback"):
                cluster.use("tensorfold", cluster.TF_PROFILE)
            stop.assert_not_called()

    def test_startup_failure_restores_sglang(self):
        with tempfile.TemporaryDirectory() as directory, \
             mock.patch.object(cluster, "STATE", Path(directory)), \
             mock.patch.object(cluster, "validate"), \
             mock.patch.object(cluster, "sync_config"), \
             mock.patch.object(cluster, "active", side_effect=[False, True]), \
             mock.patch.object(cluster, "memory", return_value=115), \
             mock.patch.object(cluster, "inspect", return_value=None), \
             mock.patch.object(cluster, "stop_other_models"), \
             mock.patch.object(cluster, "stop") as stop, \
             mock.patch.object(cluster, "start_sglang") as restore:
            def execute(args, *unused, **kwargs):
                if "up" in args:
                    raise subprocess.CalledProcessError(1, args)
                return subprocess.CompletedProcess(args, 0, "", "")
            with mock.patch.object(cluster, "run", side_effect=execute):
                with self.assertRaises(subprocess.CalledProcessError):
                    cluster.use("tensorfold", cluster.TF_PROFILE)
            restore.assert_called_once_with()
            self.assertIn(mock.call("tensorfold", True), stop.call_args_list)

    def test_switch_back_stops_tensorfold_before_sglang_start(self):
        events = []
        with mock.patch.object(cluster, "validate", side_effect=lambda *a: events.append("validate")), \
             mock.patch.object(cluster, "active", return_value=False), \
             mock.patch.object(cluster, "stop", side_effect=lambda *a: events.append("stop_tf")), \
             mock.patch.object(cluster, "stop_other_models", side_effect=lambda: events.append("stop_other")), \
             mock.patch.object(cluster, "start_sglang", side_effect=lambda: events.append("start_sg")):
            cluster.use("sglang", cluster.SG_PROFILE)
        self.assertEqual(events, ["validate", "stop_tf", "stop_other", "start_sg"])

    def test_startup_failure_keeps_previously_stopped_services_off(self):
        with tempfile.TemporaryDirectory() as directory, \
             mock.patch.object(cluster, "STATE", Path(directory)), \
             mock.patch.object(cluster, "validate"), \
             mock.patch.object(cluster, "sync_config"), \
             mock.patch.object(cluster, "active", return_value=False), \
             mock.patch.object(cluster, "memory", return_value=115), \
             mock.patch.object(cluster, "inspect", return_value=None), \
             mock.patch.object(cluster, "stop_other_models"), \
             mock.patch.object(cluster, "stop"), \
             mock.patch.object(cluster, "start_sglang") as start_sg:
            commands = []

            def execute(args, *unused, **kwargs):
                commands.append(args)
                if "up" in args:
                    raise subprocess.CalledProcessError(1, args)
                return subprocess.CompletedProcess(args, 0, "", "")

            with mock.patch.object(cluster, "run", side_effect=execute):
                with self.assertRaises(subprocess.CalledProcessError):
                    cluster.use("tensorfold", cluster.TF_PROFILE)
            start_sg.assert_not_called()
            self.assertFalse(any("start" in args for args in commands))

    def test_missing_or_truncated_engram_is_not_ready(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "view/model"
            engram = root / "view/engram"
            model.mkdir(parents=True)
            engram.mkdir()
            for name in ["config.json", "tokenizer.json", "tokenizer_config.json",
                         "model.safetensors.index.json"]:
                (model / name).write_text("{}")
            (model / "chat_template.jinja").write_bytes(b"fixture")
            import hashlib
            manifest = root / "tensorfold"
            manifest.mkdir()
            (manifest / "model-files.json").write_text(json.dumps({"files": [
                {"path": "model/main.safetensors", "size": 3},
                {"path": "engram/model-00047-of-00048.safetensors", "size": 5},
                {"path": "engram/model-00048-of-00048.safetensors", "size": 7},
            ]}))
            (model / "main.safetensors").write_bytes(b"abc")
            (engram / "model-00047-of-00048.safetensors").write_bytes(b"12345")
            with mock.patch.object(cluster, "ROOT", root), \
                 mock.patch.object(cluster, "MODEL_VIEW", root / "view"), \
                 mock.patch.object(cluster, "TEMPLATE_SHA", hashlib.sha256(b"fixture").hexdigest()):
                self.assertIn("engram/model-00048-of-00048.safetensors", cluster.model_errors(0))
                (engram / "model-00048-of-00048.safetensors").write_bytes(b"short")
                self.assertTrue(cluster.model_errors(0))
                (engram / "model-00048-of-00048.safetensors").write_bytes(b"1234567")
                self.assertEqual(cluster.model_errors(0), [])


if __name__ == "__main__":
    unittest.main()
