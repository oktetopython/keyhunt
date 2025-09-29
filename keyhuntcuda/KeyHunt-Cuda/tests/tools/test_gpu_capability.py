import json
import os
import subprocess
import unittest


class GpuCapabilityTest(unittest.TestCase):
    def test_gpu_capability_script_returns_json(self):
        repo_root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], text=True
        ).strip()
        project_root = os.path.join(repo_root, "keyhuntcuda", "keyhunt-cuda")

        result = subprocess.run(
            [os.path.join(project_root, "tools", "gpu_capability.sh")],
            capture_output=True,
            text=True,
            check=True,
            cwd=project_root,
        )

        payload = result.stdout.strip()
        self.assertTrue(payload)
        data = json.loads(payload)
        self.assertIn("generated_at", data)
        self.assertIn("status", data)
        self.assertIn("gpus", data)


if __name__ == "__main__":
    unittest.main()
