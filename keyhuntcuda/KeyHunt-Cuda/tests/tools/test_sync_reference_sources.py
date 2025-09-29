import os
import subprocess
import tempfile
import unittest


class SyncReferenceSourcesTest(unittest.TestCase):
    def test_sync_reference_sources_generates_snapshot(self):
        repo_root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], text=True
        ).strip()
        project_root = os.path.join(repo_root, "keyhuntcuda", "keyhunt-cuda")

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = os.path.join(tmpdir, "snapshots")
            report = os.path.join(tmpdir, "report.md")
            config = os.path.join(tmpdir, "sources.yml")

            with open(config, "w", encoding="utf-8") as fh:
                fh.write("keyhuntcuda/keyhunt-cuda/README.md\n")

            result = subprocess.run(
                [
                    os.path.join(project_root, "tools", "sync_reference_sources.sh"),
                    "--apply",
                    "--config",
                    config,
                    "--output-dir",
                    out_dir,
                    "--report",
                    report,
                ],
                cwd=project_root,
                capture_output=True,
                text=True,
                check=True,
            )

            self.assertIn("Snapshot stored at", result.stdout)
            self.assertTrue(os.path.isfile(report))

            with open(report, "r", encoding="utf-8") as fh:
                lines = [line.strip() for line in fh if line.strip()]

            self.assertGreaterEqual(len(lines), 2)
            header = lines[0]
            self.assertEqual(header, "| Snapshot | Digest | Commit |")

            snapshot_dirs = [
                entry
                for entry in os.listdir(out_dir)
                if os.path.isdir(os.path.join(out_dir, entry))
            ]
            self.assertEqual(len(snapshot_dirs), 1)
            snapshot = os.path.join(out_dir, snapshot_dirs[0])
            self.assertTrue(os.path.isfile(os.path.join(snapshot, "SHA256SUMS.txt")))


if __name__ == "__main__":
    unittest.main()
