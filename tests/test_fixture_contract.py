"""检查采集资料的身份、形状和可交叉引用性。"""

import json
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read_fixture(name):
    return json.loads((ROOT / "fixtures" / name).read_text(encoding="utf-8"))


class FixtureContractTests(unittest.TestCase):
    def test_record_manifest(self):
        manifest = read_fixture("manifest.json")
        primary = read_fixture("primary.json")
        revision = read_fixture("revision.json")
        self.assertEqual(manifest["fixture"], "servo_scan")
        self.assertEqual(manifest["records"], ["primary.json", "revision.json"])
        self.assertTrue(manifest["synthetic"])
        self.assertEqual(revision["replaces"], primary["capture"])
        self.assertNotEqual(revision["capture"], primary["capture"])

    def test_related_input_records(self):
        primary = read_fixture("primary.json")
        revision = read_fixture("revision.json")
        initial = primary["initial_state"]
        self.assertEqual({len(v) for v in initial.values()}, {2})
        blocks = primary["blocks"] + [revision["arrival"]]
        by_id = {block["id"]: block for block in blocks}
        self.assertEqual(set(revision["redelivery"]) - by_id.keys(), set())
        for block in blocks:
            self.assertEqual(np.asarray(block["a"]).shape, np.asarray(block["b"]).shape)
            self.assertEqual(np.asarray(block["a"]).shape[1], len(block["axis_map"]))
            self.assertFalse(set(block["axis_map"]) - initial.keys())
        self.assertIn(primary["handoff"]["axis"], initial)
