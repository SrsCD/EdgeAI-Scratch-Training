"""Feasible group split must not be rejected by the pilot allocator."""
import unittest

from scripts.stage3b_export_approved import choose_splits, MIN_SUPPORT, NAMES


class PilotSplitFixtureTests(unittest.TestCase):
    def test_independent_single_class_groups_cover_all_splits(self):
        items=[{"review":{"final_group_id":f"group_{cid}_{i}",
                          "boxes":[{"class_id":cid}]}}
               for cid in range(6) for i in range(30)]
        assignment,sizes,support=choose_splits(items)
        self.assertEqual(sum(sizes.values()),len(items))
        self.assertEqual(len(assignment),len(items))
        for split in ("train","test","val"):
            for name in NAMES:
                self.assertGreaterEqual(support[split][name],MIN_SUPPORT[split])


if __name__=="__main__":unittest.main()
