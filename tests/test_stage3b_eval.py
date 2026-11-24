"""Synthetic software fixtures only; these are never model performance data."""
from __future__ import annotations

import sys
from pathlib import Path
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from edgeai_scratch.detection_eval import evaluate,iou  # noqa: E402


class EvaluationFixtureTests(unittest.TestCase):
    def test_known_tp_fp_fn_ap_iou_and_timing(self):
        gt={"A":[{"class_id":0,"box":(.2,.2,.2,.2)},
                 {"class_id":5,"box":(.8,.8,.1,.1)}],
            "B":[{"class_id":0,"box":(.4,.4,.2,.2)}]}
        preds=[{"image_id":"A","class_id":0,"confidence":.9,"box":(.2,.2,.2,.2)},
               {"image_id":"A","class_id":5,"confidence":.8,"box":(.5,.8,.1,.1)},
               {"image_id":"B","class_id":0,"confidence":.7,"box":(.4,.4,.2,.2)},
               {"image_id":"A","class_id":0,"confidence":.6,"box":(.2,.2,.2,.2)}]
        result=evaluate(gt,preds,{"A":10.0,"B":20.0},.25,.5)
        overall=result["overall"]
        self.assertEqual((overall["tp"],overall["fp"],overall["fn"]),(2,2,1))
        self.assertAlmostEqual(overall["precision"],.5)
        self.assertAlmostEqual(overall["recall"],2/3)
        self.assertAlmostEqual(overall["f1"],4/7)
        self.assertAlmostEqual(result["per_class"][0]["ap50"],1.0)
        self.assertAlmostEqual(result["per_class"][0]["ap75"],1.0)
        self.assertAlmostEqual(result["per_class"][5]["ap50"],0.0)
        self.assertAlmostEqual(overall["map50"],.5)
        self.assertAlmostEqual(overall["map50_95"],.5)
        self.assertAlmostEqual(overall["mean_matched_iou"],1.0)
        self.assertAlmostEqual(overall["timing"]["mean_latency_ms"],15.0)
        self.assertAlmostEqual(overall["timing"]["fps_model_only"],1000/15)
        self.assertEqual(result["confusion_raw"][0][0],2)
        self.assertEqual(result["confusion_raw"][5][6],1)
        self.assertEqual(result["confusion_raw"][6][0],1)
        self.assertEqual(result["confusion_raw"][6][5],1)

    def test_wrong_class_is_confusion_and_detection_fp_fn(self):
        gt={"A":[{"class_id":0,"box":(.5,.5,.2,.2)}]}
        preds=[{"image_id":"A","class_id":1,"confidence":.9,"box":(.5,.5,.2,.2)}]
        result=evaluate(gt,preds,{"A":5.0})
        self.assertEqual((result["overall"]["tp"],result["overall"]["fp"],result["overall"]["fn"]),(0,1,1))
        self.assertEqual(result["confusion_raw"][0][1],1)

    def test_iou(self):
        self.assertAlmostEqual(iou((.5,.5,.2,.2),(.5,.5,.2,.2)),1.0)
        self.assertEqual(iou((.1,.1,.1,.1),(.9,.9,.1,.1)),0.0)


if __name__=="__main__":unittest.main()
