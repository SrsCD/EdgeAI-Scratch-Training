"""Serve only the six pending Stage 5C images on localhost."""
import os
from pathlib import Path
import runpy

os.environ["STAGE3_REVIEW_BASE"] = str(Path(__file__).resolve().parents[1] /
                                        "data/experimental/stage5c_v1/review_queue")
runpy.run_path(str(Path(__file__).with_name("stage3b_review_server.py")), run_name="__main__")
