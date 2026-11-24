"""Run the existing local review UI against only the Stage 3C ten-image queue."""
import os,runpy
from pathlib import Path
os.environ['STAGE3_REVIEW_BASE']=str(Path(__file__).resolve().parents[1]/'data/experimental/stage3c_v1/review_sample')
runpy.run_path(str(Path(__file__).with_name('stage3b_review_server.py')),run_name='__main__')
