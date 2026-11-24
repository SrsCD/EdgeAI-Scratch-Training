"""Future explicitly invoked experimental scratch run; Stage 3C does not run it."""
import argparse,hashlib,json,platform,time
from datetime import datetime,timezone
from pathlib import Path
import torch,yaml
from stage3c_preflight import main as preflight
ROOT=Path(__file__).resolve().parents[1]
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--smoke',action='store_true',help='One epoch on a tiny train-only fraction; no validation or test')
    parser.add_argument('--batch',type=int,help='Recorded throughput-tuning override of initial batch 2')
    args_cli=parser.parse_args()
    preflight()
    config=yaml.safe_load((ROOT/'configs/stage3c_train.yaml').read_text())
    if not torch.cuda.is_available() or '3050' not in torch.cuda.get_device_name(0):
        raise SystemExit('Expected RTX 3050 CUDA device is unavailable')
    if config['model']!='configs/yolo11n.yaml' or config['pretrained'] is not False or config['resume'] is not False:
        raise SystemExit('Scratch initialization contract changed')
    from ultralytics import YOLO
    model=YOLO(str(ROOT/config['model']))
    if getattr(model,'ckpt',None) or getattr(model,'ckpt_path',None) or getattr(model.model,'pt_path',None):
        raise SystemExit('Unexpected pretrained checkpoint')
    args={k:v for k,v in config.items() if k!='model'}
    args['data']=str(ROOT/config['data']);args['project']=str(ROOT/config['project']);args['exist_ok']=True
    if args_cli.batch:
        if args_cli.batch<2 or args_cli.batch>16:raise SystemExit('Batch override must be 2..16')
        args['batch']=args_cli.batch
        args['name']=config['name']+f'_b{args_cli.batch}'
    if args_cli.smoke:
        args.update(name='stage4a_scratch_smoke'+(f'_b{args_cli.batch}' if args_cli.batch else ''),epochs=1,patience=1,val=False,
                    fraction=.006 if args_cli.batch else .002,plots=False,save=True)
    run_dir=ROOT/'runs'/args['name']
    if run_dir.exists():raise SystemExit(f'Run directory already exists: {run_dir}')
    run_dir.mkdir(parents=True)
    digest=lambda p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
    started=datetime.now(timezone.utc).isoformat();t0=time.monotonic()
    evidence={
        'status':'EXPERIMENTAL / NOT FULLY HUMAN-VERIFIED','initialization':'architecture YAML random weights',
        'pretrained':False,'resume':False,'source':'data/experimental/stage3c_v1',
        'mode':'smoke' if args_cli.smoke else 'full','started_utc':started,
        'ledger_sha256':digest('data/experimental/stage3c_v1/ledger.csv'),
        'split_stats_sha256':digest('data/experimental/stage3c_v1/split_stats.csv'),
        'architecture_sha256':digest('configs/yolo11n.yaml'),
        'data_yaml_sha256':digest('configs/stage3c_data.yaml'),
        'train_config_sha256':digest('configs/stage3c_train.yaml'),
        'gpu':torch.cuda.get_device_name(0),'vram_bytes':torch.cuda.get_device_properties(0).total_memory,
        'torch':torch.__version__,'python':platform.python_version(),
        'effective_arguments':args,'test_used_for_training_or_selection':False,
        'manual_review':'10 Hold; zero Approve'}
    (run_dir/'stage4a_provenance.json').write_text(json.dumps(evidence,indent=2)+'\n')
    try:
        model.train(**args)
    except BaseException as exc:
        evidence.update(outcome='interrupted_or_failed',error=repr(exc),duration_seconds=time.monotonic()-t0)
        (run_dir/'stage4a_provenance.json').write_text(json.dumps(evidence,indent=2)+'\n')
        raise
    save_dir=Path(model.trainer.save_dir)
    if save_dir.resolve()!=run_dir.resolve():raise SystemExit('Unexpected run directory suffix; inspect output')
    evidence.update(outcome='completed',duration_seconds=time.monotonic()-t0,
                    finished_utc=datetime.now(timezone.utc).isoformat(),
                    peak_vram_bytes=torch.cuda.max_memory_allocated(0))
    (run_dir/'stage4a_provenance.json').write_text(json.dumps(evidence,indent=2)+'\n')
    if not args_cli.smoke:
        (save_dir/'EXPERIMENTAL_PROVISIONAL_LABELS.json').write_text(json.dumps(evidence,indent=2)+'\n')
if __name__=='__main__':main()
