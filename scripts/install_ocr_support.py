"""Install optional registry OCR wheels into project cache without changing base packages."""
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from run_record import snapshot_run
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args()
    target=ROOT/'data/cache/ocr-python'
    if target.exists():raise ValueError('OCR dependency cache already exists; inspect rather than overwrite')
    out=snapshot_run(args.output,{'executable':sys.executable,'target':str(target),'preserve_base_packages':True},[ROOT/'requirements-ocr.txt'])
    command=[sys.executable,'-m','pip','install','--no-deps','--only-binary=:all:','--target',str(target),'-r',str(ROOT/'requirements-ocr.txt'),'--report',str(out/'pip-report.json')]
    (out/'command.json').write_text(json.dumps(command,indent=2))
    with (out/'install.log').open('x') as log:
        process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        for line in process.stdout:log.write(line);log.flush();print(line,end='',flush=True)
        code=process.wait()
    if code:raise RuntimeError('Optional OCR dependency install failed; inspect saved log')
    print('OCR_DEPENDENCIES_INSTALLED',flush=True)
