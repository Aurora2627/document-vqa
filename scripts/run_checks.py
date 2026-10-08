"""VS Code test runner: parse all saved Python and exercise model contracts."""
import ast,json,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
if __name__=='__main__':
    for folder in ['src','scripts','tests']:
        for path in (ROOT/folder).glob('*.py'):ast.parse(path.read_text(),filename=str(path))
    suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    report={'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'skipped':len(result.skipped),'passed':result.wasSuccessful(),'launch':'VS Code'}
    (ROOT/'reports/tests.json').write_text(json.dumps(report,indent=2))
    sys.exit(0 if result.wasSuccessful() else 1)
