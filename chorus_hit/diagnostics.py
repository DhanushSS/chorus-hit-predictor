"""Versioned output and execution identity shared by exploratory commands."""
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
import platform
import math
import subprocess
from .audit import atomic_json, sha256_file
from .config import ROOT
from .supervision import supervise


def new_output(path):
    path=Path(path)
    path.mkdir(parents=True,exist_ok=False)
    return path


def identity(script, **protocol):
    return {'created_utc':datetime.now(timezone.utc).isoformat(),
            'dataset_sha256':sha256_file(ROOT/'data/chorus_features.csv'),
            'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'dirty_tree':bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()),
            'code_files':{str(p.relative_to(ROOT)):sha256_file(p) for p in [Path(script),*sorted((ROOT/'chorus_hit').glob('*.py'))]},
            'python':platform.python_version(),'platform':platform.platform(),
            'packages':{p:version(p) for p in ['numpy','pandas','scipy','scikit-learn','joblib']},
            'protocol':protocol}


def bounded_command(command, output, seconds):
    from .supervision import require_supported_platform
    require_supported_platform()
    if not math.isfinite(seconds) or seconds<=0: raise ValueError('Timeout must be positive')
    output=new_output(output)
    atomic_json(output/'execution.json',{'status':'running','budget_scope':'per_invocation','timeout_seconds':seconds})
    try:
        code=supervise(command,seconds,ROOT)
        if code: raise RuntimeError(f'Diagnostic worker exited with status {code}; inspect its captured records')
    except BaseException as exc:
        atomic_json(output/'execution.json',{'status':'incomplete','error':f'{type(exc).__name__}: {exc}','timeout_seconds':seconds})
        raise
    atomic_json(output/'execution.json',{'status':'complete','timeout_seconds':seconds})
