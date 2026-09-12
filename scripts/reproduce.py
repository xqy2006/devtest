"""Exercise the released loader using synthetic matching-Electron caches."""
from pathlib import Path
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'evidence'
EVIDENCE.mkdir(exist_ok=True)
WINDOWS = platform.system() == 'Windows'
D8 = ROOT / 'd8' / ('d8.exe' if WINDOWS else 'd8')
ELECTRON = ROOT / 'node_modules' / 'electron' / 'dist' / ('electron.exe' if WINDOWS else 'electron')


def run_logged(command, destination, *, env=None, timeout=120):
    with destination.open('wb') as stream:
        try:
            process = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=timeout)
            return process.returncode
        except subprocess.TimeoutExpired:
            stream.write(b'\nPROBE_TIMEOUT\n')
            return 124


def main():
    env = dict(os.environ, ELECTRON_RUN_AS_NODE='1')
    for mode in ('default', 'eager'):
        code = run_logged([str(ELECTRON), str(ROOT / 'scripts/generate.cjs'), mode], EVIDENCE / f'generate-{mode}.log', env=env)
        if code:
            print((EVIDENCE / f'generate-{mode}.log').read_text(errors='replace'))
            return 1
    run_logged([str(D8), '--version'], EVIDENCE / 'd8-version.log')
    results = []
    for fixture in sorted((EVIDENCE / 'fixtures').glob('*.jsc')):
        script = f'print("BEFORE_LOAD"); loadjsc({json.dumps(fixture.as_posix())}); print("AFTER_LOAD");'
        command = [str(D8), '-e', script]
        log = EVIDENCE / f'{fixture.stem}.log'
        code = run_logged(command, log)
        output = log.read_text(errors='replace')
        row = dict(label=fixture.stem, bytes=fixture.stat().st_size, exit=code, before='BEFORE_LOAD\n' in output, after='AFTER_LOAD\n' in output, printed=output.count('Start SharedFunctionInfo'), sha256=hashlib.sha256(fixture.read_bytes()).hexdigest(), tail=output[-1800:])
        results.append(row)
        print(json.dumps({key: row[key] for key in ('label', 'bytes', 'exit', 'after', 'printed')}), flush=True)
    failing = [r for r in results if r['exit'] or not r['after']]
    if failing and shutil.which('gdb'):
        target = min(failing, key=lambda row: row['bytes'])
        fixture = EVIDENCE / 'fixtures' / f'{target["label"]}.jsc'
        command = ['gdb', '-q', '-batch', '-ex', 'set pagination off', '-ex', 'run', '-ex', 'thread apply all bt', '--args', str(D8), '-e', f'loadjsc({json.dumps(fixture.as_posix())});']
        run_logged(command, EVIDENCE / 'gdb-smallest-failure.log')
    summary = dict(platform=platform.platform(), total=len(results), failures=len(failing), results=results)
    (EVIDENCE / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f'{len(failing)}/{len(results)} loader failures')
    return 1 if failing else 0


if __name__ == '__main__':
    sys.exit(main())
