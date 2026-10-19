"""Developer tools gated by explicit per-action authorization.

Running tests is NOT an OS sandbox. Use a VM/container without secrets or
network access for untrusted repos; this module requires a consent callback.
"""
import ast
from pathlib import Path
import subprocess
import sys
from .agent import SafeTools,MAX_OUTPUT

class DevTools(SafeTools):
    def __init__(self,root,approval=None):
        super().__init__(root)
        self.approval=approval or (lambda name,args:False)

    def _write_path(self,rel):
        if not isinstance(rel,str) or not 0<len(rel)<=512 or Path(rel).is_absolute() or '..' in Path(rel).parts:
            raise PermissionError('Unsafe path')
        raw=self.root/rel
        if raw.suffix.lower() not in {'.py','.txt','.md','.json','.jsonl','.cpp','.hpp','.h','.c','.cmake'}:
            raise PermissionError('Extension not in allowlist')
        if any(part.startswith('.') for part in Path(rel).parts):
            raise PermissionError('Hidden paths disallowed')
        parent=raw.parent.resolve(strict=True)
        if not parent.is_relative_to(self.root):raise PermissionError('Outside root')
        if raw.is_symlink():raise PermissionError('Cannot write symlink')
        if raw.exists() and not raw.resolve().is_relative_to(self.root):
            raise PermissionError('Outside root')
        return raw

    def execute(self,name,arguments):
        if name not in {'write_file','run_tests'}:
            return super().execute(name,arguments)
        if not isinstance(arguments,dict):raise ValueError('Arguments must be object')
        if not bool(self.approval(name,arguments)):
            raise PermissionError(f'Explicit approval required for {name}')
        if name=='write_file':
            if set(arguments)!={'path','content'}:raise ValueError('Expected path and content')
            data=arguments['content']
            if not isinstance(data,str) or len(data.encode('utf-8'))>65536:
                raise ValueError('Text must be 64 KiB or less')
            path=self._write_path(arguments['path'])
            path.write_text(data,encoding='utf-8')
            return f'Wrote {path.relative_to(self.root)} ({len(data)} chars)'
        if name=='run_tests':
            if set(arguments)!={'path'} or arguments['path'] not in {'tests','.'}:
                raise ValueError('Only tests/ or workspace root permitted')
            # NOT safe for malicious test code: subprocess can access the host.
            cmd=[sys.executable,'-m','unittest','discover','-s',arguments['path'],'-v']
            try:
                p=subprocess.run(cmd,cwd=self.root,capture_output=True,text=True,timeout=20,check=False)
                return str({'returncode':p.returncode,'stdout':p.stdout[-MAX_OUTPUT//2:],
                            'stderr':p.stderr[-MAX_OUTPUT//2:]})
            except subprocess.TimeoutExpired:
                return 'Tests timed out after 20 s'
        raise AssertionError('Unreachable')
