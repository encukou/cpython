from dataclasses import dataclass
from pathlib import Path
import subprocess
import sysconfig
import shlex
import ast

getvar = sysconfig.get_config_var

if getvar('CC') != 'gcc':
    raise Exception('This script currently works with gcc only. Ports welcome.')

CFLAGS = getvar('CFLAGS')

def run(*args, **kwargs):
    kwargs.setdefault('check', True)
    return subprocess.run(args, **kwargs)

projectpath = Path('.')
while not projectpath.joinpath('Include').exists():
    if projectpath == projectpath.parent:
        raise Exception('Could not find project root')
    projectpath = projectpath.parent

gcc_proc = subprocess.Popen(
    [
        'gcc',
        '-E', '-dD',
        '-I.', '-IInclude/',
        *shlex.split(CFLAGS),
        #'-DPy_LIMITED_API=3',
        'Include/Python.h',
    ],
    stdout=subprocess.PIPE,
    encoding='utf-8',
    cwd=projectpath,
    errors='backslashreplace',
)

@dataclass
class Definition:
    kind: str
    filename: str
    lineno: int
    content: str

@dataclass
class Entry:
    name: str
    definitions: list[Definition]

entries = {}

def get_entry(name):
    try:
        return entries[name]
    except KeyError:
        entry = entries[name] = Entry(name, [])
        return entry

filename = '<start>'
lineno = 0
with gcc_proc.stdout as gcc_stdout:
    for line in gcc_stdout:
        line = line.rstrip('\n')
        if line.startswith('# '):
            # print(line)
            try:
                marks = set()
                rest = line
                while rest.endswith((' 0', ' 1', ' 2', ' 3', ' 4')):
                    marks.add(int(rest[-1]))
                    rest = rest[:-2]
                _hash, lineno, filename = rest.rsplit(None, 2)
                filename = ast.literal_eval(filename)
                lineno = int(lineno)
            except:
                print('Line was:', line)
            continue
        if line.startswith('#'):
            if filename.startswith(('Include', '.')):
                #print(filename, lineno, line)
                if line.startswith('#define'):
                    try:
                        _define, name, content = line.split(None, 2)
                    except ValueError:
                        _define, name = line.split(None, 1)
                        content = ''
                    get_entry(name).definitions.append(Definition(
                        kind='macro',
                        filename=filename,
                        lineno=lineno,
                        content=content,
                    ))
                elif line.startswith('#undef'):
                    get_entry(name).definitions.append(Definition(
                        kind='undef',
                        filename=filename,
                        lineno=lineno,
                        content=None,
                    ))
                else:
                    print(line)
        lineno += 1

if gcc_proc.wait():
    raise Exception(f'gcc failed with return code {gcc_proc.returncode}')

for entry in entries.values():
    print(entry)
