from dataclasses import dataclass
from pathlib import Path
import subprocess
import sysconfig
import shlex
import ast
import re

getvar = sysconfig.get_config_var

if getvar('CC') != 'gcc':
    raise Exception('This script currently works with gcc only. Ports welcome.')

CFLAGS = getvar('CFLAGS')

projectpath = Path('.')
while not projectpath.joinpath('Include').exists():
    if projectpath == projectpath.parent:
        raise Exception('Could not find project root')
    projectpath = projectpath.parent

@dataclass
class Location:
    filename: str
    lineno: int

    def __repr__(self):
        return f'<{self.filename}:{self.lineno}>'

@dataclass(frozen=True)
class Variant:
    limited_api: str

    def __repr__(self):
        return f'<{self.limited_api}>'

@dataclass
class Definition:
    kind: str
    args: str
    body: str
    variants: dict[Variant, Location]

    def has_same_content_as(self, other):
        return (
            self.kind == other.kind
            and self.args == other.args
            and self.body == other.body
        )

@dataclass
class Entry:
    name: str
    definitions: list[Definition]

def add_entries(entries, limited_api=None):
    if limited_api is None:
        defines = []
    else:
        defines = [f'-DPy_LIMITED_API={limited_api}']
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

    def get_entry(name):
        try:
            return entries[name]
        except KeyError:
            entry = entries[name] = Entry(name, [])
            return entry

    def add_definition(kind, name, args=None, body=None):
        variant = Variant(limited_api)
        location = Location(
            filename=filename,
            lineno=lineno,
        )
        definition = Definition(
            kind=kind,
            args=args,
            body=body,
            variants={variant: location},
        )
        entry = get_entry(name)
        for old_definition in reversed(entry.definitions):
            if variant in old_definition.variants:
                break
            if old_definition.has_same_content_as(definition):
                old_definition.variants[variant] = location
                return
        entry.definitions.append(definition)

    def split_macro_def(content):
        match = re.fullmatch(
            r'([a-zA-Z0-9_-]*)\s*(\([^)]*\))?(.*)',
            content,
        )
        name = match[1]
        args = match[2]
        body = match[3].strip()
        return name, args, body

    filename = '<start>'
    lineno = 0
    with gcc_proc.stdout as gcc_stdout:
        for line in gcc_stdout:
            line = line.rstrip('\n')
            try:
                if line.startswith('# '):
                    marks = set()
                    rest = line
                    while rest.endswith((' 0', ' 1', ' 2', ' 3', ' 4')):
                        marks.add(int(rest[-1]))
                        rest = rest[:-2]
                    _hash, lineno, filename = rest.rsplit(None, 2)
                    filename = ast.literal_eval(filename)
                    lineno = int(lineno)
                    continue
                if line.startswith('#'):
                    if filename.startswith(('Include', '.')):
                        #print(filename, lineno, line)
                        if line.startswith('#define'):
                            content = line[len('#define'):].strip()
                            name, args, body = split_macro_def(content)
                            add_definition(
                                kind='define',
                                name=name, args=args, body=body,
                            )
                        elif line.startswith('#undef'):
                            content = line[len('#undef'):].strip()
                            name, args, body = split_macro_def(content)
                            add_definition(
                                kind='undef',
                                name=name, args=args, body=body,
                            )
                        else:
                            print(line)
            except:
                print('Line was:', line)
                raise
            lineno += 1

    if gcc_proc.wait():
        raise Exception(f'gcc failed with return code {gcc_proc.returncode}')

entries = {}
add_entries(entries)
add_entries(entries, '3')
add_entries(entries, '0x03020000')

for entry in entries.values():
    print(entry.name)
    for definition in entry.definitions:
        print('   ', definition)
