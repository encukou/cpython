from dataclasses import dataclass, field
from pathlib import Path
import subprocess
import sysconfig
import shlex
import ast
import re

# https://en.wikipedia.org/wiki/ANSI_escape_code#CSI_(Control_Sequence_Introducer)_sequences
CSI_RE = re.compile(r'\033\[[0-?]*[ -/]*[@-~]')

def csi(code):
    result = '\033' + code
    assert CSI_RE.fullmatch(result)
    return result

INTENSE = csi('[1m')
DIM = csi('[2m')
RED = csi('[1;31m')
CYAN = csi('[36m')
RESET = csi('[m')

getvar = sysconfig.get_config_var

if getvar('CC') != 'gcc':
    raise Exception('This script currently works with gcc only. Ports welcome.')

CFLAGS = getvar('CFLAGS')

projectpath = Path('.')
while not projectpath.joinpath('Include').exists():
    if projectpath == projectpath.parent:
        raise Exception('Could not find project root')
    projectpath = projectpath.parent

@dataclass(frozen=True)
class Location:
    filename: str
    lineno: int

    def __repr__(self):
        return f'<{self}>'

    def __str__(self):
        return f'{self.filename}:{self.lineno}'

@dataclass(frozen=True)
class Variant:
    limited_api: str

    def cflags(self):
        if self.limited_api is None:
            return []
        else:
            return [f'-DPy_LIMITED_API={hex(self.limited_api)}']

    def __repr__(self):
        if self.limited_api:
            if self.limited_api == 3:
                return f'<{3}>'
            else:
                major = (self.limited_api & 0xFF000000) >> 24
                minor = (self.limited_api & 0x00FF0000) >> 16
                rest = self.limited_api & ~0xFFFF0000
                if not rest:
                    return f'<{major}.{minor}>'
                else:
                    return f'<{self.limited_api:x}>'
        else:
            return '<cpy>'

@dataclass
class Definition:
    kind: str
    name: str
    args: str
    variants: dict[Variant, Location] = field(default_factory=dict, compare=False)

    def __str__(self):
        return CSI_RE.sub('', self.colorized)

    @property
    def colorized(self):
        return repr(self)


@dataclass
class MacroDef(Definition):
    kind = 'define'
    body: str = ''

    @property
    def colorized(self):
        return f'#define {INTENSE}{self.name}{RESET}{self.args or ''} {self.body}'

@dataclass
class MacroUndef(Definition):
    kind = 'undef'

    @property
    def colorized(self):
        return f'#undef {DIM}{self.name}{RESET}{self.args or ''}'


@dataclass
class Entry:
    name: str
    definitions: list[Definition]

def add_entries(entries, variant=None):
    gcc_proc = subprocess.Popen(
        [
            'gcc',
            '-E', '-dD',
            '-I.', '-IInclude/',
            *shlex.split(CFLAGS),
            *variant.cflags(),
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

    def add_definition(definition):
        location = Location(
            filename=filename,
            lineno=lineno,
        )
        entry = get_entry(definition.name)
        for old_definition in reversed(entry.definitions):
            if variant in old_definition.variants:
                break
            if old_definition == definition:
                old_definition.variants[variant] = location
                return
        definition.variants[variant] = location
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
                            add_definition(MacroDef(
                                kind = 'define',
                                name=name, args=args, body=body,
                            ))
                        elif line.startswith('#undef'):
                            content = line[len('#undef'):].strip()
                            name, args, body = split_macro_def(content)
                            assert not body
                            add_definition(MacroUndef(
                                kind='undef',
                                name=name, args=args,
                            ))
                        else:
                            print(line)
            except:
                print('Line was:', line)
                raise
            lineno += 1

    if gcc_proc.wait():
        raise Exception(f'gcc failed with return code {gcc_proc.returncode}')

entries = {}
add_entries(entries, Variant(None))
add_entries(entries, Variant(3))
add_entries(entries, Variant(0x03020000))
add_entries(entries, Variant(0x03030000))
add_entries(entries, Variant(0x03040000))
add_entries(entries, Variant(0x03050000))
add_entries(entries, Variant(0x03060000))
add_entries(entries, Variant(0x03070000))
add_entries(entries, Variant(0x03080000))
add_entries(entries, Variant(0x03090000))
add_entries(entries, Variant(0x030a0000))
add_entries(entries, Variant(0x030b0000))
add_entries(entries, Variant(0x030c0000))
add_entries(entries, Variant(0x030d0000))

for entry in entries.values():
    color = CYAN
    if not entry.name.startswith(('Py', 'PY', '_Py', '_PY')):
        color = RED
    print(f'{color}{entry.name}{RESET}')
    for definition in entry.definitions:
        loc_to_var = {}
        for variant, location in definition.variants.items():
            loc_to_var.setdefault(location, []).append(variant)
        print('   ', definition.colorized)
        for location, variants in loc_to_var.items():
            print('       ', variants, location)
