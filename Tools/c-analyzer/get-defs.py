from dataclasses import dataclass, field, KW_ONLY
from functools import lru_cache, total_ordering, cached_property, partial
from pathlib import Path
import concurrent.futures
import collections.abc
import subprocess
import contextlib
import sysconfig
import shlex
import json
import sys
import ast
import re

from clang.cindex import Index
import clang.cindex

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

@dataclass(frozen=True, order=True)
class Location:
    filename: str
    lineno: int

    def __repr__(self):
        return f'<{self}>'

    def __str__(self):
        return f'{self.filename}:{self.lineno}'

@lru_cache
@total_ordering
@dataclass(frozen=True)
class Variant:
    limited_api: str

    def cflags(self):
        if self.limited_api is None:
            return []
        else:
            return [f'-DPy_LIMITED_API={hex(self.limited_api)}']

    def __str__(self):
        if self.limited_api:
            if self.limited_api == 3:
                return f'L3'
            else:
                major = (self.limited_api & 0xFF000000) >> 24
                minor = (self.limited_api & 0x00FF0000) >> 16
                rest = self.limited_api & ~0xFFFF0000
                if not rest:
                    return f'L{major}.{minor}'
                else:
                    return f'L{self.limited_api:x}'
        else:
            return 'cpy'

    def __lt__(self, other):
        return (self.limited_api or 0) < (other.limited_api or 0)

    @cached_property
    def next(self):
        if self.limited_api == 3:
            return Variant(limited_api=0x03020000)
        elif self.limited_api:
            major = (self.limited_api & 0xFF000000) >> 24
            minor = (self.limited_api & 0x00FF0000) >> 16
            rest = self.limited_api & ~0xFFFF0000
            return Variant(limited_api=(major << 24) | ((minor+1) << 16) | rest)

    def __repr__(self):
        return f'<{self}>'

@dataclass
class Definition:
    name: str
    variants: dict[Variant, Location] = field(default_factory=dict, compare=False)

    def __str__(self):
        return CSI_RE.sub('', self.colorized)

    @property
    def colorized(self):
        return repr(self)


@dataclass
class MacroDef(Definition):
    body: str = ''
    args: str = ''

    @property
    def colorized(self):
        return f'#define {INTENSE}{self.name}{RESET}{self.args or ""} {self.body}'

@dataclass
class MacroUndef(Definition):
    args: str = ''
    @property
    def colorized(self):
        return f'#undef {DIM}{self.name}{RESET}{self.args or ""}'

@dataclass
class FuncDef(Definition):
    _: KW_ONLY
    name: str
    type: object
    params: list
    body: object
    variadic: bool

    @property
    def colorized(self):
        params = ', '.join(str(p) for p in self.params)
        if self.variadic:
            params += ', ...'
        return (
            f'fn {DIM}{self.type}{RESET}{INTENSE}{self.name}{RESET}'
            + f'({DIM}{params}{RESET}) {self.body}'
        )

@dataclass
class Entry:
    name: str
    definitions: list[Definition]

class Entries(dict):
    def add_definition(self, definition, *, variant, location):
        name = definition.name
        try:
            entry = self[name]
        except KeyError:
            entry = self[name] = Entry(name, [])

        # Find the oldest definition that
        # - has the same content (__eq__)
        # - doesn't include this variant yet
        # - isn't followed by another definition of this variant (e.g. #undef)
        # If there is one, only add a new variant/location to it.
        # Otherwise, append this `definition` to our entry.
        definition_to_join = None
        for old_definition in reversed(entry.definitions):
            if variant in old_definition.variants:
                break
            if old_definition == definition:
                definition_to_join = old_definition
        if definition_to_join:
            definition_to_join.variants[variant] = location
        else:
            definition.variants[variant] = location
            entry.definitions.append(definition)

def add_entries(entries, variant=None):
    gcc_proc = subprocess.Popen(
        [
            'gcc',
            '-E', '-dD',
            '-I.', '-IInclude/',
            *shlex.split(CFLAGS),
            *variant.cflags(),
            'Include/Python.h',
            '-D_Float32=float',
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding='utf-8',
        cwd=projectpath,
        errors='backslashreplace',
    )

    clang_proc = subprocess.Popen(
        [
            'clang',
            '-Xclang', '-ast-dump=json', '-fsyntax-only',
            '-I.', '-IInclude/',
            *shlex.split(CFLAGS),
            *variant.cflags(),
            'Include/Python.h',
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
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
        entries.add_definition(definition, location=location, variant=variant)

    def split_macro_def(content):
        match = re.fullmatch(
            r'([a-zA-Z0-9_-]*)(\([^)]*\))?(.*)',
            content,
        )
        name = match[1]
        args = match[2]
        body = match[3].strip()
        return name, args, body

    filename = '<start>'
    lineno = 0
    with contextlib.ExitStack() as exitstack:
        executor = exitstack.enter_context(concurrent.futures.ThreadPoolExecutor(1))
        ast_future = executor.submit(json.load, clang_proc.stdout)
        exitstack.callback(gcc_proc.stdout.close)
        exitstack.callback(gcc_proc.stderr.close)
        exitstack.callback(clang_proc.stdout.close)
        exitstack.callback(clang_proc.stderr.close)

        for orig_line in exitstack.enter_context(gcc_proc.stdout):
            line = orig_line.rstrip('\n')
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
                                name=name, args=args, body=body,
                            ))
                        elif line.startswith('#undef'):
                            content = line[len('#undef'):].strip()
                            name, args, body = split_macro_def(content)
                            assert not body
                            add_definition(MacroUndef(
                                name=name, args=args,
                            ))
                        else:
                            print(line)
            except:
                print('Line was:', repr(line), f'({filename}:{lineno})')
                raise
            lineno += 1

        for name, proc in ('gcc', gcc_proc), ('clang', clang_proc):
            for line in proc.stderr:
                print(line, file=sys.stderr, end='')
            if proc.wait():
                print(file=sys.stderr, end='')
                raise Exception(f'{name} failed with return code {proc.returncode}')

    index = Index.create()
    args = [
        '-I.', '-IInclude/',
        '-xc',
        #'-I/usr/lib64/clang/15.0.7/include/', '-I/usr/include/', '-I/usr/include/linux/',
        #*shlex.split(CFLAGS),
        *variant.cflags(),
        'Include/Python.h.c',
    ]
    tu = index.parse(None, args)
    if not tu:
        raise Exception('unable to load input')
    for diag in tu.diagnostics:
        print({
            "severity": diag.severity,
            "location": diag.location,
            "spelling": diag.spelling,
            "ranges": diag.ranges,
            "fixits": list(diag.fixits),
        })
    dump_info(tu)

    exit('.')

    clang_translation_unit = ASTNode(ast_future.result(1))
    clang_translation_unit.dump()

    add_clang_ast(entries, clang_translation_unit, variant=variant)

def dump_info(node, depth=0):
    #children = [get_info(c, depth + 1) for c in node.get_children()]
    print(node, dir(node))
    print(clang.cindex, dir(clang.cindex))
    print({
        "kind": type(node).__name__,
        "spelling": node.spelling,
        #"location": node.get_extent(),
        "c": list(node),
    })
    return
    return {
        #"id": get_cursor_id(node),
        "usr": node.get_usr(),
        "extent.start": node.extent.start,
        "extent.end": node.extent.end,
        "is_definition": node.is_definition(),
        #"definition id": get_cursor_id(node.get_definition()),
        #"children": children,
    }

class ASTNode(collections.abc.Mapping):
    _registry = {}
    @classmethod
    def register(cls):
        def decorator(cls):
            cls._registry[cls.__name__] = cls
            return cls
        return decorator

    def __new__(cls, data):
        if cls is __class__:
            try:
                cls = cls._registry[data['kind']]
            except KeyError:
                return super().__new__(cls)
            else:
                return super().__new__(cls)
        return cls.__new__(cls, data)

    def __init__(self, data):
        self._data = data

    def __getitem__(self, key): return self._data[key]
    def __iter__(self): return iter(self._data)
    def __len__(self): return len(self._data)

    @cached_property
    def inner(self):
        return tuple(ASTNode(d) for d in self.get('inner', ()))

    @cached_property
    def kind(self):
        return self.get('kind', None)

    def __str__(self):
        return str(self.kind)

    def dump(self, indent=0):
        print('  ' * indent + str(self), type(self), self.get('name'))
        for child in self.inner:
            child.dump(indent=indent + 1)

@ASTNode.register()
class TypedefDecl(ASTNode):
    @cached_property
    def name(self):
        return self['name']

@ASTNode.register()
class FunctionDecl(ASTNode):
    @cached_property
    def name(self):
        return self['name']

    @cached_property
    def location(self):
        return make_location(self['loc'])

    @cached_property
    def type(self):
        return self['type']['qualType']

    @cached_property
    def params(self):
        params = []
        for child in self.inner:
            match child:
                case ParmVarDecl():
                    params.append(child)
        return params

    @cached_property
    def variadic(self):
        return self.get('variadic')

    @cached_property
    def body(self):
        for child in self.inner:
            match child:
                case CompoundStatement():
                    return child

@ASTNode.register()
class ParmVarDecl(ASTNode):
    def __str__(self):
        return f'{self.type} {self.name}'

    @cached_property
    def name(self):
        return self.get('name', '$')

    @cached_property
    def type(self):
        return self.get('type', {}).get('qualType')

@ASTNode.register()
class CompoundStatement(ASTNode):
    pass

def make_location(loc):
    match loc:
        case {'file': filename, 'line': lineno}:
            return Location(filename=filename, lineno=lineno)
        case {'line': lineno}:
            return Location(filename=f'<???>', lineno=lineno)
        case {'expansionLoc': inner}:
            return make_location(inner)
        case {}:
            return Location(filename=f'<???>', lineno=-1)
    raise ValueError(loc)

def add_clang_ast(entries, clang_translation_unit, variant):
    for child in clang_translation_unit.inner:
        match child:
            case FunctionDecl():
                entries.add_definition(
                    FuncDef(
                        name=child.name,
                        type=child.type,
                        params=child.params,
                        body=child.body,
                        variadic=child.variadic,
                    ),
                    location=(l:=child.location),
                    variant=variant,
                )

entries = Entries()
add_entries(entries, Variant(None))
add_entries(entries, Variant(3))
# add_entries(entries, Variant(0x03020000))
# add_entries(entries, Variant(0x03030000))
# add_entries(entries, Variant(0x03040000))
# add_entries(entries, Variant(0x03050000))
# add_entries(entries, Variant(0x03060000))
# add_entries(entries, Variant(0x03070000))
# add_entries(entries, Variant(0x03080000))
# add_entries(entries, Variant(0x03090000))
# add_entries(entries, Variant(0x030a0000))
# add_entries(entries, Variant(0x030b0000))
# add_entries(entries, Variant(0x030c0000))
# add_entries(entries, Variant(0x030d0000))

@lru_cache
def format_variant_range(variants):
    variants = sorted(variants)
    results = []
    for variant in variants:
        if results and results[-1][-1].next == variant:
            results[-1][-1] = variant
        else:
            results.append([variant, variant])
    return '[' + ', '.join(
        'L' if start == Variant(3) and stop == Variant(0x030d0000)
        else f'{start}+' if stop == Variant(0x030d0000)
        else f'{start}' if start == stop
        else f'{start}-{stop}'
        for start, stop in results
    ) + ']'

for entry in entries.values():
    color = CYAN
    if not entry.name.startswith(('Py', 'PY', '_Py', '_PY')):
        color = RED
    print(f'{color}{entry.name}{RESET}')
    for definition in entry.definitions:
        loc_to_var = {}
        for variant, location in definition.variants.items():
            loc_to_var.setdefault(location, []).append(variant)
        print(' ' * 3, definition.colorized)
        for location, variants in loc_to_var.items():
            print(' ' * 7, format_variant_range(frozenset(variants)), location)
