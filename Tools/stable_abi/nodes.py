import dataclasses
import typing
import collections.abc

@dataclasses.dataclass
class ABIDef(collections.abc.Mapping):
    entries: dict

    def __init__(self, entries):
        self.entries = {}
        for entry in entries:
            if entry.name in self.entries:
                raise ValueError(f'Duplicate entry: {entry.name}')
            self.entries[entry.name] = entry

    def __getitem__(self, name):
        return self.entries[name]

    def __iter__(self):
        return iter(self.entries)

    def __len__(self):
        return len(self.entries)

    def dump(self):
        return ''.join(e.dump() for e in self.entries.values())

    @property
    def functions(self):
        return (
            e for e in self.entries.values()
            if isinstance(e, FunctionInfo) and not e.hard_removed
        )

    @property
    def data(self):
        return (
            e for e in self.entries.values()
            if isinstance(e, DataInfo) and not e.hard_removed
        )

@dataclasses.dataclass
class StructInfo:
    name: str
    changes: typing.Sequence = ()
    fields: typing.Sequence = ()
    abi_only: bool = False
    hard_removed: bool = False

    @classmethod
    def make(cls, name, entries):
        return cls(name, **handle_entries(entries))

    def dump(self):
        return f'struct {self.name}\n' + _indent(
            _flags_repr(self)
            + ''.join(e.dump() for e in self.changes)
            + (''.join(e.dump() for e in self.fields.values()) if self.fields else '')
        )

@dataclasses.dataclass
class TypeInfo:
    pass

@dataclasses.dataclass
class ChangeInfo:
    type: str
    version_info: tuple

    @classmethod
    def make(cls, name, version_string):
        return cls(name, tuple(int(v) for v in version_string.split('.')))

    def dump(self):
        return f'{self.type} {".".join(str(v) for v in self.version_info)}\n'

@dataclasses.dataclass
class IfdefInfo:
    name: str

    @classmethod
    def make(cls, name):
        return cls(name)

    def dump(self):
        return f'ifdef {self.name}\n'

@dataclasses.dataclass
class FieldInfo:
    type: TypeInfo
    abi_only: bool = False
    hard_removed: bool = False
    changes: typing.Sequence = ()

    @classmethod
    def make(cls, typ, entries=()):
        return cls(typ, **handle_entries(entries))

    @property
    def name(self):
        return self.type.name

    def dump(self):
        return f'field {self.type.dump()} # {self.name}\n' + _indent(
            _flags_repr(self)
            + ''.join(e.dump() for e in self.changes)
        )

@dataclasses.dataclass
class FunctionInfo:
    name: str
    return_type: object = None  # XXX
    args: typing.Mapping = dataclasses.field(default_factory=dict)
    abi_only: bool = False
    hard_removed: bool = False
    changes: typing.Sequence = ()
    ifdef: object = None

    @classmethod
    def make(cls, name, entries=()):
        return cls(name, **handle_entries(entries))

    def dump(self):
        return f'function {self.name}\n' + _indent(
            _flags_repr(self)
            + ''.join(e.dump() for e in self.changes)
            + ''.join(e.dump() for e in self.args.values())
            + (self.return_type.dump() + '\n' if self.return_type else '')
            + (self.ifdef.dump() if self.ifdef else '')
        )

@dataclasses.dataclass
class DataInfo:
    name: str
    abi_only: bool = False
    hard_removed: bool = False
    changes: typing.Sequence = ()
    ifdef: object = None

    @classmethod
    def make(cls, name, entries=()):
        return cls(name, **handle_entries(entries))

    def dump(self):
        return f'data {self.name}\n' + _indent(
            _flags_repr(self)
            + ''.join(e.dump() for e in self.changes)
            + (self.ifdef.dump() if self.ifdef else '')
        )

@dataclasses.dataclass
class ConstInfo:
    name: str
    abi_only: bool = False
    hard_removed: bool = False
    changes: typing.Sequence = ()

    @classmethod
    def make(cls, name, entries=()):
        return cls(name, **handle_entries(entries))

    def dump(self):
        return f'const {self.name}\n' + _indent(
            _flags_repr(self)
            + ''.join(e.dump() for e in self.changes)
        )

@dataclasses.dataclass
class TypedefInfo:
    name: str
    abi_only: bool = False
    hard_removed: bool = False
    changes: typing.Sequence = ()

    @classmethod
    def make(cls, name, entries=()):
        return cls(name, **handle_entries(entries))
    def dump(self):
        return f'typedef {self.name}\n' + _indent(
            _flags_repr(self)
            + ''.join(e.dump() for e in self.changes)
        )

@dataclasses.dataclass
class MacroInfo:
    name: str
    abi_only: bool = False
    hard_removed: bool = False
    changes: typing.Sequence = ()

    @classmethod
    def make(cls, name, entries=()):
        changes = []
        return cls(name, **handle_entries(entries))

    def dump(self):
        return f'macro {self.name}\n' + _indent(
            _flags_repr(self)
            + ''.join(e.dump() for e in self.changes)
        )

@dataclasses.dataclass
class ArgInfo:
    type: str
    abi_only: bool = False
    hard_removed: bool = False
    changes: typing.Sequence = ()

    @classmethod
    def make(cls, typ, entries=()):
        return cls(typ, **handle_entries(entries))

    @property
    def name(self):
        return self.type.name

    def dump(self):
        return f'arg {self.type.dump()} # {self.name}\n' + _indent(
            _flags_repr(self)
            + ''.join(e.dump() for e in self.changes)
        )

@dataclasses.dataclass
class ReturnInfo:
    name: str
    abi_only: bool = False
    hard_removed: bool = False
    changes: typing.Sequence = ()

    @classmethod
    def make(cls, name, entries=()):
        changes = []
        return cls(name, **handle_entries(entries))

    def dump(self):
        return f'return {self.name.dump()}\n' + _indent(
            _flags_repr(self)
            + ''.join(e.dump() for e in self.changes)
        )


@dataclasses.dataclass
class FlagInfo:
    name: str


@dataclasses.dataclass
class TypeName:
    name: str
    qualifiers: typing.Sequence = ()

    def dump(self):
        return f'{" ".join(self.qualifiers)} {self.name if self.name else ""}'


@dataclasses.dataclass
class TypeDecl:
    specqual: object
    declarator: object = None

    def dump(self):
        return f'{self.specqual.dump()} {self.declarator.dump()}'

    @property
    def name(self):
        return self.declarator.name


@dataclasses.dataclass
class Pointer:
    qualifiers: list
    declarator: object = None

    def dump(self):
        return f'{" ".join(self.qualifiers)} *{self.declarator.dump() if self.declarator else ""}'

    @property
    def name(self):
        return self.declarator.name

@dataclasses.dataclass
class FuncDecl:
    declarator: object
    params: typing.Sequence

    def dump(self):
        if isinstance(self.params, NoParams):
            params_repr = 'void'
        else:
            params_repr = ", ".join(p.dump() for p in self.params)
        return f'({self.declarator.dump()})({params_repr})'

    @property
    def name(self):
        return self.declarator.name

@dataclasses.dataclass
class NoParams:
    def dump(self):
        return 'void'

    def __iter__(self):
        return iter(())

def _flags_repr(node):
    fr = []
    if node.abi_only:
        fr.append('abi_only\n')
    if node.hard_removed:
        fr.append('hard_removed\n')
    return ''.join(fr)

def _indent(string):
    if not string:
        return ''
    result = '  ' + '\n  '.join(string.split('\n'))
    if result.endswith('\n  '):
        result = result[:-2]
    return result

def _add_named_entry(mapping, entry, entry_category):
    if entry.name in mapping:
        raise ValueError(f'Repeated {entry_category} {entry.name}')
    mapping[entry.name] = entry

def handle_entries(entries):
    result = {}
    for entry in entries:
        if isinstance(entry, FlagInfo):
            result[entry.name] = True
        elif isinstance(entry, ChangeInfo):
            result.setdefault('changes', []).append(entry)
        elif isinstance(entry, FieldInfo):
            fields = result.setdefault('fields', {})
            _add_named_entry(fields, entry, 'field')
        elif isinstance(entry, ArgInfo):
            args = result.setdefault('args', {})
            _add_named_entry(args, entry, 'arg')
        elif isinstance(entry, ReturnInfo):
            result['return_type'] = entry
        elif isinstance(entry, IfdefInfo):
            result['ifdef'] = entry
        else:
            raise TypeError(type(entry))
    return result
