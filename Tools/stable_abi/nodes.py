import dataclasses
import typing

@dataclasses.dataclass
class StructInfo:
    name: str
    changes: typing.Sequence = ()
    fields: typing.Sequence = ()
    abi_only: bool = False

    @classmethod
    def make(cls, name, entries):
        return cls(name, **handle_entries(entries))

    def dump(self):
        return f'struct {self.name}\n' + _indent(
            _abionly_repr(self)
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
class FieldInfo:
    type: TypeInfo
    abi_only: bool = False
    changes: typing.Sequence = ()

    @classmethod
    def make(cls, typ, entries=()):
        return cls(typ, **handle_entries(entries))

    @property
    def name(self):
        return self.type.name

    def dump(self):
        return f'field {self.type.dump()} # {self.name}\n' + _indent(
            _abionly_repr(self)
            + ''.join(e.dump() for e in self.changes)
        )

@dataclasses.dataclass
class FunctionInfo:
    name: str
    return_type: object = None  # XXX
    args: typing.Sequence = ()
    abi_only: bool = False
    changes: typing.Sequence = ()

    @classmethod
    def make(cls, name, entries=()):
        return cls(name, **handle_entries(entries))

    def dump(self):
        return f'function {self.name}\n' + _indent(
            _abionly_repr(self)
            + ''.join(e.dump() for e in self.changes)
            + (''.join(e.dump() for e in self.args) if self.args else '')
            + (self.return_type.dump() + '\n' if self.return_type else '')
        )

@dataclasses.dataclass
class DataInfo:
    name: str
    abi_only: bool = False
    changes: typing.Sequence = ()

    @classmethod
    def make(cls, name, entries=()):
        return cls(name, **handle_entries(entries))

    def dump(self):
        return f'data {self.name}\n' + _indent(
            _abionly_repr(self)
            + ''.join(e.dump() for e in self.changes)
        )

@dataclasses.dataclass
class TypedefInfo:
    name: str
    abi_only: bool = False
    changes: typing.Sequence = ()

    @classmethod
    def make(cls, name, entries=()):
        return cls(name, **handle_entries(entries))
    def dump(self):
        return f'typedef {self.name}\n' + _indent(
            _abionly_repr(self)
            + ''.join(e.dump() for e in self.changes)
        )

@dataclasses.dataclass
class MacroInfo:
    name: str
    abi_only: bool = False
    changes: typing.Sequence = ()

    @classmethod
    def make(cls, name, entries=()):
        changes = []
        return cls(name, **handle_entries(entries))

    def dump(self):
        return f'macro {self.name}\n' + _indent(
            _abionly_repr(self)
            + ''.join(e.dump() for e in self.changes)
        )

@dataclasses.dataclass
class ArgInfo:
    type: str
    abi_only: bool = False
    changes: typing.Sequence = ()

    @classmethod
    def make(cls, typ, entries=()):
        return cls(typ, **handle_entries(entries))

    @property
    def name(self):
        return self.type.name

    def dump(self):
        return f'arg {self.type.dump()} # {self.name}\n' + _indent(
            _abionly_repr(self)
            + ''.join(e.dump() for e in self.changes)
        )

@dataclasses.dataclass
class ReturnInfo:
    name: str
    abi_only: bool = False
    changes: typing.Sequence = ()

    @classmethod
    def make(cls, name, entries=()):
        changes = []
        return cls(name, **handle_entries(entries))

    def dump(self):
        return f'return {self.name.dump()}\n' + _indent(
            _abionly_repr(self)
            + ''.join(e.dump() for e in self.changes)
        )


@dataclasses.dataclass
class AbiOnly:
    pass


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

def _abionly_repr(node):
    if node.abi_only:
        return 'abi_only\n'
    return ''

def _indent(string):
    if not string:
        return ''
    result = '  ' + '\n  '.join(string.split('\n'))
    if result.endswith('\n  '):
        result = result[:-2]
    return result

def handle_entries(entries):
    result = {}
    for entry in entries:
        if isinstance(entry, AbiOnly):
            result['abi_only'] = True
        elif isinstance(entry, ChangeInfo):
            result.setdefault('changes', []).append(entry)
        elif isinstance(entry, FieldInfo):
            fields = result.setdefault('fields', {})
            # XXX:
            #if entry.name in fields:
            #    raise ValueError(f'Repeated field {entry.name}')
            fields[entry.name] = entry
        elif isinstance(entry, ArgInfo):
            # XXX: Handle repeats
            result.setdefault('args', []).append(entry)
        elif isinstance(entry, ReturnInfo):
            result['return_type'] = entry
        else:
            raise TypeError(type(entry))
    return result
