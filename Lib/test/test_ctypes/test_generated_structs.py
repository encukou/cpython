"""Test CTypes structs, unions, bitfields against C equivalents.

The types here are auto-converted to C source at
`Modules/_ctypes/_ctypes_test_generated.c.h`, which is compiled into
_ctypes_test.

Run this module to regenerate the files:

./python Lib/test/test_ctypes/test_generated_structs.py > Modules/_ctypes/_ctypes_test_generated.c.h
"""

import unittest
from test.support import import_helper
import re
from dataclasses import dataclass
from functools import cached_property
import sys

import ctypes
from ctypes import Structure, Union
from ctypes import sizeof, alignment, pointer, string_at
_ctypes_test = import_helper.import_module("_ctypes_test")


# ctypes erases the difference between `c_int` and e.g.`c_int16`.
# To keep it, we'll use custom subclasses with the C name stashed in `_c_name`:
class c_bool(ctypes.c_bool):
    _c_name = '_Bool'

# To do it for all the other types, use some metaprogramming:
for c_name, ctypes_name in {
    'signed char': 'c_byte',
    'short': 'c_short',
    'int': 'c_int',
    'long': 'c_long',
    'long long': 'c_longlong',
    'unsigned char': 'c_ubyte',
    'unsigned short': 'c_ushort',
    'unsigned int': 'c_uint',
    'unsigned long': 'c_ulong',
    'unsigned long long': 'c_ulonglong',
    **{f'{u}int{n}_t': f'c_{u}int{n}'
       for u in ('', 'u')
       for n in (8, 16, 32, 64)}
}.items():
    ctype = getattr(ctypes, ctypes_name)
    newtype = type(ctypes_name, (ctype,), {'_c_name': c_name})
    globals()[ctypes_name] = newtype


# Register structs and unions to test

TESTCASES = {}
def register(name=None, set_name=False):
    def decorator(cls, name=name):
        if name is None:
            name = cls.__name__
        assert name.isascii()  # will be used in _PyUnicode_EqualToASCIIString
        assert name.isidentifier()  # will be used as a C identifier
        assert name not in TESTCASES
        TESTCASES[name] = cls
        if set_name:
            cls.__name__ = name
        return cls
    return decorator

@register()
class SingleInt(Structure):
    _fields_ = [('a', c_int)]

@register()
class SingleInt_Union(Union):
    _fields_ = [('a', c_int)]


@register()
class SingleU32(Structure):
    _fields_ = [('a', c_uint32)]


@register()
class SimpleStruct(Structure):
    _fields_ = [('x', c_int32), ('y', c_int8), ('z', c_uint16)]


@register()
class SimpleUnion(Union):
    _fields_ = [('x', c_int32), ('y', c_int8), ('z', c_uint16)]


@register()
class ManyTypes(Structure):
    _fields_ = [
        ('i8', c_int8), ('u8', c_uint8),
        ('i16', c_int16), ('u16', c_uint16),
        ('i32', c_int32), ('u32', c_uint32),
        ('i64', c_int64), ('u64', c_uint64),
    ]


@register()
class ManyTypesU(Union):
    _fields_ = [
        ('i8', c_int8), ('u8', c_uint8),
        ('i16', c_int16), ('u16', c_uint16),
        ('i32', c_int32), ('u32', c_uint32),
        ('i64', c_int64), ('u64', c_uint64),
    ]


@register()
class Nested(Structure):
    _fields_ = [
        ('a', SimpleStruct), ('b', SimpleUnion), ('anon', SimpleStruct),
    ]
    _anonymous_ = ['anon']


@register()
class Packed1(Structure):
    _fields_ = [('a', c_int8), ('b', c_int64)]
    _pack_ = 1


@register()
class Packed2(Structure):
    _fields_ = [('a', c_int8), ('b', c_int64)]
    _pack_ = 2


@register()
class Packed3(Structure):
    _fields_ = [('a', c_int8), ('b', c_int64)]
    _pack_ = 4


@register()
class Packed4(Structure):
    _fields_ = [('a', c_int8), ('b', c_int64)]
    _pack_ = 8

@register()
class X86_32EdgeCase(Structure):
    # On a Pentium, long long (int64) is 32-bit aligned,
    # so these are packed tightly.
    _fields_ = [('a', c_int32), ('b', c_int64), ('c', c_int32)]

@register()
class MSBitFieldExample(Structure):
    # From https://learn.microsoft.com/en-us/cpp/c-language/c-bit-fields
    _fields_ = [
        ('a', c_uint, 4),
        ('b', c_uint, 5),
        ('c', c_uint, 7)]

@register()
class MSStraddlingExample(Structure):
    # From https://learn.microsoft.com/en-us/cpp/c-language/c-bit-fields
    _fields_ = [
        ('first', c_uint, 9),
        ('second', c_uint, 7),
        ('may_straddle', c_uint, 30),
        ('last', c_uint, 18)]

@register()
class IntBits(Structure):
    _fields_ = [("A", c_int, 1),
                ("B", c_int, 2),
                ("C", c_int, 3),
                ("D", c_int, 4),
                ("E", c_int, 5),
                ("F", c_int, 6),
                ("G", c_int, 7),
                ("H", c_int, 8),
                ("I", c_int, 9)]

@register()
class Bits(Structure):
    _fields_ = [*IntBits._fields_,

                ("M", c_short, 1),
                ("N", c_short, 2),
                ("O", c_short, 3),
                ("P", c_short, 4),
                ("Q", c_short, 5),
                ("R", c_short, 6),
                ("S", c_short, 7)]

@register()
class IntBits_MSVC(Structure):
    _layout_ = "ms"
    _fields_ = [("A", c_int, 1),
                ("B", c_int, 2),
                ("C", c_int, 3),
                ("D", c_int, 4),
                ("E", c_int, 5),
                ("F", c_int, 6),
                ("G", c_int, 7),
                ("H", c_int, 8),
                ("I", c_int, 9)]

@register()
class Bits_MSVC(Structure):
    _layout_ = "ms"
    _fields_ = [*IntBits_MSVC._fields_,

                ("M", c_short, 1),
                ("N", c_short, 2),
                ("O", c_short, 3),
                ("P", c_short, 4),
                ("Q", c_short, 5),
                ("R", c_short, 6),
                ("S", c_short, 7)]

# Skipped for now -- we don't always match the alignment
#@register()
class IntBits_Union(Union):
    _fields_ = [("A", c_int, 1),
                ("B", c_int, 2),
                ("C", c_int, 3),
                ("D", c_int, 4),
                ("E", c_int, 5),
                ("F", c_int, 6),
                ("G", c_int, 7),
                ("H", c_int, 8),
                ("I", c_int, 9)]

# Skipped for now -- we don't always match the alignment
#@register()
class BitsUnion(Union):
    _fields_ = [*IntBits_Union._fields_,

                ("M", c_short, 1),
                ("N", c_short, 2),
                ("O", c_short, 3),
                ("P", c_short, 4),
                ("Q", c_short, 5),
                ("R", c_short, 6),
                ("S", c_short, 7)]

@register()
class I64Bits(Structure):
    _fields_ = [("a", c_int64, 1),
                ("b", c_int64, 62),
                ("c", c_int64, 1)]

@register()
class U64Bits(Structure):
    _fields_ = [("a", c_uint64, 1),
                ("b", c_uint64, 62),
                ("c", c_uint64, 1)]

for n in 8, 16, 32, 64:
    for signedness in '', 'u':
        ctype = globals()[f'c_{signedness}int{n}']

        @register(f'Struct331_{signedness}{n}', set_name=True)
        class _cls(Structure):
            _fields_ = [("a", ctype, 3),
                        ("b", ctype, 3),
                        ("c", ctype, 1)]

        @register(f'Struct1x1_{signedness}{n}', set_name=True)
        class _cls(Structure):
            _fields_ = [("a", ctype, 1),
                        ("b", ctype, n-2),
                        ("c", ctype, 1)]

        @register(f'Struct1nx1_{signedness}{n}', set_name=True)
        class _cls(Structure):
            _fields_ = [("a", ctype, 1),
                        ("full", ctype),
                        ("b", ctype, n-2),
                        ("c", ctype, 1)]

        @register(f'Struct3xx_{signedness}{n}', set_name=True)
        class _cls(Structure):
            _fields_ = [("a", ctype, 3),
                        ("b", ctype, n-2),
                        ("c", ctype, n-2)]

@register()
class Mixed1(Structure):
    _fields_ = [("a", c_byte, 4),
                ("b", c_int, 4)]

@register()
class Mixed2(Structure):
    _fields_ = [("a", c_byte, 4),
                ("b", c_int32, 32)]

@register()
class Mixed3(Structure):
    _fields_ = [("a", c_byte, 4),
                ("b", c_ubyte, 4)]

@register()
class Mixed4(Structure):
    _fields_ = [("a", c_short, 4),
                ("b", c_short, 4),
                ("c", c_int, 24),
                ("d", c_short, 4),
                ("e", c_short, 4),
                ("f", c_int, 24)]

@register()
class Mixed5(Structure):
    _fields_ = [('A', c_uint, 1),
                ('B', c_ushort, 16)]

@register()
class Mixed6(Structure):
    _fields_ = [('A', c_ulonglong, 1),
                ('B', c_uint, 32)]

@register()
class Mixed7(Structure):
    _fields_ = [("A", c_uint32),
                ('B', c_uint32, 20),
                ('C', c_uint64, 24)]

@register()
class Mixed8_a(Structure):
    _fields_ = [("A", c_uint32),
                ("B", c_uint32, 32),
                ("C", c_ulonglong, 1)]

@register()
class Mixed8_b(Structure):
    _fields_ = [("A", c_uint32),
                ("B", c_uint32),
                ("C", c_ulonglong, 1)]

@register()
class Mixed9(Structure):
    _fields_ = [("A", c_uint8),
                ("B", c_uint32, 1)]

@register()
class Mixed10(Structure):
    _fields_ = [("A", c_uint32, 1),
                ("B", c_uint64, 1)]

@register()
class Example_gh_95496(Structure):
    _fields_ = [("A", c_uint32, 1),
                ("B", c_uint64, 1)]

@register()
class Example_gh_84039_bad(Structure):
    _pack_ = 1
    _fields_ = [("a0", c_uint8, 1),
                ("a1", c_uint8, 1),
                ("a2", c_uint8, 1),
                ("a3", c_uint8, 1),
                ("a4", c_uint8, 1),
                ("a5", c_uint8, 1),
                ("a6", c_uint8, 1),
                ("a7", c_uint8, 1),
                ("b0", c_uint16, 4),
                ("b1", c_uint16, 12)]

@register()
class Example_gh_84039_good_a(Structure):
    _pack_ = 1
    _fields_ = [("a0", c_uint8, 1),
                ("a1", c_uint8, 1),
                ("a2", c_uint8, 1),
                ("a3", c_uint8, 1),
                ("a4", c_uint8, 1),
                ("a5", c_uint8, 1),
                ("a6", c_uint8, 1),
                ("a7", c_uint8, 1)]

@register()
class Example_gh_84039_good(Structure):
    _pack_ = 1
    _fields_ = [("a", Example_gh_84039_good_a),
                ("b0", c_uint16, 4),
                ("b1", c_uint16, 12)]

@register()
class Example_gh_73939(Structure):
    _pack_ = 1
    _fields_ = [("P", c_uint16),
                ("L", c_uint16, 9),
                ("Pro", c_uint16, 1),
                ("G", c_uint16, 1),
                ("IB", c_uint16, 1),
                ("IR", c_uint16, 1),
                ("R", c_uint16, 3),
                ("T", c_uint32, 10),
                ("C", c_uint32, 20),
                ("R2", c_uint32, 2)]

@register()
class Example_gh_86098(Structure):
    _fields_ = [("a", c_uint8, 8),
                ("b", c_uint8, 8),
                ("c", c_uint32, 16)]

@register()
class Example_gh_86098_pack(Structure):
    _pack_ = 1
    _fields_ = [("a", c_uint8, 8),
                ("b", c_uint8, 8),
                ("c", c_uint32, 16)]

@register()
class Example_gh_59324_17(Structure):
    _fields_ = [("a", c_int32, 17),
                ("b", c_int16, 1)]

@register()
class Example_gh_59324_17_Swap(Structure):
    _swappedbytes_ = True
    _fields_ = [("a", c_int32, 13),
                ("b", c_int16, 5)]

@register()
class Example_gh_59324_13(Structure):
    _fields_ = [("a", c_int32, 13),
                ("b", c_int16, 5)]

@register()
class AnonBitfields(Structure):
    class X(Structure):
        _fields_ = [("a", c_byte, 4),
                    ("b", c_ubyte, 4)]
    _anonymous_ = ["_"]
    _fields_ = [("_", X), ('y', c_byte)]

@register()
class StructWithArrays(Structure):
    _fields_ = [("a", c_byte * 3),
                ("b", c_int32 * 5)]


@register()
class NestedStructWithArrays(Structure):
    _fields_ = [("anon", StructWithArrays),
                ("y", StructWithArrays * 3)]
    _anonymous_ = ['anon']

class GeneratedTest(unittest.TestCase):
    def test_generated_data(self):
        """Check that a ctypes struct/union matches its C equivalent.

        This compares with data from get_generated_test_data(), a list of:
        - name (str)
        - size (int)
        - alignment (int)
        - for each field, three snapshots of memory, as bytes:
            - memory after the field is set to -1
            - memory after the field is set to 1
            - memory after the field is set to 0

        or:
        - None
        - reason to skip the test (str)

        This does depend on the C compiler keeping padding bits zero.
        Common compilers seem to do so.
        """
        for name, cls in TESTCASES.items():
            with self.subTest(name=name):
                expected = iter(_ctypes_test.get_generated_test_data(name))
                expected_name = next(expected)
                if expected_name is None:
                    self.skipTest(next(expected))
                self.assertEqual(name, expected_name)
                self.assertEqual(sizeof(cls), next(expected))
                with self.subTest('alignment'):
                    self.assertEqual(alignment(cls), next(expected))
                obj = cls()
                ptr = pointer(obj)
                for field in iterfields(cls):
                    for value in -1, 1, 0:
                        with self.subTest(field=field.c_getter, value=value):
                            field.set_to(obj, value)
                            py_mem = string_at(ptr, sizeof(obj))
                            c_mem = next(expected)
                            if py_mem != c_mem:
                                # Generate a helpful failure message
                                lines, requires = dump_ctype(cls)
                                m = "\n".join([str(field), 'in:', *lines])
                                self.assertEqual(py_mem.hex(), c_mem.hex(), m)


# The rest of this file is generating C code from a ctypes type.
# This is only meant for (and tested with) the known inputs in this file!

def c_str_repr(string):
    """Return a string as a C literal"""
    return '"' + re.sub('([\"\'\\\\\n])', r'\\\1', string) + '"'

def dump_simple_ctype(tp, variable_name='', end=''):
    """Get C type name or declaration of a scalar type

    variable_name: if given, declare the given variable
    end: a semicolon, array length, and/or bitfield specification
         to tack on to the end
    """
    return f'{tp._c_name}{maybe_space(variable_name)}{end}'


def dump_ctype(tp, struct_or_union_tag='', variable_name='', end=''):
    """Get C type name or declaration of a ctype

    struct_or_union_tag: name of the struct or union
    variable_name: if given, declare the given variable
    end: a semicolon, array length, and/or bitfield specification
         to tack on to the end
    """
    if issubclass(tp, (Structure, Union)):
        requires = set()
        attributes = []
        pushes = []
        pops = []
        pack = getattr(tp, '_pack_', None)
        if pack is not None:
            pushes.append(f'#pragma pack(push, {pack})')
            pops.append(f'#pragma pack(pop)')
        layout = getattr(tp, '_layout_', None)
        if layout == 'ms' or pack:
            # The 'ms_struct' attribute only works on x86 and PowerPC
            requires.add(
                'defined(MS_WIN32) || ('
                    '(defined(__x86_64__) || defined(__i386__) || defined(__ppc64__)) && ('
                    'defined(__GNUC__) || defined(__clang__)))'
                )
            attributes.append('ms_struct')
        if getattr(tp, '_swappedbytes_', False):
            requires.add('defined(__GNUC__) || defined(__clang__)')
            if sys.byteorder == 'big':
                other = 'little'
            else:
                other = 'big'
            attributes.append(f'scalar_storage_order("{other}-endian")')
        if attributes:
            a = f' GCC_ATTR({", ".join(attributes)})'
        else:
            a = ''
        lines = [f'{struct_or_union(tp)}{a}{maybe_space(struct_or_union_tag)} ' +'{']
        for fielddesc in tp._fields_:
            f_name, f_tp, f_bits = unpack_field_desc(*fielddesc)
            if f_name in getattr(tp, '_anonymous_', ()):
                f_name = ''
            if f_bits is None:
                subend = ';'
            else:
                if f_tp not in (c_int, c_uint):
                    # XLC can reportedly only handle int & unsigned int
                    # bitfields (the only types required by C spec)
                    requires.add('!defined(__xlc__)')
                subend = f' :{f_bits};'
            sub_lines, sub_requires = dump_ctype(
                f_tp, variable_name=f_name, end=subend)
            requires.update(sub_requires)
            for line in sub_lines:
                lines.append('    ' + line)
        lines.append(f'}}{maybe_space(variable_name)}{end}')
        return [*pushes, *lines, *reversed(pops)], requires
    elif length := getattr(tp, '_length_', None):
        f_tp = tp._type_
        return dump_ctype(
                f_tp, variable_name=variable_name, end=f'[{length}]{end}')
    else:
        return [dump_simple_ctype(tp, variable_name, end)], set()

def struct_or_union(cls):
    if issubclass(cls, Structure):
         return 'struct'
    if issubclass(cls, Union):
        return 'union'
    raise TypeError(cls)

def maybe_space(string):
    if string:
        return ' ' + string
    return string

def unpack_field_desc(f_name, f_tp, f_bits=None):
    """Unpack a _fields_ entry into a (name, type, bits) triple"""
    return f_name, f_tp, f_bits

class InfoBase:
    @cached_property
    def root(self):
        if self.parent is None:
            return self
        else:
            return self.parent

    def __repr__(self):
        qname = f'{self.root.parent_type.__name__}{self.c_getter}'
        try:
            desc = self.descriptor
        except AttributeError:
            desc = '???'
        return f'<{type(self).__name__} for {qname}: {desc}>'

@dataclass(repr=False)
class FieldInfo(InfoBase):
    """Information about a (possibly nested) struct/union field"""
    name: str
    tp: type
    bits: int | None  # number if this is a bit field
    parent_type: type
    parent: 'InfoBase | None'

    @cached_property
    def c_getter(self):
        if self.name in getattr(self.parent_type, '_anonymous_', ()):
            g = ''
        else:
            g = f'.{self.name}'
        if self.parent:
            return self.parent.c_getter + g
        else:
            return g

    def get_from(self, obj):
        if self.parent:
            obj = self.parent.get_from(obj)
        return getattr(obj, self.name)

    def set_to(self, obj, new):
        """Set the field on a given Structure/Union instance"""
        if self.parent:
            obj = self.parent.get_from(obj)
        setattr(obj, self.name, new)

    @cached_property
    def descriptor(self):
        return getattr(self.parent_type, self.name)

@dataclass(repr=False)
class ElementInfo(InfoBase):
    """Information about a (possibly nested) array element"""
    index: int
    tp: type
    parent_type: type
    parent: 'InfoBase | None'

    bits = None

    @cached_property
    def c_getter(self):
        g = f'[{self.index}]'
        if self.parent:
            return self.parent.c_getter + g
        else:
            return g

    def get_from(self, obj):
        if self.parent:
            obj = self.parent.get_from(obj)
        return obj[self.index]

    def set_to(self, obj, new):
        """Set the field on a given Structure/Union instance"""
        if self.parent:
            obj = self.parent.get_from(obj)
        obj[self.index] = new

    @cached_property
    def root(self):
        if self.parent is None:
            return self
        else:
            return self.parent

def iterfields(tp, parent=None):
    """Get *leaf* fields of a structure or union, as FieldInfo"""
    fields = getattr(tp, '_fields_', None)
    if fields:
        for fielddesc in fields:
            f_name, f_tp, f_bits = unpack_field_desc(*fielddesc)
            sub = FieldInfo(f_name, f_tp, f_bits, tp, parent)
            yield from iterfields(f_tp, sub)
        return

    length = getattr(tp, '_length_', None)
    if length:
        for i in range(length):
            sub = ElementInfo(i, tp._type_, tp, parent)
            yield from iterfields(tp._type_, sub)
        return

    yield parent


if __name__ == '__main__':
    # Dump C source to stdout
    def output(string):
        print(re.compile(r'^ +$', re.MULTILINE).sub('', string).lstrip('\n'))
    output("""
        /* Generated by Lib/test/test_ctypes/test_generated_structs.py */


        // Append VALUE to the result.
        #define APPEND(ITEM) {                          \\
            PyObject *item = ITEM;                      \\
            if (!item) {                                \\
                Py_DECREF(result);                      \\
                return NULL;                            \\
            }                                           \\
            int rv = PyList_Append(result, item);       \\
            Py_DECREF(item);                            \\
            if (rv < 0) {                               \\
                Py_DECREF(result);                      \\
                return NULL;                            \\
            }                                           \\
        }

        // Set TARGET, and append a snapshot of `value`'s
        // memory to the result.
        #define SET_AND_APPEND(TYPE, TARGET, VAL) {     \\
            TYPE v = VAL;                               \\
            TARGET = v;                                 \\
            APPEND(PyBytes_FromStringAndSize(           \\
                (char*)&value, sizeof(value)));         \\
        }

        // Set a field to -1, 1 and 0; append a snapshot of the memory
        // after each of the operations.
        #define TEST_FIELD(TYPE, TARGET) {              \\
            SET_AND_APPEND(TYPE, TARGET, -1)            \\
            SET_AND_APPEND(TYPE, TARGET, 1)             \\
            SET_AND_APPEND(TYPE, TARGET, 0)             \\
        }

        #if defined(__GNUC__) || defined(__clang__)
        #define GCC_ATTR(X) __attribute__((X))
        #else
        #define GCC_ATTR(X) /* */
        #endif

        static PyObject *
        get_generated_test_data(PyObject *self, PyObject *name)
        {
            if (!PyUnicode_Check(name)) {
                PyErr_SetString(PyExc_TypeError, "need a string");
                return NULL;
            }
            PyObject *result = PyList_New(0);
            if (!result) {
                return NULL;
            }
    """)
    for name, cls in TESTCASES.items():
        output("""
            if (PyUnicode_CompareWithASCIIString(name, %s) == 0) {
            """ % c_str_repr(name))
        lines, requires = dump_ctype(cls, struct_or_union_tag=name, end=';')
        if requires:
            output(f"""
            #if {" && ".join(f'({r})' for r in sorted(requires))}
            """)
        for line in lines:
            output('                ' + line)
        typename = f'{struct_or_union(cls)} {name}'
        output(f"""
                {typename} value = {{0}};
                APPEND(PyUnicode_FromString({c_str_repr(name)}));
                APPEND(PyLong_FromLong(sizeof({typename})));
                APPEND(PyLong_FromLong(_Alignof({typename})));
        """.rstrip())
        for field in iterfields(cls):
            f_tp = dump_simple_ctype(field.tp)
            output(f"""\
                TEST_FIELD({f_tp}, value{field.c_getter});
            """.rstrip())
        if requires:
            output(f"""
            #else
                APPEND(Py_NewRef(Py_None));
                APPEND(PyUnicode_FromString("skipped on this compiler"));
            #endif
            """)
        output("""
                return result;
            }
        """)

    output("""
            Py_DECREF(result);
            PyErr_Format(PyExc_ValueError, "unknown testcase %R", name);
            return NULL;
        }

        #undef GCC_ATTR
        #undef TEST_FIELD
        #undef SET_AND_APPEND
        #undef APPEND
    """)
