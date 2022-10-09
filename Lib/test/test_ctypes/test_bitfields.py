from ctypes import *
from test.test_ctypes import need_symbol
from test import support
import unittest
import os
import sys

import _ctypes_test

class BITS(Structure):
    _fields_ = [("A", c_int, 1),
                ("B", c_int, 2),
                ("C", c_int, 3),
                ("D", c_int, 4),
                ("E", c_int, 5),
                ("F", c_int, 6),
                ("G", c_int, 7),
                ("H", c_int, 8),
                ("I", c_int, 9),

                ("M", c_short, 1),
                ("N", c_short, 2),
                ("O", c_short, 3),
                ("P", c_short, 4),
                ("Q", c_short, 5),
                ("R", c_short, 6),
                ("S", c_short, 7)]

func = CDLL(_ctypes_test.__file__).unpack_bitfields
func.argtypes = POINTER(BITS), c_char

##for n in "ABCDEFGHIMNOPQRS":
##    print n, hex(getattr(BITS, n).size), getattr(BITS, n).offset

class C_Test(unittest.TestCase):

    def test_ints(self):
        for i in range(512):
            for name in "ABCDEFGHI":
                b = BITS()
                setattr(b, name, i)
                self.assertEqual(getattr(b, name), func(byref(b), name.encode('ascii')))

    def test_shorts(self):
        b = BITS()
        name = "M"
        if func(byref(b), name.encode('ascii')) == 999:
            self.skipTest("Compiler does not support signed short bitfields")
        for i in range(256):
            for name in "MNOPQRS":
                b = BITS()
                setattr(b, name, i)
                self.assertEqual(getattr(b, name), func(byref(b), name.encode('ascii')))

signed_int_types = (c_byte, c_short, c_int, c_long, c_longlong)
unsigned_int_types = (c_ubyte, c_ushort, c_uint, c_ulong, c_ulonglong)
int_types = unsigned_int_types + signed_int_types

class BitFieldTest(unittest.TestCase):

    def test_longlong(self):
        class X(Structure):
            _fields_ = [("a", c_longlong, 1),
                        ("b", c_longlong, 62),
                        ("c", c_longlong, 1)]

        self.assertEqual(sizeof(X), sizeof(c_longlong))
        x = X()
        x.a, x.b, x.c = -1, 7, -1
        self.assertEqual((x.a, x.b, x.c), (-1, 7, -1))

    def test_ulonglong(self):
        class X(Structure):
            _fields_ = [("a", c_ulonglong, 1),
                        ("b", c_ulonglong, 62),
                        ("c", c_ulonglong, 1)]

        self.assertEqual(sizeof(X), sizeof(c_longlong))
        x = X()
        self.assertEqual((x.a, x.b, x.c), (0, 0, 0))
        x.a, x.b, x.c = 7, 7, 7
        self.assertEqual((x.a, x.b, x.c), (1, 7, 1))

    def test_signed(self):
        for c_typ in signed_int_types:
            class X(Structure):
                _fields_ = [("dummy", c_typ),
                            ("a", c_typ, 3),
                            ("b", c_typ, 3),
                            ("c", c_typ, 1)]
            self.assertEqual(sizeof(X), sizeof(c_typ)*2)

            x = X()
            self.assertEqual((c_typ, x.a, x.b, x.c), (c_typ, 0, 0, 0))
            x.a = -1
            self.assertEqual((c_typ, x.a, x.b, x.c), (c_typ, -1, 0, 0))
            x.a, x.b = 0, -1
            self.assertEqual((c_typ, x.a, x.b, x.c), (c_typ, 0, -1, 0))


    def test_unsigned(self):
        for c_typ in unsigned_int_types:
            class X(Structure):
                _fields_ = [("a", c_typ, 3),
                            ("b", c_typ, 3),
                            ("c", c_typ, 1)]
            self.assertEqual(sizeof(X), sizeof(c_typ))

            x = X()
            self.assertEqual((c_typ, x.a, x.b, x.c), (c_typ, 0, 0, 0))
            x.a = -1
            self.assertEqual((c_typ, x.a, x.b, x.c), (c_typ, 7, 0, 0))
            x.a, x.b = 0, -1
            self.assertEqual((c_typ, x.a, x.b, x.c), (c_typ, 0, 7, 0))


    def fail_fields(self, *fields):
        return self.get_except(type(Structure), "X", (),
                               {"_fields_": fields})

    def test_nonint_types(self):
        # bit fields are not allowed on non-integer types.
        result = self.fail_fields(("a", c_char_p, 1))
        self.assertEqual(result, (TypeError, 'bit fields not allowed for type c_char_p'))

        result = self.fail_fields(("a", c_void_p, 1))
        self.assertEqual(result, (TypeError, 'bit fields not allowed for type c_void_p'))

        if c_int != c_long:
            result = self.fail_fields(("a", POINTER(c_int), 1))
            self.assertEqual(result, (TypeError, 'bit fields not allowed for type LP_c_int'))

        result = self.fail_fields(("a", c_char, 1))
        self.assertEqual(result, (TypeError, 'bit fields not allowed for type c_char'))

        class Dummy(Structure):
            _fields_ = []

        result = self.fail_fields(("a", Dummy, 1))
        self.assertEqual(result, (TypeError, 'bit fields not allowed for type Dummy'))

    @need_symbol('c_wchar')
    def test_c_wchar(self):
        result = self.fail_fields(("a", c_wchar, 1))
        self.assertEqual(result,
                (TypeError, 'bit fields not allowed for type c_wchar'))

    def test_single_bitfield_size(self):
        for c_typ in int_types:
            result = self.fail_fields(("a", c_typ, -1))
            self.assertEqual(result, (ValueError, 'number of bits invalid for bit field'))

            result = self.fail_fields(("a", c_typ, 0))
            self.assertEqual(result, (ValueError, 'number of bits invalid for bit field'))

            class X(Structure):
                _fields_ = [("a", c_typ, 1)]
            self.assertEqual(sizeof(X), sizeof(c_typ))

            class X(Structure):
                _fields_ = [("a", c_typ, sizeof(c_typ)*8)]
            self.assertEqual(sizeof(X), sizeof(c_typ))

            result = self.fail_fields(("a", c_typ, sizeof(c_typ)*8 + 1))
            self.assertEqual(result, (ValueError, 'number of bits invalid for bit field'))

    def test_multi_bitfields_size(self):
        class X(Structure):
            _fields_ = [("a", c_short, 1),
                        ("b", c_short, 14),
                        ("c", c_short, 1)]
        self.assertEqual(sizeof(X), sizeof(c_short))

        class X(Structure):
            _fields_ = [("a", c_short, 1),
                        ("a1", c_short),
                        ("b", c_short, 14),
                        ("c", c_short, 1)]
        self.assertEqual(sizeof(X), sizeof(c_short)*3)
        self.assertEqual(X.a.offset, 0)
        self.assertEqual(X.a1.offset, sizeof(c_short))
        self.assertEqual(X.b.offset, sizeof(c_short)*2)
        self.assertEqual(X.c.offset, sizeof(c_short)*2)

        class X(Structure):
            _fields_ = [("a", c_short, 3),
                        ("b", c_short, 14),
                        ("c", c_short, 14)]
        self.assertEqual(sizeof(X), sizeof(c_short)*3)
        self.assertEqual(X.a.offset, sizeof(c_short)*0)
        self.assertEqual(X.b.offset, sizeof(c_short)*1)
        self.assertEqual(X.c.offset, sizeof(c_short)*2)


    def get_except(self, func, *args, **kw):
        try:
            func(*args, **kw)
        except Exception as detail:
            return detail.__class__, str(detail)

    def test_mixed_1(self):
        class X(Structure):
            _fields_ = [("a", c_byte, 4),
                        ("b", c_int, 4)]
        if os.name == "nt":
            self.assertEqual(sizeof(X), sizeof(c_int)*2)
        else:
            self.assertEqual(sizeof(X), sizeof(c_int))

    def test_mixed_2(self):
        class X(Structure):
            _fields_ = [("a", c_byte, 4),
                        ("b", c_int, 32)]
        self.assertEqual(sizeof(X), alignment(c_int)+sizeof(c_int))

    def test_mixed_3(self):
        class X(Structure):
            _fields_ = [("a", c_byte, 4),
                        ("b", c_ubyte, 4)]
        self.assertEqual(sizeof(X), sizeof(c_byte))

    def test_mixed_4(self):
        class X(Structure):
            _fields_ = [("a", c_short, 4),
                        ("b", c_short, 4),
                        ("c", c_int, 24),
                        ("d", c_short, 4),
                        ("e", c_short, 4),
                        ("f", c_int, 24)]
        # MSVC does NOT combine c_short and c_int into one field, GCC
        # does (unless GCC is run with '-mms-bitfields' which
        # produces code compatible with MSVC).
        if os.name == "nt":
            self.assertEqual(sizeof(X), sizeof(c_int) * 4)
        else:
            self.assertEqual(sizeof(X), sizeof(c_int) * 2)

    def test_mixed_5(self):
        class X(Structure):
            _fields_ = [
                ('A', c_uint, 1),
                ('B', c_ushort, 16)]
        a = X()
        a.A = 0
        a.B = 1
        self.assertEqual(1, a.B)

    def test_mixed_6(self):
        class X(Structure):
            _fields_ = [
                ('A', c_ulonglong, 1),
                ('B', c_uint, 32)]
        a = X()
        a.A = 0
        a.B = 1
        self.assertEqual(1, a.B)

    def test_mixed_7(self):
        class X(Structure):
            _fields_ = [
                ("A", c_uint),
                ('B', c_uint, 20),
                ('C', c_ulonglong, 24)]
        self.assertEqual(16, sizeof(X))

    def test_mixed_8(self):
        class Foo(Structure):
            _fields_ = [
                ("A", c_uint),
                ("B", c_uint, 32),
                ("C", c_ulonglong, 1),
                ]

        class Bar(Structure):
            _fields_ = [
                ("A", c_uint),
                ("B", c_uint),
                ("C", c_ulonglong, 1),
                ]
        self.assertEqual(sizeof(Foo), sizeof(Bar))

    def test_mixed_9(self):
        class X(Structure):
            _fields_ = [
                ("A", c_uint8),
                ("B", c_uint, 1),
                ]
        if sys.platform == 'win32':
            self.assertEqual(8, sizeof(X))
        else:
            self.assertEqual(4, sizeof(X))

    def test_mixed_10(self):
        class X(Structure):
            _fields_ = [
                ("A", c_uint32, 1),
                ("B", c_uint64, 1),
                ]
        if sys.platform == 'win32':
            self.assertEqual(8, alignment(X))
            self.assertEqual(16, sizeof(X))
        else:
            self.assertEqual(8, alignment(X))
            self.assertEqual(8, sizeof(X))

    def test_gh_95496(self):
        for field_width in range(1, 33):
            class TestStruct(Structure):
                _fields_ = [
                    ("Field1", c_uint32, field_width),
                    ("Field2", c_uint8, 8)
                ]

            cmd = TestStruct()
            cmd.Field2 = 1
            self.assertEqual(1, cmd.Field2)

    def test_gh_84039(self):
        class Bad(Structure):
            _pack_ = 1
            _fields_ = [
                ("a0", c_uint8, 1),
                ("a1", c_uint8, 1),
                ("a2", c_uint8, 1),
                ("a3", c_uint8, 1),
                ("a4", c_uint8, 1),
                ("a5", c_uint8, 1),
                ("a6", c_uint8, 1),
                ("a7", c_uint8, 1),
                ("b0", c_uint16, 4),
                ("b1", c_uint16, 12),
            ]


        class GoodA(Structure):
            _pack_ = 1
            _fields_ = [
                ("a0", c_uint8, 1),
                ("a1", c_uint8, 1),
                ("a2", c_uint8, 1),
                ("a3", c_uint8, 1),
                ("a4", c_uint8, 1),
                ("a5", c_uint8, 1),
                ("a6", c_uint8, 1),
                ("a7", c_uint8, 1),
            ]


        class Good(Structure):
            _pack_ = 1
            _fields_ = [
                ("a", GoodA),
                ("b0", c_uint16, 4),
                ("b1", c_uint16, 12),
            ]

        self.assertEqual(3, sizeof(Bad))
        self.assertEqual(3, sizeof(Good))

    def test_gh_73939(self):
        class MyStructure(Structure):
            _pack_      = 1
            _fields_    = [
                            ("P",       c_uint16),
                            ("L",       c_uint16, 9),
                            ("Pro",     c_uint16, 1),
                            ("G",       c_uint16, 1),
                            ("IB",      c_uint16, 1),
                            ("IR",      c_uint16, 1),
                            ("R",       c_uint16, 3),
                            ("T",       c_uint32, 10),
                            ("C",       c_uint32, 20),
                            ("R2",      c_uint32, 2)
                        ]
        self.assertEqual(8, sizeof(MyStructure))

    def test_gh_86098(self):
        class X(Structure):
            _fields_ = [
                ("a", c_uint8, 8),
                ("b", c_uint8, 8),
                ("c", c_uint32, 16)
            ]
        self.assertEqual(4, sizeof(X))

    def test_anon_bitfields(self):
        # anonymous bit-fields gave a strange error message
        class X(Structure):
            _fields_ = [("a", c_byte, 4),
                        ("b", c_ubyte, 4)]
        class Y(Structure):
            _anonymous_ = ["_"]
            _fields_ = [("_", X)]

    @need_symbol('c_uint32')
    def test_uint32(self):
        class X(Structure):
            _fields_ = [("a", c_uint32, 32)]
        x = X()
        x.a = 10
        self.assertEqual(x.a, 10)
        x.a = 0xFDCBA987
        self.assertEqual(x.a, 0xFDCBA987)

    @need_symbol('c_uint64')
    def test_uint64(self):
        class X(Structure):
            _fields_ = [("a", c_uint64, 64)]
        x = X()
        x.a = 10
        self.assertEqual(x.a, 10)
        x.a = 0xFEDCBA9876543211
        self.assertEqual(x.a, 0xFEDCBA9876543211)

    @need_symbol('c_uint32')
    def test_uint32_swap_little_endian(self):
        # Issue #23319
        class Little(LittleEndianStructure):
            _fields_ = [("a", c_uint32, 24),
                        ("b", c_uint32, 4),
                        ("c", c_uint32, 4)]
        b = bytearray(4)
        x = Little.from_buffer(b)
        x.a = 0xabcdef
        x.b = 1
        x.c = 2
        self.assertEqual(b, b'\xef\xcd\xab\x21')

    @need_symbol('c_uint32')
    def test_uint32_swap_big_endian(self):
        # Issue #23319
        class Big(BigEndianStructure):
            _fields_ = [("a", c_uint32, 24),
                        ("b", c_uint32, 4),
                        ("c", c_uint32, 4)]
        b = bytearray(4)
        x = Big.from_buffer(b)
        x.a = 0xabcdef
        x.b = 1
        x.c = 2
        self.assertEqual(b, b'\xab\xcd\xef\x12')

if __name__ == "__main__":
    unittest.main()
