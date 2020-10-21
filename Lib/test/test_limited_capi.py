import unittest
from ctypes import pythonapi, PYFUNCTYPE, CFUNCTYPE, cast, pointer
from ctypes import py_object, c_char_p, c_int, c_size_t, c_wchar_p, c_void_p
from ctypes import POINTER
import sys

def get_func(name, result_type, *arg_types):
    proto = PYFUNCTYPE(result_type, *arg_types)
    raw_func = getattr(pythonapi, name)
    return cast(raw_func, proto)

PyThreadState = c_void_p

class TestLimitedCAPI(unittest.TestCase):

    def test_Py_AddPendingCall(self):
        func = get_func('Py_AddPendingCall', c_int, CFUNCTYPE(c_int, c_void_p),
                        c_void_p)
        # can't test this in process

    def test_Py_AtExit(self):
        func = get_func('Py_AtExit', c_int, CFUNCTYPE(None))
        # XXX can't test this in process

    def test_Py_BuildValue(self):
        func = get_func('Py_BuildValue', py_object, c_char_p, c_int, c_int)
        self.assertEqual(func(b"(ii)", 1, 2), (1, 2))

    # XXX not exported!
    # def test_Py_CompileString(self):
    #     func = get_func('Py_CompileString', py_object, c_char_p, c_int)
    #     result = func(b"123 + 654", b"filename.py", 0)
    #     assert eval(result) == 777

    def test_Py_DecodeLocale(self):
        func = get_func('Py_DecodeLocale', c_wchar_p, c_char_p, POINTER(c_size_t))
        # XXX - need to free the result
        # result = func(b"abcd", pointer(c_size_t(0)))
        # PyMem_RawFree(result)

    def test_Py_DecRef(self):
        incref = get_func('Py_IncRef', None, py_object)
        decref = get_func('Py_DecRef', None, py_object)
        obj = object()
        incref(obj)
        decref(obj)

    def test_Py_EncodeLocale(self):
        func = get_func('Py_EncodeLocale', c_char_p, c_wchar_p, POINTER(c_size_t))
        # XXX - need to free the result
        # result = func("abcd", pointer(c_size_t(0)))
        # PyMem_RawFree(result)

    def test_Py_EndInterpreter(self):
        func = get_func('Py_EndInterpreter', None, c_wchar_p, POINTER(c_size_t))
        # can't test this in process

    def test_Py_EnterRecursiveCall(self):
        func = get_func('Py_EnterRecursiveCall', c_int, c_char_p)
        # can't test easily

    def test_Py_Exit(self):
        func = get_func('Py_Exit', None, c_int)
        # can't test this in process

    def test_Py_FatalError(self):
        func = get_func('Py_FatalError', None, c_char_p)
        # can't test this in process

    def test_Py_Finalize(self):
        func = get_func('Py_Finalize', None)
        # can't test this in process

    def test_Py_FinalizeEx(self):
        func = get_func('Py_FinalizeEx', c_int)
        # can't test this in process

    def test_Py_GenericAlias(self):
        pythonapi.Py_GenericAlias
        # not documented

    def test_Py_GenericAliasType(self):
        pythonapi.Py_GenericAliasType
        # not documented; is data(!!!)  (XXX)

    def test_Py_GetArgcArgv(self):
        func = get_func('Py_GetArgcArgv', None, POINTER(c_int),
                        POINTER(POINTER(c_wchar_p)))
        number = c_int()
        value = POINTER(c_wchar_p)()
        func(pointer(number), pointer(value))
        self.assertGreater(number.value, 0)

    def test_Py_GetBuildInfo(self):
        func = get_func('Py_GetBuildInfo', c_char_p)
        result = func()
        self.assertIn(result.decode(), sys.version)

    def test_Py_GetCompiler(self):
        func = get_func('Py_GetCompiler', c_char_p)
        result = func()
        self.assertIn(result.decode(), sys.version)

    def test_Py_GetCopyright(self):
        func = get_func('Py_GetCopyright', c_char_p)
        result = func()
        self.assertEqual(result.decode(), sys.copyright)

    def test_Py_GetExecPrefix(self):
        func = get_func('Py_GetExecPrefix', c_wchar_p)
        result = func()
        self.assertEqual(result, sys.exec_prefix)

    def test_Py_GetPath(self):
        func = get_func('Py_GetPath', c_wchar_p)
        result = func()

    def test_Py_GetPlatform(self):
        func = get_func('Py_GetPlatform', c_char_p)
        result = func()
        self.assertEqual(result.decode(), sys.platform)

    def test_Py_GetPrefix(self):
        func = get_func('Py_GetPrefix', c_wchar_p)
        result = func()
        self.assertEqual(result, sys.prefix)

    def test_Py_GetProgramFullPath(self):
        func = get_func('Py_GetProgramFullPath', c_wchar_p)
        result = func()
        self.assertEqual(result, sys.executable)

    def test_Py_GetProgramName(self):
        func = get_func('Py_GetProgramName', c_wchar_p)
        result = func()

    def test_Py_GetPythonHome(self):
        func = get_func('Py_GetPythonHome', c_wchar_p)
        result = func()

    # XXX Py_GetRecursionLimit - not documented!

    def test_Py_GetVersion(self):
        func = get_func('Py_GetVersion', c_char_p)
        result = func()
        self.assertIn(result.decode(), sys.version)

    def test_Py_Initialize(self):
        func = get_func('Py_Initialize', None)
        # can't test easily

    def test_Py_InitializeEx(self):
        func = get_func('Py_InitializeEx', None, c_int)
        # can't test easily

    def test_Py_IsInitialized(self):
        func = get_func('Py_IsInitialized', c_int)
        result = func()
        self.assertTrue(result)

    def test_Py_LeaveRecursiveCall(self):
        func = get_func('Py_LeaveRecursiveCall', None)
        # can't test easily

    def test_Py_Main(self):
        func = get_func('Py_Main', c_int, c_int, POINTER(c_wchar_p))
        # can't test easily

    # XXX Py_MakePendingCalls - not documented!

    def test_Py_NewInterpreter(self):
        func = get_func('Py_NewInterpreter', PyThreadState)
        # can't test easily

    def test_Py_ReprEnter(self):
        func = get_func('Py_ReprEnter', c_int, py_object)
        # can't test easily

    def test_Py_ReprEnter(self):
        func = get_func('Py_ReprLeave', None, py_object)
        # can't test easily

    def test_Py_SetPath(self):
        func = get_func('Py_SetPath', None, c_wchar_p)
        # can't test easily

    def test_Py_SetProgramName(self):
        func = get_func('Py_SetProgramName', None, c_wchar_p)
        # can't test easily

    def test_Py_SetPythonHome(self):
        func = get_func('Py_SetPythonHome', None, c_wchar_p)
        # can't reset easily

    # XXX Py_SetRecursionLimit -- not documented!

    # XXX Py_SymtableString -- not documented!

    def test_Py_VaBuildValue(self):
        # ctypes doesn't do va_list
        pythonapi.Py_VaBuildValue



















