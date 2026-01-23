// Entry point of the Python C API.
// C extensions should only #include <Python.h>, and not include directly
// the other Python header files included by <Python.h>.

#ifndef Py_PYTHON_H
#define Py_PYTHON_H

// Since this is a "meta-include" file, "#ifdef __cplusplus / extern "C" {"
// is not needed.


// Include Python header files
#include "patchlevel.h"
#include "pyconfig.h"
#include "pymacconfig.h"


// Include standard header files
// When changing these files, remember to update Doc/extending/extending.rst.
#include <assert.h>               // assert()
#include <inttypes.h>             // uintptr_t
#include <limits.h>               // INT_MAX
#include <math.h>                 // HUGE_VAL
#include <stdarg.h>               // va_list
#include <wchar.h>                // wchar_t
#ifdef HAVE_SYS_TYPES_H
#  include <sys/types.h>          // ssize_t
#endif

// <errno.h>, <stdio.h>, <stdlib.h> and <string.h> headers are no longer used
// by Python, but kept for the backward compatibility of existing third party C
// extensions. They are not included by limited C API version 3.11 and newer.
//
// The <ctype.h> and <unistd.h> headers are not included by limited C API
// version 3.13 and newer.
#if !defined(Py_LIMITED_API) || Py_LIMITED_API+0 < 0x030b0000
#  include <errno.h>              // errno
#  include <stdio.h>              // FILE*
#  include <stdlib.h>             // getenv()
#  include <string.h>             // memcpy()
#endif
#if !defined(Py_LIMITED_API) || Py_LIMITED_API+0 < 0x030d0000
#  include <ctype.h>              // tolower()
#  ifndef MS_WINDOWS
#    include <unistd.h>           // close()
#  endif
#endif

#if defined(Py_GIL_DISABLED)
#  if defined(Py_LIMITED_API) && !defined(_Py_OPAQUE_PYOBJECT)
#    error "Py_LIMITED_API is not currently supported in the free-threaded build"
#  endif

#  if defined(_MSC_VER)
#    include <intrin.h>             // __readgsqword()
#  endif

#  if defined(__MINGW32__)
#    include <intrin.h>             // __readgsqword()
#  endif
#endif // Py_GIL_DISABLED

#ifdef _MSC_VER
// Ignore MSC warning C4201: "nonstandard extension used: nameless
// struct/union".  (Only generated for C standard versions less than C11, which
// we don't *officially* support.)
__pragma(warning(push))
__pragma(warning(disable: 4201))
#endif


// Include Python header files
#include "pyport.h"
#include "pymacro.h"
#include "pymath.h"
#include "pymem.h"
#include "pytypedefs.h"
#include "pybuffer.h"
#include "pystats.h"
#include "pyatomic.h"
#include "cpython/pylock.h"
#include "critical_section.h"
#include "object.h"
#include "refcount.h"
#include "objimpl.h"
#include "typeslots.h"
#include "pyhash.h"
#include "cpython/pydebug.h"
#include "bytearrayobject.h"
#include "bytesobject.h"
#include "unicodeobject.h"
#include "pyerrors.h"
#include "longobject.h"
#include "cpython/longintrepr.h"
#include "boolobject.h"
#include "floatobject.h"
#include "complexobject.h"
#include "rangeobject.h"
#include "memoryobject.h"
#include "tupleobject.h"
#include "listobject.h"
#include "dictobject.h"
#include "cpython/odictobject.h"
#include "enumobject.h"
#include "setobject.h"
#include "methodobject.h"
#include "moduleobject.h"
#include "cpython/monitoring.h"
#include "cpython/funcobject.h"
#include "cpython/classobject.h"
#include "fileobject.h"
#include "pycapsule.h"
#include "cpython/code.h"
#include "pyframe.h"
#include "traceback.h"
#include "sliceobject.h"
#include "cpython/cellobject.h"
#include "iterobject.h"
#include "cpython/initconfig.h"
#include "pystate.h"
#include "cpython/genobject.h"
#include "descrobject.h"
#include "genericaliasobject.h"
#include "warnings.h"
#include "weakrefobject.h"
#include "structseq.h"
#include "cpython/picklebufobject.h"
#include "cpython/pytime.h"
#include "codecs.h"
#include "pythread.h"
#include "cpython/context.h"
#include "modsupport.h"
#include "compile.h"
#include "pythonrun.h"
#include "pylifecycle.h"
#include "ceval.h"
#include "sysmodule.h"
#include "audit.h"
#include "osmodule.h"
#include "intrcheck.h"
#include "import.h"
#include "abstract.h"
#include "bltinmodule.h"
#include "cpython/pyctype.h"
#include "pystrtod.h"
#include "pystrcmp.h"
#include "fileutils.h"
#include "cpython/pyfpe.h"
#include "cpython/tracemalloc.h"


#ifndef Py_NO_POISON
#define _PyDict_GetItem_backcompat PyDict_GetItem
#pragma GCC poison PyDict_GetItem

#define _PyDict_GetItemString_backcompat PyDict_GetItemString
#pragma GCC poison PyDict_GetItemString

#define _PyImport_AddModule_backcompat PyImport_AddModule
#pragma GCC poison PyImport_AddModule

#define _PyList_GetItem_backcompat PyList_GetItem
#pragma GCC poison PyList_GetItem

#undef PY_FORMAT_SIZE_T
#pragma GCC poison PY_FORMAT_SIZE_T

#pragma GCC poison PY_UNICODE_TYPE

#pragma GCC poison PyCode_GetFirstFree

#pragma GCC poison PyCode_New

#pragma GCC poison PyCode_NewWithPosOnlyArgs

#define _PyImport_ImportModuleNoBlock_backcompat PyImport_ImportModuleNoBlock
#pragma GCC poison PyImport_ImportModuleNoBlock

//pragma GCC poison PyMem_DEL

//pragma GCC poison PyMem_Del

//pragma GCC poison PyMem_FREE

//pragma GCC poison PyMem_MALLOC

//pragma GCC poison PyMem_NEW

//pragma GCC poison PyMem_REALLOC

//pragma GCC poison PyMem_RESIZE

#define _PyModule_GetFilename_backcompat PyModule_GetFilename
#pragma GCC poison PyModule_GetFilename

#define _PyOS_AfterFork_backcompat PyOS_AfterFork
#pragma GCC poison PyOS_AfterFork

//pragma GCC poison PyObject_DEL

//pragma GCC poison PyObject_Del

//pragma GCC poison PyObject_FREE

//pragma GCC poison PyObject_MALLOC

//pragma GCC poison PyObject_REALLOC

//pragma GCC poison PySlice_GetIndicesEx

#define _PyThread_ReInitTLS_backcompat PyThread_ReInitTLS
#pragma GCC poison PyThread_ReInitTLS

#define _PyThread_create_key_backcompat PyThread_create_key
#pragma GCC poison PyThread_create_key

#define _PyThread_delete_key_backcompat PyThread_delete_key
#pragma GCC poison PyThread_delete_key

#define _PyThread_delete_key_value_backcompat PyThread_delete_key_value
#pragma GCC poison PyThread_delete_key_value

#define _PyThread_get_key_value_backcompat PyThread_get_key_value
#pragma GCC poison PyThread_get_key_value

#define _PyThread_set_key_value_backcompat PyThread_set_key_value
#pragma GCC poison PyThread_set_key_value

#define _PyUnicode_AsDecodedObject_backcompat PyUnicode_AsDecodedObject
#pragma GCC poison PyUnicode_AsDecodedObject

#define _PyUnicode_AsDecodedUnicode_backcompat PyUnicode_AsDecodedUnicode
#pragma GCC poison PyUnicode_AsDecodedUnicode

#define _PyUnicode_AsEncodedObject_backcompat PyUnicode_AsEncodedObject
#pragma GCC poison PyUnicode_AsEncodedObject

#define _PyUnicode_AsEncodedUnicode_backcompat PyUnicode_AsEncodedUnicode
#pragma GCC poison PyUnicode_AsEncodedUnicode

//pragma GCC poison PyUnicode_IS_READY

//pragma GCC poison PyUnicode_READY

#pragma GCC poison PyWeakref_GET_OBJECT

#define _PyWeakref_GetObject_backcompat PyWeakref_GetObject
#pragma GCC poison PyWeakref_GetObject

#pragma GCC poison Py_UNICODE

#pragma GCC poison _PyCode_GetExtra

#pragma GCC poison _PyCode_SetExtra

#pragma GCC poison _PyEval_RequestCodeExtraIndex

//pragma GCC poison _PyHASH_BITS

//pragma GCC poison _PyHASH_IMAG

//pragma GCC poison _PyHASH_INF

//pragma GCC poison _PyHASH_MODULUS

//pragma GCC poison _PyHASH_MULTIPLIER

//pragma GCC poison _PyObject_EXTRA_INIT

#pragma GCC poison _PyThreadState_UncheckedGet

#pragma GCC poison _PyUnicode_AsString

#pragma GCC poison _Py_HashPointer

//pragma GCC poison _Py_T_OBJECT

//pragma GCC poison _Py_WRITE_RESTRICTED

#define _PyDict_GetItemWithError_backcompat PyDict_GetItemWithError
#pragma GCC poison PyDict_GetItemWithError

#define _PyDict_SetDefault_backcompat PyDict_SetDefault
#pragma GCC poison PyDict_SetDefault

#define _PyMapping_HasKey_backcompat PyMapping_HasKey

#pragma GCC poison PyMapping_HasKey

#define _PyMapping_HasKeyString_backcompat PyMapping_HasKeyString

#pragma GCC poison PyMapping_HasKeyString

#define _PyObject_HasAttr_backcompat PyObject_HasAttr
#pragma GCC poison PyObject_HasAttr

#define _PyObject_HasAttrString_backcompat PyObject_HasAttrString
#pragma GCC poison PyObject_HasAttrString

#pragma GCC poison T_SHORT

#pragma GCC poison T_INT

#pragma GCC poison T_LONG

#pragma GCC poison T_FLOAT

#pragma GCC poison T_DOUBLE

#pragma GCC poison T_STRING

//pragma GCC poison T_OBJECT

#pragma GCC poison T_CHAR

#pragma GCC poison T_BYTE

#pragma GCC poison T_UBYTE

#pragma GCC poison T_USHORT

#pragma GCC poison T_UINT

#pragma GCC poison T_ULONG

#pragma GCC poison T_STRING_INPLACE

#pragma GCC poison T_BOOL

#pragma GCC poison T_OBJECT_EX

#pragma GCC poison T_LONGLONG

#pragma GCC poison T_ULONGLONG

#pragma GCC poison T_PYSSIZET

#pragma GCC poison T_NONE

#pragma GCC poison READONLY

#pragma GCC poison PY_AUDIT_READ

#pragma GCC poison READ_RESTRICTED

//pragma GCC poison PY_WRITE_RESTRICTED

#pragma GCC poison RESTRICTED

//pragma GCC poison Py_IS_NAN

//pragma GCC poison Py_IS_INFINITY

//pragma GCC poison Py_IS_FINITE

//pragma GCC poison Py_MEMCPY
#endif // Py_NO_POISON

#ifdef _MSC_VER
__pragma(warning(pop))  // warning(disable: 4201)
#endif

#endif /* !Py_PYTHON_H */
