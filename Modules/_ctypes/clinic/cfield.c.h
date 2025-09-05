/*[clinic input]
preserve
[clinic start generated code]*/

#if defined(Py_BUILD_CORE) && !defined(Py_BUILD_CORE_MODULE)
#  include "pycore_gc.h"          // PyGC_Head
#  include "pycore_runtime.h"     // _Py_ID()
#endif
#include "pycore_abstract.h"      // _PyNumber_Index()
#include "pycore_modsupport.h"    // _PyArg_UnpackKeywords()

static PyObject *
PyCField_new_impl(PyTypeObject *type, PyObject *name, PyObject *proto,
                  Py_ssize_t byte_size, Py_ssize_t byte_offset,
                  Py_ssize_t index, int _internal_use, Py_ssize_t bit_size,
                  Py_ssize_t bit_offset);

static PyObject *
PyCField_new(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
    PyObject *return_value = NULL;
    #if defined(Py_BUILD_CORE) && !defined(Py_BUILD_CORE_MODULE)

    #define NUM_KEYWORDS 8
    static struct {
        PyGC_Head _this_is_not_used;
        PyObject_VAR_HEAD
        Py_hash_t ob_hash;
        PyObject *ob_item[NUM_KEYWORDS];
    } _kwtuple = {
        .ob_base = PyVarObject_HEAD_INIT(&PyTuple_Type, NUM_KEYWORDS)
        .ob_hash = -1,
        .ob_item = { &_Py_ID(name), &_Py_ID(type), &_Py_ID(byte_size), &_Py_ID(byte_offset), &_Py_ID(index), &_Py_ID(_internal_use), &_Py_ID(bit_size), &_Py_ID(bit_offset), },
    };
    #undef NUM_KEYWORDS
    #define KWTUPLE (&_kwtuple.ob_base.ob_base)

    #else  // !Py_BUILD_CORE
    #  define KWTUPLE NULL
    #endif  // !Py_BUILD_CORE

    static const char * const _keywords[] = {"name", "type", "byte_size", "byte_offset", "index", "_internal_use", "bit_size", "bit_offset", NULL};
    static _PyArg_Parser _parser = {
        .keywords = _keywords,
        .fname = "CField",
        .kwtuple = KWTUPLE,
    };
    #undef KWTUPLE
    PyObject *argsbuf[8];
    PyObject * const *fastargs;
    Py_ssize_t nargs = PyTuple_GET_SIZE(args);
    Py_ssize_t noptargs = nargs + (kwargs ? PyDict_GET_SIZE(kwargs) : 0) - 6;
    PyObject *name;
    PyObject *proto;
    Py_ssize_t byte_size;
    Py_ssize_t byte_offset;
    Py_ssize_t index;
    int _internal_use;
    Py_ssize_t bit_size = -1;
    Py_ssize_t bit_offset = -1;

    fastargs = _PyArg_UnpackKeywords(_PyTuple_CAST(args)->ob_item, nargs, kwargs, NULL, &_parser,
            /*minpos*/ 0, /*maxpos*/ 0, /*minkw*/ 6, /*varpos*/ 0, argsbuf);
    if (!fastargs) {
        goto exit;
    }
    if (!PyUnicode_Check(fastargs[0])) {
        _PyArg_BadArgument("CField", "argument 'name'", "str", fastargs[0]);
        goto exit;
    }
    name = fastargs[0];
    proto = fastargs[1];
    {
        Py_ssize_t ival = -1;
        PyObject *iobj = _PyNumber_Index(fastargs[2]);
        if (iobj != NULL) {
            ival = PyLong_AsSsize_t(iobj);
            Py_DECREF(iobj);
        }
        if (ival == -1 && PyErr_Occurred()) {
            goto exit;
        }
        byte_size = ival;
    }
    {
        Py_ssize_t ival = -1;
        PyObject *iobj = _PyNumber_Index(fastargs[3]);
        if (iobj != NULL) {
            ival = PyLong_AsSsize_t(iobj);
            Py_DECREF(iobj);
        }
        if (ival == -1 && PyErr_Occurred()) {
            goto exit;
        }
        byte_offset = ival;
    }
    {
        Py_ssize_t ival = -1;
        PyObject *iobj = _PyNumber_Index(fastargs[4]);
        if (iobj != NULL) {
            ival = PyLong_AsSsize_t(iobj);
            Py_DECREF(iobj);
        }
        if (ival == -1 && PyErr_Occurred()) {
            goto exit;
        }
        index = ival;
    }
    _internal_use = PyObject_IsTrue(fastargs[5]);
    if (_internal_use < 0) {
        goto exit;
    }
    if (!noptargs) {
        goto skip_optional_kwonly;
    }
    if (fastargs[6]) {
        if (!_Py_convert_optional_to_non_negative_ssize_t(fastargs[6], &bit_size)) {
            goto exit;
        }
        if (!--noptargs) {
            goto skip_optional_kwonly;
        }
    }
    if (!_Py_convert_optional_to_non_negative_ssize_t(fastargs[7], &bit_offset)) {
        goto exit;
    }
skip_optional_kwonly:
    return_value = PyCField_new_impl(type, name, proto, byte_size, byte_offset, index, _internal_use, bit_size, bit_offset);

exit:
    return return_value;
}

PyDoc_STRVAR(_ctypes_CField__replace__doc__,
"_replace($self, /, byte_offset=unchanged, index=unchanged, *,\n"
"         name=unchanged, type=unchanged, byte_size=unchanged,\n"
"         bit_size=None, bit_offset=None)\n"
"--\n"
"\n"
"Create a copy of this field with the given attributes modified.");

#define _CTYPES_CFIELD__REPLACE_METHODDEF    \
    {"_replace", _PyCFunction_CAST(_ctypes_CField__replace), METH_METHOD|METH_FASTCALL|METH_KEYWORDS, _ctypes_CField__replace__doc__},

static PyObject *
_ctypes_CField__replace_impl(CFieldObject *self,
                             PyTypeObject *defining_class,
                             Py_ssize_t byte_offset, Py_ssize_t index,
                             PyObject *name, PyObject *proto,
                             Py_ssize_t byte_size, Py_ssize_t bit_size,
                             Py_ssize_t bit_offset);

static PyObject *
_ctypes_CField__replace(PyObject *self, PyTypeObject *defining_class, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames)
{
    PyObject *return_value = NULL;
    #if defined(Py_BUILD_CORE) && !defined(Py_BUILD_CORE_MODULE)

    #define NUM_KEYWORDS 7
    static struct {
        PyGC_Head _this_is_not_used;
        PyObject_VAR_HEAD
        Py_hash_t ob_hash;
        PyObject *ob_item[NUM_KEYWORDS];
    } _kwtuple = {
        .ob_base = PyVarObject_HEAD_INIT(&PyTuple_Type, NUM_KEYWORDS)
        .ob_hash = -1,
        .ob_item = { &_Py_ID(byte_offset), &_Py_ID(index), &_Py_ID(name), &_Py_ID(type), &_Py_ID(byte_size), &_Py_ID(bit_size), &_Py_ID(bit_offset), },
    };
    #undef NUM_KEYWORDS
    #define KWTUPLE (&_kwtuple.ob_base.ob_base)

    #else  // !Py_BUILD_CORE
    #  define KWTUPLE NULL
    #endif  // !Py_BUILD_CORE

    static const char * const _keywords[] = {"byte_offset", "index", "name", "type", "byte_size", "bit_size", "bit_offset", NULL};
    static _PyArg_Parser _parser = {
        .keywords = _keywords,
        .fname = "_replace",
        .kwtuple = KWTUPLE,
    };
    #undef KWTUPLE
    PyObject *argsbuf[7];
    Py_ssize_t noptargs = nargs + (kwnames ? PyTuple_GET_SIZE(kwnames) : 0) - 0;
    Py_ssize_t byte_offset = _CFieldObject_CAST(self)->byte_offset;
    Py_ssize_t index = _CFieldObject_CAST(self)->index;
    PyObject *name = _CFieldObject_CAST(self)->name;
    PyObject *proto = NULL;
    Py_ssize_t byte_size = _CFieldObject_CAST(self)->byte_size;
    Py_ssize_t bit_size = _CFieldObject_CAST(self)->bitfield_size ?  _CFieldObject_CAST(self)->bitfield_size : -1;
    Py_ssize_t bit_offset = _CFieldObject_CAST(self)->bitfield_size ?  _CFieldObject_CAST(self)->bit_offset : -1;

    args = _PyArg_UnpackKeywords(args, nargs, NULL, kwnames, &_parser,
            /*minpos*/ 0, /*maxpos*/ 2, /*minkw*/ 0, /*varpos*/ 0, argsbuf);
    if (!args) {
        goto exit;
    }
    if (!noptargs) {
        goto skip_optional_pos;
    }
    if (args[0]) {
        {
            Py_ssize_t ival = -1;
            PyObject *iobj = _PyNumber_Index(args[0]);
            if (iobj != NULL) {
                ival = PyLong_AsSsize_t(iobj);
                Py_DECREF(iobj);
            }
            if (ival == -1 && PyErr_Occurred()) {
                goto exit;
            }
            byte_offset = ival;
        }
        if (!--noptargs) {
            goto skip_optional_pos;
        }
    }
    if (args[1]) {
        {
            Py_ssize_t ival = -1;
            PyObject *iobj = _PyNumber_Index(args[1]);
            if (iobj != NULL) {
                ival = PyLong_AsSsize_t(iobj);
                Py_DECREF(iobj);
            }
            if (ival == -1 && PyErr_Occurred()) {
                goto exit;
            }
            index = ival;
        }
        if (!--noptargs) {
            goto skip_optional_pos;
        }
    }
skip_optional_pos:
    if (!noptargs) {
        goto skip_optional_kwonly;
    }
    if (args[2]) {
        if (!PyUnicode_Check(args[2])) {
            _PyArg_BadArgument("_replace", "argument 'name'", "str", args[2]);
            goto exit;
        }
        name = args[2];
        if (!--noptargs) {
            goto skip_optional_kwonly;
        }
    }
    if (args[3]) {
        proto = args[3];
        if (!--noptargs) {
            goto skip_optional_kwonly;
        }
    }
    if (args[4]) {
        {
            Py_ssize_t ival = -1;
            PyObject *iobj = _PyNumber_Index(args[4]);
            if (iobj != NULL) {
                ival = PyLong_AsSsize_t(iobj);
                Py_DECREF(iobj);
            }
            if (ival == -1 && PyErr_Occurred()) {
                goto exit;
            }
            byte_size = ival;
        }
        if (!--noptargs) {
            goto skip_optional_kwonly;
        }
    }
    if (args[5]) {
        if (!_Py_convert_optional_to_non_negative_ssize_t(args[5], &bit_size)) {
            goto exit;
        }
        if (!--noptargs) {
            goto skip_optional_kwonly;
        }
    }
    if (!_Py_convert_optional_to_non_negative_ssize_t(args[6], &bit_offset)) {
        goto exit;
    }
skip_optional_kwonly:
    return_value = _ctypes_CField__replace_impl((CFieldObject *)self, defining_class, byte_offset, index, name, proto, byte_size, bit_size, bit_offset);

exit:
    return return_value;
}
/*[clinic end generated code: output=7c92d2ee0b4fa6cb input=a9049054013a1b77]*/
