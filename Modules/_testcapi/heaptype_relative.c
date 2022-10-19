#define Py_LIMITED_API 0x030c0000 // 3.12
#include "parts.h"
#include <stdalign.h>             // alignof
#include <stddef.h>               // max_align_t
#include <string.h>               // memset

#ifdef LIMITED_API_AVAILABLE

static PyType_Slot empty_slots[] = {
    {0, NULL},
};

static PyObject *
make_sized_heaptypes(PyObject *module, PyObject *args)
{
    PyObject *base = NULL;
    PyObject *sub = NULL;
    PyObject *instance = NULL;
    PyObject *result = NULL;

    int extra_base_size, basicsize;

    int r = PyArg_ParseTuple(args, "ii", &extra_base_size, &basicsize);
    if (!r) {
        goto finally;
    }

    PyType_Spec base_spec = {
        .name = "_testcapi.Base",
        .basicsize = sizeof(PyObject) + extra_base_size,
        .flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
        .slots = empty_slots,
    };
    PyType_Spec sub_spec = {
        .name = "_testcapi.Sub",
        .basicsize = basicsize,
        .flags = Py_TPFLAGS_DEFAULT,
        .slots = empty_slots,
    };

    base = PyType_FromMetaclass(NULL, module, &base_spec, NULL);
    if (!base) {
        goto finally;
    }
    sub = PyType_FromMetaclass(NULL, module, &sub_spec, base);
    if (!sub) {
        goto finally;
    }
    instance = PyObject_CallNoArgs(sub);
    if (!instance) {
        goto finally;
    }
    void *data_ptr = PyObject_GetTypeData(instance, (PyTypeObject *)sub);
    if (!data_ptr) {
        goto finally;
    }
    Py_ssize_t data_size = PyObject_GetTypeDataSize((PyTypeObject *)sub);
    if (data_size < 0) {
        goto finally;
    }

    result = Py_BuildValue("OOOln", base, sub, instance, (long)data_ptr,
                           data_size);
  finally:
    Py_XDECREF(base);
    Py_XDECREF(sub);
    Py_XDECREF(instance);
    return result;
}

static PyObject *
var_heaptype_set_data_to_3s(
    PyObject *self, PyTypeObject *defining_class,
    PyObject **args, Py_ssize_t nargs, PyObject *kwnames)
{
    void *data_ptr = PyObject_GetTypeData(self, defining_class);
    if (!data_ptr) {
        return NULL;
    }
    Py_ssize_t data_size = PyObject_GetTypeDataSize(defining_class);
    if (data_size < 0) {
        return NULL;
    }
    memset(data_ptr, 3, data_size);
    Py_RETURN_NONE;
}

static PyObject *
var_heaptype_get_data(PyObject *self, PyTypeObject *defining_class,
                      PyObject **args, Py_ssize_t nargs, PyObject *kwnames)
{
    void *data_ptr = PyObject_GetTypeData(self, defining_class);
    if (!data_ptr) {
        return NULL;
    }
    Py_ssize_t data_size = PyObject_GetTypeDataSize(defining_class);
    if (data_size < 0) {
        return NULL;
    }
    return PyBytes_FromStringAndSize(data_ptr, data_size);
}

static PyMethodDef var_heaptype_methods[] = {
    {"set_data_to_3s", _PyCFunction_CAST(var_heaptype_set_data_to_3s),
        METH_METHOD | METH_FASTCALL | METH_KEYWORDS},
    {"get_data", _PyCFunction_CAST(var_heaptype_get_data),
        METH_METHOD | METH_FASTCALL | METH_KEYWORDS},
    {NULL},
};

static PyObject *
subclass_var_heaptype(PyObject *module, PyObject *args)
{
    PyObject *result = NULL;

    PyObject *base; // borrowed from args
    int basicsize, itemsize;
    long pfunc;

    int r = PyArg_ParseTuple(args, "Oiil", &base, &basicsize, &itemsize, &pfunc);
    if (!r) {
        goto finally;
    }

    PyType_Slot slots[] = {
        {Py_tp_methods, var_heaptype_methods},
        {Py_slot_inherit_itemsize, (void*)pfunc},
        {0, NULL},
    };

    PyType_Spec sub_spec = {
        .name = "_testcapi.Sub",
        .basicsize = basicsize,
        .itemsize = itemsize,
        .flags = Py_TPFLAGS_DEFAULT,
        .slots = slots,
    };

    result = PyType_FromMetaclass(NULL, module, &sub_spec, base);
  finally:
    return result;
}


static PyMethodDef TestMethods[] = {
    {"make_sized_heaptypes", make_sized_heaptypes, METH_VARARGS},
    {"subclass_var_heaptype", subclass_var_heaptype, METH_VARARGS},
    {NULL},
};

int
_PyTestCapi_Init_HeaptypeRelative(PyObject *m) {
    if (PyModule_AddFunctions(m, TestMethods) < 0) {
        return -1;
    }

    PyModule_AddIntConstant(m, "alignof_max_align_t", alignof(max_align_t));

    return 0;
}

#endif // LIMITED_API_AVAILABLE
