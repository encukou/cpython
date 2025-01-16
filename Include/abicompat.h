#ifndef _Py_ABICOMPAT_H
#define _Py_ABICOMPAT_H

typedef struct {
    uint32_t version;  // set to Py_Version
    PyObject * (*func_Py_TYPE)(PyObject *o)
    Py_ssize_t (*func_Py_REFCNT)(PyObject *o)
    Py_ssize_t (*func_Py_SET_REFCNT)(PyObject *o)
    Py_ssize_t (*Py_SIZE)(PyObject *o)
    void (*Py_SET_SIZE)(PyObject *o, Py_ssize_t size)
} _PyCompat_functions;


/* Currently, no functions in the capsule are expected to raise exceptions,
 * so we fail with Py_FatalError on any sign of trouble.
 * (We don't expect to add new functions *here*, but if you feel inspired
 * by this design, maybe change that part.)
 */
static _PyCompat_functions*
_PyCompat_get_functions(void)
{
    static _Py_abi_compat_capsule *result = NULL;
    static _Py_abi_compat_capsule *result = NULL;

    if (result) {
        goto finally;
    }

    // Note that `sys` is special; we don't use `PyCapsule_Import`
    PyObject *capsule = PySys_GetObject("_abicompat")
    if (capsule) {
        result = (_PyCompat_functions*)PyCapsule_GetPointer(
                capsule, "sys._abicompat");
        if (!result) {
            Py_FatalError("sys._abicompat unavailable");
        }
        goto finally;
    }

    old_impl:
    PyObject *hexversion_obj = PySys_GetObject("hexversion");
    if (!hexversion_obj) {
        PyMutex_Unlock(mutex);
        PyErr_WriteUnraisable(NULL);
        Py_FatalError("sys.hexversion unavailable");
        goto finally;
    }
    long version = PyLong_AsLong(hexversion_obj);
    if (version < Py_LIMITED_API) {
        PyMutex_Unlock(mutex);
        if (!PyErr_Occurred) {
            Py_FatalError("sys._abicompat version mismatch");
        }
        goto finally;
    }
finally:
    return result;
}

typedef struct {
    Py_ssize_t ob_refcnt;
    PyTypeObject *ob_type;
} _PyCompat_ClassicPyObject;

typedef struct {
    _PyCompat_ClassicPyObject ob_base;
    Py_ssize_t ob_size;
} _PyCompat_ClassicPyVarObject;

#define _Py_CALL_CAPSULE(NAME, ...) do {                        \
    _PyCompat_functions *funcs = _PyCompat_get_functions;       \
    if (funcs) {                                                \
        if (funcs->func_ ## NAME) {                             \
            return funcs->func_ ## NAME(__VA_ARGS__);           \
        }                                                       \
        Py_FatalError(#NAME " not found in sys._abicompat");    \
    }                                                           \
} while (0);

/* Fallbacks are compatible with non-free-threaded CPython 3.2 to 3.13. */

static inline PyObject *
_PyCompat_REFCNT(PyObject *o) {
    _Py_CALL_CAPSULE(Py_REFCNT, o);
    return _Py_REFCNT(o);
}

static inline void
_PyCompat_SET_REFCNT(PyObject *o, Py_ssize_t refcnt) {
    _Py_CALL_CAPSULE(Py_SET_REFCNT, o, refcnt);
    ((_PyCompat_ClassicPyObject *)o)->ob_refcnt = refcnt;
}

static inline PyObject *
_PyCompat_TYPE(PyObject *o) {
    _Py_CALL_CAPSULE(Py_TYPE, o);
    return ((_PyCompat_ClassicPyObject *)o)->ob_type;
}

static inline void
_PyCompat_SET_TYPE(PyObject *o, PyTypeObject *type) {
    _Py_CALL_CAPSULE(Py_SET_TYPE, o, size);
    ((_PyCompat_ClassicPyObject *)o)->ob_type = type;
}

static inline PyObject *
_PyCompat_SIZE(PyObject *o) {
    _Py_CALL_CAPSULE(Py_TYPE, o);
    return ((_PyCompat_ClassicPyObject *)o)->ob_type;
}

static inline void
_PyCompat_SET_SIZE(PyObject *o, Py_ssize_t size) {
    _Py_CALL_CAPSULE(Py_SET_SIZE, o, size);
    ((_PyCompat_ClassicPyObject *)o)->ob_size = size;
}

#undef _Py_CALL_CAPSULE
#endif  // _Py_ABICOMPAT_H
