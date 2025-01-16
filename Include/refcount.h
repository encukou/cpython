#ifndef Py_REFCOUNT_H
#define Py_REFCOUNT_H
#ifdef __cplusplus
extern "C" {
#endif

// Functions exported for the stable ABI
PyAPI_FUNC(Py_ssize_t) Py_REFCNT(PyObject *ob);
PyAPI_FUNC(void) _Py_SetRefcnt(PyObject *ob, Py_ssize_t refcnt);
PyAPI_FUNC(void) _Py_Dealloc(PyObject *);
PyAPI_FUNC(void) Py_IncRef(PyObject *);
PyAPI_FUNC(void) Py_DecRef(PyObject *);
PyAPI_FUNC(void) _Py_IncRef(PyObject *);
PyAPI_FUNC(void) _Py_DecRef(PyObject *);
PyAPI_FUNC(PyObject*) Py_NewRef(PyObject *obj);
PyAPI_FUNC(PyObject*) Py_XNewRef(PyObject *obj);

#if defined(Py_LIMITED_API)
    // Stable ABI implements Py_INCREF() as a function call on limited C API.
    // _Py_IncRef() was added to Python 3.10, use Py_IncRef() on older Python
    // versions.
    // Py_IncRef() accepts NULL whereas _Py_IncRef() doesn't.
    // Same for Py_DECREF.

    // Since limited API 3.11, we require Py_INCREF users to cast to PyObject *.
#   if Py_LIMITED_API+0 >= _Py_PACK_VERSION(3, 11)
#      define Py_INCREF(op) _Py_IncRef(op)
#      define Py_DECREF(op) _Py_DecRef(_PyObject_CAST(op))
#   elif Py_LIMITED_API+0 >= _Py_PACK_VERSION(3, 10)
#      define Py_INCREF(op) _Py_IncRef(_PyObject_CAST(op))
#      define Py_DECREF(op) _Py_DecRef(_PyObject_CAST(op))
#   else
       // Py_IncRef & Py_DecRef are exported as functions since before 3.0.
#      define Py_INCREF(op) Py_IncRef(_PyObject_CAST(op))
#      define Py_DECREF(op) Py_DecRef(_PyObject_CAST(op))
#   endif
#endif

// See comments in internal/refcount.h
// They are needed for `PyModuleDef_HEAD_INIT`.
#define _Py_STATICALLY_ALLOCATED_FLAG (1 << 7)
#if SIZEOF_VOID_P > 4
#   define _Py_IMMORTAL_INITIAL_REFCNT (3UL << 30)
#   define _Py_STATIC_IMMORTAL_INITIAL_REFCNT ((Py_ssize_t)(_Py_IMMORTAL_INITIAL_REFCNT | (((Py_ssize_t)_Py_STATICALLY_ALLOCATED_FLAG) << 32)))
#else
#   define _Py_IMMORTAL_INITIAL_REFCNT ((Py_ssize_t)(5L << 28))
#   define _Py_IMMORTAL_MINIMUM_REFCNT ((Py_ssize_t)(1L << 30))
#   define _Py_STATIC_IMMORTAL_INITIAL_REFCNT ((Py_ssize_t)(7L << 28))
#   define _Py_STATIC_IMMORTAL_MINIMUM_REFCNT ((Py_ssize_t)(6L << 28))
#endif


/* Safely decref `op` and set `op` to NULL, especially useful in tp_clear
 * and tp_dealloc implementations.
 *
 * Note that "the obvious" code can be deadly:
 *
 *     Py_XDECREF(op);
 *     op = NULL;
 *
 * Typically, `op` is something like self->containee, and `self` is done
 * using its `containee` member.  In the code sequence above, suppose
 * `containee` is non-NULL with a refcount of 1.  Its refcount falls to
 * 0 on the first line, which can trigger an arbitrary amount of code,
 * possibly including finalizers (like __del__ methods or weakref callbacks)
 * coded in Python, which in turn can release the GIL and allow other threads
 * to run, etc.  Such code may even invoke methods of `self` again, or cause
 * cyclic gc to trigger, but-- oops! --self->containee still points to the
 * object being torn down, and it may be in an insane state while being torn
 * down.  This has in fact been a rich historic source of miserable (rare &
 * hard-to-diagnose) segfaulting (and other) bugs.
 *
 * The safe way is:
 *
 *      Py_CLEAR(op);
 *
 * That arranges to set `op` to NULL _before_ decref'ing, so that any code
 * triggered as a side-effect of `op` getting torn down no longer believes
 * `op` points to a valid object.
 *
 * There are cases where it's safe to use the naive code, but they're brittle.
 * For example, if `op` points to a Python integer, you know that destroying
 * one of those can't cause problems -- but in part that relies on that
 * Python integers aren't currently weakly referencable.  Best practice is
 * to use Py_CLEAR() even if you can't think of a reason for why you need to.
 *
 * gh-98724: Use a temporary variable to only evaluate the macro argument once,
 * to avoid the duplication of side effects if the argument has side effects.
 *
 * gh-99701: If the PyObject* type is used with casting arguments to PyObject*,
 * the code can be miscompiled with strict aliasing because of type punning.
 * With strict aliasing, a compiler considers that two pointers of different
 * types cannot read or write the same memory which enables optimization
 * opportunities.
 *
 * If available, use _Py_TYPEOF() to use the 'op' type for temporary variables,
 * and so avoid type punning. Otherwise, use memcpy() which causes type erasure
 * and so prevents the compiler to reuse an old cached 'op' value after
 * Py_CLEAR().
 */
#ifdef _Py_TYPEOF
#define Py_CLEAR(op) \
    do { \
        _Py_TYPEOF(op)* _tmp_op_ptr = &(op); \
        _Py_TYPEOF(op) _tmp_old_op = (*_tmp_op_ptr); \
        if (_tmp_old_op != NULL) { \
            *_tmp_op_ptr = _Py_NULL; \
            Py_DECREF(_tmp_old_op); \
        } \
    } while (0)
#else
#define Py_CLEAR(op) \
    do { \
        PyObject **_tmp_op_ptr = _Py_CAST(PyObject**, &(op)); \
        PyObject *_tmp_old_op = (*_tmp_op_ptr); \
        if (_tmp_old_op != NULL) { \
            PyObject *_null_ptr = _Py_NULL; \
            memcpy(_tmp_op_ptr, &_null_ptr, sizeof(PyObject*)); \
            Py_DECREF(_tmp_old_op); \
        } \
    } while (0)
#endif


#ifndef Py_LIMITED_API
#  define Py_CPYTHON_REFCOUNT_H
#  include "cpython/refcount.h"
#  undef Py_CPYTHON_REFCOUNT_H
#endif

/* Function to use in case the object pointer can be NULL: */
static inline void Py_XINCREF(PyObject *op)
{
    if (op != _Py_NULL) {
        Py_INCREF(op);
    }
}

static inline void Py_XDECREF(PyObject *op)
{
    if (op != _Py_NULL) {
        Py_DECREF(op);
    }
}

static inline PyObject* _Py_NewRef(PyObject *obj)
{
    Py_INCREF(obj);
    return obj;
}

static inline PyObject* _Py_XNewRef(PyObject *obj)
{
    Py_XINCREF(obj);
    return obj;
}

// Py_NewRef() and Py_XNewRef() are exported as functions for the stable ABI.
// Names overridden with macros by static inline functions for best
// performances.
//
// For many other functions, ws cast to PyObject* before calling an inline
// function. This is not done in limited API 3.11 or above.
#if !defined(Py_LIMITED_API) || Py_LIMITED_API+0 < _Py_PACK_VERSION(3, 11)
#  define Py_NewRef(obj) _Py_NewRef(_PyObject_CAST(obj))
#  define Py_XNewRef(obj) _Py_XNewRef(_PyObject_CAST(obj))
#  define Py_XINCREF(op) Py_XINCREF(_PyObject_CAST(op))
#  define Py_XDECREF(op) Py_XDECREF(_PyObject_CAST(op))
#  define Py_SET_REFCNT(ob, refcnt) Py_SET_REFCNT(_PyObject_CAST(ob), (refcnt))
#else
#  define Py_NewRef(obj) _Py_NewRef(obj)
#  define Py_XNewRef(obj) _Py_XNewRef(obj)
#endif

#if defined(Py_LIMITED_API) && Py_LIMITED_API+0 < _Py_PACK_VERSION(3, 14)
#  define Py_REFCNT(op) _PyCompat_REFCNT(_PyObject_CAST(op))
#  define Py_SET_REFCNT(op, v) _PyCompat_SET_REFCNT(_PyObject_CAST((op), v))
#endif

#ifdef __cplusplus
}
#endif
#endif   // !Py_REFCOUNT_H
