#ifndef Py_CPYTHON_REFCOUNT_H
#  error "this header file must not be included directly"
#endif

/*
Immortalization:

The following indicates the immortalization strategy depending on the amount
of available bits in the reference count field. All strategies are backwards
compatible but the specific reference count value or immortalization check
might change depending on the specializations for the underlying system.

Proper deallocation of immortal instances requires distinguishing between
statically allocated immortal instances vs those promoted by the runtime to be
immortal. The latter should be the only instances that require
cleanup during runtime finalization.
*/

/* Leave the low bits for refcount overflow for old stable ABI code */
// See Include/refcount.h for:
//  #define _Py_STATICALLY_ALLOCATED_FLAG

#if SIZEOF_VOID_P > 4
/*
In 64+ bit systems, any object whose 32 bit reference count is >= 2**31
will be treated as immortal.

Using the lower 32 bits makes the value backwards compatible by allowing
C-Extensions without the updated checks in Py_INCREF and Py_DECREF to safely
increase and decrease the objects reference count.

In order to offer sufficient resilience to C extensions using the stable ABI
compiled against 3.11 or earlier, we set the initial value near the
middle of the range (2**31, 2**32). That way the the refcount can be
off by ~1 billion without affecting immortality.

Reference count increases will use saturated arithmetic, taking advantage of
having all the lower 32 bits set, which will avoid the reference count to go
beyond the refcount limit. Immortality checks for reference count decreases will
be done by checking the bit sign flag in the lower 32 bits.

*/
// See Include/refcount.h for:
//  #define _Py_IMMORTAL_INITIAL_REFCNT
//  #define _Py_STATIC_IMMORTAL_INITIAL_REFCNT

#else
/*
In 32 bit systems, an object will be treated as immortal if its reference
count equals or exceeds _Py_IMMORTAL_MINIMUM_REFCNT (2**30).

Using the lower 30 bits makes the value backwards compatible by allowing
C-Extensions without the updated checks in Py_INCREF and Py_DECREF to safely
increase and decrease the objects reference count. The object would lose its
immortality, but the execution would still be correct.

Reference count increases and decreases will first go through an immortality
check by comparing the reference count field to the minimum immortality refcount.
*/
// See Include/refcount.h for.
//  #define _Py_IMMORTAL_INITIAL_REFCNT
//  #define _Py_IMMORTAL_MINIMUM_REFCNT
//  #define _Py_STATIC_IMMORTAL_INITIAL_REFCNT
#endif

// Py_GIL_DISABLED builds indicate immortal objects using `ob_ref_local`, which is
// always 32-bits.
#ifdef Py_GIL_DISABLED
#define _Py_IMMORTAL_REFCNT_LOCAL UINT32_MAX
#endif


#ifdef Py_GIL_DISABLED
   // The shared reference count uses the two least-significant bits to store
   // flags. The remaining bits are used to store the reference count.
#  define _Py_REF_SHARED_SHIFT        2
#  define _Py_REF_SHARED_FLAG_MASK    0x3

   // The shared flags are initialized to zero.
#  define _Py_REF_SHARED_INIT         0x0
#  define _Py_REF_MAYBE_WEAKREF       0x1
#  define _Py_REF_QUEUED              0x2
#  define _Py_REF_MERGED              0x3

   // Create a shared field from a refcnt and desired flags
#  define _Py_REF_SHARED(refcnt, flags) \
              (((refcnt) << _Py_REF_SHARED_SHIFT) + (flags))
#endif  // Py_GIL_DISABLED


static inline Py_ssize_t _Py_REFCNT(PyObject *ob) {
    #if !defined(Py_GIL_DISABLED)
        return ob->ob_refcnt;
    #else
        uint32_t local = _Py_atomic_load_uint32_relaxed(&ob->ob_ref_local);
        if (local == _Py_IMMORTAL_REFCNT_LOCAL) {
            return _Py_IMMORTAL_INITIAL_REFCNT;
        }
        Py_ssize_t shared = _Py_atomic_load_ssize_relaxed(&ob->ob_ref_shared);
        return _Py_STATIC_CAST(Py_ssize_t, local) +
               Py_ARITHMETIC_RIGHT_SHIFT(Py_ssize_t, shared, _Py_REF_SHARED_SHIFT);
    #endif
}
#define Py_REFCNT(ob) _Py_REFCNT(_PyObject_CAST(ob))

static inline Py_ALWAYS_INLINE int _Py_IsImmortal(PyObject *op)
{
#if defined(Py_GIL_DISABLED)
    return (_Py_atomic_load_uint32_relaxed(&op->ob_ref_local) ==
            _Py_IMMORTAL_REFCNT_LOCAL);
#elif SIZEOF_VOID_P > 4
    return _Py_CAST(PY_INT32_T, op->ob_refcnt) < 0;
#else
    return op->ob_refcnt >= _Py_IMMORTAL_MINIMUM_REFCNT;
#endif
}
#define _Py_IsImmortal(op) _Py_IsImmortal(_PyObject_CAST(op))

static inline Py_ALWAYS_INLINE int _Py_IsStaticImmortal(PyObject *op)
{
#if defined(Py_GIL_DISABLED) || SIZEOF_VOID_P > 4
    return (op->ob_flags & _Py_STATICALLY_ALLOCATED_FLAG) != 0;
#else
    return op->ob_refcnt >= _Py_STATIC_IMMORTAL_MINIMUM_REFCNT;
#endif
}
#define _Py_IsStaticImmortal(op) _Py_IsStaticImmortal(_PyObject_CAST(op))


static inline void Py_SET_REFCNT(PyObject *ob, Py_ssize_t refcnt) {
    assert(refcnt >= 0);
    // This immortal check is for code that is unaware of immortal objects.
    // The runtime tracks these objects and we should avoid as much
    // as possible having extensions inadvertently change the refcnt
    // of an immortalized object.
    if (_Py_IsImmortal(ob)) {
        return;
    }
#ifndef Py_GIL_DISABLED
#if SIZEOF_VOID_P > 4
    ob->ob_refcnt = (PY_UINT32_T)refcnt;
#else
    ob->ob_refcnt = refcnt;
#endif
#else
    if (_Py_IsOwnedByCurrentThread(ob)) {
        if ((size_t)refcnt > (size_t)UINT32_MAX) {
            // On overflow, make the object immortal
            ob->ob_tid = _Py_UNOWNED_TID;
            ob->ob_ref_local = _Py_IMMORTAL_REFCNT_LOCAL;
            ob->ob_ref_shared = 0;
        }
        else {
            // Set local refcount to desired refcount and shared refcount
            // to zero, but preserve the shared refcount flags.
            ob->ob_ref_local = _Py_STATIC_CAST(uint32_t, refcnt);
            ob->ob_ref_shared &= _Py_REF_SHARED_FLAG_MASK;
        }
    }
    else {
        // Set local refcount to zero and shared refcount to desired refcount.
        // Mark the object as merged.
        ob->ob_tid = _Py_UNOWNED_TID;
        ob->ob_ref_local = 0;
        ob->ob_ref_shared = _Py_REF_SHARED(refcnt, _Py_REF_MERGED);
    }
#endif  // Py_GIL_DISABLED
}

/*
The macros Py_INCREF(op) and Py_DECREF(op) are used to increment or decrement
reference counts.  Py_DECREF calls the object's deallocator function when
the refcount falls to 0; for
objects that don't contain references to other objects or heap memory
this can be the standard function free().  Both macros can be used
wherever a void expression is allowed.  The argument must not be a
NULL pointer.  If it may be NULL, use Py_XINCREF/Py_XDECREF instead.
The macro _Py_NewReference(op) initialize reference counts to 1, and
in special builds (Py_REF_DEBUG, Py_TRACE_REFS) performs additional
bookkeeping appropriate to the special build.

We assume that the reference count field can never overflow; this can
be proven when the size of the field is the same as the pointer size, so
we ignore the possibility.  Provided a C int is at least 32 bits (which
is implicitly assumed in many parts of this code), that's enough for
about 2**31 references to an object.

XXX The following became out of date in Python 2.2, but I'm not sure
XXX what the full truth is now.  Certainly, heap-allocated type objects
XXX can and should be deallocated.
Type objects should never be deallocated; the type pointer in an object
is not considered to be a reference to the type object, to save
complications in the deallocation function.  (This is actually a
decision that's up to the implementer of each new type so if you want,
you can count such references to the type object.)
*/

#if defined(Py_REF_DEBUG)
PyAPI_FUNC(void) _Py_NegativeRefcount(const char *filename, int lineno,
                                      PyObject *op);
PyAPI_FUNC(void) _Py_INCREF_IncRefTotal(void);
PyAPI_FUNC(void) _Py_DECREF_DecRefTotal(void);
#endif  // Py_REF_DEBUG && !Py_LIMITED_API


static inline Py_ALWAYS_INLINE void Py_INCREF(PyObject *op)
{
    // Non-limited C API and limited C API for Python 3.9 and older access
    // directly PyObject.ob_refcnt.
#if defined(Py_GIL_DISABLED)
    uint32_t local = _Py_atomic_load_uint32_relaxed(&op->ob_ref_local);
    uint32_t new_local = local + 1;
    if (new_local == 0) {
        _Py_INCREF_IMMORTAL_STAT_INC();
        // local is equal to _Py_IMMORTAL_REFCNT_LOCAL: do nothing
        return;
    }
    if (_Py_IsOwnedByCurrentThread(op)) {
        _Py_atomic_store_uint32_relaxed(&op->ob_ref_local, new_local);
    }
    else {
        _Py_atomic_add_ssize(&op->ob_ref_shared, (1 << _Py_REF_SHARED_SHIFT));
    }
#elif SIZEOF_VOID_P > 4
    PY_UINT32_T cur_refcnt = op->ob_refcnt;
    if (((int32_t)cur_refcnt) < 0) {
        // the object is immortal
        _Py_INCREF_IMMORTAL_STAT_INC();
        return;
    }
    op->ob_refcnt = cur_refcnt + 1;
#else  // SIZEOF_VOID_P
    if (_Py_IsImmortal(op)) {
        _Py_INCREF_IMMORTAL_STAT_INC();
        return;
    }
    op->ob_refcnt++;
#endif // SIZEOF_VOID_P
    _Py_INCREF_STAT_INC();
#ifdef Py_REF_DEBUG
    _Py_INCREF_IncRefTotal();
#endif  // Py_GIL_DISABLED
}
#define Py_INCREF(op) Py_INCREF(_PyObject_CAST(op))


#if !defined(Py_LIMITED_API) && defined(Py_GIL_DISABLED)
// Implements Py_DECREF on objects not owned by the current thread.
PyAPI_FUNC(void) _Py_DecRefShared(PyObject *);
PyAPI_FUNC(void) _Py_DecRefSharedDebug(PyObject *, const char *, int);

// Called from Py_DECREF by the owning thread when the local refcount reaches
// zero. The call will deallocate the object if the shared refcount is also
// zero. Otherwise, the thread gives up ownership and merges the reference
// count fields.
PyAPI_FUNC(void) _Py_MergeZeroLocalRefcount(PyObject *);
#endif

#if defined(Py_GIL_DISABLED) && defined(Py_REF_DEBUG)
static inline void Py_DECREF(const char *filename, int lineno, PyObject *op)
{
    uint32_t local = _Py_atomic_load_uint32_relaxed(&op->ob_ref_local);
    if (local == _Py_IMMORTAL_REFCNT_LOCAL) {
        _Py_DECREF_IMMORTAL_STAT_INC();
        return;
    }
    _Py_DECREF_STAT_INC();
    _Py_DECREF_DecRefTotal();
    if (_Py_IsOwnedByCurrentThread(op)) {
        if (local == 0) {
            _Py_NegativeRefcount(filename, lineno, op);
        }
        local--;
        _Py_atomic_store_uint32_relaxed(&op->ob_ref_local, local);
        if (local == 0) {
            _Py_MergeZeroLocalRefcount(op);
        }
    }
    else {
        _Py_DecRefSharedDebug(op, filename, lineno);
    }
}
#define Py_DECREF(op) Py_DECREF(__FILE__, __LINE__, _PyObject_CAST(op))

#elif defined(Py_GIL_DISABLED)
static inline void Py_DECREF(PyObject *op)
{
    uint32_t local = _Py_atomic_load_uint32_relaxed(&op->ob_ref_local);
    if (local == _Py_IMMORTAL_REFCNT_LOCAL) {
        _Py_DECREF_IMMORTAL_STAT_INC();
        return;
    }
    _Py_DECREF_STAT_INC();
    if (_Py_IsOwnedByCurrentThread(op)) {
        local--;
        _Py_atomic_store_uint32_relaxed(&op->ob_ref_local, local);
        if (local == 0) {
            _Py_MergeZeroLocalRefcount(op);
        }
    }
    else {
        _Py_DecRefShared(op);
    }
}
#define Py_DECREF(op) Py_DECREF(_PyObject_CAST(op))

#elif defined(Py_REF_DEBUG)
static inline void Py_DECREF(const char *filename, int lineno, PyObject *op)
{
#if SIZEOF_VOID_P > 4
    /* If an object has been freed, it will have a negative full refcnt
     * If it has not it been freed, will have a very large refcnt */
    if (op->ob_refcnt_full <= 0 || op->ob_refcnt > (((PY_UINT32_T)-1) - (1<<20))) {
#else
    if (op->ob_refcnt <= 0) {
#endif
        _Py_NegativeRefcount(filename, lineno, op);
    }
    if (_Py_IsImmortal(op)) {
        _Py_DECREF_IMMORTAL_STAT_INC();
        return;
    }
    _Py_DECREF_STAT_INC();
    _Py_DECREF_DecRefTotal();
    if (--op->ob_refcnt == 0) {
        _Py_Dealloc(op);
    }
}
#define Py_DECREF(op) Py_DECREF(__FILE__, __LINE__, _PyObject_CAST(op))

#else
static inline Py_ALWAYS_INLINE void Py_DECREF(PyObject *op)
{
    // Non-limited C API and limited C API for Python 3.9 and older access
    // directly PyObject.ob_refcnt.
    if (_Py_IsImmortal(op)) {
        _Py_DECREF_IMMORTAL_STAT_INC();
        return;
    }
    _Py_DECREF_STAT_INC();
    if (--op->ob_refcnt == 0) {
        _Py_Dealloc(op);
    }
}
#define Py_DECREF(op) Py_DECREF(_PyObject_CAST(op))
#endif
