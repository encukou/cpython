#ifndef Py_CRITICAL_SECTION_H
#define Py_CRITICAL_SECTION_H
#ifdef __cplusplus
extern "C" {
#endif


/* This is more convoluted than it needs to be!
 * I'm trying out more ideas at once.
 */



// NOTE: the contents of these structs are private and may change betweeen
// Python releases without a deprecation period.
// The size (and alignment) are fixed, however; if they change we need to add
// "_v2" API.
struct PyCriticalSection_v1 {
    // Tagged pointer to an outer active critical section (or 0).
    uintptr_t _cs_prev;

    // Mutex used to protect critical section
    struct PyMutex *_cs_mutex;
};
struct PyCriticalSection2_v1 {
    struct PyCriticalSection_v1 _cs_base;
    struct PyMutex *_cs_mutex2;
};

PyAPI_FUNC(void) PyCriticalSection_Begin_v1(
    struct PyCriticalSection_v1 *c, PyObject *op);
PyAPI_FUNC(void) PyCriticalSection_End_v1(
    struct PyCriticalSection_v1 *c);
PyAPI_FUNC(void) PyCriticalSection2_Begin_v1(
    struct PyCriticalSection2_v1 *c, PyObject *a, PyObject *b);
PyAPI_FUNC(void) PyCriticalSection2_End_v1(
    struct PyCriticalSection2_v1 *c);


// For demo purposes, we add a "v0", which puts the critical section
// on the heap. An extension can use this API instead of "v1".
struct PyCriticalSection_v0 {
    struct PyCriticalSection_v1 *_cs;
};
struct PyCriticalSection2_v0 {
    struct PyCriticalSection2_v1 *_cs2;
};

PyAPI_FUNC(void) PyCriticalSection_Begin_v0(
    struct PyCriticalSection_v0 *c, PyObject *op);
PyAPI_FUNC(void) PyCriticalSection_End_v0(
    struct PyCriticalSection_v0 *c);
PyAPI_FUNC(void) PyCriticalSection2_Begin_v0(
    struct PyCriticalSection2_v0 *c, PyObject *a, PyObject *b);
PyAPI_FUNC(void) PyCriticalSection2_End_v0(
    struct PyCriticalSection2_v0 *c);

#if defined(Py_LIMITED_API)
#  define _Py_CRITICAL_SECTION_VERSION v0
#elif defined(Py_GIL_DISABLED)
#  define _Py_CRITICAL_SECTION_VERSION v1
#else
#  define _Py_CRITICAL_SECTION_VERSION noop
#  define _Py_CS_NOOPS
#endif

#ifdef _Py_CS_CORE
typedef struct PyCriticalSection_v1 PyCriticalSection;
typedef struct PyCriticalSection2_v1 PyCriticalSection2;
#else
#define _Py_CS_TYPEDEF1(NAME, VERSION) \
    typedef struct NAME ## _ ## VERSION NAME
#define _Py_CS_TYPEDEF(NAME, VERSION) _Py_CS_TYPEDEF1(NAME, VERSION)
_Py_CS_TYPEDEF(PyCriticalSection, _Py_CRITICAL_SECTION_VERSION);
_Py_CS_TYPEDEF(PyCriticalSection2, _Py_CRITICAL_SECTION_VERSION);
#undef _Py_CS_TYPEDEF
#undef _Py_CS_TYPEDEF1
#endif

#define _Py_CS_FUNCNAME2(NAME, VERSION) NAME ## VERSION
#define _Py_CS_FUNCNAME1(NAME, VERSION) _Py_CS_FUNCNAME2(NAME, VERSION)
#define _Py_CS_FUNCNAME(NAME) _Py_CS_FUNCNAME1(NAME ## _, _Py_CRITICAL_SECTION_VERSION)

#define PyCriticalSection_Begin _Py_CS_FUNCNAME(PyCriticalSection_Begin)
#define PyCriticalSection_End _Py_CS_FUNCNAME(PyCriticalSection_End)
#define PyCriticalSection2_Begin _Py_CS_FUNCNAME(PyCriticalSection2_Begin)
#define PyCriticalSection2_End _Py_CS_FUNCNAME(PyCriticalSection2_End)


#ifdef _Py_CS_NOOPS

# define Py_BEGIN_CRITICAL_SECTION(op)      \
    {
# define Py_END_CRITICAL_SECTION()          \
    }
# define Py_BEGIN_CRITICAL_SECTION2(a, b)   \
    {
# define Py_END_CRITICAL_SECTION2()         \
    }

#else /* _Py_CS_NOOPS */

# define Py_BEGIN_CRITICAL_SECTION(op)                                  \
    {                                                                   \
        PyCriticalSection _py_cs;                                       \
        PyCriticalSection_Begin(&_py_cs, _PyObject_CAST(op))

# define Py_END_CRITICAL_SECTION()                                      \
        PyCriticalSection_End(&_py_cs);                                 \
    }

# define Py_BEGIN_CRITICAL_SECTION2(a, b)                               \
    {                                                                   \
        PyCriticalSection2 _py_cs2;                                     \
        PyCriticalSection2_Begin(&_py_cs2, _PyObject_CAST(a), _PyObject_CAST(b))

# define Py_END_CRITICAL_SECTION2()                                     \
        PyCriticalSection2_End(&_py_cs2);                               \
    }

#endif /* _Py_CS_NOOPS */


#ifndef Py_LIMITED_API
#  define Py_CPYTHON_CRITICAL_SECTION_H
#  include "cpython/critical_section.h"
#  undef Py_CPYTHON_CRITICAL_SECTION_H
#endif

#undef _Py_CS_NOOPS

#ifdef __cplusplus
}
#endif
#endif /* !Py_CRITICAL_SECTION_H */
