/* Common code for use by all hashlib related modules. */

// RHEL: use OpenSSL to turn off unsupported modules under FIPS mode
#include <openssl/crypto.h>
#include <openssl/err.h>
#if OPENSSL_VERSION_NUMBER >= 0x30000000L
// For EVP_default_properties_is_fips_enabled
#include <openssl/evp.h>
#endif

/*
 * Given a PyObject* obj, fill in the Py_buffer* viewp with the result
 * of PyObject_GetBuffer.  Sets an exception and issues the erraction
 * on any errors, e.g. 'return NULL' or 'goto error'.
 */
#define GET_BUFFER_VIEW_OR_ERROR(obj, viewp, erraction) do { \
        if (PyUnicode_Check((obj))) { \
            PyErr_SetString(PyExc_TypeError, \
                            "Strings must be encoded before hashing");\
            erraction; \
        } \
        if (!PyObject_CheckBuffer((obj))) { \
            PyErr_SetString(PyExc_TypeError, \
                            "object supporting the buffer API required"); \
            erraction; \
        } \
        if (PyObject_GetBuffer((obj), (viewp), PyBUF_SIMPLE) == -1) { \
            erraction; \
        } \
        if ((viewp)->ndim > 1) { \
            PyErr_SetString(PyExc_BufferError, \
                            "Buffer must be single dimension"); \
            PyBuffer_Release((viewp)); \
            erraction; \
        } \
    } while(0)

#define GET_BUFFER_VIEW_OR_ERROUT(obj, viewp) \
    GET_BUFFER_VIEW_OR_ERROR(obj, viewp, return NULL)

/*
 * Helper code to synchronize access to the hash object when the GIL is
 * released around a CPU consuming hashlib operation. All code paths that
 * access a mutable part of obj must be enclosed in an ENTER_HASHLIB /
 * LEAVE_HASHLIB block or explicitly acquire and release the lock inside
 * a PY_BEGIN / END_ALLOW_THREADS block if they wish to release the GIL for
 * an operation.
 */

#include "pythread.h"
#define ENTER_HASHLIB(obj) \
    if ((obj)->lock) { \
        if (!PyThread_acquire_lock((obj)->lock, 0)) { \
            Py_BEGIN_ALLOW_THREADS \
            PyThread_acquire_lock((obj)->lock, 1); \
            Py_END_ALLOW_THREADS \
        } \
    }
#define LEAVE_HASHLIB(obj) \
    if ((obj)->lock) { \
        PyThread_release_lock((obj)->lock); \
    }

/* TODO(gps): We should probably make this a module or EVPobject attribute
 * to allow the user to optimize based on the platform they're using. */
#define HASHLIB_GIL_MINSIZE 2048

__attribute__((__unused__))
static int
_Py_hashlib_fips_error(PyObject *exc, char *name) {
#if OPENSSL_VERSION_NUMBER >= 0x30000000L
    if (EVP_default_properties_is_fips_enabled(NULL)) {
#else
    if (FIPS_mode()) {
#endif
        PyErr_Format(exc, "%s is not available in FIPS mode", name);
        return 1;
    }
    return 0;
}

#define FAIL_RETURN_IN_FIPS_MODE(exc, name) do { \
    if (_Py_hashlib_fips_error(exc, name)) return NULL; \
} while (0)
