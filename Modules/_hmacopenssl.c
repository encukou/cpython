/* Module that wraps all OpenSSL MHAC algorithm */

/* Copyright (C) 2019 Red Hat, Inc. Red Hat, Inc. and/or its affiliates
 *
 * Based on _hashopenssl.c, which is:
 * Copyright (C) 2005-2010   Gregory P. Smith (greg@krypto.org)
 * Licensed to PSF under a Contributor Agreement.
 *
 * Derived from a skeleton of shamodule.c containing work performed by:
 *
 * Andrew Kuchling (amk@amk.ca)
 * Greg Stein (gstein@lyra.org)
 *
 */

#define PY_SSIZE_T_CLEAN

#include "Python.h"
#include "structmember.h"
#include "hashlib.h"
#include "pystrhex.h"
#include "_hashopenssl.h"


#include <openssl/hmac.h>

typedef struct hmacopenssl_state {
    PyTypeObject *HmacType;
} hmacopenssl_state;


typedef struct {
    PyObject_HEAD
    PyObject *name;  /* name of the hash algorithm */
    HMAC_CTX *ctx;   /* OpenSSL hmac context */
    PyThread_type_lock lock;  /* HMAC context lock */
} HmacObject;

#include "clinic/_hmacopenssl.c.h"
/*[clinic input]
module _hmacopenssl
class _hmacopenssl.HMAC "HmacObject *" "PyModule_GetState(module)->HmacType"
[clinic start generated code]*/
/*[clinic end generated code: output=da39a3ee5e6b4b0d input=204b7f45847f57b4]*/


/*[clinic input]
_hmacopenssl.new

    key: Py_buffer
    *
    digestmod: str

Return a new hmac object.
[clinic start generated code]*/

static PyObject *
_hmacopenssl_new_impl(PyObject *module, Py_buffer *key,
                      const char *digestmod)
/*[clinic end generated code: output=46f1cb4e02921922 input=be8c0c2e4fad508c]*/
{
    hmacopenssl_state *state;

    if (digestmod == NULL) {
        PyErr_SetString(PyExc_ValueError, "digestmod must be specified");
        return NULL;
    }

    state = PyModule_GetState(module);
    if (state == NULL) {
        return NULL;
    }

    /* name mut be lowercase */
    for (int i=0; digestmod[i]; i++) {
        if (
            ((digestmod[i] < 'a') || (digestmod[i] > 'z'))
            && ((digestmod[i] < '0') || (digestmod[i] > '9'))
            && digestmod[i] != '-'
        ) {
            PyErr_SetString(PyExc_ValueError, "digestmod must be lowercase");
            return NULL;
        }
    }

    const EVP_MD *digest = EVP_get_digestbyname(digestmod);
    if (!digest) {
        PyErr_SetString(PyExc_ValueError, "unknown hash function");
        return NULL;
    }

    PyObject *name = NULL;
    HMAC_CTX *ctx = NULL;
    HmacObject *retval = NULL;

    name = PyUnicode_FromFormat("hmac-%s", digestmod);
    if (name == NULL) {
        goto error;
    }

    ctx = HMAC_CTX_new();
    if (ctx == NULL) {
        _setException(PyExc_ValueError);
        goto error;
    }

    int r = HMAC_Init_ex(
        ctx,
        (const char*)key->buf,
        key->len,
        digest,
        NULL /*impl*/);
    if (r == 0) {
        _setException(PyExc_ValueError);
        goto error;
    }

    retval = (HmacObject *)PyObject_New(HmacObject, state->HmacType);
    if (retval == NULL) {
        goto error;
    }

    retval->name = name;
    retval->ctx = ctx;
    retval->lock = NULL;

    return (PyObject*)retval;

error:
    if (ctx) HMAC_CTX_free(ctx);
    if (name) Py_DECREF(name);
    if (retval) PyObject_Del(name);
    return NULL;
}

/*[clinic input]
_hmacopenssl.HMAC.copy

Return a copy (“clone”) of the HMAC object.
[clinic start generated code]*/

static PyObject *
_hmacopenssl_HMAC_copy_impl(HmacObject *self)
/*[clinic end generated code: output=fe5ee41faf30dcf0 input=f5ed20feec42d8d0]*/
{
    HmacObject *retval;

    retval = (HmacObject *)PyObject_New(HmacObject, (PyTypeObject *)PyObject_Type((PyObject *)self));
    if (retval == NULL) {
        return NULL;
    }

    Py_INCREF(self->name);
    retval->name = self->name;

    int r = HMAC_CTX_copy(retval->ctx, self->ctx);
    if (r == 0) {
        PyObject_Del(retval);
        return _setException(PyExc_ValueError);
    }

    return (PyObject *)retval;
}

static void
_hmac_dealloc(HmacObject *self)
{
    if (self->lock != NULL) {
        PyThread_free_lock(self->lock);
    }
    HMAC_CTX_free(self->ctx);
    Py_XDECREF(self->name);
    PyObject_Del(self);
}

static PyObject *
_hmac_repr(HmacObject *self)
{
    return PyUnicode_FromFormat("<%U HMAC object @ %p>", self->name, self);
}

/*[clinic input]
_hmacopenssl.HMAC.update

    msg: Py_buffer

Update the HMAC object with msg.
[clinic start generated code]*/

static PyObject *
_hmacopenssl_HMAC_update_impl(HmacObject *self, Py_buffer *msg)
/*[clinic end generated code: output=0efeee663a98cee5 input=0683d64f35808cb9]*/
{
    if (self->lock == NULL && msg->len >= HASHLIB_GIL_MINSIZE) {
        self->lock = PyThread_allocate_lock();
        /* fail? lock = NULL and we fail over to non-threaded code. */
    }

    int r;

    if (self->lock != NULL) {
        Py_BEGIN_ALLOW_THREADS
        PyThread_acquire_lock(self->lock, 1);
        r = HMAC_Update(self->ctx, (const unsigned char*)msg->buf, msg->len);
        PyThread_release_lock(self->lock);
        Py_END_ALLOW_THREADS
    } else {
        r = HMAC_Update(self->ctx, (const unsigned char*)msg->buf, msg->len);
    }

    if (r == 0) {
        _setException(PyExc_ValueError);
        return NULL;
    }
    Py_RETURN_NONE;
}

static unsigned int
_digest_size(HmacObject *self)
{
    const EVP_MD *md = HMAC_CTX_get_md(self->ctx);
    if (md == NULL) {
        _setException(PyExc_ValueError);
        return 0;
    }
    return EVP_MD_size(md);
}

static int
_digest(HmacObject *self, unsigned char *buf, unsigned int len)
{
    HMAC_CTX *temp_ctx = HMAC_CTX_new();
    if (temp_ctx == NULL) {
        PyErr_NoMemory();
        return 0;
    }
    int r = HMAC_CTX_copy(temp_ctx, self->ctx);
    if (r == 0) {
        _setException(PyExc_ValueError);
        return 0;
    }
    r = HMAC_Final(temp_ctx, buf, &len);
    HMAC_CTX_free(temp_ctx);
    if (r == 0) {
        _setException(PyExc_ValueError);
        return 0;
    }
    return 1;
}

/*[clinic input]
_hmacopenssl.HMAC.digest

Return the digest of the bytes passed to the update() method so far.
[clinic start generated code]*/

static PyObject *
_hmacopenssl_HMAC_digest_impl(HmacObject *self)
/*[clinic end generated code: output=3aa6dbfc46ec4957 input=bf769a10b1d9edd9]*/
{
    unsigned int digest_size = _digest_size(self);
    if (digest_size == 0) {
        return _setException(PyExc_ValueError);
    }
    unsigned char buf[digest_size]; /* FIXME: C99 feature */
    int r = _digest(self, buf, digest_size);
    if (r == 0) {
        return NULL;
    }
    return PyBytes_FromStringAndSize((const char *)buf, digest_size);
}

/*[clinic input]
_hmacopenssl.HMAC.hexdigest

Return hexadecimal digest of the bytes passed to the update() method so far.

This may be used to exchange the value safely in email or other non-binary
environments.
[clinic start generated code]*/

static PyObject *
_hmacopenssl_HMAC_hexdigest_impl(HmacObject *self)
/*[clinic end generated code: output=630f6fa89f9f1e48 input=b8e60ec8b811c4cd]*/
{
    unsigned int digest_size = _digest_size(self);
    if (digest_size == 0) {
        return _setException(PyExc_ValueError);
    }
    unsigned char buf[digest_size]; /* FIXME: C99 feature */
    int r = _digest(self, buf, digest_size);
    if (r == 0) {
        return NULL;
    }
    return _Py_strhex((const char *)buf, digest_size);
}



static PyObject *
_hmacopenssl_get_digest_size(HmacObject *self, void *closure)
{
    unsigned int digest_size = _digest_size(self);
    if (digest_size == 0) {
        return _setException(PyExc_ValueError);
    }
    return PyLong_FromLong(digest_size);
}

static PyObject *
_hmacopenssl_get_block_size(HmacObject *self, void *closure)
{
    const EVP_MD *md = HMAC_CTX_get_md(self->ctx);
    if (md == NULL) {
        return _setException(PyExc_ValueError);
    }
    return PyLong_FromLong(EVP_MD_size(md));
}

static PyMethodDef Hmac_methods[] = {
    _HMACOPENSSL_HMAC_UPDATE_METHODDEF
    _HMACOPENSSL_HMAC_DIGEST_METHODDEF
    _HMACOPENSSL_HMAC_HEXDIGEST_METHODDEF
    _HMACOPENSSL_HMAC_COPY_METHODDEF
    {NULL, NULL}  /* sentinel */
};

static PyGetSetDef Hmac_getset[] = {
    {"digest_size", (getter)_hmacopenssl_get_digest_size, NULL, NULL, NULL},
    {"block_size", (getter)_hmacopenssl_get_block_size, NULL, NULL, NULL},
    {NULL}  /* Sentinel */
};

static PyMemberDef Hmac_members[] = {
    {"name", T_OBJECT, offsetof(HmacObject, name), READONLY, PyDoc_STR("HMAC name")},
};

PyDoc_STRVAR(hmactype_doc,
"The object used to calculate HMAC of a message.\n\
\n\
Methods:\n\
\n\
update() -- updates the current digest with an additional string\n\
digest() -- return the current digest value\n\
hexdigest() -- return the current digest as a string of hexadecimal digits\n\
copy() -- return a copy of the current hash object\n\
\n\
Attributes:\n\
\n\
name -- the name, including the hash algorithm used by this object\n\
digest_size -- number of bytes in digest() output\n");

static PyType_Slot HmacType_slots[] = {
    {Py_tp_doc, hmactype_doc},
    {Py_tp_repr, (reprfunc)_hmac_repr},
    {Py_tp_dealloc,(destructor)_hmac_dealloc},
    {Py_tp_methods, Hmac_methods},
    {Py_tp_getset, Hmac_getset},
    {Py_tp_members, Hmac_members},
    {0, NULL}
};

PyType_Spec HmacType_spec = {
    "_hmacopenssl.HMAC",    /* name */
    sizeof(HmacObject),     /* basicsize */
    .flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .slots = HmacType_slots,
};


static struct PyMethodDef hmacopenssl_functions[] = {
    _HMACOPENSSL_NEW_METHODDEF
    {NULL,      NULL}            /* Sentinel */
};



/* Initialize this module. */

static int
hmacopenssl_exec(PyObject *m) {
    /* TODO build EVP_functions openssl_* entries dynamically based
     * on what hashes are supported rather than listing many
     * and having some unsupported.  Only init appropriate
     * constants. */
    PyObject *temp;

    temp = PyType_FromSpec(&HmacType_spec);
    if (temp == NULL) {
        goto fail;
    }

    if (PyModule_AddObject(m, "HMAC", temp) == -1) {
        goto fail;
    }

    return 0;

fail:
    Py_XDECREF(temp);
    return -1;
}

static PyModuleDef_Slot hmacopenssl_slots[] = {
    {Py_mod_exec, hmacopenssl_exec},
    {0, NULL},
};

static struct PyModuleDef _hmacopenssl_def = {
    PyModuleDef_HEAD_INIT,  /* m_base */
    .m_name = "_hmacopenssl",
    .m_methods = hmacopenssl_functions,
    .m_slots = hmacopenssl_slots,
    .m_size = sizeof(hmacopenssl_state)
};


PyMODINIT_FUNC
PyInit__hmacopenssl(void)
{
    return PyModuleDef_Init(&_hmacopenssl_def);
}
