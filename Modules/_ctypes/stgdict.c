#ifndef Py_BUILD_CORE_BUILTIN
#  define Py_BUILD_CORE_MODULE 1
#endif

#include "Python.h"
// windows.h must be included before pycore internal headers
#ifdef MS_WIN32
#  include <windows.h>
#endif

#include "pycore_call.h"          // _PyObject_CallNoArgs()
#include "pycore_dict.h"          // _PyDict_SizeOf()
#include <ffi.h>
#ifdef MS_WIN32
#  include <malloc.h>
#endif
#include "ctypes.h"

/* This file relates to StgInfo -- type-specific information for ctypes.
 * See ctypes.h for details.
 */

int
PyCStgInfo_clone(StgInfo *dst_info, StgInfo *src_info)
{
    Py_ssize_t size;

    ctype_clear_stginfo(dst_info);
    PyMem_Free(dst_info->ffi_type_pointer.elements);
    PyMem_Free(dst_info->format);
    dst_info->format = NULL;
    PyMem_Free(dst_info->shape);
    dst_info->shape = NULL;
    dst_info->ffi_type_pointer.elements = NULL;

    memcpy(dst_info, src_info, sizeof(StgInfo));

    Py_XINCREF(dst_info->proto);
    Py_XINCREF(dst_info->argtypes);
    Py_XINCREF(dst_info->converters);
    Py_XINCREF(dst_info->restype);
    Py_XINCREF(dst_info->checker);
    Py_XINCREF(dst_info->module);

    if (src_info->format) {
        dst_info->format = PyMem_Malloc(strlen(src_info->format) + 1);
        if (dst_info->format == NULL) {
            PyErr_NoMemory();
            return -1;
        }
        strcpy(dst_info->format, src_info->format);
    }
    if (src_info->shape) {
        dst_info->shape = PyMem_Malloc(sizeof(Py_ssize_t) * src_info->ndim);
        if (dst_info->shape == NULL) {
            PyErr_NoMemory();
            return -1;
        }
        memcpy(dst_info->shape, src_info->shape,
               sizeof(Py_ssize_t) * src_info->ndim);
    }

    if (src_info->ffi_type_pointer.elements == NULL)
        return 0;
    size = sizeof(ffi_type *) * (src_info->length + 1);
    dst_info->ffi_type_pointer.elements = PyMem_Malloc(size);
    if (dst_info->ffi_type_pointer.elements == NULL) {
        PyErr_NoMemory();
        return -1;
    }
    memcpy(dst_info->ffi_type_pointer.elements,
           src_info->ffi_type_pointer.elements,
           size);
    return 0;
}

/* descr is the descriptor for a field marked as anonymous.  Get all the
 _fields_ descriptors from descr->proto, create new descriptors with offset
 and index adjusted, and stuff them into type.
 */
static int
MakeFields(PyObject *type, CFieldObject *descr,
           Py_ssize_t index, Py_ssize_t offset)
{
    Py_ssize_t i;
    PyObject *fields;
    PyObject *fieldlist;

    fields = PyObject_GetAttrString(descr->proto, "_fields_");
    if (fields == NULL)
        return -1;
    fieldlist = PySequence_Fast(fields, "_fields_ must be a sequence");
    Py_DECREF(fields);
    if (fieldlist == NULL)
        return -1;

    ctypes_state *st = get_module_state_by_class(Py_TYPE(descr));
    PyTypeObject *cfield_tp = st->PyCField_Type;
    for (i = 0; i < PySequence_Fast_GET_SIZE(fieldlist); ++i) {
        PyObject *pair = PySequence_Fast_GET_ITEM(fieldlist, i); /* borrowed */
        PyObject *fname, *ftype, *bits;
        CFieldObject *fdescr;
        CFieldObject *new_descr;
        /* Convert to PyArg_UnpackTuple... */
        if (!PyArg_ParseTuple(pair, "OO|O", &fname, &ftype, &bits)) {
            Py_DECREF(fieldlist);
            return -1;
        }
        fdescr = (CFieldObject *)PyObject_GetAttr(descr->proto, fname);
        if (fdescr == NULL) {
            Py_DECREF(fieldlist);
            return -1;
        }
        if (!Py_IS_TYPE(fdescr, cfield_tp)) {
            PyErr_SetString(PyExc_TypeError, "unexpected type");
            Py_DECREF(fdescr);
            Py_DECREF(fieldlist);
            return -1;
        }
        if (fdescr->anonymous) {
            int rc = MakeFields(type, fdescr,
                                index + fdescr->index,
                                offset + fdescr->offset);
            Py_DECREF(fdescr);
            if (rc == -1) {
                Py_DECREF(fieldlist);
                return -1;
            }
            continue;
        }
        new_descr = (CFieldObject *)cfield_tp->tp_alloc(cfield_tp, 0);
        if (new_descr == NULL) {
            Py_DECREF(fdescr);
            Py_DECREF(fieldlist);
            return -1;
        }
        assert(Py_IS_TYPE(new_descr, cfield_tp));
        new_descr->size = fdescr->size;
        new_descr->offset = fdescr->offset + offset;
        new_descr->index = fdescr->index + index;
        new_descr->proto = Py_XNewRef(fdescr->proto);
        new_descr->getfunc = fdescr->getfunc;
        new_descr->setfunc = fdescr->setfunc;

        Py_DECREF(fdescr);

        if (-1 == PyObject_SetAttr(type, fname, (PyObject *)new_descr)) {
            Py_DECREF(fieldlist);
            Py_DECREF(new_descr);
            return -1;
        }
        Py_DECREF(new_descr);
    }
    Py_DECREF(fieldlist);
    return 0;
}

/* Iterate over the names in the type's _anonymous_ attribute, if present,
 */
static int
MakeAnonFields(PyObject *type)
{
    PyObject *anon;
    PyObject *anon_names;
    Py_ssize_t i;

    if (PyObject_GetOptionalAttr(type, &_Py_ID(_anonymous_), &anon) < 0) {
        return -1;
    }
    if (anon == NULL) {
        return 0;
    }
    anon_names = PySequence_Fast(anon, "_anonymous_ must be a sequence");
    Py_DECREF(anon);
    if (anon_names == NULL)
        return -1;

    ctypes_state *st = get_module_state_by_def(Py_TYPE(type));
    PyTypeObject *cfield_tp = st->PyCField_Type;
    for (i = 0; i < PySequence_Fast_GET_SIZE(anon_names); ++i) {
        PyObject *fname = PySequence_Fast_GET_ITEM(anon_names, i); /* borrowed */
        CFieldObject *descr = (CFieldObject *)PyObject_GetAttr(type, fname);
        if (descr == NULL) {
            Py_DECREF(anon_names);
            return -1;
        }
        if (!Py_IS_TYPE(descr, cfield_tp)) {
            PyErr_Format(PyExc_AttributeError,
                         "'%U' is specified in _anonymous_ but not in "
                         "_fields_",
                         fname);
            Py_DECREF(anon_names);
            Py_DECREF(descr);
            return -1;
        }
        descr->anonymous = 1;

        /* descr is in the field descriptor. */
        if (-1 == MakeFields(type, (CFieldObject *)descr,
                             ((CFieldObject *)descr)->index,
                             ((CFieldObject *)descr)->offset)) {
            Py_DECREF(descr);
            Py_DECREF(anon_names);
            return -1;
        }
        Py_DECREF(descr);
    }

    Py_DECREF(anon_names);
    return 0;
}

/*
  Retrieve the (optional) _pack_ attribute from a type, the _fields_ attribute,
  and initialize StgInfo.  Used for Structure and Union subclasses.
*/
int
PyCStructUnionType_update_stginfo(PyObject *type, PyObject *fields, int isStruct)
{
    Py_ssize_t len, i;
    Py_ssize_t aligned_size;
    PyObject *tmp;
    Py_ssize_t ffi_ofs;
    int arrays_seen = 0;

    int retval = -1;
    // The following are NULL or hold strong references.
    // They're cleared on error.
    PyObject *prop_obj = NULL;
    PyObject *layout_fields = NULL;
    PyObject *layout = NULL;
    PyObject *format_spec_obj = NULL;

    if (fields == NULL) {
        return 0;
    }

    ctypes_state *st = get_module_state_by_def(Py_TYPE(type));
    StgInfo *stginfo;
    if (PyStgInfo_FromType(st, type, &stginfo) < 0) {
        goto error;
    }
    if (!stginfo) {
        PyErr_SetString(PyExc_TypeError,
                        "ctypes state is not initialized");
        goto error;
    }
    PyObject *base = (PyObject *)((PyTypeObject *)type)->tp_base;
    StgInfo *baseinfo;
    if (PyStgInfo_FromType(st, base, &baseinfo) < 0) {
        goto error;
    }

    /* If this structure/union is already marked final we cannot assign
       _fields_ anymore. */

    if (stginfo->flags & DICTFLAG_FINAL) {/* is final ? */
        PyErr_SetString(PyExc_AttributeError,
                        "_fields_ is final");
        goto error;
    }

    PyObject *layout_class;
    if (PyObject_GetOptionalAttr(type, &_Py_ID(_layout_), &tmp) < 0) {
        goto error;
    }
    if (!tmp || PyUnicode_Check(tmp)) {
        layout_class = _PyImport_GetModuleAttrString("ctypes._layout",
                                                     "default_layout");
        Py_XDECREF(tmp);
    }
    else {
        layout_class = tmp;
    }
    if (!layout_class) {
        goto error;
    }
    PyObject *kwnames = PyTuple_Pack(
        2,
        &_Py_ID(is_struct),
        &_Py_ID(base));
    if (!kwnames) {
        goto error;
    }
    PyObject *base_arg = Py_NewRef(baseinfo ? base : Py_None);
    layout = PyObject_Vectorcall(
        layout_class,
        1 + (PyObject*[]){
            NULL,
            /* positional args */
            type,
            fields,
            /* keyword args */
            Py_GetConstantBorrowed(
                isStruct ? Py_CONSTANT_TRUE : Py_CONSTANT_FALSE),
            base_arg},
        2 | PY_VECTORCALL_ARGUMENTS_OFFSET,
        kwnames);
    Py_DECREF(kwnames);
    Py_DECREF(base_arg);
    Py_DECREF(layout_class);
    if (!layout) {
        goto error;
    }

    tmp = PyObject_GetAttrString(layout, "forced_align");
    if (!tmp) {
        goto error;
    }
    int forced_alignment = PyLong_AsInt(tmp);
    Py_DECREF(tmp);
    if (forced_alignment < 0) {
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_ValueError,
                            "_align_ must be a non-negative integer");
        }
        goto error;
    }

    tmp = PyObject_GetAttrString(layout, "align");
    if (!tmp) {
        goto error;
    }
    Py_ssize_t total_align = PyLong_AsInt(tmp);
    Py_DECREF(tmp);
    if (total_align < 0) {
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_ValueError,
                            "total_align must be a non-negative integer");
        }
        goto error;
    }

    tmp = PyObject_GetAttrString(layout, "size");
    if (!tmp) {
        goto error;
    }
    Py_ssize_t total_size = PyLong_AsInt(tmp);
    Py_DECREF(tmp);
    if (total_size < 0) {
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_ValueError,
                            "size must be a non-negative integer");
        }
        goto error;
    }

    format_spec_obj = PyObject_GetAttrString(layout, "format_spec");
    if (!format_spec_obj) {
        goto error;
    }
    Py_ssize_t format_spec_size;
    const char *format_spec = PyUnicode_AsUTF8AndSize(format_spec_obj,
                                                      &format_spec_size);
    if (!format_spec) {
        goto error;
    }

    PyObject *layout_fields_obj = PyObject_GetAttrString(layout, "fields");
    if (!layout_fields_obj) {
        goto error;
    }
    layout_fields = PySequence_Tuple(layout_fields_obj);
    Py_DECREF(layout_fields_obj);
    if (!layout_fields) {
        goto error;
    }
    Py_CLEAR(layout);

    len = PySequence_Size(fields);
    if (len == -1) {
        if (PyErr_ExceptionMatches(PyExc_TypeError)) {
            PyErr_SetString(PyExc_TypeError,
                            "'_fields_' must be a sequence of pairs");
        }
        goto error;
    }

    if (len != PySequence_Fast_GET_SIZE(layout_fields)) {
        PyErr_Format(PyExc_ValueError,
                        "number of '_fields_' must match result of layout... for now. want %zd, got %zd",
                        len, PySequence_Fast_GET_SIZE(layout_fields));
        goto error;
    }

    if (stginfo->format) {
        PyMem_Free(stginfo->format);
        stginfo->format = NULL;
    }
    stginfo->format = PyMem_Malloc(format_spec_size + 1);
    if (!stginfo->format) {
        PyErr_NoMemory();
        goto error;
    }
    memcpy(stginfo->format, format_spec, format_spec_size + 1);

    if (stginfo->ffi_type_pointer.elements) {
        PyMem_Free(stginfo->ffi_type_pointer.elements);
    }

    if (baseinfo) {
        stginfo->ffi_type_pointer.type = FFI_TYPE_STRUCT;
        stginfo->ffi_type_pointer.elements = PyMem_New(ffi_type *, baseinfo->length + len + 1);
        if (stginfo->ffi_type_pointer.elements == NULL) {
            PyErr_NoMemory();
            goto error;
        }
        memset(stginfo->ffi_type_pointer.elements, 0,
               sizeof(ffi_type *) * (baseinfo->length + len + 1));
        if (baseinfo->length > 0) {
            memcpy(stginfo->ffi_type_pointer.elements,
                   baseinfo->ffi_type_pointer.elements,
                   sizeof(ffi_type *) * (baseinfo->length));
        }
        ffi_ofs = baseinfo->length;
    } else {
        stginfo->ffi_type_pointer.type = FFI_TYPE_STRUCT;
        stginfo->ffi_type_pointer.elements = PyMem_New(ffi_type *, len + 1);
        if (stginfo->ffi_type_pointer.elements == NULL) {
            PyErr_NoMemory();
            goto error;
        }
        memset(stginfo->ffi_type_pointer.elements, 0,
               sizeof(ffi_type *) * (len + 1));
        ffi_ofs = 0;
    }

    for (i = 0; i < len; ++i) {
        prop_obj = PySequence_Fast_GET_ITEM(layout_fields, i);
        if (!prop_obj) {
            goto error;
        }
        Py_INCREF(prop_obj);
        if (!PyType_IsSubtype(Py_TYPE(prop_obj), st->PyCField_Type)) {
            PyErr_Format(PyExc_TypeError,
                        "fields must be of type CField, got %T", prop_obj);
            goto error;

        }
        CFieldObject *prop = (CFieldObject *)prop_obj; // borrow from prop_obj

        if (prop->index != i) {
            PyErr_Format(PyExc_ValueError,
                        "field %R index mismatch (expected %d, got %d)",
                         prop->name, i, prop->index);
            goto error;
        }

        if (PyCArrayTypeObject_Check(st, prop->proto)) {
            arrays_seen = 1;
        }

        StgInfo *info;
        if (PyStgInfo_FromType(st, prop->proto, &info) < 0) {
            goto error;
        }
        assert(info);

        stginfo->ffi_type_pointer.elements[ffi_ofs + i] = &info->ffi_type_pointer;
        if (info->flags & (TYPEFLAG_ISPOINTER | TYPEFLAG_HASPOINTER))
            stginfo->flags |= TYPEFLAG_HASPOINTER;
        info->flags |= DICTFLAG_FINAL; /* mark field type final */

        assert(prop);

        if (-1 == PyObject_SetAttr(type, prop->name, prop_obj)) {
            goto error;
        }
        Py_CLEAR(prop_obj);
    }
    Py_CLEAR(layout_fields);

    /* Adjust the size according to the alignment requirements */
    aligned_size = ((total_size + total_align - 1) / total_align) * total_align;

    stginfo->ffi_type_pointer.alignment = Py_SAFE_DOWNCAST(total_align,
                                                           Py_ssize_t,
                                                           unsigned short);
    stginfo->ffi_type_pointer.size = aligned_size;

    stginfo->size = aligned_size;
    stginfo->align = total_align;
    stginfo->length = ffi_ofs + len;

/*
 * The value of MAX_STRUCT_SIZE depends on the platform Python is running on.
 */
#if defined(__aarch64__) || defined(__arm__) || defined(_M_ARM64)
#  define MAX_STRUCT_SIZE 32
#elif defined(__powerpc64__)
#  define MAX_STRUCT_SIZE 64
#else
#  define MAX_STRUCT_SIZE 16
#endif

    if (arrays_seen && (total_size <= MAX_STRUCT_SIZE)) {
        /*
         * See bpo-22273 and gh-110190. Arrays are normally treated as
         * pointers, which is fine when an array name is being passed as
         * parameter, but not when passing structures by value that contain
         * arrays.
         * Small structures passed by value are passed in registers, and in
         * order to do this, libffi needs to know the true type of the array
         * members of structs. Treating them as pointers breaks things.
         *
         * Small structures have different sizes depending on the platform
         * where Python is running on:
         *
         *      * x86-64: 16 bytes or less
         *      * Arm platforms (both 32 and 64 bit): 32 bytes or less
         *      * PowerPC 64 Little Endian: 64 bytes or less
         *
         * In that case, there can't be more than 16, 32 or 64 elements after
         * unrolling arrays, as we (will) disallow bitfields.
         * So we can collect the true ffi_type values in a fixed-size local
         * array on the stack and, if any arrays were seen, replace the
         * ffi_type_pointer.elements with a more accurate set, to allow
         * libffi to marshal them into registers correctly.
         * It means one more loop over the fields, but if we got here,
         * the structure is small, so there aren't too many of those.
         *
         * Although the passing in registers is specific to the above
         * platforms, the array-in-struct vs. pointer problem is general.
         * But we restrict the type transformation to small structs
         * nonetheless.
         *
         * Note that although a union may be small in terms of memory usage, it
         * could contain many overlapping declarations of arrays, e.g.
         *
         * union {
         *     unsigned int_8 foo [16];
         *     unsigned uint_8 bar [16];
         *     unsigned int_16 baz[8];
         *     unsigned uint_16 bozz[8];
         *     unsigned int_32 fizz[4];
         *     unsigned uint_32 buzz[4];
         * }
         *
         * which is still only 16 bytes in size. We need to convert this into
         * the following equivalent for libffi:
         *
         * union {
         *     struct { int_8 e1; int_8 e2; ... int_8 e_16; } f1;
         *     struct { uint_8 e1; uint_8 e2; ... uint_8 e_16; } f2;
         *     struct { int_16 e1; int_16 e2; ... int_16 e_8; } f3;
         *     struct { uint_16 e1; uint_16 e2; ... uint_16 e_8; } f4;
         *     struct { int_32 e1; int_32 e2; ... int_32 e_4; } f5;
         *     struct { uint_32 e1; uint_32 e2; ... uint_32 e_4; } f6;
         * }
         *
         * The same principle applies for a struct 32 or 64 bytes in size.
         *
         * So the struct/union needs setting up as follows: all non-array
         * elements copied across as is, and all array elements replaced with
         * an equivalent struct which has as many fields as the array has
         * elements, plus one NULL pointer.
         */

        Py_ssize_t num_ffi_type_pointers = 0;  /* for the dummy fields */
        Py_ssize_t num_ffi_types = 0;  /* for the dummy structures */
        size_t alloc_size;  /* total bytes to allocate */
        void *type_block;  /* to hold all the type information needed */
        ffi_type **element_types;  /* of this struct/union */
        ffi_type **dummy_types;  /* of the dummy struct elements */
        ffi_type *structs;  /* point to struct aliases of arrays */
        Py_ssize_t element_index;  /* index into element_types for this */
        Py_ssize_t dummy_index = 0; /* index into dummy field pointers */
        Py_ssize_t struct_index = 0; /* index into dummy structs */

        /* first pass to see how much memory to allocate */
        for (i = 0; i < len; ++i) {
            PyObject *name, *desc;
            PyObject *pair = PySequence_GetItem(fields, i);
            int bitsize = 0;

            if (pair == NULL) {
                goto error;
            }
            if (!PyArg_ParseTuple(pair, "UO|i", &name, &desc, &bitsize)) {
                PyErr_SetString(PyExc_TypeError,
                    "'_fields_' must be a sequence of (name, C type) pairs");
                Py_DECREF(pair);
                goto error;
            }

            StgInfo *info;
            if (PyStgInfo_FromType(st, desc, &info) < 0) {
                Py_DECREF(pair);
                goto error;
            }
            if (info == NULL) {
                Py_DECREF(pair);
                PyErr_Format(PyExc_TypeError,
                    "second item in _fields_ tuple (index %zd) must be a C type",
                    i);
                goto error;
            }

            if (!PyCArrayTypeObject_Check(st, desc)) {
                /* Not an array. Just need an ffi_type pointer. */
                num_ffi_type_pointers++;
            }
            else {
                /* It's an array. */
                Py_ssize_t length = info->length;

                StgInfo *einfo;
                if (PyStgInfo_FromType(st, info->proto, &einfo) < 0) {
                    Py_DECREF(pair);
                    goto error;
                }
                if (einfo == NULL) {
                    Py_DECREF(pair);
                    PyErr_Format(PyExc_TypeError,
                        "second item in _fields_ tuple (index %zd) must be a C type",
                        i);
                    goto error;
                }
                /*
                 * We need one extra ffi_type to hold the struct, and one
                 * ffi_type pointer per array element + one for a NULL to
                 * mark the end.
                 */
                num_ffi_types++;
                num_ffi_type_pointers += length + 1;
            }
            Py_DECREF(pair);
        }

        /*
         * At this point, we know we need storage for some ffi_types and some
         * ffi_type pointers. We'll allocate these in one block.
         * There are three sub-blocks of information: the ffi_type pointers to
         * this structure/union's elements, the ffi_type_pointers to the
         * dummy fields standing in for array elements, and the
         * ffi_types representing the dummy structures.
         */
        alloc_size = (ffi_ofs + 1 + len + num_ffi_type_pointers) * sizeof(ffi_type *) +
                        num_ffi_types * sizeof(ffi_type);
        type_block = PyMem_Malloc(alloc_size);

        if (type_block == NULL) {
            PyErr_NoMemory();
            goto error;
        }
        /*
         * the first block takes up ffi_ofs + len + 1 which is the pointers *
         * for this struct/union. The second block takes up
         * num_ffi_type_pointers, so the sum of these is ffi_ofs + len + 1 +
         * num_ffi_type_pointers as allocated above. The last bit is the
         * num_ffi_types structs.
         */
        element_types = (ffi_type **) type_block;
        dummy_types = &element_types[ffi_ofs + len + 1];
        structs = (ffi_type *) &dummy_types[num_ffi_type_pointers];

        if (num_ffi_types > 0) {
            memset(structs, 0, num_ffi_types * sizeof(ffi_type));
        }
        if (ffi_ofs && (baseinfo != NULL)) {
            memcpy(element_types,
                baseinfo->ffi_type_pointer.elements,
                ffi_ofs * sizeof(ffi_type *));
        }
        element_index = ffi_ofs;

        /* second pass to actually set the type pointers */
        for (i = 0; i < len; ++i) {
            PyObject *name, *desc;
            PyObject *pair = PySequence_GetItem(fields, i);
            int bitsize = 0;

            if (pair == NULL) {
                PyMem_Free(type_block);
                goto error;
            }
            /* In theory, we made this call in the first pass, so it *shouldn't*
             * fail. However, you never know, and the code above might change
             * later - keeping the check in here is a tad defensive but it
             * will affect program size only slightly and performance hardly at
             * all.
             */
            if (!PyArg_ParseTuple(pair, "UO|i", &name, &desc, &bitsize)) {
                PyErr_SetString(PyExc_TypeError,
                                "'_fields_' must be a sequence of (name, C type) pairs");
                Py_DECREF(pair);
                PyMem_Free(type_block);
                goto error;
            }

            StgInfo *info;
            if (PyStgInfo_FromType(st, desc, &info) < 0) {
                Py_DECREF(pair);
                PyMem_Free(type_block);
                goto error;
            }

            /* Possibly this check could be avoided, but see above comment. */
            if (info == NULL) {
                Py_DECREF(pair);
                PyMem_Free(type_block);
                PyErr_Format(PyExc_TypeError,
                             "second item in _fields_ tuple (index %zd) must be a C type",
                             i);
                goto error;
            }

            assert(element_index < (ffi_ofs + len)); /* will be used below */
            if (!PyCArrayTypeObject_Check(st, desc)) {
                /* Not an array. Just copy over the element ffi_type. */
                element_types[element_index++] = &info->ffi_type_pointer;
            }
            else {
                Py_ssize_t length = info->length;
                StgInfo *einfo;
                if (PyStgInfo_FromType(st, info->proto, &einfo) < 0) {
                    Py_DECREF(pair);
                    PyMem_Free(type_block);
                    goto error;
                }
                if (einfo == NULL) {
                    Py_DECREF(pair);
                    PyMem_Free(type_block);
                    PyErr_Format(PyExc_TypeError,
                                 "second item in _fields_ tuple (index %zd) must be a C type",
                                 i);
                    goto error;
                }
                element_types[element_index++] = &structs[struct_index];
                structs[struct_index].size = length * einfo->ffi_type_pointer.size;
                structs[struct_index].alignment = einfo->ffi_type_pointer.alignment;
                structs[struct_index].type = FFI_TYPE_STRUCT;
                structs[struct_index].elements = &dummy_types[dummy_index];
                ++struct_index;
                /* Copy over the element's type, length times. */
                while (length > 0) {
                    assert(dummy_index < (num_ffi_type_pointers));
                    dummy_types[dummy_index++] = &einfo->ffi_type_pointer;
                    length--;
                }
                assert(dummy_index < (num_ffi_type_pointers));
                dummy_types[dummy_index++] = NULL;
            }
            Py_DECREF(pair);
        }

        element_types[element_index] = NULL;
        /*
         * Replace the old elements with the new, taking into account
         * base class elements where necessary.
         */
        assert(stginfo->ffi_type_pointer.elements);
        PyMem_Free(stginfo->ffi_type_pointer.elements);
        stginfo->ffi_type_pointer.elements = element_types;
    }

    /* We did check that this flag was NOT set above, it must not
       have been set until now. */
    if (stginfo->flags & DICTFLAG_FINAL) {
        PyErr_SetString(PyExc_AttributeError,
                        "Structure or union cannot contain itself");
        goto error;
    }
    stginfo->flags |= DICTFLAG_FINAL;

    retval = MakeAnonFields(type);
error:
    Py_XDECREF(prop_obj);
    Py_XDECREF(layout_fields);
    Py_XDECREF(layout);
    Py_XDECREF(format_spec_obj);
    return retval;
}
