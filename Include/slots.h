#ifndef _Py_HAVE_SLOTS_H
#define _Py_HAVE_SLOTS_H

struct PySlot {
    uint16_t sl_id;
    uint16_t sl_flags;
    union {
        uint32_t sl_array_size;
    };
    union {
        void *sl_ptr;
        void (*sl_func)(void);
        Py_ssize_t sl_size;
        int64_t sl_int64;
        uint64_t sl_uint64;
    };
};

#define Py_SLOT_OPTIONAL 0x01
#define Py_SLOT_HAS_FALLBACK 0x02
#define Py_SLOT_SKIP_IF_NULL 0x04
#define Py_SLOT_STATIC 0x08
#define Py_SLOT_SIZED_ARRAY 0x10
#define Py_SLOT_INTPTR 0x20

#define PySlot_DATA(NAME, VALUE) \
    {.sl_id=Py_ ## NAME, .sl_flags=Py_SLOT_INTPTR, .sl_ptr=(void*)(VALUE)}

#define PySlot_FUNC(NAME, VALUE) \
    {.sl_id=Py_ ## NAME, .sl_func=(VALUE)}

#define PySlot_SIZE(NAME, VALUE) \
    {.sl_id=Py_ ## NAME, .sl_size=(Py_ssize_t)(VALUE)}

#define PySlot_INT64(NAME, VALUE) \
    {.sl_id=Py_ ## NAME, .sl_int64=(int64_t)(VALUE)}

#define PySlot_UINT64(NAME, VALUE) \
    {.sl_id=Py_ ## NAME, .sl_uint64=(uint64_t)(VALUE)}

#define PySlot_STATIC(NAME, VALUE) \
    {.sl_id=Py_ ## NAME, .sl_flags=Py_SLOT_STATIC, .sl_ptr=(VALUE)}

#define PySlot_END {.sl_id=0}


#define PySlot_INTPTR(NAME, VALUE) \
    {Py ## NAME, Py_SLOT_INTPTR, {0}, {(void*)VALUE}}

#define PySlot_INTPTR_STATIC(NAME, VALUE) \
    {Py ## NAME, Py_SLOT_INTPTR | Py_SLOT_STATIC, {0}, {(void*)VALUE}}

#endif  // _Py_HAVE_SLOTS_H

