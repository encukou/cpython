#ifndef _Py_HAVE_SLOTS_H
#define _Py_HAVE_SLOTS_H

struct PySlot {
    uint16_t sl_id;
    uint16_t sl_flags;
    _Py_ANONYMOUS union {
        uint32_t _sl_reserved;
    };
    _Py_ANONYMOUS union {
        void *sl_ptr;
        void (*sl_func)(void);
        Py_ssize_t sl_size;
        int64_t sl_int64;
        uint64_t sl_uint64;
    };
};

#define PySlot_OPTIONAL 0x01
#define PySlot_HAS_FALLBACK 0x02
#define PySlot_STATIC 0x04
#define PySlot_INTPTR 0x08


#define PySlot_END {Py_slot_end}

#ifndef __cplusplus

/* Convenience macros.
 *
 * For C99 we use designated initializers.
 */

#define PySlot_PTR(NAME, VALUE) \
    {NAME, .sl_flags=0, .sl_ptr=(void*)VALUE}

#define PySlot_FUNC(NAME, VALUE) \
    {NAME, PySlot_STATIC, .sl_func=(VALUE)}

#define PySlot_SIZE(NAME, VALUE) \
    {NAME, PySlot_STATIC, .sl_size=(Py_ssize_t)(VALUE)}

#define PySlot_INT64(NAME, VALUE) \
    {NAME, PySlot_STATIC, .sl_int64=(int64_t)(VALUE)}

#define PySlot_UINT64(NAME, VALUE) \
    {NAME, PySlot_STATIC, .sl_uint64=(uint64_t)(VALUE)}

#define PySlot_STATIC_PTR(NAME, VALUE) \
    {NAME, PySlot_STATIC, .sl_ptr=(void*)VALUE}

#else

/* For C++ we define functions; you can use dynamic initialization even for
 * static data.
 */

static inline PySlot
PySlot_PTR(uint16_t name, void *value)
{
    PySlot result = {name, 0};
    result.sl_ptr = value;
    return result;
}

static inline PySlot
PySlot_FUNC(uint16_t name, void (*value)(void))
{
    PySlot result = {name, PySlot_STATIC};
    result.sl_func = value;
    return result;
}

static inline PySlot
PySlot_SIZE(uint16_t name, Py_ssize_t value)
{
    PySlot result = {name, PySlot_STATIC};
    result.sl_size = value;
    return result;
}

static inline PySlot
PySlot_INT64(uint16_t name, int64_t value)
{
    PySlot result = {name, PySlot_STATIC};
    result.sl_int64 = value;
    return result;
}

static inline PySlot
PySlot_UINT64(uint16_t name, uint64_t value)
{
    PySlot result = {name, PySlot_STATIC};
    result.sl_uint64 = value;
    return result;
}

static inline PySlot
PySlot_STATIC_PTR(uint16_t name, void *value)
{
    PySlot result = {name, PySlot_STATIC};
    result.sl_ptr = value;
    return result;
}

#endif


#endif  // _Py_HAVE_SLOTS_H
