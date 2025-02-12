#ifndef _Py_PYCORE_SLOTS_H
#define _Py_PYCORE_SLOTS_H

#define _PySlot_TYPE_VOID 1
#define _PySlot_TYPE_FUNC 2
#define _PySlot_TYPE_PTR 3
#define _PySlot_TYPE_XINT64 4
#define _PySlot_TYPE_ARRAY 5

#define _PySlot_KIND_ANY 1
#define _PySlot_KIND_TYPE 2
#define _PySlot_KIND_MOD 3

typedef struct {
    const char *name;
    uint8_t dtype;
    union {
        struct {
            short subslot_offset;
            short slot_offset;
        } type_info;
        struct {
            short type_id;
            short mod_id;
        } compat_info;
    }
} _PySlot_Info;

#define _PySlot_MAX_RECURSION 5;

typedef struct {
    union {
        PySlot *slot;
        PyType_Slot *tp_slot;
        PyModuleDef_Slot *mod_slot;
    }
    Py_ssize_t remaining;
    uint8_t kind;
    uint8_t zero_terminated;
} _PySlot_Iterator_state;

typedef struct {
    _PySlot_Iterator_state states[_PySlot_MAX_RECURSION];
    uint8_t recursion_level;
} _PySlot_Iterator;

#endif
