/* Common handling of type/module slots
 */

#include "Python.h"

#include "pycore_slots.h"           // _PySlot_Info

// Initialize a pre-allocated iterator.
// On error, return -1 with exception set. (Currently doesn't happen.)
// Currently the iteration does not malloc and needs no cleanup.
int
_PySlotIterator_Init(_PySlotIterator *it, PySlot *slots, Py_ssize_t n_slots,
                     int kind)
{
    memset(it, 0, sizeof(_PySlotIterator));
    it->state = it->states;
    it->state->slot = slots;
    it->state->slot_struct_type = _PySlot_KIND_SLOT;
    if (n_slots < 0) {
        it->state->zero_terminated = true;
    }
    else {
        it->state->remaining = n_slots;
    }
    it->kind = kind;
    return 0;
}

static char*
kind_name(_PySlotIterator *it)
{
    switch (it->kind) {
        case _PySlot_KIND_TYPE: return "type";
        case _PySlot_KIND_MOD: return "module";
    }
    Py_UNREACHABLE();
    return "<thing>";
}


int _PySlotIterator_SetDuplicateError(_PySlotIterator *it, const char *name)
{
    uint16_t id = it->last_slot->sl_id;
    _PySlot_Info *info = &_PySlot_InfoTable[id];
    PyErr_Format(
        PyExc_SystemError,
        "%s%s%s has multiple %s (%d) slots",
        kind_name(it),
        name ? " " : "",
        name,
        info->name,
        id);
    return -1;
}

static int unwind(_PySlotIterator *it);

// Advance `it` to the next entry.
// Return 0 if there are no more entries, 1 if there are. Currently can't fail.
static int
advance(_PySlotIterator *it)
{
    if (!it->state->zero_terminated) {
        if (it->state->remaining == 0) {
            if (unwind(it)) {
                return 0;
            }
            return advance(it);
        }
        it->state->remaining--;
    }
    it->state->any_slot++;
    return 1;
}

// Return from one level of nested slots.
// Return 0 if there are no more entries, 1 if there are. Currently can't fail.
static int
unwind(_PySlotIterator *it)
{
    if (it->recursion_level == 0) {
        return 0;
    }
    it->recursion_level--;
    it->state = &it->states[it->recursion_level];
    return advance(it);
}

// Copy a slot to scratch space. Call this before modifying the slot.
static void
to_scratch(PySlot **result, PySlot *scratch)
{
    if (*result != scratch) {
        memcpy(scratch, *result, sizeof(PySlot));
        *result = scratch;
    }
}

// Get the next slot in the iteartor, and its slot info.
// Return 0 if there are no more entries, 1 if there are.
// On error, return -1 with exception set.
int
_PySlotIterator_Next(_PySlotIterator *it, PySlot **result, _PySlot_Info **info)
{
    assert(it);
    assert(result);
    assert(info);
    *result = NULL;
    do {
        if (advance(it) == 0) {
            return 0;
        }
        switch (it->state->slot_struct_type) {
            case _PySlot_KIND_SLOT: {
                it->last_slot = it->state->slot;
            } break;
            case _PySlot_KIND_TYPE: {
                it->last_slot = &it->scratch;
                memset(it->last_slot, 0, sizeof(it->scratch));
                it->scratch.sl_id = it->state->tp_slot->slot;
                it->scratch.sl_flags = Py_SLOT_INTPTR;
                it->scratch.sl_ptr = (void*)it->state->tp_slot->pfunc;
                it->last_slot = &it->scratch;
            } break;
            case _PySlot_KIND_MOD: {
                memset(&it->scratch, 0, sizeof(it->scratch));
                it->scratch.sl_id = it->state->mod_slot->slot;
                it->scratch.sl_flags = Py_SLOT_INTPTR;
                it->scratch.sl_ptr = (void*)it->state->mod_slot->value;
                it->last_slot = &it->scratch;
            } break;
            default: {
                Py_UNREACHABLE();
            } break;
        }
        uint16_t flags = it->last_slot->sl_flags;
        if ((flags & Py_SLOT_SKIP_IF_NULL)
            && it->last_slot->sl_ptr == NULL
            && it->last_slot->sl_func == NULL
            && it->last_slot->sl_size == 0
            && it->last_slot->sl_int64 == 0
            )
        {
            continue;
        }
        if (it->state->ignoring_fallbacks) {
            if (!(flags & Py_SLOT_HAS_FALLBACK)) {
                it->state->ignoring_fallbacks = false;
            }
            continue;
        }
        uint16_t id = it->last_slot->sl_id;
        if (id >= _Py_slot_COUNT) {
            if (flags & (Py_SLOT_OPTIONAL | Py_SLOT_HAS_FALLBACK)) {
                continue;
            }
            PyErr_Format(PyExc_SystemError,
                         "unknown slot ID %u", (unsigned int)id);
            return -1;
        }
        if (id == 0) {
            if (flags & Py_SLOT_OPTIONAL) {
                continue;
            }
            if (flags) {
                PyErr_Format(PyExc_SystemError,
                            "invalid flags for Py_slot_end: 0x%x",
                             (unsigned int)flags);
                return -1;
            }
            if (it->state->zero_terminated) {
                PyErr_Format(PyExc_SystemError,
                            "Py_slot_end in slot array of explicit size");
                return -1;
            }
            if (unwind(it)) {
                return 0;
            }
            continue;
        }
        *info = &_PySlot_InfoTable[id];

        // Resolve a legacy ambiguous slot number
        // Save the original slot definition for error messages.
        uint16_t orig_id = id;
        _PySlot_Info *orig_info = &_PySlot_InfoTable[id];
        if ((*info)->kind == _PySlot_KIND_COMPAT) {
            switch (it->kind) {
                case _PySlot_KIND_TYPE: {
                    id = (*info)->compat_info.type_id;
                } break;
                case _PySlot_KIND_MOD: {
                    id = (*info)->compat_info.mod_id;
                } break;
                default: {
                    Py_UNREACHABLE();
                } break;
            }
            (*info) = &_PySlot_InfoTable[id];
        }

        if (((*info)->kind != it->kind) && (id != Py_slot_subslots)) {
            PyErr_Format(PyExc_SystemError,
                        "Py_%s (slot %d) is not compatible with %s",
                        orig_info->name, orig_id,
                        it->kind == _PySlot_KIND_TYPE ? "types"
                        : it->kind == _PySlot_KIND_MOD ? "modules"
                        : "this");
            return -1;
        }
        if (flags & Py_SLOT_SIZED_ARRAY) {
            if ((*info)->dtype != _PySlot_TYPE_ARRAY) {
                PyErr_Format(PyExc_SystemError,
                            "Py_%s (slot %d) is not compatible with "
                             "Py_SLOT_SIZED_ARRAY",
                            orig_info->name, orig_id);
                return -1;
            }
        }

        if ((*info)->subslots) {
            it->recursion_level++;
            if (it->recursion_level >= _PySlot_MAX_NESTING) {
                PyErr_Format(PyExc_SystemError,
                            "Py_%s (slot %d): too many levels of nested slots",
                            orig_info->name, orig_id);
                return -1;
            }
            it->state = &it->states[it->recursion_level];
            memset(it->state, 0, sizeof(_PySlotIterator_state));
            it->state->slot = it->last_slot->sl_ptr;
            it->state->slot_struct_type = (*info)->kind;
            it->state->zero_terminated = !(flags & Py_SLOT_SIZED_ARRAY);
            if (!it->state->zero_terminated) {
                it->state->remaining = it->last_slot->sl_array_size;
            }
            continue;
        }

        if (flags & Py_SLOT_INTPTR) {
            switch ((*info)->dtype) {
                case _PySlot_TYPE_SIZE: {
                    to_scratch(result, &it->scratch);
                    it->scratch.sl_size = (intptr_t)it->scratch.sl_ptr;
                } break;
                case _PySlot_TYPE_INT64: {
                    to_scratch(result, &it->scratch);
                    it->scratch.sl_int64 = (intptr_t)it->scratch.sl_ptr;
                } break;
                case _PySlot_TYPE_UINT64: {
                    to_scratch(result, &it->scratch);
                    it->scratch.sl_uint64 = (intptr_t)it->scratch.sl_ptr;
                } break;
            }
        }

        if (flags & Py_SLOT_HAS_FALLBACK) {
            it->state->ignoring_fallbacks = true;
        }

        *result = it->last_slot;
        return 1;
    } while(false);
    Py_UNREACHABLE ();
}
