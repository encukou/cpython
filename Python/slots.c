/* Common handling of type/module slots
 */

#include "Python.h"

#include "pycore_slots.h"           // _PySlot_Info

#include <stdio.h>

// Iterating through a recursive structure doesn't look great in a debugger.
// Define this to get a trace on stderr.
#if 0
#define MSG(...) { \
    fprintf(stderr, "slotiter: " __VA_ARGS__); fprintf(stderr, "\n");}
#else
#define MSG(...)
#endif

static char*
kind_name(int kind)
{
    switch (kind) {
        case _PySlot_KIND_TYPE: return "type";
        case _PySlot_KIND_MOD: return "module";
    }
    Py_UNREACHABLE();
    return "<thing>";
}

// Initialize a pre-allocated iterator.
// On error, return -1 with exception set.
// Currently the iteration does not malloc and needs no cleanup.
int
_PySlotIterator_Init(_PySlotIterator *it, PySlot *slots, Py_ssize_t n_slots,
                     int kind)
{
    MSG("");
    MSG("init (%s slot iterator, size %d)", kind_name(kind), (int)n_slots);
    memset(it, 0, sizeof(_PySlotIterator));
    it->state = it->states;
    it->state->slot = slots;
    it->state->slot_struct_kind = _PySlot_KIND_SLOT;
    if (n_slots < 0) {
        if (slots) {
            it->state->zero_terminated = true;
        }
        else {
            MSG("NULL slots, treating as n_slots=0");
            it->state->remaining = 0;
        }
    }
    else {
        if (slots) {
            it->state->remaining = n_slots;
        }
        else {
            PyErr_SetString(
                PyExc_SystemError,
                "PySlot array with explicit size must not be NULL");
            return -1;
        }
    }
    it->kind = kind;
    return 0;
}


int _PySlotIterator_SetDuplicateError(_PySlotIterator *it, PySlot *slot,
                                      const char *name)
{
    uint16_t id = slot->sl_id;
    _PySlot_Info *info = &_PySlot_InfoTable[id];
    PyErr_Format(
        PyExc_SystemError,
        "%s%s%s has multiple Py_%s (%d) slots",
        kind_name(it->kind),
        name ? " " : "",
        name ? name : "",
        info->name,
        id);
    return -1;
}


int _PySlotIterator_RejectNull(_PySlotIterator *it, PySlot *slot,
                               const char *name)
{
    uint16_t id = slot->sl_id;
    _PySlot_Info *info = &_PySlot_InfoTable[id];
    PyErr_Format(
        PyExc_SystemError,
        "Py_%s (%d) slot for %s%s%s must not be NULL",
        info->name,
        id,
        kind_name(it->kind),
        name ? " " : "",
        name ? name : "");
    return -1;
}

static int unwind(_PySlotIterator *it);

// Advance `it` to the next entry.
// Return 0 if there are no more entries, 1 if there are. Currently can't fail.
static int
advance(_PySlotIterator *it)
{
    MSG("advance lv=%d", (int)it->recursion_level);
    if (!it->state->zero_terminated) {
        it->state->remaining--;
        MSG("sized array; remaining=%d", (int)it->state->remaining);
        if (it->state->remaining == 0) {
            if (unwind(it) == 0) {
                return 0;
            }
            return advance(it);
        }
    }
    switch (it->state->slot_struct_kind) {
        case _PySlot_KIND_SLOT: it->state->slot++; break;
        case _PySlot_KIND_TYPE: it->state->tp_slot++; break;
        case _PySlot_KIND_MOD: it->state->mod_slot++; break;
        default:
            Py_UNREACHABLE();
    }
    return 1;
}

// Return from one level of nested slots.
// Return 0 if there are no more entries, 1 if there are. Currently can't fail.
static int
unwind(_PySlotIterator *it)
{
    MSG("unwind from level %d", (int)it->recursion_level);
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
    MSG("to_scratch");
        memcpy(scratch, *result, sizeof(PySlot));
        *result = scratch;
    }
}

// Get the next slot in the iteartor, and its slot info.
// Return 0 if there are no more entries, 1 if there are.
// On error, return -1 with exception set.
int
_PySlotIterator_Next(_PySlotIterator *it, PySlot **p_result, _PySlot_Info **info)
{
    MSG("next");
    assert(it);
    assert(p_result);
    assert(info);

    *p_result = NULL;

    PySlot *result = NULL;
    while (true) {
        if (!it->state->zero_terminated) {
            MSG("slots remaining: %d", (int)it->state->remaining);
            if (!it->state->remaining) {
                MSG("at end of sized array");
                if (unwind(it) == 0) {
                    MSG("end (last level unwound)");
                    return 0;
                }
                continue;
            }
        }

        switch (it->state->slot_struct_kind) {
            case _PySlot_KIND_SLOT: {
                MSG("PySlot structure");
                result = it->state->slot;
            } break;
            case _PySlot_KIND_TYPE: {
                MSG("type slot structure");
                result = &it->scratch;
                memset(result, 0, sizeof(it->scratch));
                result->sl_id = it->state->tp_slot->slot;
                result->sl_flags = PySlot_INTPTR;
                result->sl_ptr = (void*)it->state->tp_slot->pfunc;
            } break;
            case _PySlot_KIND_MOD: {
                MSG("module slot structure");
                result = &it->scratch;
                memset(result, 0, sizeof(it->scratch));
                result->sl_id = it->state->mod_slot->slot;
                result->sl_flags = PySlot_INTPTR;
                result->sl_ptr = (void*)it->state->mod_slot->value;
            } break;
            default: {
                Py_UNREACHABLE();
            } break;
        }
        uint16_t flags = result->sl_flags;
        MSG("slot %d flags 0x%x", (int)result->sl_id, (unsigned)flags);
        if ((flags & PySlot_SKIP_IF_NULL)
            && result->sl_ptr == NULL
            && result->sl_func == NULL
            && result->sl_size == 0
            && result->sl_int64 == 0
            )
        {
            MSG("skipped (NULL)");
            continue;
        }
        if (it->state->ignoring_fallbacks) {
            if (!(flags & PySlot_HAS_FALLBACK)) {
                MSG("stopping to ignore fallbacks");
                it->state->ignoring_fallbacks = false;
            }
            MSG("skipped (ignoring fallbacks)");
            continue;
        }
        if (result->sl_id >= _Py_slot_COUNT) {
            if (flags & (PySlot_OPTIONAL | PySlot_HAS_FALLBACK)) {
                MSG("skipped (unknown slot)");
                continue;
            }
            MSG("error (unknown slot)");
            PyErr_Format(PyExc_SystemError,
                         "unknown slot ID %u", (unsigned int)result->sl_id);
            return -1;
        }
        if (result->sl_id == 0) {
            flags &= ~PySlot_INTPTR;
            MSG("sentinel slot, flags %x", (unsigned)flags);
            if (flags == PySlot_OPTIONAL) {
                MSG("skipped (optional sentinel)");
                continue;
            }
            if (flags) {
                MSG("error (bad flags on sentinel)");
                PyErr_Format(PyExc_SystemError,
                            "invalid flags for Py_slot_end: 0x%x",
                             (unsigned int)flags);
                return -1;
            }
            if (!it->state->zero_terminated) {
                MSG("error (sentinel in sized array)");
                PyErr_Format(PyExc_SystemError,
                            "Py_slot_end in slot array of explicit size");
                return -1;
            }
            if (unwind(it) == 0) {
                MSG("end (last level unwound)");
                return 0;
            }
            continue;
        }
        *info = &_PySlot_InfoTable[result->sl_id];
        MSG("slot %d: %s", (int)result->sl_id, (*info)->name);

        // Resolve a legacy ambiguous slot number
        // Save the original slot definition for error messages.
        uint16_t orig_id = result->sl_id;
        _PySlot_Info *orig_info = &_PySlot_InfoTable[result->sl_id];
        if ((*info)->kind == _PySlot_KIND_COMPAT) {
            MSG("resolving compat slot");
            to_scratch(&result, &it->scratch);
            switch (it->kind) {
                case _PySlot_KIND_TYPE: {
                    result->sl_id = (*info)->compat_info.type_id;
                } break;
                case _PySlot_KIND_MOD: {
                    result->sl_id = (*info)->compat_info.mod_id;
                } break;
                default: {
                    Py_UNREACHABLE();
                } break;
            }
            (*info) = &_PySlot_InfoTable[result->sl_id];
            MSG("slot %d: %s", (int)result->sl_id, (*info)->name);
        }

        if (((*info)->kind != it->kind) && (result->sl_id != Py_slot_subslots)) {
            MSG("error (bad slot kind)");
            PyErr_Format(PyExc_SystemError,
                         "Py_%s (slot %d) is not compatible with %ss",
                         orig_info->name,
                         orig_id,
                         kind_name(it->kind));
            return -1;
        }
        if (flags & PySlot_SIZED_ARRAY) {
            if ((*info)->dtype != _PySlot_TYPE_ARRAY) {
                MSG("error (array size for non-array)");
                PyErr_Format(PyExc_SystemError,
                            "Py_%s (slot %d) is not compatible with "
                             "PySlot_SIZED_ARRAY",
                            orig_info->name, orig_id);
                return -1;
            }
        }

        if ((*info)->subslots) {
            if (result->sl_ptr == NULL) {
                if (flags & PySlot_SIZED_ARRAY) {
                    PyErr_SetString(
                        PyExc_SystemError,
                        "slot array with explicit size must not be NULL");
                    return -1;
                }
                MSG("NULL subslots; skipping");
                advance(it);
                continue;
            }
            it->recursion_level++;
            MSG("recursing into level %d", it->recursion_level);
            if (it->recursion_level >= _PySlot_MAX_NESTING) {
                MSG("error (too much recursion)");
                PyErr_Format(PyExc_SystemError,
                            "Py_%s (slot %d): too many levels of nested slots",
                            orig_info->name, orig_id);
                return -1;
            }
            it->state = &it->states[it->recursion_level];
            memset(it->state, 0, sizeof(_PySlotIterator_state));
            it->state->slot = result->sl_ptr;
            it->state->slot_struct_kind = (*info)->kind;
            it->state->zero_terminated = !(flags & PySlot_SIZED_ARRAY);
            if (!it->state->zero_terminated) {
                it->state->remaining = result->sl_array_size;
            }
            continue;
        }

        if (flags & PySlot_INTPTR) {
            MSG("casting from intptr");
            switch ((*info)->dtype) {
                case _PySlot_TYPE_SIZE: {
                    to_scratch(&result, &it->scratch);
                    it->scratch.sl_size = (intptr_t)it->scratch.sl_ptr;
                } break;
                case _PySlot_TYPE_INT64: {
                    to_scratch(&result, &it->scratch);
                    it->scratch.sl_int64 = (intptr_t)it->scratch.sl_ptr;
                } break;
                case _PySlot_TYPE_UINT64: {
                    to_scratch(&result, &it->scratch);
                    it->scratch.sl_uint64 = (intptr_t)it->scratch.sl_ptr;
                } break;
            }
        }

        if (flags & PySlot_HAS_FALLBACK) {
            MSG("starting to ignore fallbacks");
            it->state->ignoring_fallbacks = true;
        }

        *p_result = result;
        advance(it);
        MSG("result: %d (%s)", (int)result->sl_id, (*info)->name);
        return 1;
    }
    Py_UNREACHABLE ();
}
