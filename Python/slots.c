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
_PySlotIterator_InitWithKind(_PySlotIterator *it, PySlot *slots,
                             Py_ssize_t n_slots,
                             int result_kind, int slot_struct_kind)
{
    MSG("");
    MSG("init (%s slot iterator, size %d)", kind_name(kind), (int)n_slots);
    memset(it, 0, sizeof(_PySlotIterator));
    it->state = it->states;
    it->state->slot = slots;
    it->state->slot_struct_kind = slot_struct_kind;
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
    it->kind = result_kind;
    return 0;
}

static Py_ssize_t
seen_index(uint16_t id)
{
    return id / sizeof(unsigned int);
}

static unsigned int
seen_mask(uint16_t id)
{
    return ((unsigned int)1) << (id % sizeof(unsigned int));
}

bool
_PySlotIterator_SawSlot(_PySlotIterator *it, int id)
{
    assert (id > 0);
    assert (id < _Py_slot_COUNT);
    return it->seen[seen_index(id)] & seen_mask(id);
}

int
_PySlotIterator_ValidateCurrentSlot(_PySlotIterator *it)
{
    const _PySlot_Info *info = it->info;
    int id = it->current.sl_id;

    if (it->info->kind != it->kind) {
        MSG("error (bad slot kind)");
        PyErr_Format(PyExc_SystemError,
                        "Py_%s (slot %d) is not compatible with %ss",
                        info->name,
                        id,
                        kind_name(it->kind));
        return -1;
    }

    if (it->info->null_handling != _PySlot_NULL_ALLOW) {
        bool is_null = false;
        switch (it->info->dtype) {
            case _PySlot_TYPE_PTR: {
                is_null = it->current.sl_ptr == NULL;
            } break;
            case _PySlot_TYPE_FUNC: {
                is_null = it->current.sl_func == NULL;
            } break;
            default: {
                Py_UNREACHABLE();
            } break;
        }
        if (is_null) {
            if (it->info->null_handling == _PySlot_NULL_DEPRECATED) {
                if (PyErr_WarnFormat(
                    PyExc_DeprecationWarning,
                    1,
                    "NULL value in slot Py_%s is deprecated",
                    it->info->name) < 0)
                {
                    return -1;
                }
            }
            else {
                PyErr_Format(PyExc_SystemError,
                             "NULL not allowed for slot %s",
                             it->info->name);
                return -1;
            }
        }
    }

    if (info->reject_duplicates || info->deprecate_duplicates) {
        if (_PySlotIterator_SawSlot(it, id)) {
            MSG("slot was seen before");
            if (info->reject_duplicates) {
                MSG("error (duplicate rejected)");
                PyErr_Format(
                    PyExc_SystemError,
                    "%s%s%s has multiple Py_%s (%d) slots",
                    kind_name(it->kind),
                    it->name ? " " : "",
                    it->name ? it->name : "",
                    info->name,
                    (int)it->current.sl_id);
                return -1;
            }
            MSG("deprecated duplicate");
            if (PyErr_WarnFormat(
                    PyExc_DeprecationWarning,
                    0,
                    "%s%s%s has multiple Py_%s (%d) slots. This is deprecated.",
                    kind_name(it->kind),
                    it->name ? " " : "",
                    it->name ? it->name : "",
                    info->name,
                    (int)it->current.sl_id) < 0) {
                return -1;
            }
        }
    }
    it->seen[seen_index(id)] |= seen_mask(id);
    return 0;
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

// Get the next slot in the iterator, and set to the info.
// Return 0 if there are no more entries, the slot ID if there are.
// On error, return -1 with exception set.
int
_PySlotIterator_Next(_PySlotIterator *it)
{
    MSG("next");
    assert(it);

    it->current.sl_id = -1;

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
                it->current = *it->state->slot;  /* struct copy */
            } break;
            case _PySlot_KIND_TYPE: {
                MSG("type slot structure");
                memset(&it->current, 0, sizeof(it->current));
                it->current.sl_id = it->state->tp_slot->slot;
                it->current.sl_flags = PySlot_INTPTR;
                it->current.sl_ptr = (void*)it->state->tp_slot->pfunc;
            } break;
            case _PySlot_KIND_MOD: {
                MSG("module slot structure");
                memset(&it->current, 0, sizeof(it->current));
                it->current.sl_id = it->state->mod_slot->slot;
                it->current.sl_flags = PySlot_INTPTR;
                it->current.sl_ptr = (void*)it->state->mod_slot->value;
            } break;
            default: {
                Py_UNREACHABLE();
            } break;
        }
        PySlot *const result = &it->current;
        uint16_t flags = result->sl_flags;
        MSG("slot %d flags 0x%x @%p", (int)result->sl_id, (unsigned)flags,
            it->state->slot);
        if ((flags & PySlot_SKIP_IF_NULL)
            && result->sl_ptr == NULL
            && result->sl_func == NULL
            && result->sl_size == 0
            && result->sl_int64 == 0
            )
        {
            MSG("skipped (NULL)");
            advance(it);
            continue;
        }
        if (it->state->ignoring_fallbacks) {
            if (!(flags & PySlot_HAS_FALLBACK)) {
                MSG("stopping to ignore fallbacks");
                it->state->ignoring_fallbacks = false;
            }
            MSG("skipped (ignoring fallbacks)");
            advance(it);
            continue;
        }
        if (result->sl_id >= _Py_slot_COUNT) {
            if (flags & (PySlot_OPTIONAL | PySlot_HAS_FALLBACK)) {
                MSG("skipped (unknown slot)");
                advance(it);
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
                advance(it);
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
        it->info = &_PySlot_InfoTable[result->sl_id];
        MSG("slot %d: %s", (int)result->sl_id, it->info->name);

        if (it->info->is_name) {
            assert(it->info->dtype == _PySlot_TYPE_PTR);
            it->name = result->sl_ptr;
        }

        // Resolve a legacy ambiguous slot number
        // Save the original slot definition for error messages.
        uint16_t orig_id = result->sl_id;
        _PySlot_Info *orig_info = &_PySlot_InfoTable[result->sl_id];
        if (it->info->kind == _PySlot_KIND_COMPAT) {
            MSG("resolving compat slot");
            switch (it->kind) {
                case _PySlot_KIND_TYPE: {
                    result->sl_id = it->info->compat_info.type_id;
                } break;
                case _PySlot_KIND_MOD: {
                    result->sl_id = it->info->compat_info.mod_id;
                } break;
                default: {
                    Py_UNREACHABLE();
                } break;
            }
            it->info = &_PySlot_InfoTable[result->sl_id];
            MSG("slot %d: %s", (int)result->sl_id, it->info->name);
        }

        if (it->info->subslots) {
            if (result->sl_ptr == NULL) {
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
            it->state->slot_struct_kind = it->info->kind;
            it->state->zero_terminated = true;
            continue;
        }

        if (flags & PySlot_INTPTR) {
            MSG("casting from intptr");
            switch (it->info->dtype) {
                case _PySlot_TYPE_SIZE: {
                    result->sl_size = (Py_ssize_t)(intptr_t)result->sl_ptr;
                } break;
                case _PySlot_TYPE_INT64: {
                    result->sl_int64 = (int64_t)(intptr_t)result->sl_ptr;
                } break;
                case _PySlot_TYPE_UINT64: {
                    result->sl_uint64 = (uint64_t)(intptr_t)result->sl_ptr;
                } break;
                case _PySlot_TYPE_PTR:
                case _PySlot_TYPE_FUNC:
                    break;
                default:
                    Py_UNREACHABLE();
            }
        }

        if (flags & PySlot_HAS_FALLBACK) {
            MSG("starting to ignore fallbacks");
            it->state->ignoring_fallbacks = true;
        }

        advance(it);
        MSG("result: %d (%s)", (int)result->sl_id, it->info->name);
        assert (result->sl_id > 0);
        assert (result->sl_id <= INT_MAX);
        return (int)result->sl_id;
    }
    Py_UNREACHABLE ();
}
