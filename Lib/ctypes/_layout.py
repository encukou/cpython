import sys
import warnings
import struct

from _ctypes import CField, buffer_info
import ctypes

def round_down(n, multiple):
    assert n >= 0
    assert multiple > 0
    return (n // multiple) * multiple

def round_up(n, multiple):
    assert n >= 0
    assert multiple > 0
    return ((n + multiple - 1) // multiple) * multiple

def LOW_BIT(offset):
    return offset & 0xFFFF

def NUM_BITS(bitsize):
    return bitsize >> 16

def BUILD_SIZE(bitsize, offset):
    assert(0 <= offset)
    assert(offset <= 0xFFFF)
    ## We don't support zero length bitfields.
    ## And GET_BITFIELD uses NUM_BITS(size)==0,
    ## to figure out whether we are handling a bitfield.
    assert(0 < bitsize)
    result = (bitsize << 16) + offset
    assert(bitsize == NUM_BITS(result))
    assert(offset == LOW_BIT(result))
    return result

def build_size(bit_size, bit_offset, big_endian, type_size):
    if big_endian:
        return BUILD_SIZE(bit_size, 8 * type_size - bit_offset - bit_size)
    return BUILD_SIZE(bit_size, bit_offset)

_INT_MAX = (1 << (ctypes.sizeof(ctypes.c_int) * 8) - 1) - 1

class _structunion_layout:
    """Compute the layout of a struct or union

    This is a callable that returns an object with attributes:
    - fields: sequence of CField objects
    - size: total size of the aggregate
    - align: total alignment requirement of the aggregate
    - format_spec: buffer format specification (as a string, UTF-8 but
      best kept ASCII-only)

    Technically this is a class, that might change.
    """
    def __init__(self, cls, fields, is_struct, base, _ms):
        align = getattr(cls, '_align_', 1)
        if align < 0:
            raise ValueError('_align_ must be a non-negative integer')
        elif align == 0:
            # Setting `_align_ = 0` amounts to using the default alignment
            align == 1

        if base:
            align = max(ctypes.alignment(base), align)

        swapped_bytes = hasattr(cls, '_swappedbytes_')
        if swapped_bytes:
            big_endian = sys.byteorder == 'little'
        else:
            big_endian = sys.byteorder == 'big'

        pack = getattr(cls, '_pack_', None)
        if pack is not None:
            try:
                pack = int(pack)
            except (TypeError, ValueError):
                raise ValueError("_pack_ must be an integer")
            if pack < 0:
                raise ValueError("_pack_ must be a non-negative integer")
            if pack > _INT_MAX:
                raise ValueError("_pack_ too big")
            if not _ms:
                raise ValueError('_pack_ is not compatible with gcc-sysv layout')

        self.fields = []

        if is_struct:
            format_spec_parts = ["T{"]
        else:
            format_spec_parts = ["B"]

        last_field_bit_size = 0  # used in MS layout only

        # `8 * next_byte_offset + next_bit_offset` points to where the
        # next field would start.
        next_bit_offset = 0
        next_byte_offset = 0

        # size if this was a struct (sum of field sizes, plus padding)
        struct_size = 0
        # max of field sizes; only meaningful for unions
        union_size = 0

        if base:
            struct_size = ctypes.sizeof(base)
            if _ms:
                next_byte_offset = struct_size
            else:
                next_bit_offset = struct_size * 8

        last_size = struct_size
        last_field = None
        for i, field in enumerate(fields):
            if not is_struct:
                # Unions start fresh each time
                last_field_bit_size = 0
                next_bit_offset = 0
                next_byte_offset = 0

            # Unpack the field
            field = tuple(field)
            try:
                name, ctype = field
                is_bitfield = False
                type_size = ctypes.sizeof(ctype)
                bit_size = type_size * 8
            except ValueError:
                name, ctype, bit_size = field
                is_bitfield = True
                if bit_size <= 0:
                    raise ValueError(f'number of bits invalid for bit field {name!r}')
                type_size = ctypes.sizeof(ctype)

            type_bit_size = type_size * 8
            type_align = ctypes.alignment(ctype) or 1
            type_bit_align = type_align * 8

            if not _ms:
                # We don't use next_byte_offset here
                assert pack is None
                assert next_byte_offset == 0

                # Determine whether the bit field, if placed at the next
                # free bit, fits within a single object of its specified type.
                # That is: determine a "slot", sized & aligned for the
                # specified type, which contains the bitfield's beginning:
                slot_start_bit = round_down(next_bit_offset, type_bit_align)
                slot_end_bit = slot_start_bit + type_bit_size
                # And see if it also contains the bitfield's last bit:
                field_end_bit = next_bit_offset + bit_size
                if field_end_bit > slot_end_bit:
                    # It doesn't: add padding (bump up to the next
                    # alignment boundary)
                    next_bit_offset = round_up(next_bit_offset, type_bit_align)

                offset = round_down(next_bit_offset, type_bit_align) // 8
                if is_bitfield:
                    effective_bit_offset = next_bit_offset - 8 * offset
                    size = build_size(bit_size, effective_bit_offset,
                                      big_endian, type_size)
                    assert effective_bit_offset <= type_bit_size
                else:
                    assert offset == next_bit_offset / 8
                    size = type_size

                next_bit_offset += bit_size
                struct_size = round_up(next_bit_offset, 8) // 8
            else:
                if pack:
                    type_align = min(pack, type_align)

                # next_byte_offset points to end of current bitfield.
                # next_bit_offset is generally non-positive,
                # and 8 * next_byte_offset + next_bit_offset points just behind
                # the end of the last field we placed.
                if (
                    (0 < next_bit_offset + bit_size)
                    or (type_bit_size != last_field_bit_size)
                ):
                    # Close the previous bitfield (if any)
                    # and start a new bitfield
                    next_byte_offset = round_up(next_byte_offset, type_align)

                    next_byte_offset += type_size

                    last_field_bit_size = type_bit_size
                    # Reminder: 8 * (next_byte_offset) + next_bit_offset
                    # points to where we would start a
                    # new field.  I.e. just behind where we placed the last
                    # field plus an allowance for alignment.
                    next_bit_offset = - last_field_bit_size

                assert type_bit_size == last_field_bit_size
                assert type_bit_size > 0

                offset = next_byte_offset - last_field_bit_size // 8
                if is_bitfield:
                    assert 0 <= (last_field_bit_size + next_bit_offset)
                    size = build_size(bit_size,
                                      last_field_bit_size + next_bit_offset,
                                      big_endian, type_size)
                else:
                    size = type_size
                assert (last_field_bit_size + next_bit_offset) < type_bit_size

                next_bit_offset += bit_size
                struct_size = next_byte_offset

            assert((not is_bitfield) or (LOW_BIT(size) <= size * 8))

            # Add the format spec parts
            if is_struct:
                padding = offset - last_size
                format_spec_parts.append(padding_spec(padding))

                fieldfmt, bf_ndim, bf_shape = buffer_info(ctype)

                if bf_shape:
                    format_spec_parts.extend((
                        "(",
                        ','.join(str(n) for n in bf_shape),
                        ")",
                    ))

                if fieldfmt is None:
                    fieldfmt = "B"
                format_spec_parts.append(f"{fieldfmt}:{name}:")

            last_field = CField(
                name=name,
                type=ctype,
                size=size,
                offset=offset,
                bit_size=bit_size if is_bitfield else None,
                index=i,
            )
            self.fields.append(last_field)
            align = max(align, type_align)
            last_size = struct_size
            if not is_struct:
                union_size = max(struct_size, union_size)

        if is_struct:
            total_size = struct_size
        else:
            total_size = union_size

        # Adjust the size according to the alignment requirements
        aligned_size = round_up(total_size, align)

        # Finish up the format spec
        if is_struct:
            padding = aligned_size - total_size
            format_spec_parts.append(padding_spec(padding))
            format_spec_parts.append("}")

        self.size = aligned_size
        self.align = align
        self.format_spec = "".join(format_spec_parts)


def padding_spec(padding):
    if padding <= 0:
        return ""
    if padding == 1:
        return "x"
    return f"{padding}x"


def default_layout(cls, *args, **kwargs):
    layout = getattr(cls, '_layout_', None)
    if layout is None:
        if sys.platform == 'win32' or getattr(cls, '_pack_', None):
            return _structunion_layout(cls, *args, **kwargs, _ms=True)
        return _structunion_layout(cls, *args, **kwargs, _ms=False)
    elif layout == 'ms':
        return _structunion_layout(cls, *args, **kwargs, _ms=True)
    elif layout == 'gcc-sysv':
        return _structunion_layout(cls, *args, **kwargs, _ms=False)
    else:
        raise ValueError(f'unknown _layout_: {layout!r}')
