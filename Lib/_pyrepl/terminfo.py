"""Pure Python curses-like terminal capability queries."""

from dataclasses import dataclass, field
import errno
import os
from pathlib import Path
import re
import struct
from typing import Callable, Self, Literal, cast
from functools import partial
import operator


# Terminfo constants
MAGIC16 = 0o432  # Magic number for 16-bit terminfo format
MAGIC32 = 0o1036  # Magic number for 32-bit terminfo format

# Special values for absent/cancelled capabilities
ABSENT_BOOLEAN = -1
ABSENT_NUMERIC = -1
CANCELLED_NUMERIC = -2
ABSENT_STRING = None
CANCELLED_STRING = None


# Standard string capability names from ncurses Caps file
# This matches the order used by ncurses when compiling terminfo
# fmt: off
_STRING_NAMES: tuple[str, ...] = (
    "cbt", "bel", "cr", "csr", "tbc", "clear", "el", "ed", "hpa", "cmdch",
    "cup", "cud1", "home", "civis", "cub1", "mrcup", "cnorm", "cuf1", "ll",
    "cuu1", "cvvis", "dch1", "dl1", "dsl", "hd", "smacs", "blink", "bold",
    "smcup", "smdc", "dim", "smir", "invis", "prot", "rev", "smso", "smul",
    "ech", "rmacs", "sgr0", "rmcup", "rmdc", "rmir", "rmso", "rmul", "flash",
    "ff", "fsl", "is1", "is2", "is3", "if", "ich1", "il1", "ip", "kbs", "ktbc",
    "kclr", "kctab", "kdch1", "kdl1", "kcud1", "krmir", "kel", "ked", "kf0",
    "kf1", "kf10", "kf2", "kf3", "kf4", "kf5", "kf6", "kf7", "kf8", "kf9",
    "khome", "kich1", "kil1", "kcub1", "kll", "knp", "kpp", "kcuf1", "kind",
    "kri", "khts", "kcuu1", "rmkx", "smkx", "lf0", "lf1", "lf10", "lf2", "lf3",
    "lf4", "lf5", "lf6", "lf7", "lf8", "lf9", "rmm", "smm", "nel", "pad", "dch",
    "dl", "cud", "ich", "indn", "il", "cub", "cuf", "rin", "cuu", "pfkey",
    "pfloc", "pfx", "mc0", "mc4", "mc5", "rep", "rs1", "rs2", "rs3", "rf", "rc",
    "vpa", "sc", "ind", "ri", "sgr", "hts", "wind", "ht", "tsl", "uc", "hu",
    "iprog", "ka1", "ka3", "kb2", "kc1", "kc3", "mc5p", "rmp", "acsc", "pln",
    "kcbt", "smxon", "rmxon", "smam", "rmam", "xonc", "xoffc", "enacs", "smln",
    "rmln", "kbeg", "kcan", "kclo", "kcmd", "kcpy", "kcrt", "kend", "kent",
    "kext", "kfnd", "khlp", "kmrk", "kmsg", "kmov", "knxt", "kopn", "kopt",
    "kprv", "kprt", "krdo", "kref", "krfr", "krpl", "krst", "kres", "ksav",
    "kspd", "kund", "kBEG", "kCAN", "kCMD", "kCPY", "kCRT", "kDC", "kDL",
    "kslt", "kEND", "kEOL", "kEXT", "kFND", "kHLP", "kHOM", "kIC", "kLFT",
    "kMSG", "kMOV", "kNXT", "kOPT", "kPRV", "kPRT", "kRDO", "kRPL", "kRIT",
    "kRES", "kSAV", "kSPD", "kUND", "rfi", "kf11", "kf12", "kf13", "kf14",
    "kf15", "kf16", "kf17", "kf18", "kf19", "kf20", "kf21", "kf22", "kf23",
    "kf24", "kf25", "kf26", "kf27", "kf28", "kf29", "kf30", "kf31", "kf32",
    "kf33", "kf34", "kf35", "kf36", "kf37", "kf38", "kf39", "kf40", "kf41",
    "kf42", "kf43", "kf44", "kf45", "kf46", "kf47", "kf48", "kf49", "kf50",
    "kf51", "kf52", "kf53", "kf54", "kf55", "kf56", "kf57", "kf58", "kf59",
    "kf60", "kf61", "kf62", "kf63", "el1", "mgc", "smgl", "smgr", "fln", "sclk",
    "dclk", "rmclk", "cwin", "wingo", "hup","dial", "qdial", "tone", "pulse",
    "hook", "pause", "wait", "u0", "u1", "u2", "u3", "u4", "u5", "u6", "u7",
    "u8", "u9", "op", "oc", "initc", "initp", "scp", "setf", "setb", "cpi",
    "lpi", "chr", "cvr", "defc", "swidm", "sdrfq", "sitm", "slm", "smicm",
    "snlq", "snrmq", "sshm", "ssubm", "ssupm", "sum", "rwidm", "ritm", "rlm",
    "rmicm", "rshm", "rsubm", "rsupm", "rum", "mhpa", "mcud1", "mcub1", "mcuf1",
    "mvpa", "mcuu1", "porder", "mcud", "mcub", "mcuf", "mcuu", "scs", "smgb",
    "smgbp", "smglp", "smgrp", "smgt", "smgtp", "sbim", "scsd", "rbim", "rcsd",
    "subcs", "supcs", "docr", "zerom", "csnm", "kmous", "minfo", "reqmp",
    "getm", "setaf", "setab", "pfxl", "devt", "csin", "s0ds", "s1ds", "s2ds",
    "s3ds", "smglr", "smgtb", "birep", "binel", "bicr", "colornm", "defbi",
    "endbi", "setcolor", "slines", "dispc", "smpch", "rmpch", "smsc", "rmsc",
    "pctrm", "scesc", "scesa", "ehhlm", "elhlm", "elohlm", "erhlm", "ethlm",
    "evhlm", "sgr1", "slength", "OTi2", "OTrs", "OTnl", "OTbc", "OTko", "OTma",
    "OTG2", "OTG3", "OTG1", "OTG4", "OTGR", "OTGL", "OTGU", "OTGD", "OTGH",
    "OTGV", "OTGC","meml", "memu", "box1"
)
# fmt: on


def _get_terminfo_dirs() -> list[Path]:
    """Get list of directories to search for terminfo files.

    Based on ncurses behavior in:
    - ncurses/tinfo/db_iterator.c:_nc_next_db()
    - ncurses/tinfo/read_entry.c:_nc_read_entry()
    """
    dirs = []

    terminfo = os.environ.get("TERMINFO")
    if terminfo:
        dirs.append(terminfo)

    try:
        home = Path.home()
        dirs.append(str(home / ".terminfo"))
    except RuntimeError:
        pass

    # Check TERMINFO_DIRS
    terminfo_dirs = os.environ.get("TERMINFO_DIRS", "")
    if terminfo_dirs:
        for d in terminfo_dirs.split(":"):
            if d:
                dirs.append(d)

    dirs.extend(
        [
            "/etc/terminfo",
            "/lib/terminfo",
            "/usr/lib/terminfo",
            "/usr/share/terminfo",
            "/usr/share/lib/terminfo",
            "/usr/share/misc/terminfo",
            "/usr/local/lib/terminfo",
            "/usr/local/share/terminfo",
        ]
    )

    return [Path(d) for d in dirs if Path(d).is_dir()]


def _validate_terminal_name_or_raise(terminal_name: str) -> None:
    if not isinstance(terminal_name, str):
        raise TypeError("`terminal_name` must be a string")

    if not terminal_name:
        raise ValueError("`terminal_name` cannot be empty")

    if "\x00" in terminal_name:
        raise ValueError("NUL character found in `terminal_name`")

    t = Path(terminal_name)
    if len(t.parts) > 1:
        raise ValueError("`terminal_name` cannot contain path separators")


def _read_terminfo_file(terminal_name: str) -> bytes:
    """Find and read terminfo file for given terminal name.

    Terminfo files are stored in directories using the first character
    of the terminal name as a subdirectory.
    """
    _validate_terminal_name_or_raise(terminal_name)
    first_char = terminal_name[0].lower()
    filename = terminal_name

    for directory in _get_terminfo_dirs():
        path = directory / first_char / filename
        if path.is_file():
            return path.read_bytes()

        # Try with hex encoding of first char (for special chars)
        hex_dir = "%02x" % ord(first_char)
        path = directory / hex_dir / filename
        if path.is_file():
            return path.read_bytes()

    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), filename)


# Hard-coded terminal capabilities for common terminals
# This is a minimal subset needed by PyREPL
_TERMINAL_CAPABILITIES = {
    # ANSI/xterm-compatible terminals
    "ansi": {
        # Bell
        "bel": b"\x07",
        # Cursor movement
        "cub": b"\x1b[%p1%dD",  # Move cursor left N columns
        "cud": b"\x1b[%p1%dB",  # Move cursor down N rows
        "cuf": b"\x1b[%p1%dC",  # Move cursor right N columns
        "cuu": b"\x1b[%p1%dA",  # Move cursor up N rows
        "cub1": b"\x08",  # Move cursor left 1 column
        "cud1": b"\n",  # Move cursor down 1 row
        "cuf1": b"\x1b[C",  # Move cursor right 1 column
        "cuu1": b"\x1b[A",  # Move cursor up 1 row
        "cup": b"\x1b[%i%p1%d;%p2%dH",  # Move cursor to row, column
        "hpa": b"\x1b[%i%p1%dG",  # Move cursor to column
        # Clear operations
        "clear": b"\x1b[H\x1b[2J",  # Clear screen and home cursor
        "el": b"\x1b[K",  # Clear to end of line
        # Insert/delete
        "dch": b"\x1b[%p1%dP",  # Delete N characters
        "dch1": b"\x1b[P",  # Delete 1 character
        "ich": b"\x1b[%p1%d@",  # Insert N characters
        "ich1": b"",  # Insert 1 character
        # Cursor visibility
        "civis": b"\x1b[?25l",  # Make cursor invisible
        "cnorm": b"\x1b[?12l\x1b[?25h",  # Make cursor normal (visible)
        # Scrolling
        "ind": b"\n",  # Scroll up one line
        "ri": b"\x1bM",  # Scroll down one line
        # Keypad mode
        "smkx": b"\x1b[?1h\x1b=",  # Enable keypad mode
        "rmkx": b"\x1b[?1l\x1b>",  # Disable keypad mode
        # Padding (not used in modern terminals)
        "pad": b"",
        # Function keys and special keys
        "kdch1": b"\x1b[3~",  # Delete key
        "kcud1": b"\x1bOB",  # Down arrow
        "kend": b"\x1bOF",  # End key
        "kent": b"\x1bOM",  # Enter key
        "khome": b"\x1bOH",  # Home key
        "kich1": b"\x1b[2~",  # Insert key
        "kcub1": b"\x1bOD",  # Left arrow
        "knp": b"\x1b[6~",  # Page down
        "kpp": b"\x1b[5~",  # Page up
        "kcuf1": b"\x1bOC",  # Right arrow
        "kcuu1": b"\x1bOA",  # Up arrow
        # Function keys F1-F20
        "kf1": b"\x1bOP",
        "kf2": b"\x1bOQ",
        "kf3": b"\x1bOR",
        "kf4": b"\x1bOS",
        "kf5": b"\x1b[15~",
        "kf6": b"\x1b[17~",
        "kf7": b"\x1b[18~",
        "kf8": b"\x1b[19~",
        "kf9": b"\x1b[20~",
        "kf10": b"\x1b[21~",
        "kf11": b"\x1b[23~",
        "kf12": b"\x1b[24~",
        "kf13": b"\x1b[1;2P",
        "kf14": b"\x1b[1;2Q",
        "kf15": b"\x1b[1;2R",
        "kf16": b"\x1b[1;2S",
        "kf17": b"\x1b[15;2~",
        "kf18": b"\x1b[17;2~",
        "kf19": b"\x1b[18;2~",
        "kf20": b"\x1b[19;2~",
    },
    # Dumb terminal - minimal capabilities
    "dumb": {
        "bel": b"\x07",  # Bell
        "cud1": b"\n",  # Move down 1 row (newline)
        "ind": b"\n",  # Scroll up one line (newline)
    },
    # Linux console
    "linux": {
        # Bell
        "bel": b"\x07",
        # Cursor movement
        "cub": b"\x1b[%p1%dD",  # Move cursor left N columns
        "cud": b"\x1b[%p1%dB",  # Move cursor down N rows
        "cuf": b"\x1b[%p1%dC",  # Move cursor right N columns
        "cuu": b"\x1b[%p1%dA",  # Move cursor up N rows
        "cub1": b"\x08",  # Move cursor left 1 column (backspace)
        "cud1": b"\n",  # Move cursor down 1 row (newline)
        "cuf1": b"\x1b[C",  # Move cursor right 1 column
        "cuu1": b"\x1b[A",  # Move cursor up 1 row
        "cup": b"\x1b[%i%p1%d;%p2%dH",  # Move cursor to row, column
        "hpa": b"\x1b[%i%p1%dG",  # Move cursor to column
        # Clear operations
        "clear": b"\x1b[H\x1b[J",  # Clear screen and home cursor (different from ansi!)
        "el": b"\x1b[K",  # Clear to end of line
        # Insert/delete
        "dch": b"\x1b[%p1%dP",  # Delete N characters
        "dch1": b"\x1b[P",  # Delete 1 character
        "ich": b"\x1b[%p1%d@",  # Insert N characters
        "ich1": b"\x1b[@",  # Insert 1 character
        # Cursor visibility
        "civis": b"\x1b[?25l\x1b[?1c",  # Make cursor invisible
        "cnorm": b"\x1b[?25h\x1b[?0c",  # Make cursor normal
        # Scrolling
        "ind": b"\n",  # Scroll up one line
        "ri": b"\x1bM",  # Scroll down one line
        # Keypad mode
        "smkx": b"\x1b[?1h\x1b=",  # Enable keypad mode
        "rmkx": b"\x1b[?1l\x1b>",  # Disable keypad mode
        # Function keys and special keys
        "kdch1": b"\x1b[3~",  # Delete key
        "kcud1": b"\x1b[B",  # Down arrow
        "kend": b"\x1b[4~",  # End key (different from ansi!)
        "khome": b"\x1b[1~",  # Home key (different from ansi!)
        "kich1": b"\x1b[2~",  # Insert key
        "kcub1": b"\x1b[D",  # Left arrow
        "knp": b"\x1b[6~",  # Page down
        "kpp": b"\x1b[5~",  # Page up
        "kcuf1": b"\x1b[C",  # Right arrow
        "kcuu1": b"\x1b[A",  # Up arrow
        # Function keys
        "kf1": b"\x1b[[A",
        "kf2": b"\x1b[[B",
        "kf3": b"\x1b[[C",
        "kf4": b"\x1b[[D",
        "kf5": b"\x1b[[E",
        "kf6": b"\x1b[17~",
        "kf7": b"\x1b[18~",
        "kf8": b"\x1b[19~",
        "kf9": b"\x1b[20~",
        "kf10": b"\x1b[21~",
        "kf11": b"\x1b[23~",
        "kf12": b"\x1b[24~",
        "kf13": b"\x1b[25~",
        "kf14": b"\x1b[26~",
        "kf15": b"\x1b[28~",
        "kf16": b"\x1b[29~",
        "kf17": b"\x1b[31~",
        "kf18": b"\x1b[32~",
        "kf19": b"\x1b[33~",
        "kf20": b"\x1b[34~",
    },
}

# Map common TERM values to capability sets
_TERM_ALIASES = {
    "xterm": "ansi",
    "xterm-color": "ansi",
    "xterm-256color": "ansi",
    "screen": "ansi",
    "screen-256color": "ansi",
    "tmux": "ansi",
    "tmux-256color": "ansi",
    "vt100": "ansi",
    "vt220": "ansi",
    "rxvt": "ansi",
    "rxvt-unicode": "ansi",
    "rxvt-unicode-256color": "ansi",
    "unknown": "dumb",
}


@dataclass
class TermInfo:
    terminal_name: str | bytes | None
    fallback: bool = True

    _capabilities: dict[str, 'Tparm'] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initialize terminal capabilities for the given terminal type.

        Based on ncurses implementation in:
        - ncurses/tinfo/lib_setup.c:setupterm() and _nc_setupterm()
        - ncurses/tinfo/lib_setup.c:TINFO_SETUP_TERM()

        This version first attempts to read terminfo database files like ncurses,
        then, if `fallback` is True, falls back to hardcoded capabilities for
        common terminal types.
        """
        # If termstr is None or empty, try to get from environment
        if not self.terminal_name:
            self.terminal_name = os.environ.get("TERM") or "ANSI"

        if isinstance(self.terminal_name, bytes):
            self.terminal_name = self.terminal_name.decode("ascii")

        try:
            self._parse_terminfo_file(self.terminal_name)
        except (OSError, ValueError):
            if not self.fallback:
                raise

            term_type = _TERM_ALIASES.get(
                self.terminal_name, self.terminal_name
            )
            if term_type not in _TERMINAL_CAPABILITIES:
                term_type = "dumb"
            self._capabilities = {
                name: Tparm.from_bytes(cap)
                for name, cap in _TERMINAL_CAPABILITIES[term_type].items()
            }

    def _parse_terminfo_file(self, terminal_name: str) -> None:
        """Parse a terminfo file.

        Populate the _capabilities dict for easy retrieval

        Based on ncurses implementation in:
        - ncurses/tinfo/read_entry.c:_nc_read_termtype()
        - ncurses/tinfo/read_entry.c:_nc_read_file_entry()
        - ncurses/tinfo/lib_ti.c:tigetstr()
        """
        data = _read_terminfo_file(terminal_name)
        too_short = f"TermInfo file for {terminal_name!r} too short"
        offset = 12
        if len(data) < offset:
            raise ValueError(too_short)

        magic, name_size, bool_count, num_count, str_count, str_size = (
            struct.unpack("<Hhhhhh", data[:offset])
        )

        if magic == MAGIC16:
            number_size = 2
        elif magic == MAGIC32:
            number_size = 4
        else:
            raise ValueError(
                f"TermInfo file for {terminal_name!r} uses unknown magic"
            )

        # Skip data than PyREPL doesn't need:
        # - names (`|`-separated ASCII strings)
        # - boolean capabilities (bytes with value 0 or 1)
        # - numbers (little-endian integers, `number_size` bytes each)
        offset += name_size
        offset += bool_count
        if offset % 2:
            # Align to even byte boundary for numbers
            offset += 1
        offset += num_count * number_size
        if offset > len(data):
            raise ValueError(too_short)

        # Read string offsets
        end_offset = offset + 2 * str_count
        if offset > len(data):
            raise ValueError(too_short)
        string_offset_data = data[offset:end_offset]
        string_offsets = [
            off for [off] in struct.iter_unpack("<h", string_offset_data)
        ]
        offset = end_offset

        # Read string table
        if offset + str_size > len(data):
            raise ValueError(too_short)
        string_table = data[offset : offset + str_size]

        # Extract strings from string table
        capabilities = {}
        for cap, off in zip(_STRING_NAMES, string_offsets):
            if off < 0:
                # CANCELLED_STRING; we do not store those
                continue
            elif off < len(string_table):
                # Find null terminator
                end = string_table.find(0, off)
                if end >= 0:
                    capabilities[cap] = Tparm.from_bytes(string_table[off:end])
            # in other cases this is ABSENT_STRING; we don't store those.

        # Note: we don't support extended capabilities since PyREPL doesn't
        # need them.

        self._capabilities = capabilities

    def get(self, cap: str) -> Tparm | None:
        """Get terminal capability string by name.
        """
        if not isinstance(cap, str):
            raise TypeError(f"`cap` must be a string, not {type(cap)}")

        return self._capabilities.get(cap)


def tparm(cap: Tparm, *params: int) -> bytes:
    """Parameterize a terminal capability string.

    Based on ncurses implementation in:
    - ncurses/tinfo/lib_tparm.c:tparm()
    - ncurses/tinfo/lib_tparm.c:tparam_internal()

    The ncurses version implements a full stack-based interpreter for
    terminfo parameter strings. This pure Python version implements only
    the subset of parameter substitution operations needed by PyREPL:
    - %i (increment parameters for 1-based indexing)
    - %p[1-9]%d (parameter substitution)
    - %p[1-9]%{n}%+%d (parameter plus constant)
    """
    if not isinstance(cap, Tparm):
        raise TypeError(f"`cap` must be Tparm, not {type(cap)}")

    return cap(*params)

type TparmPart = bytes | Callable[['TparmState'], bytes | None]

@dataclass
class Tparm:
    pieces: tuple[TparmPart, ...]
    source: bytes

    def __call__(self, *params: int) -> bytes:
        result = []
        state = TparmState(params)
        # Note: in ncurses, variable values persist across calls to tparm.
        # We create fresh state every time.
        for part in self.pieces:
            match part:
                case bytes():
                    if state.active:
                        result.append(part)
                case ConditionalOp():
                    part(state)
                case _:
                    if state.active:
                        piece = part(state)
                        if piece:
                            result.append(piece)
        return b''.join(result)

    @classmethod
    def from_const(cls, src: bytes) -> Self:
        return cls((src,), src)

    @classmethod
    def from_bytes(cls, src: bytes) -> Self:
        parts: list[TparmPart] = []
        pos = 0
        while pos < len(src):
            percent_pos = src.find(b'%', pos)
            if percent_pos == -1:
                parts.append(src[pos:])
                break
            part = src[pos:percent_pos]
            if part:
                parts.append(part)
            pos = percent_pos + 2
            control = src[pos-1:pos]
            def append_binop(func: Callable[[int, int], int]) -> None:
                parts.append(partial(TparmState.binary_op, op=func))
            match control:
                #fmt: off
                case b'?': parts.append(TparmState.cond_if)
                case b't': parts.append(TparmState.cond_then)
                case b'e': parts.append(TparmState.cond_else)
                case b';': parts.append(TparmState.cond_endif)
                case b'%': parts.append(b'%')
                case b'c': parts.append(TparmState.output_byte)
                case b's': parts.append(TparmState.output_decimal)
                case b'l': parts.append(TparmState.output_length)
                case b'+': append_binop(operator.add)
                case b'-': append_binop(operator.sub)
                case b'*': append_binop(operator.mul)
                case b'/': append_binop(lambda a, b: int(a / b))
                case b'm': append_binop(lambda a, b: a - (int(a / b) * b))
                case b'&': append_binop(operator.and_)
                case b'|': append_binop(operator.or_)
                case b'^': append_binop(operator.xor)
                case b'=': append_binop(lambda a, b: int(a == b))
                case b'>': append_binop(lambda a, b: int(a > b))
                case b'<': append_binop(lambda a, b: int(a < b))
                case b'A': append_binop(lambda a, b: 1 if a and b else 0)
                case b'O': append_binop(lambda a, b: 1 if a or b else 0)
                case b'!': parts.append(TparmState.not_op)
                case b'~': parts.append(TparmState.inv_op)
                case b'i': parts.append(TparmState.inc2)
                #fmt: on
                case b'p':
                    n = int(src[pos:pos+1].decode('ascii'))
                    pos += 1
                    if src[pos:pos+2] == b'%d':
                        parts.append(partial(TparmState.output_param, index=n-1))
                        pos += 2
                    else:
                        parts.append(partial(TparmState.push_param, index=n-1))
                case b'P':
                    n = src[pos]
                    pos += 1
                    parts.append(partial(TparmState.store_var, name=n))
                case b'g':
                    n = src[pos]
                    pos += 1
                    parts.append(partial(TparmState.push_var, name=n))
                case b"'":
                    n = src[pos]
                    pos += 2
                    parts.append(partial(TparmState.push, value=n))
                case b"{":
                    end = src.index(b'}', pos)
                    n = int(src[pos:end].decode('ascii'))
                    pos = end + 1
                    parts.append(partial(TparmState.push, value=n))
                case _:
                    match = re.compile(b'[doxXs]').search(src, pos)
                    if match:
                        end = match.start()
                        fmt = src[pos:end+1]
                        if fmt.startswith(b'%:'):
                            fmt = b'%' + fmt[2:]
                        parts.append(partial(TparmState.format, fmt=fmt))
                        pos = end + 1
        return cls(tuple(parts), src)

@dataclass
class ConditionalOp:
    _func: Callable[['TparmState'], None]

    def __call__(self, state):
        return self._func(state)

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return partial(self, instance)

@dataclass
class TparmState:
    params: tuple[int, ...]
    variables: dict[int, int] = field(default_factory=dict)
    stack: list[int] = field(default_factory=list)
    active: bool = True
    if_stack: list[Literal['cond', 'then', 'wait', 'skip']] = field(default_factory=list)

    def push(self, value: int) -> None:
        self.stack.append(value)

    def pop(self) -> int:
        try:
            return self.stack.pop()
        except IndexError:
            return 0

    def get_param(self, n: int) -> int:
        if 0 <= n < len(self.params):
            return self.params[n]
        return 0

    @ConditionalOp
    def cond_if(self):
        self.if_stack.append('cond')

    @ConditionalOp
    def cond_then(self):
        if self.active:
            if self.pop():
                self.if_stack[-1] = 'then'
            else:
                self.if_stack[-1] = 'wait'
        self._set_active()

    @ConditionalOp
    def cond_else(self):
        if self.if_stack[-1] == 'wait':
            self.if_stack[-1] = 'cond'
        elif self.if_stack[-1] == 'then':
            self.if_stack[-1] = 'skip'
        self._set_active()

    @ConditionalOp
    def cond_endif(self):
        self.if_stack.pop()
        self._set_active()

    def _set_active(self):
        self.active = all(e in {'cond', 'then'} for e in self.if_stack)

    def output_byte(self) -> bytes:
        char = self.pop()
        if char == 0:
            char = 0x80
        char &= 0xff
        if char == 0:
            # ncurses outputs a zero byte, ending the string
            return b''
        return bytes([char])

    def output_decimal(self) -> bytes | None:
        val = self.pop()
        if val:
            return str(val).encode('ascii')
        return None

    def output_length(self) -> bytes:
        val = self.pop()
        return str(len(str(val))).encode('ascii')

    def binary_op(self, op: Callable[[int, int], int]) -> None:
        b = self.pop()
        a = self.pop()
        self.push(op(a, b))

    def not_op(self) -> None:
        self.push(0 if self.pop() else 1)

    def inv_op(self) -> None:
        self.push(~self.pop())

    def push_param(self, index: int) -> None:
        self.push(self.get_param(index))

    def output_param(self, index: int) -> bytes | None:
        val = self.get_param(index)
        if val:
            return str(val).encode('ascii')
        return None

    def store_var(self, name: int) -> None:
        self.variables[name] = self.pop()

    def push_var(self, name: int) -> None:
        self.push(self.variables.get(name, 0))

    def inc2(self) -> None:
        a = self.get_param(0) + 1
        b = self.get_param(1) + 1
        self.params = (a, b, *self.params[2:])
        self.push(a)
        self.push(b)

    def format(self, fmt: bytes) -> bytes:
        val = self.pop()
        if val < 0 and fmt[-1] in b'xXo':
            val = val & 0xFFFFFFFF
        return fmt % val
