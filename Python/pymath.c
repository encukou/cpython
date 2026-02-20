#include "Python.h"


#ifdef HAVE_GCC_ASM_FOR_X87
// Inline assembly for getting and setting the 387 FPU control word on
// GCC/x86.
_Py_NO_SANITIZE_MEMORY
unsigned short _Py_get_387controlword(void) {
    unsigned short cw;
    __asm__ __volatile__ ("fnstcw %0" : "=m" (cw));
    return cw;
}

void _Py_set_387controlword(unsigned short cw) {
    __asm__ __volatile__ ("fldcw %0" : : "m" (cw));
}
#endif  // HAVE_GCC_ASM_FOR_X87
