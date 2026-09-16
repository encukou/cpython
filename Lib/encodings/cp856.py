""" Python Character Mapping Codec cp856 generated from 'MAPPINGS/VENDORS/MISC/CP856.TXT' with gencodec.py.

"""#"

import codecs

### Codec APIs

class Codec(codecs.Codec):

    def encode(self,input,errors='strict'):
        return codecs.charmap_encode(input,errors,encoding_table)

    def decode(self,input,errors='strict'):
        return codecs.charmap_decode(input,errors,decoding_table)

class IncrementalEncoder(codecs.IncrementalEncoder):
    def encode(self, input, final=False):
        return codecs.charmap_encode(input,self.errors,encoding_table)[0]

class IncrementalDecoder(codecs.IncrementalDecoder):
    def decode(self, input, final=False):
        return codecs.charmap_decode(input,self.errors,decoding_table)[0]

class StreamWriter(Codec,codecs.StreamWriter):
    pass

class StreamReader(Codec,codecs.StreamReader):
    pass

### encodings module API

def getregentry():
    return codecs.CodecInfo(
        name='cp856',
        encode=Codec().encode,
        decode=Codec().decode,
        incrementalencoder=IncrementalEncoder,
        incrementaldecoder=IncrementalDecoder,
        streamreader=StreamReader,
        streamwriter=StreamWriter,
    )


### Decoding Table

decoding_table = (
    ''.join(chr(x) for x in range(0x80)) +
    '\u05d0'   #  0x80 -> HEBREW LETTER ALEF
    '\u05d1'   #  0x81 -> HEBREW LETTER BET
    '\u05d2'   #  0x82 -> HEBREW LETTER GIMEL
    '\u05d3'   #  0x83 -> HEBREW LETTER DALET
    '\u05d4'   #  0x84 -> HEBREW LETTER HE
    '\u05d5'   #  0x85 -> HEBREW LETTER VAV
    '\u05d6'   #  0x86 -> HEBREW LETTER ZAYIN
    '\u05d7'   #  0x87 -> HEBREW LETTER HET
    '\u05d8'   #  0x88 -> HEBREW LETTER TET
    '\u05d9'   #  0x89 -> HEBREW LETTER YOD
    '\u05da'   #  0x8A -> HEBREW LETTER FINAL KAF
    '\u05db'   #  0x8B -> HEBREW LETTER KAF
    '\u05dc'   #  0x8C -> HEBREW LETTER LAMED
    '\u05dd'   #  0x8D -> HEBREW LETTER FINAL MEM
    '\u05de'   #  0x8E -> HEBREW LETTER MEM
    '\u05df'   #  0x8F -> HEBREW LETTER FINAL NUN
    '\u05e0'   #  0x90 -> HEBREW LETTER NUN
    '\u05e1'   #  0x91 -> HEBREW LETTER SAMEKH
    '\u05e2'   #  0x92 -> HEBREW LETTER AYIN
    '\u05e3'   #  0x93 -> HEBREW LETTER FINAL PE
    '\u05e4'   #  0x94 -> HEBREW LETTER PE
    '\u05e5'   #  0x95 -> HEBREW LETTER FINAL TSADI
    '\u05e6'   #  0x96 -> HEBREW LETTER TSADI
    '\u05e7'   #  0x97 -> HEBREW LETTER QOF
    '\u05e8'   #  0x98 -> HEBREW LETTER RESH
    '\u05e9'   #  0x99 -> HEBREW LETTER SHIN
    '\u05ea'   #  0x9A -> HEBREW LETTER TAV
    '\ufffe'   #  0x9B -> UNDEFINED
    '\xa3'     #  0x9C -> POUND SIGN
    '\ufffe'   #  0x9D -> UNDEFINED
    '\xd7'     #  0x9E -> MULTIPLICATION SIGN
    '\ufffe'   #  0x9F -> UNDEFINED
    '\ufffe'   #  0xA0 -> UNDEFINED
    '\ufffe'   #  0xA1 -> UNDEFINED
    '\ufffe'   #  0xA2 -> UNDEFINED
    '\ufffe'   #  0xA3 -> UNDEFINED
    '\ufffe'   #  0xA4 -> UNDEFINED
    '\ufffe'   #  0xA5 -> UNDEFINED
    '\ufffe'   #  0xA6 -> UNDEFINED
    '\ufffe'   #  0xA7 -> UNDEFINED
    '\ufffe'   #  0xA8 -> UNDEFINED
    '\xae'     #  0xA9 -> REGISTERED SIGN
    '\xac'     #  0xAA -> NOT SIGN
    '\xbd'     #  0xAB -> VULGAR FRACTION ONE HALF
    '\xbc'     #  0xAC -> VULGAR FRACTION ONE QUARTER
    '\ufffe'   #  0xAD -> UNDEFINED
    '\xab'     #  0xAE -> LEFT-POINTING DOUBLE ANGLE QUOTATION MARK
    '\xbb'     #  0xAF -> RIGHT-POINTING DOUBLE ANGLE QUOTATION MARK
    '\u2591'   #  0xB0 -> LIGHT SHADE
    '\u2592'   #  0xB1 -> MEDIUM SHADE
    '\u2593'   #  0xB2 -> DARK SHADE
    '\u2502'   #  0xB3 -> BOX DRAWINGS LIGHT VERTICAL
    '\u2524'   #  0xB4 -> BOX DRAWINGS LIGHT VERTICAL AND LEFT
    '\ufffe'   #  0xB5 -> UNDEFINED
    '\ufffe'   #  0xB6 -> UNDEFINED
    '\ufffe'   #  0xB7 -> UNDEFINED
    '\xa9'     #  0xB8 -> COPYRIGHT SIGN
    '\u2563'   #  0xB9 -> BOX DRAWINGS DOUBLE VERTICAL AND LEFT
    '\u2551'   #  0xBA -> BOX DRAWINGS DOUBLE VERTICAL
    '\u2557'   #  0xBB -> BOX DRAWINGS DOUBLE DOWN AND LEFT
    '\u255d'   #  0xBC -> BOX DRAWINGS DOUBLE UP AND LEFT
    '\xa2'     #  0xBD -> CENT SIGN
    '\xa5'     #  0xBE -> YEN SIGN
    '\u2510'   #  0xBF -> BOX DRAWINGS LIGHT DOWN AND LEFT
    '\u2514'   #  0xC0 -> BOX DRAWINGS LIGHT UP AND RIGHT
    '\u2534'   #  0xC1 -> BOX DRAWINGS LIGHT UP AND HORIZONTAL
    '\u252c'   #  0xC2 -> BOX DRAWINGS LIGHT DOWN AND HORIZONTAL
    '\u251c'   #  0xC3 -> BOX DRAWINGS LIGHT VERTICAL AND RIGHT
    '\u2500'   #  0xC4 -> BOX DRAWINGS LIGHT HORIZONTAL
    '\u253c'   #  0xC5 -> BOX DRAWINGS LIGHT VERTICAL AND HORIZONTAL
    '\ufffe'   #  0xC6 -> UNDEFINED
    '\ufffe'   #  0xC7 -> UNDEFINED
    '\u255a'   #  0xC8 -> BOX DRAWINGS DOUBLE UP AND RIGHT
    '\u2554'   #  0xC9 -> BOX DRAWINGS DOUBLE DOWN AND RIGHT
    '\u2569'   #  0xCA -> BOX DRAWINGS DOUBLE UP AND HORIZONTAL
    '\u2566'   #  0xCB -> BOX DRAWINGS DOUBLE DOWN AND HORIZONTAL
    '\u2560'   #  0xCC -> BOX DRAWINGS DOUBLE VERTICAL AND RIGHT
    '\u2550'   #  0xCD -> BOX DRAWINGS DOUBLE HORIZONTAL
    '\u256c'   #  0xCE -> BOX DRAWINGS DOUBLE VERTICAL AND HORIZONTAL
    '\xa4'     #  0xCF -> CURRENCY SIGN
    '\ufffe'   #  0xD0 -> UNDEFINED
    '\ufffe'   #  0xD1 -> UNDEFINED
    '\ufffe'   #  0xD2 -> UNDEFINED
    '\ufffe'   #  0xD3 -> UNDEFINEDS
    '\ufffe'   #  0xD4 -> UNDEFINED
    '\ufffe'   #  0xD5 -> UNDEFINED
    '\ufffe'   #  0xD6 -> UNDEFINEDE
    '\ufffe'   #  0xD7 -> UNDEFINED
    '\ufffe'   #  0xD8 -> UNDEFINED
    '\u2518'   #  0xD9 -> BOX DRAWINGS LIGHT UP AND LEFT
    '\u250c'   #  0xDA -> BOX DRAWINGS LIGHT DOWN AND RIGHT
    '\u2588'   #  0xDB -> FULL BLOCK
    '\u2584'   #  0xDC -> LOWER HALF BLOCK
    '\xa6'     #  0xDD -> BROKEN BAR
    '\ufffe'   #  0xDE -> UNDEFINED
    '\u2580'   #  0xDF -> UPPER HALF BLOCK
    '\ufffe'   #  0xE0 -> UNDEFINED
    '\ufffe'   #  0xE1 -> UNDEFINED
    '\ufffe'   #  0xE2 -> UNDEFINED
    '\ufffe'   #  0xE3 -> UNDEFINED
    '\ufffe'   #  0xE4 -> UNDEFINED
    '\ufffe'   #  0xE5 -> UNDEFINED
    '\xb5'     #  0xE6 -> MICRO SIGN
    '\ufffe'   #  0xE7 -> UNDEFINED
    '\ufffe'   #  0xE8 -> UNDEFINED
    '\ufffe'   #  0xE9 -> UNDEFINED
    '\ufffe'   #  0xEA -> UNDEFINED
    '\ufffe'   #  0xEB -> UNDEFINED
    '\ufffe'   #  0xEC -> UNDEFINED
    '\ufffe'   #  0xED -> UNDEFINED
    '\xaf'     #  0xEE -> MACRON
    '\xb4'     #  0xEF -> ACUTE ACCENT
    '\xad'     #  0xF0 -> SOFT HYPHEN
    '\xb1'     #  0xF1 -> PLUS-MINUS SIGN
    '\u2017'   #  0xF2 -> DOUBLE LOW LINE
    '\xbe'     #  0xF3 -> VULGAR FRACTION THREE QUARTERS
    '\xb6'     #  0xF4 -> PILCROW SIGN
    '\xa7'     #  0xF5 -> SECTION SIGN
    '\xf7'     #  0xF6 -> DIVISION SIGN
    '\xb8'     #  0xF7 -> CEDILLA
    '\xb0'     #  0xF8 -> DEGREE SIGN
    '\xa8'     #  0xF9 -> DIAERESIS
    '\xb7'     #  0xFA -> MIDDLE DOT
    '\xb9'     #  0xFB -> SUPERSCRIPT ONE
    '\xb3'     #  0xFC -> SUPERSCRIPT THREE
    '\xb2'     #  0xFD -> SUPERSCRIPT TWO
    '\u25a0'   #  0xFE -> BLACK SQUARE
    '\xa0'     #  0xFF -> NO-BREAK SPACE
)

### Encoding table
encoding_table=codecs.charmap_build(decoding_table)
