""" Python Character Mapping Codec cp1255 generated from 'MAPPINGS/VENDORS/MICSFT/WINDOWS/CP1255.TXT' with gencodec.py.

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
        name='cp1255',
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
    '\u20ac'   #  0x80 -> EURO SIGN
    '\ufffe'   #  0x81 -> UNDEFINED
    '\u201a'   #  0x82 -> SINGLE LOW-9 QUOTATION MARK
    '\u0192'   #  0x83 -> LATIN SMALL LETTER F WITH HOOK
    '\u201e'   #  0x84 -> DOUBLE LOW-9 QUOTATION MARK
    '\u2026'   #  0x85 -> HORIZONTAL ELLIPSIS
    '\u2020'   #  0x86 -> DAGGER
    '\u2021'   #  0x87 -> DOUBLE DAGGER
    '\u02c6'   #  0x88 -> MODIFIER LETTER CIRCUMFLEX ACCENT
    '\u2030'   #  0x89 -> PER MILLE SIGN
    '\ufffe'   #  0x8A -> UNDEFINED
    '\u2039'   #  0x8B -> SINGLE LEFT-POINTING ANGLE QUOTATION MARK
    '\ufffe'   #  0x8C -> UNDEFINED
    '\ufffe'   #  0x8D -> UNDEFINED
    '\ufffe'   #  0x8E -> UNDEFINED
    '\ufffe'   #  0x8F -> UNDEFINED
    '\ufffe'   #  0x90 -> UNDEFINED
    '\u2018'   #  0x91 -> LEFT SINGLE QUOTATION MARK
    '\u2019'   #  0x92 -> RIGHT SINGLE QUOTATION MARK
    '\u201c'   #  0x93 -> LEFT DOUBLE QUOTATION MARK
    '\u201d'   #  0x94 -> RIGHT DOUBLE QUOTATION MARK
    '\u2022'   #  0x95 -> BULLET
    '\u2013'   #  0x96 -> EN DASH
    '\u2014'   #  0x97 -> EM DASH
    '\u02dc'   #  0x98 -> SMALL TILDE
    '\u2122'   #  0x99 -> TRADE MARK SIGN
    '\ufffe'   #  0x9A -> UNDEFINED
    '\u203a'   #  0x9B -> SINGLE RIGHT-POINTING ANGLE QUOTATION MARK
    '\ufffe'   #  0x9C -> UNDEFINED
    '\ufffe'   #  0x9D -> UNDEFINED
    '\ufffe'   #  0x9E -> UNDEFINED
    '\ufffe'   #  0x9F -> UNDEFINED
    '\xa0'     #  0xA0 -> NO-BREAK SPACE
    '\xa1'     #  0xA1 -> INVERTED EXCLAMATION MARK
    '\xa2'     #  0xA2 -> CENT SIGN
    '\xa3'     #  0xA3 -> POUND SIGN
    '\u20aa'   #  0xA4 -> NEW SHEQEL SIGN
    '\xa5'     #  0xA5 -> YEN SIGN
    '\xa6'     #  0xA6 -> BROKEN BAR
    '\xa7'     #  0xA7 -> SECTION SIGN
    '\xa8'     #  0xA8 -> DIAERESIS
    '\xa9'     #  0xA9 -> COPYRIGHT SIGN
    '\xd7'     #  0xAA -> MULTIPLICATION SIGN
    '\xab'     #  0xAB -> LEFT-POINTING DOUBLE ANGLE QUOTATION MARK
    '\xac'     #  0xAC -> NOT SIGN
    '\xad'     #  0xAD -> SOFT HYPHEN
    '\xae'     #  0xAE -> REGISTERED SIGN
    '\xaf'     #  0xAF -> MACRON
    '\xb0'     #  0xB0 -> DEGREE SIGN
    '\xb1'     #  0xB1 -> PLUS-MINUS SIGN
    '\xb2'     #  0xB2 -> SUPERSCRIPT TWO
    '\xb3'     #  0xB3 -> SUPERSCRIPT THREE
    '\xb4'     #  0xB4 -> ACUTE ACCENT
    '\xb5'     #  0xB5 -> MICRO SIGN
    '\xb6'     #  0xB6 -> PILCROW SIGN
    '\xb7'     #  0xB7 -> MIDDLE DOT
    '\xb8'     #  0xB8 -> CEDILLA
    '\xb9'     #  0xB9 -> SUPERSCRIPT ONE
    '\xf7'     #  0xBA -> DIVISION SIGN
    '\xbb'     #  0xBB -> RIGHT-POINTING DOUBLE ANGLE QUOTATION MARK
    '\xbc'     #  0xBC -> VULGAR FRACTION ONE QUARTER
    '\xbd'     #  0xBD -> VULGAR FRACTION ONE HALF
    '\xbe'     #  0xBE -> VULGAR FRACTION THREE QUARTERS
    '\xbf'     #  0xBF -> INVERTED QUESTION MARK
    '\u05b0'   #  0xC0 -> HEBREW POINT SHEVA
    '\u05b1'   #  0xC1 -> HEBREW POINT HATAF SEGOL
    '\u05b2'   #  0xC2 -> HEBREW POINT HATAF PATAH
    '\u05b3'   #  0xC3 -> HEBREW POINT HATAF QAMATS
    '\u05b4'   #  0xC4 -> HEBREW POINT HIRIQ
    '\u05b5'   #  0xC5 -> HEBREW POINT TSERE
    '\u05b6'   #  0xC6 -> HEBREW POINT SEGOL
    '\u05b7'   #  0xC7 -> HEBREW POINT PATAH
    '\u05b8'   #  0xC8 -> HEBREW POINT QAMATS
    '\u05b9'   #  0xC9 -> HEBREW POINT HOLAM
    '\ufffe'   #  0xCA -> UNDEFINED
    '\u05bb'   #  0xCB -> HEBREW POINT QUBUTS
    '\u05bc'   #  0xCC -> HEBREW POINT DAGESH OR MAPIQ
    '\u05bd'   #  0xCD -> HEBREW POINT METEG
    '\u05be'   #  0xCE -> HEBREW PUNCTUATION MAQAF
    '\u05bf'   #  0xCF -> HEBREW POINT RAFE
    '\u05c0'   #  0xD0 -> HEBREW PUNCTUATION PASEQ
    '\u05c1'   #  0xD1 -> HEBREW POINT SHIN DOT
    '\u05c2'   #  0xD2 -> HEBREW POINT SIN DOT
    '\u05c3'   #  0xD3 -> HEBREW PUNCTUATION SOF PASUQ
    '\u05f0'   #  0xD4 -> HEBREW LIGATURE YIDDISH DOUBLE VAV
    '\u05f1'   #  0xD5 -> HEBREW LIGATURE YIDDISH VAV YOD
    '\u05f2'   #  0xD6 -> HEBREW LIGATURE YIDDISH DOUBLE YOD
    '\u05f3'   #  0xD7 -> HEBREW PUNCTUATION GERESH
    '\u05f4'   #  0xD8 -> HEBREW PUNCTUATION GERSHAYIM
    '\ufffe'   #  0xD9 -> UNDEFINED
    '\ufffe'   #  0xDA -> UNDEFINED
    '\ufffe'   #  0xDB -> UNDEFINED
    '\ufffe'   #  0xDC -> UNDEFINED
    '\ufffe'   #  0xDD -> UNDEFINED
    '\ufffe'   #  0xDE -> UNDEFINED
    '\ufffe'   #  0xDF -> UNDEFINED
    '\u05d0'   #  0xE0 -> HEBREW LETTER ALEF
    '\u05d1'   #  0xE1 -> HEBREW LETTER BET
    '\u05d2'   #  0xE2 -> HEBREW LETTER GIMEL
    '\u05d3'   #  0xE3 -> HEBREW LETTER DALET
    '\u05d4'   #  0xE4 -> HEBREW LETTER HE
    '\u05d5'   #  0xE5 -> HEBREW LETTER VAV
    '\u05d6'   #  0xE6 -> HEBREW LETTER ZAYIN
    '\u05d7'   #  0xE7 -> HEBREW LETTER HET
    '\u05d8'   #  0xE8 -> HEBREW LETTER TET
    '\u05d9'   #  0xE9 -> HEBREW LETTER YOD
    '\u05da'   #  0xEA -> HEBREW LETTER FINAL KAF
    '\u05db'   #  0xEB -> HEBREW LETTER KAF
    '\u05dc'   #  0xEC -> HEBREW LETTER LAMED
    '\u05dd'   #  0xED -> HEBREW LETTER FINAL MEM
    '\u05de'   #  0xEE -> HEBREW LETTER MEM
    '\u05df'   #  0xEF -> HEBREW LETTER FINAL NUN
    '\u05e0'   #  0xF0 -> HEBREW LETTER NUN
    '\u05e1'   #  0xF1 -> HEBREW LETTER SAMEKH
    '\u05e2'   #  0xF2 -> HEBREW LETTER AYIN
    '\u05e3'   #  0xF3 -> HEBREW LETTER FINAL PE
    '\u05e4'   #  0xF4 -> HEBREW LETTER PE
    '\u05e5'   #  0xF5 -> HEBREW LETTER FINAL TSADI
    '\u05e6'   #  0xF6 -> HEBREW LETTER TSADI
    '\u05e7'   #  0xF7 -> HEBREW LETTER QOF
    '\u05e8'   #  0xF8 -> HEBREW LETTER RESH
    '\u05e9'   #  0xF9 -> HEBREW LETTER SHIN
    '\u05ea'   #  0xFA -> HEBREW LETTER TAV
    '\ufffe'   #  0xFB -> UNDEFINED
    '\ufffe'   #  0xFC -> UNDEFINED
    '\u200e'   #  0xFD -> LEFT-TO-RIGHT MARK
    '\u200f'   #  0xFE -> RIGHT-TO-LEFT MARK
    '\ufffe'   #  0xFF -> UNDEFINED
)

### Encoding table
encoding_table=codecs.charmap_build(decoding_table)
