""" Python Character Mapping Codec cp1253 generated from 'MAPPINGS/VENDORS/MICSFT/WINDOWS/CP1253.TXT' with gencodec.py.

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
        name='cp1253',
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
    '\ufffe'   #  0x88 -> UNDEFINED
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
    '\ufffe'   #  0x98 -> UNDEFINED
    '\u2122'   #  0x99 -> TRADE MARK SIGN
    '\ufffe'   #  0x9A -> UNDEFINED
    '\u203a'   #  0x9B -> SINGLE RIGHT-POINTING ANGLE QUOTATION MARK
    '\ufffe'   #  0x9C -> UNDEFINED
    '\ufffe'   #  0x9D -> UNDEFINED
    '\ufffe'   #  0x9E -> UNDEFINED
    '\ufffe'   #  0x9F -> UNDEFINED
    '\xa0'     #  0xA0 -> NO-BREAK SPACE
    '\u0385'   #  0xA1 -> GREEK DIALYTIKA TONOS
    '\u0386'   #  0xA2 -> GREEK CAPITAL LETTER ALPHA WITH TONOS
    '\xa3'     #  0xA3 -> POUND SIGN
    '\xa4'     #  0xA4 -> CURRENCY SIGN
    '\xa5'     #  0xA5 -> YEN SIGN
    '\xa6'     #  0xA6 -> BROKEN BAR
    '\xa7'     #  0xA7 -> SECTION SIGN
    '\xa8'     #  0xA8 -> DIAERESIS
    '\xa9'     #  0xA9 -> COPYRIGHT SIGN
    '\ufffe'   #  0xAA -> UNDEFINED
    '\xab'     #  0xAB -> LEFT-POINTING DOUBLE ANGLE QUOTATION MARK
    '\xac'     #  0xAC -> NOT SIGN
    '\xad'     #  0xAD -> SOFT HYPHEN
    '\xae'     #  0xAE -> REGISTERED SIGN
    '\u2015'   #  0xAF -> HORIZONTAL BAR
    '\xb0'     #  0xB0 -> DEGREE SIGN
    '\xb1'     #  0xB1 -> PLUS-MINUS SIGN
    '\xb2'     #  0xB2 -> SUPERSCRIPT TWO
    '\xb3'     #  0xB3 -> SUPERSCRIPT THREE
    '\u0384'   #  0xB4 -> GREEK TONOS
    '\xb5'     #  0xB5 -> MICRO SIGN
    '\xb6'     #  0xB6 -> PILCROW SIGN
    '\xb7'     #  0xB7 -> MIDDLE DOT
    '\u0388'   #  0xB8 -> GREEK CAPITAL LETTER EPSILON WITH TONOS
    '\u0389'   #  0xB9 -> GREEK CAPITAL LETTER ETA WITH TONOS
    '\u038a'   #  0xBA -> GREEK CAPITAL LETTER IOTA WITH TONOS
    '\xbb'     #  0xBB -> RIGHT-POINTING DOUBLE ANGLE QUOTATION MARK
    '\u038c'   #  0xBC -> GREEK CAPITAL LETTER OMICRON WITH TONOS
    '\xbd'     #  0xBD -> VULGAR FRACTION ONE HALF
    '\u038e'   #  0xBE -> GREEK CAPITAL LETTER UPSILON WITH TONOS
    '\u038f'   #  0xBF -> GREEK CAPITAL LETTER OMEGA WITH TONOS
    '\u0390'   #  0xC0 -> GREEK SMALL LETTER IOTA WITH DIALYTIKA AND TONOS
    '\u0391'   #  0xC1 -> GREEK CAPITAL LETTER ALPHA
    '\u0392'   #  0xC2 -> GREEK CAPITAL LETTER BETA
    '\u0393'   #  0xC3 -> GREEK CAPITAL LETTER GAMMA
    '\u0394'   #  0xC4 -> GREEK CAPITAL LETTER DELTA
    '\u0395'   #  0xC5 -> GREEK CAPITAL LETTER EPSILON
    '\u0396'   #  0xC6 -> GREEK CAPITAL LETTER ZETA
    '\u0397'   #  0xC7 -> GREEK CAPITAL LETTER ETA
    '\u0398'   #  0xC8 -> GREEK CAPITAL LETTER THETA
    '\u0399'   #  0xC9 -> GREEK CAPITAL LETTER IOTA
    '\u039a'   #  0xCA -> GREEK CAPITAL LETTER KAPPA
    '\u039b'   #  0xCB -> GREEK CAPITAL LETTER LAMDA
    '\u039c'   #  0xCC -> GREEK CAPITAL LETTER MU
    '\u039d'   #  0xCD -> GREEK CAPITAL LETTER NU
    '\u039e'   #  0xCE -> GREEK CAPITAL LETTER XI
    '\u039f'   #  0xCF -> GREEK CAPITAL LETTER OMICRON
    '\u03a0'   #  0xD0 -> GREEK CAPITAL LETTER PI
    '\u03a1'   #  0xD1 -> GREEK CAPITAL LETTER RHO
    '\ufffe'   #  0xD2 -> UNDEFINED
    '\u03a3'   #  0xD3 -> GREEK CAPITAL LETTER SIGMA
    '\u03a4'   #  0xD4 -> GREEK CAPITAL LETTER TAU
    '\u03a5'   #  0xD5 -> GREEK CAPITAL LETTER UPSILON
    '\u03a6'   #  0xD6 -> GREEK CAPITAL LETTER PHI
    '\u03a7'   #  0xD7 -> GREEK CAPITAL LETTER CHI
    '\u03a8'   #  0xD8 -> GREEK CAPITAL LETTER PSI
    '\u03a9'   #  0xD9 -> GREEK CAPITAL LETTER OMEGA
    '\u03aa'   #  0xDA -> GREEK CAPITAL LETTER IOTA WITH DIALYTIKA
    '\u03ab'   #  0xDB -> GREEK CAPITAL LETTER UPSILON WITH DIALYTIKA
    '\u03ac'   #  0xDC -> GREEK SMALL LETTER ALPHA WITH TONOS
    '\u03ad'   #  0xDD -> GREEK SMALL LETTER EPSILON WITH TONOS
    '\u03ae'   #  0xDE -> GREEK SMALL LETTER ETA WITH TONOS
    '\u03af'   #  0xDF -> GREEK SMALL LETTER IOTA WITH TONOS
    '\u03b0'   #  0xE0 -> GREEK SMALL LETTER UPSILON WITH DIALYTIKA AND TONOS
    '\u03b1'   #  0xE1 -> GREEK SMALL LETTER ALPHA
    '\u03b2'   #  0xE2 -> GREEK SMALL LETTER BETA
    '\u03b3'   #  0xE3 -> GREEK SMALL LETTER GAMMA
    '\u03b4'   #  0xE4 -> GREEK SMALL LETTER DELTA
    '\u03b5'   #  0xE5 -> GREEK SMALL LETTER EPSILON
    '\u03b6'   #  0xE6 -> GREEK SMALL LETTER ZETA
    '\u03b7'   #  0xE7 -> GREEK SMALL LETTER ETA
    '\u03b8'   #  0xE8 -> GREEK SMALL LETTER THETA
    '\u03b9'   #  0xE9 -> GREEK SMALL LETTER IOTA
    '\u03ba'   #  0xEA -> GREEK SMALL LETTER KAPPA
    '\u03bb'   #  0xEB -> GREEK SMALL LETTER LAMDA
    '\u03bc'   #  0xEC -> GREEK SMALL LETTER MU
    '\u03bd'   #  0xED -> GREEK SMALL LETTER NU
    '\u03be'   #  0xEE -> GREEK SMALL LETTER XI
    '\u03bf'   #  0xEF -> GREEK SMALL LETTER OMICRON
    '\u03c0'   #  0xF0 -> GREEK SMALL LETTER PI
    '\u03c1'   #  0xF1 -> GREEK SMALL LETTER RHO
    '\u03c2'   #  0xF2 -> GREEK SMALL LETTER FINAL SIGMA
    '\u03c3'   #  0xF3 -> GREEK SMALL LETTER SIGMA
    '\u03c4'   #  0xF4 -> GREEK SMALL LETTER TAU
    '\u03c5'   #  0xF5 -> GREEK SMALL LETTER UPSILON
    '\u03c6'   #  0xF6 -> GREEK SMALL LETTER PHI
    '\u03c7'   #  0xF7 -> GREEK SMALL LETTER CHI
    '\u03c8'   #  0xF8 -> GREEK SMALL LETTER PSI
    '\u03c9'   #  0xF9 -> GREEK SMALL LETTER OMEGA
    '\u03ca'   #  0xFA -> GREEK SMALL LETTER IOTA WITH DIALYTIKA
    '\u03cb'   #  0xFB -> GREEK SMALL LETTER UPSILON WITH DIALYTIKA
    '\u03cc'   #  0xFC -> GREEK SMALL LETTER OMICRON WITH TONOS
    '\u03cd'   #  0xFD -> GREEK SMALL LETTER UPSILON WITH TONOS
    '\u03ce'   #  0xFE -> GREEK SMALL LETTER OMEGA WITH TONOS
    '\ufffe'   #  0xFF -> UNDEFINED
)

### Encoding table
encoding_table=codecs.charmap_build(decoding_table)
