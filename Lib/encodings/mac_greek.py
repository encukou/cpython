""" Python Character Mapping Codec mac_greek generated from 'MAPPINGS/VENDORS/APPLE/GREEK.TXT' with gencodec.py.

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
        name='mac-greek',
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
    '\xc4'     #  0x80 -> LATIN CAPITAL LETTER A WITH DIAERESIS
    '\xb9'     #  0x81 -> SUPERSCRIPT ONE
    '\xb2'     #  0x82 -> SUPERSCRIPT TWO
    '\xc9'     #  0x83 -> LATIN CAPITAL LETTER E WITH ACUTE
    '\xb3'     #  0x84 -> SUPERSCRIPT THREE
    '\xd6'     #  0x85 -> LATIN CAPITAL LETTER O WITH DIAERESIS
    '\xdc'     #  0x86 -> LATIN CAPITAL LETTER U WITH DIAERESIS
    '\u0385'   #  0x87 -> GREEK DIALYTIKA TONOS
    '\xe0'     #  0x88 -> LATIN SMALL LETTER A WITH GRAVE
    '\xe2'     #  0x89 -> LATIN SMALL LETTER A WITH CIRCUMFLEX
    '\xe4'     #  0x8A -> LATIN SMALL LETTER A WITH DIAERESIS
    '\u0384'   #  0x8B -> GREEK TONOS
    '\xa8'     #  0x8C -> DIAERESIS
    '\xe7'     #  0x8D -> LATIN SMALL LETTER C WITH CEDILLA
    '\xe9'     #  0x8E -> LATIN SMALL LETTER E WITH ACUTE
    '\xe8'     #  0x8F -> LATIN SMALL LETTER E WITH GRAVE
    '\xea'     #  0x90 -> LATIN SMALL LETTER E WITH CIRCUMFLEX
    '\xeb'     #  0x91 -> LATIN SMALL LETTER E WITH DIAERESIS
    '\xa3'     #  0x92 -> POUND SIGN
    '\u2122'   #  0x93 -> TRADE MARK SIGN
    '\xee'     #  0x94 -> LATIN SMALL LETTER I WITH CIRCUMFLEX
    '\xef'     #  0x95 -> LATIN SMALL LETTER I WITH DIAERESIS
    '\u2022'   #  0x96 -> BULLET
    '\xbd'     #  0x97 -> VULGAR FRACTION ONE HALF
    '\u2030'   #  0x98 -> PER MILLE SIGN
    '\xf4'     #  0x99 -> LATIN SMALL LETTER O WITH CIRCUMFLEX
    '\xf6'     #  0x9A -> LATIN SMALL LETTER O WITH DIAERESIS
    '\xa6'     #  0x9B -> BROKEN BAR
    '\u20ac'   #  0x9C -> EURO SIGN # before Mac OS 9.2.2, was SOFT HYPHEN
    '\xf9'     #  0x9D -> LATIN SMALL LETTER U WITH GRAVE
    '\xfb'     #  0x9E -> LATIN SMALL LETTER U WITH CIRCUMFLEX
    '\xfc'     #  0x9F -> LATIN SMALL LETTER U WITH DIAERESIS
    '\u2020'   #  0xA0 -> DAGGER
    '\u0393'   #  0xA1 -> GREEK CAPITAL LETTER GAMMA
    '\u0394'   #  0xA2 -> GREEK CAPITAL LETTER DELTA
    '\u0398'   #  0xA3 -> GREEK CAPITAL LETTER THETA
    '\u039b'   #  0xA4 -> GREEK CAPITAL LETTER LAMDA
    '\u039e'   #  0xA5 -> GREEK CAPITAL LETTER XI
    '\u03a0'   #  0xA6 -> GREEK CAPITAL LETTER PI
    '\xdf'     #  0xA7 -> LATIN SMALL LETTER SHARP S
    '\xae'     #  0xA8 -> REGISTERED SIGN
    '\xa9'     #  0xA9 -> COPYRIGHT SIGN
    '\u03a3'   #  0xAA -> GREEK CAPITAL LETTER SIGMA
    '\u03aa'   #  0xAB -> GREEK CAPITAL LETTER IOTA WITH DIALYTIKA
    '\xa7'     #  0xAC -> SECTION SIGN
    '\u2260'   #  0xAD -> NOT EQUAL TO
    '\xb0'     #  0xAE -> DEGREE SIGN
    '\xb7'     #  0xAF -> MIDDLE DOT
    '\u0391'   #  0xB0 -> GREEK CAPITAL LETTER ALPHA
    '\xb1'     #  0xB1 -> PLUS-MINUS SIGN
    '\u2264'   #  0xB2 -> LESS-THAN OR EQUAL TO
    '\u2265'   #  0xB3 -> GREATER-THAN OR EQUAL TO
    '\xa5'     #  0xB4 -> YEN SIGN
    '\u0392'   #  0xB5 -> GREEK CAPITAL LETTER BETA
    '\u0395'   #  0xB6 -> GREEK CAPITAL LETTER EPSILON
    '\u0396'   #  0xB7 -> GREEK CAPITAL LETTER ZETA
    '\u0397'   #  0xB8 -> GREEK CAPITAL LETTER ETA
    '\u0399'   #  0xB9 -> GREEK CAPITAL LETTER IOTA
    '\u039a'   #  0xBA -> GREEK CAPITAL LETTER KAPPA
    '\u039c'   #  0xBB -> GREEK CAPITAL LETTER MU
    '\u03a6'   #  0xBC -> GREEK CAPITAL LETTER PHI
    '\u03ab'   #  0xBD -> GREEK CAPITAL LETTER UPSILON WITH DIALYTIKA
    '\u03a8'   #  0xBE -> GREEK CAPITAL LETTER PSI
    '\u03a9'   #  0xBF -> GREEK CAPITAL LETTER OMEGA
    '\u03ac'   #  0xC0 -> GREEK SMALL LETTER ALPHA WITH TONOS
    '\u039d'   #  0xC1 -> GREEK CAPITAL LETTER NU
    '\xac'     #  0xC2 -> NOT SIGN
    '\u039f'   #  0xC3 -> GREEK CAPITAL LETTER OMICRON
    '\u03a1'   #  0xC4 -> GREEK CAPITAL LETTER RHO
    '\u2248'   #  0xC5 -> ALMOST EQUAL TO
    '\u03a4'   #  0xC6 -> GREEK CAPITAL LETTER TAU
    '\xab'     #  0xC7 -> LEFT-POINTING DOUBLE ANGLE QUOTATION MARK
    '\xbb'     #  0xC8 -> RIGHT-POINTING DOUBLE ANGLE QUOTATION MARK
    '\u2026'   #  0xC9 -> HORIZONTAL ELLIPSIS
    '\xa0'     #  0xCA -> NO-BREAK SPACE
    '\u03a5'   #  0xCB -> GREEK CAPITAL LETTER UPSILON
    '\u03a7'   #  0xCC -> GREEK CAPITAL LETTER CHI
    '\u0386'   #  0xCD -> GREEK CAPITAL LETTER ALPHA WITH TONOS
    '\u0388'   #  0xCE -> GREEK CAPITAL LETTER EPSILON WITH TONOS
    '\u0153'   #  0xCF -> LATIN SMALL LIGATURE OE
    '\u2013'   #  0xD0 -> EN DASH
    '\u2015'   #  0xD1 -> HORIZONTAL BAR
    '\u201c'   #  0xD2 -> LEFT DOUBLE QUOTATION MARK
    '\u201d'   #  0xD3 -> RIGHT DOUBLE QUOTATION MARK
    '\u2018'   #  0xD4 -> LEFT SINGLE QUOTATION MARK
    '\u2019'   #  0xD5 -> RIGHT SINGLE QUOTATION MARK
    '\xf7'     #  0xD6 -> DIVISION SIGN
    '\u0389'   #  0xD7 -> GREEK CAPITAL LETTER ETA WITH TONOS
    '\u038a'   #  0xD8 -> GREEK CAPITAL LETTER IOTA WITH TONOS
    '\u038c'   #  0xD9 -> GREEK CAPITAL LETTER OMICRON WITH TONOS
    '\u038e'   #  0xDA -> GREEK CAPITAL LETTER UPSILON WITH TONOS
    '\u03ad'   #  0xDB -> GREEK SMALL LETTER EPSILON WITH TONOS
    '\u03ae'   #  0xDC -> GREEK SMALL LETTER ETA WITH TONOS
    '\u03af'   #  0xDD -> GREEK SMALL LETTER IOTA WITH TONOS
    '\u03cc'   #  0xDE -> GREEK SMALL LETTER OMICRON WITH TONOS
    '\u038f'   #  0xDF -> GREEK CAPITAL LETTER OMEGA WITH TONOS
    '\u03cd'   #  0xE0 -> GREEK SMALL LETTER UPSILON WITH TONOS
    '\u03b1'   #  0xE1 -> GREEK SMALL LETTER ALPHA
    '\u03b2'   #  0xE2 -> GREEK SMALL LETTER BETA
    '\u03c8'   #  0xE3 -> GREEK SMALL LETTER PSI
    '\u03b4'   #  0xE4 -> GREEK SMALL LETTER DELTA
    '\u03b5'   #  0xE5 -> GREEK SMALL LETTER EPSILON
    '\u03c6'   #  0xE6 -> GREEK SMALL LETTER PHI
    '\u03b3'   #  0xE7 -> GREEK SMALL LETTER GAMMA
    '\u03b7'   #  0xE8 -> GREEK SMALL LETTER ETA
    '\u03b9'   #  0xE9 -> GREEK SMALL LETTER IOTA
    '\u03be'   #  0xEA -> GREEK SMALL LETTER XI
    '\u03ba'   #  0xEB -> GREEK SMALL LETTER KAPPA
    '\u03bb'   #  0xEC -> GREEK SMALL LETTER LAMDA
    '\u03bc'   #  0xED -> GREEK SMALL LETTER MU
    '\u03bd'   #  0xEE -> GREEK SMALL LETTER NU
    '\u03bf'   #  0xEF -> GREEK SMALL LETTER OMICRON
    '\u03c0'   #  0xF0 -> GREEK SMALL LETTER PI
    '\u03ce'   #  0xF1 -> GREEK SMALL LETTER OMEGA WITH TONOS
    '\u03c1'   #  0xF2 -> GREEK SMALL LETTER RHO
    '\u03c3'   #  0xF3 -> GREEK SMALL LETTER SIGMA
    '\u03c4'   #  0xF4 -> GREEK SMALL LETTER TAU
    '\u03b8'   #  0xF5 -> GREEK SMALL LETTER THETA
    '\u03c9'   #  0xF6 -> GREEK SMALL LETTER OMEGA
    '\u03c2'   #  0xF7 -> GREEK SMALL LETTER FINAL SIGMA
    '\u03c7'   #  0xF8 -> GREEK SMALL LETTER CHI
    '\u03c5'   #  0xF9 -> GREEK SMALL LETTER UPSILON
    '\u03b6'   #  0xFA -> GREEK SMALL LETTER ZETA
    '\u03ca'   #  0xFB -> GREEK SMALL LETTER IOTA WITH DIALYTIKA
    '\u03cb'   #  0xFC -> GREEK SMALL LETTER UPSILON WITH DIALYTIKA
    '\u0390'   #  0xFD -> GREEK SMALL LETTER IOTA WITH DIALYTIKA AND TONOS
    '\u03b0'   #  0xFE -> GREEK SMALL LETTER UPSILON WITH DIALYTIKA AND TONOS
    '\xad'     #  0xFF -> SOFT HYPHEN # before Mac OS 9.2.2, was undefined
)

### Encoding table
encoding_table=codecs.charmap_build(decoding_table)
