""" Python Character Mapping Codec cp1258 generated from 'MAPPINGS/VENDORS/MICSFT/WINDOWS/CP1258.TXT' with gencodec.py.

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
        name='cp1258',
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
    '\u0152'   #  0x8C -> LATIN CAPITAL LIGATURE OE
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
    '\u0153'   #  0x9C -> LATIN SMALL LIGATURE OE
    '\ufffe'   #  0x9D -> UNDEFINED
    '\ufffe'   #  0x9E -> UNDEFINED
    '\u0178'   #  0x9F -> LATIN CAPITAL LETTER Y WITH DIAERESIS
    '\xa0'     #  0xA0 -> NO-BREAK SPACE
    '\xa1'     #  0xA1 -> INVERTED EXCLAMATION MARK
    '\xa2'     #  0xA2 -> CENT SIGN
    '\xa3'     #  0xA3 -> POUND SIGN
    '\xa4'     #  0xA4 -> CURRENCY SIGN
    '\xa5'     #  0xA5 -> YEN SIGN
    '\xa6'     #  0xA6 -> BROKEN BAR
    '\xa7'     #  0xA7 -> SECTION SIGN
    '\xa8'     #  0xA8 -> DIAERESIS
    '\xa9'     #  0xA9 -> COPYRIGHT SIGN
    '\xaa'     #  0xAA -> FEMININE ORDINAL INDICATOR
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
    '\xba'     #  0xBA -> MASCULINE ORDINAL INDICATOR
    '\xbb'     #  0xBB -> RIGHT-POINTING DOUBLE ANGLE QUOTATION MARK
    '\xbc'     #  0xBC -> VULGAR FRACTION ONE QUARTER
    '\xbd'     #  0xBD -> VULGAR FRACTION ONE HALF
    '\xbe'     #  0xBE -> VULGAR FRACTION THREE QUARTERS
    '\xbf'     #  0xBF -> INVERTED QUESTION MARK
    '\xc0'     #  0xC0 -> LATIN CAPITAL LETTER A WITH GRAVE
    '\xc1'     #  0xC1 -> LATIN CAPITAL LETTER A WITH ACUTE
    '\xc2'     #  0xC2 -> LATIN CAPITAL LETTER A WITH CIRCUMFLEX
    '\u0102'   #  0xC3 -> LATIN CAPITAL LETTER A WITH BREVE
    '\xc4'     #  0xC4 -> LATIN CAPITAL LETTER A WITH DIAERESIS
    '\xc5'     #  0xC5 -> LATIN CAPITAL LETTER A WITH RING ABOVE
    '\xc6'     #  0xC6 -> LATIN CAPITAL LETTER AE
    '\xc7'     #  0xC7 -> LATIN CAPITAL LETTER C WITH CEDILLA
    '\xc8'     #  0xC8 -> LATIN CAPITAL LETTER E WITH GRAVE
    '\xc9'     #  0xC9 -> LATIN CAPITAL LETTER E WITH ACUTE
    '\xca'     #  0xCA -> LATIN CAPITAL LETTER E WITH CIRCUMFLEX
    '\xcb'     #  0xCB -> LATIN CAPITAL LETTER E WITH DIAERESIS
    '\u0300'   #  0xCC -> COMBINING GRAVE ACCENT
    '\xcd'     #  0xCD -> LATIN CAPITAL LETTER I WITH ACUTE
    '\xce'     #  0xCE -> LATIN CAPITAL LETTER I WITH CIRCUMFLEX
    '\xcf'     #  0xCF -> LATIN CAPITAL LETTER I WITH DIAERESIS
    '\u0110'   #  0xD0 -> LATIN CAPITAL LETTER D WITH STROKE
    '\xd1'     #  0xD1 -> LATIN CAPITAL LETTER N WITH TILDE
    '\u0309'   #  0xD2 -> COMBINING HOOK ABOVE
    '\xd3'     #  0xD3 -> LATIN CAPITAL LETTER O WITH ACUTE
    '\xd4'     #  0xD4 -> LATIN CAPITAL LETTER O WITH CIRCUMFLEX
    '\u01a0'   #  0xD5 -> LATIN CAPITAL LETTER O WITH HORN
    '\xd6'     #  0xD6 -> LATIN CAPITAL LETTER O WITH DIAERESIS
    '\xd7'     #  0xD7 -> MULTIPLICATION SIGN
    '\xd8'     #  0xD8 -> LATIN CAPITAL LETTER O WITH STROKE
    '\xd9'     #  0xD9 -> LATIN CAPITAL LETTER U WITH GRAVE
    '\xda'     #  0xDA -> LATIN CAPITAL LETTER U WITH ACUTE
    '\xdb'     #  0xDB -> LATIN CAPITAL LETTER U WITH CIRCUMFLEX
    '\xdc'     #  0xDC -> LATIN CAPITAL LETTER U WITH DIAERESIS
    '\u01af'   #  0xDD -> LATIN CAPITAL LETTER U WITH HORN
    '\u0303'   #  0xDE -> COMBINING TILDE
    '\xdf'     #  0xDF -> LATIN SMALL LETTER SHARP S
    '\xe0'     #  0xE0 -> LATIN SMALL LETTER A WITH GRAVE
    '\xe1'     #  0xE1 -> LATIN SMALL LETTER A WITH ACUTE
    '\xe2'     #  0xE2 -> LATIN SMALL LETTER A WITH CIRCUMFLEX
    '\u0103'   #  0xE3 -> LATIN SMALL LETTER A WITH BREVE
    '\xe4'     #  0xE4 -> LATIN SMALL LETTER A WITH DIAERESIS
    '\xe5'     #  0xE5 -> LATIN SMALL LETTER A WITH RING ABOVE
    '\xe6'     #  0xE6 -> LATIN SMALL LETTER AE
    '\xe7'     #  0xE7 -> LATIN SMALL LETTER C WITH CEDILLA
    '\xe8'     #  0xE8 -> LATIN SMALL LETTER E WITH GRAVE
    '\xe9'     #  0xE9 -> LATIN SMALL LETTER E WITH ACUTE
    '\xea'     #  0xEA -> LATIN SMALL LETTER E WITH CIRCUMFLEX
    '\xeb'     #  0xEB -> LATIN SMALL LETTER E WITH DIAERESIS
    '\u0301'   #  0xEC -> COMBINING ACUTE ACCENT
    '\xed'     #  0xED -> LATIN SMALL LETTER I WITH ACUTE
    '\xee'     #  0xEE -> LATIN SMALL LETTER I WITH CIRCUMFLEX
    '\xef'     #  0xEF -> LATIN SMALL LETTER I WITH DIAERESIS
    '\u0111'   #  0xF0 -> LATIN SMALL LETTER D WITH STROKE
    '\xf1'     #  0xF1 -> LATIN SMALL LETTER N WITH TILDE
    '\u0323'   #  0xF2 -> COMBINING DOT BELOW
    '\xf3'     #  0xF3 -> LATIN SMALL LETTER O WITH ACUTE
    '\xf4'     #  0xF4 -> LATIN SMALL LETTER O WITH CIRCUMFLEX
    '\u01a1'   #  0xF5 -> LATIN SMALL LETTER O WITH HORN
    '\xf6'     #  0xF6 -> LATIN SMALL LETTER O WITH DIAERESIS
    '\xf7'     #  0xF7 -> DIVISION SIGN
    '\xf8'     #  0xF8 -> LATIN SMALL LETTER O WITH STROKE
    '\xf9'     #  0xF9 -> LATIN SMALL LETTER U WITH GRAVE
    '\xfa'     #  0xFA -> LATIN SMALL LETTER U WITH ACUTE
    '\xfb'     #  0xFB -> LATIN SMALL LETTER U WITH CIRCUMFLEX
    '\xfc'     #  0xFC -> LATIN SMALL LETTER U WITH DIAERESIS
    '\u01b0'   #  0xFD -> LATIN SMALL LETTER U WITH HORN
    '\u20ab'   #  0xFE -> DONG SIGN
    '\xff'     #  0xFF -> LATIN SMALL LETTER Y WITH DIAERESIS
)

### Encoding table
encoding_table=codecs.charmap_build(decoding_table)
