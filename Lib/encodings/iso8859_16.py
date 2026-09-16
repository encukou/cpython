""" Python Character Mapping Codec iso8859_16 generated from 'MAPPINGS/ISO8859/8859-16.TXT' with gencodec.py.

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
        name='iso8859-16',
        encode=Codec().encode,
        decode=Codec().decode,
        incrementalencoder=IncrementalEncoder,
        incrementaldecoder=IncrementalDecoder,
        streamreader=StreamReader,
        streamwriter=StreamWriter,
    )


### Decoding Table

decoding_table = (
    ''.join(chr(x) for x in range(0xa1)) +
    '\u0104'   #  0xA1 -> LATIN CAPITAL LETTER A WITH OGONEK
    '\u0105'   #  0xA2 -> LATIN SMALL LETTER A WITH OGONEK
    '\u0141'   #  0xA3 -> LATIN CAPITAL LETTER L WITH STROKE
    '\u20ac'   #  0xA4 -> EURO SIGN
    '\u201e'   #  0xA5 -> DOUBLE LOW-9 QUOTATION MARK
    '\u0160'   #  0xA6 -> LATIN CAPITAL LETTER S WITH CARON
    '\xa7'     #  0xA7 -> SECTION SIGN
    '\u0161'   #  0xA8 -> LATIN SMALL LETTER S WITH CARON
    '\xa9'     #  0xA9 -> COPYRIGHT SIGN
    '\u0218'   #  0xAA -> LATIN CAPITAL LETTER S WITH COMMA BELOW
    '\xab'     #  0xAB -> LEFT-POINTING DOUBLE ANGLE QUOTATION MARK
    '\u0179'   #  0xAC -> LATIN CAPITAL LETTER Z WITH ACUTE
    '\xad'     #  0xAD -> SOFT HYPHEN
    '\u017a'   #  0xAE -> LATIN SMALL LETTER Z WITH ACUTE
    '\u017b'   #  0xAF -> LATIN CAPITAL LETTER Z WITH DOT ABOVE
    '\xb0'     #  0xB0 -> DEGREE SIGN
    '\xb1'     #  0xB1 -> PLUS-MINUS SIGN
    '\u010c'   #  0xB2 -> LATIN CAPITAL LETTER C WITH CARON
    '\u0142'   #  0xB3 -> LATIN SMALL LETTER L WITH STROKE
    '\u017d'   #  0xB4 -> LATIN CAPITAL LETTER Z WITH CARON
    '\u201d'   #  0xB5 -> RIGHT DOUBLE QUOTATION MARK
    '\xb6'     #  0xB6 -> PILCROW SIGN
    '\xb7'     #  0xB7 -> MIDDLE DOT
    '\u017e'   #  0xB8 -> LATIN SMALL LETTER Z WITH CARON
    '\u010d'   #  0xB9 -> LATIN SMALL LETTER C WITH CARON
    '\u0219'   #  0xBA -> LATIN SMALL LETTER S WITH COMMA BELOW
    '\xbb'     #  0xBB -> RIGHT-POINTING DOUBLE ANGLE QUOTATION MARK
    '\u0152'   #  0xBC -> LATIN CAPITAL LIGATURE OE
    '\u0153'   #  0xBD -> LATIN SMALL LIGATURE OE
    '\u0178'   #  0xBE -> LATIN CAPITAL LETTER Y WITH DIAERESIS
    '\u017c'   #  0xBF -> LATIN SMALL LETTER Z WITH DOT ABOVE
    '\xc0'     #  0xC0 -> LATIN CAPITAL LETTER A WITH GRAVE
    '\xc1'     #  0xC1 -> LATIN CAPITAL LETTER A WITH ACUTE
    '\xc2'     #  0xC2 -> LATIN CAPITAL LETTER A WITH CIRCUMFLEX
    '\u0102'   #  0xC3 -> LATIN CAPITAL LETTER A WITH BREVE
    '\xc4'     #  0xC4 -> LATIN CAPITAL LETTER A WITH DIAERESIS
    '\u0106'   #  0xC5 -> LATIN CAPITAL LETTER C WITH ACUTE
    '\xc6'     #  0xC6 -> LATIN CAPITAL LETTER AE
    '\xc7'     #  0xC7 -> LATIN CAPITAL LETTER C WITH CEDILLA
    '\xc8'     #  0xC8 -> LATIN CAPITAL LETTER E WITH GRAVE
    '\xc9'     #  0xC9 -> LATIN CAPITAL LETTER E WITH ACUTE
    '\xca'     #  0xCA -> LATIN CAPITAL LETTER E WITH CIRCUMFLEX
    '\xcb'     #  0xCB -> LATIN CAPITAL LETTER E WITH DIAERESIS
    '\xcc'     #  0xCC -> LATIN CAPITAL LETTER I WITH GRAVE
    '\xcd'     #  0xCD -> LATIN CAPITAL LETTER I WITH ACUTE
    '\xce'     #  0xCE -> LATIN CAPITAL LETTER I WITH CIRCUMFLEX
    '\xcf'     #  0xCF -> LATIN CAPITAL LETTER I WITH DIAERESIS
    '\u0110'   #  0xD0 -> LATIN CAPITAL LETTER D WITH STROKE
    '\u0143'   #  0xD1 -> LATIN CAPITAL LETTER N WITH ACUTE
    '\xd2'     #  0xD2 -> LATIN CAPITAL LETTER O WITH GRAVE
    '\xd3'     #  0xD3 -> LATIN CAPITAL LETTER O WITH ACUTE
    '\xd4'     #  0xD4 -> LATIN CAPITAL LETTER O WITH CIRCUMFLEX
    '\u0150'   #  0xD5 -> LATIN CAPITAL LETTER O WITH DOUBLE ACUTE
    '\xd6'     #  0xD6 -> LATIN CAPITAL LETTER O WITH DIAERESIS
    '\u015a'   #  0xD7 -> LATIN CAPITAL LETTER S WITH ACUTE
    '\u0170'   #  0xD8 -> LATIN CAPITAL LETTER U WITH DOUBLE ACUTE
    '\xd9'     #  0xD9 -> LATIN CAPITAL LETTER U WITH GRAVE
    '\xda'     #  0xDA -> LATIN CAPITAL LETTER U WITH ACUTE
    '\xdb'     #  0xDB -> LATIN CAPITAL LETTER U WITH CIRCUMFLEX
    '\xdc'     #  0xDC -> LATIN CAPITAL LETTER U WITH DIAERESIS
    '\u0118'   #  0xDD -> LATIN CAPITAL LETTER E WITH OGONEK
    '\u021a'   #  0xDE -> LATIN CAPITAL LETTER T WITH COMMA BELOW
    '\xdf'     #  0xDF -> LATIN SMALL LETTER SHARP S
    '\xe0'     #  0xE0 -> LATIN SMALL LETTER A WITH GRAVE
    '\xe1'     #  0xE1 -> LATIN SMALL LETTER A WITH ACUTE
    '\xe2'     #  0xE2 -> LATIN SMALL LETTER A WITH CIRCUMFLEX
    '\u0103'   #  0xE3 -> LATIN SMALL LETTER A WITH BREVE
    '\xe4'     #  0xE4 -> LATIN SMALL LETTER A WITH DIAERESIS
    '\u0107'   #  0xE5 -> LATIN SMALL LETTER C WITH ACUTE
    '\xe6'     #  0xE6 -> LATIN SMALL LETTER AE
    '\xe7'     #  0xE7 -> LATIN SMALL LETTER C WITH CEDILLA
    '\xe8'     #  0xE8 -> LATIN SMALL LETTER E WITH GRAVE
    '\xe9'     #  0xE9 -> LATIN SMALL LETTER E WITH ACUTE
    '\xea'     #  0xEA -> LATIN SMALL LETTER E WITH CIRCUMFLEX
    '\xeb'     #  0xEB -> LATIN SMALL LETTER E WITH DIAERESIS
    '\xec'     #  0xEC -> LATIN SMALL LETTER I WITH GRAVE
    '\xed'     #  0xED -> LATIN SMALL LETTER I WITH ACUTE
    '\xee'     #  0xEE -> LATIN SMALL LETTER I WITH CIRCUMFLEX
    '\xef'     #  0xEF -> LATIN SMALL LETTER I WITH DIAERESIS
    '\u0111'   #  0xF0 -> LATIN SMALL LETTER D WITH STROKE
    '\u0144'   #  0xF1 -> LATIN SMALL LETTER N WITH ACUTE
    '\xf2'     #  0xF2 -> LATIN SMALL LETTER O WITH GRAVE
    '\xf3'     #  0xF3 -> LATIN SMALL LETTER O WITH ACUTE
    '\xf4'     #  0xF4 -> LATIN SMALL LETTER O WITH CIRCUMFLEX
    '\u0151'   #  0xF5 -> LATIN SMALL LETTER O WITH DOUBLE ACUTE
    '\xf6'     #  0xF6 -> LATIN SMALL LETTER O WITH DIAERESIS
    '\u015b'   #  0xF7 -> LATIN SMALL LETTER S WITH ACUTE
    '\u0171'   #  0xF8 -> LATIN SMALL LETTER U WITH DOUBLE ACUTE
    '\xf9'     #  0xF9 -> LATIN SMALL LETTER U WITH GRAVE
    '\xfa'     #  0xFA -> LATIN SMALL LETTER U WITH ACUTE
    '\xfb'     #  0xFB -> LATIN SMALL LETTER U WITH CIRCUMFLEX
    '\xfc'     #  0xFC -> LATIN SMALL LETTER U WITH DIAERESIS
    '\u0119'   #  0xFD -> LATIN SMALL LETTER E WITH OGONEK
    '\u021b'   #  0xFE -> LATIN SMALL LETTER T WITH COMMA BELOW
    '\xff'     #  0xFF -> LATIN SMALL LETTER Y WITH DIAERESIS
)

### Encoding table
encoding_table=codecs.charmap_build(decoding_table)
