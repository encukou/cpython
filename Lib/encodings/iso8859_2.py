""" Python Character Mapping Codec iso8859_2 generated from 'MAPPINGS/ISO8859/8859-2.TXT' with gencodec.py.

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
        name='iso8859-2',
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
    '\u02d8'   #  0xA2 -> BREVE
    '\u0141'   #  0xA3 -> LATIN CAPITAL LETTER L WITH STROKE
    '\xa4'     #  0xA4 -> CURRENCY SIGN
    '\u013d'   #  0xA5 -> LATIN CAPITAL LETTER L WITH CARON
    '\u015a'   #  0xA6 -> LATIN CAPITAL LETTER S WITH ACUTE
    '\xa7'     #  0xA7 -> SECTION SIGN
    '\xa8'     #  0xA8 -> DIAERESIS
    '\u0160'   #  0xA9 -> LATIN CAPITAL LETTER S WITH CARON
    '\u015e'   #  0xAA -> LATIN CAPITAL LETTER S WITH CEDILLA
    '\u0164'   #  0xAB -> LATIN CAPITAL LETTER T WITH CARON
    '\u0179'   #  0xAC -> LATIN CAPITAL LETTER Z WITH ACUTE
    '\xad'     #  0xAD -> SOFT HYPHEN
    '\u017d'   #  0xAE -> LATIN CAPITAL LETTER Z WITH CARON
    '\u017b'   #  0xAF -> LATIN CAPITAL LETTER Z WITH DOT ABOVE
    '\xb0'     #  0xB0 -> DEGREE SIGN
    '\u0105'   #  0xB1 -> LATIN SMALL LETTER A WITH OGONEK
    '\u02db'   #  0xB2 -> OGONEK
    '\u0142'   #  0xB3 -> LATIN SMALL LETTER L WITH STROKE
    '\xb4'     #  0xB4 -> ACUTE ACCENT
    '\u013e'   #  0xB5 -> LATIN SMALL LETTER L WITH CARON
    '\u015b'   #  0xB6 -> LATIN SMALL LETTER S WITH ACUTE
    '\u02c7'   #  0xB7 -> CARON
    '\xb8'     #  0xB8 -> CEDILLA
    '\u0161'   #  0xB9 -> LATIN SMALL LETTER S WITH CARON
    '\u015f'   #  0xBA -> LATIN SMALL LETTER S WITH CEDILLA
    '\u0165'   #  0xBB -> LATIN SMALL LETTER T WITH CARON
    '\u017a'   #  0xBC -> LATIN SMALL LETTER Z WITH ACUTE
    '\u02dd'   #  0xBD -> DOUBLE ACUTE ACCENT
    '\u017e'   #  0xBE -> LATIN SMALL LETTER Z WITH CARON
    '\u017c'   #  0xBF -> LATIN SMALL LETTER Z WITH DOT ABOVE
    '\u0154'   #  0xC0 -> LATIN CAPITAL LETTER R WITH ACUTE
    '\xc1'     #  0xC1 -> LATIN CAPITAL LETTER A WITH ACUTE
    '\xc2'     #  0xC2 -> LATIN CAPITAL LETTER A WITH CIRCUMFLEX
    '\u0102'   #  0xC3 -> LATIN CAPITAL LETTER A WITH BREVE
    '\xc4'     #  0xC4 -> LATIN CAPITAL LETTER A WITH DIAERESIS
    '\u0139'   #  0xC5 -> LATIN CAPITAL LETTER L WITH ACUTE
    '\u0106'   #  0xC6 -> LATIN CAPITAL LETTER C WITH ACUTE
    '\xc7'     #  0xC7 -> LATIN CAPITAL LETTER C WITH CEDILLA
    '\u010c'   #  0xC8 -> LATIN CAPITAL LETTER C WITH CARON
    '\xc9'     #  0xC9 -> LATIN CAPITAL LETTER E WITH ACUTE
    '\u0118'   #  0xCA -> LATIN CAPITAL LETTER E WITH OGONEK
    '\xcb'     #  0xCB -> LATIN CAPITAL LETTER E WITH DIAERESIS
    '\u011a'   #  0xCC -> LATIN CAPITAL LETTER E WITH CARON
    '\xcd'     #  0xCD -> LATIN CAPITAL LETTER I WITH ACUTE
    '\xce'     #  0xCE -> LATIN CAPITAL LETTER I WITH CIRCUMFLEX
    '\u010e'   #  0xCF -> LATIN CAPITAL LETTER D WITH CARON
    '\u0110'   #  0xD0 -> LATIN CAPITAL LETTER D WITH STROKE
    '\u0143'   #  0xD1 -> LATIN CAPITAL LETTER N WITH ACUTE
    '\u0147'   #  0xD2 -> LATIN CAPITAL LETTER N WITH CARON
    '\xd3'     #  0xD3 -> LATIN CAPITAL LETTER O WITH ACUTE
    '\xd4'     #  0xD4 -> LATIN CAPITAL LETTER O WITH CIRCUMFLEX
    '\u0150'   #  0xD5 -> LATIN CAPITAL LETTER O WITH DOUBLE ACUTE
    '\xd6'     #  0xD6 -> LATIN CAPITAL LETTER O WITH DIAERESIS
    '\xd7'     #  0xD7 -> MULTIPLICATION SIGN
    '\u0158'   #  0xD8 -> LATIN CAPITAL LETTER R WITH CARON
    '\u016e'   #  0xD9 -> LATIN CAPITAL LETTER U WITH RING ABOVE
    '\xda'     #  0xDA -> LATIN CAPITAL LETTER U WITH ACUTE
    '\u0170'   #  0xDB -> LATIN CAPITAL LETTER U WITH DOUBLE ACUTE
    '\xdc'     #  0xDC -> LATIN CAPITAL LETTER U WITH DIAERESIS
    '\xdd'     #  0xDD -> LATIN CAPITAL LETTER Y WITH ACUTE
    '\u0162'   #  0xDE -> LATIN CAPITAL LETTER T WITH CEDILLA
    '\xdf'     #  0xDF -> LATIN SMALL LETTER SHARP S
    '\u0155'   #  0xE0 -> LATIN SMALL LETTER R WITH ACUTE
    '\xe1'     #  0xE1 -> LATIN SMALL LETTER A WITH ACUTE
    '\xe2'     #  0xE2 -> LATIN SMALL LETTER A WITH CIRCUMFLEX
    '\u0103'   #  0xE3 -> LATIN SMALL LETTER A WITH BREVE
    '\xe4'     #  0xE4 -> LATIN SMALL LETTER A WITH DIAERESIS
    '\u013a'   #  0xE5 -> LATIN SMALL LETTER L WITH ACUTE
    '\u0107'   #  0xE6 -> LATIN SMALL LETTER C WITH ACUTE
    '\xe7'     #  0xE7 -> LATIN SMALL LETTER C WITH CEDILLA
    '\u010d'   #  0xE8 -> LATIN SMALL LETTER C WITH CARON
    '\xe9'     #  0xE9 -> LATIN SMALL LETTER E WITH ACUTE
    '\u0119'   #  0xEA -> LATIN SMALL LETTER E WITH OGONEK
    '\xeb'     #  0xEB -> LATIN SMALL LETTER E WITH DIAERESIS
    '\u011b'   #  0xEC -> LATIN SMALL LETTER E WITH CARON
    '\xed'     #  0xED -> LATIN SMALL LETTER I WITH ACUTE
    '\xee'     #  0xEE -> LATIN SMALL LETTER I WITH CIRCUMFLEX
    '\u010f'   #  0xEF -> LATIN SMALL LETTER D WITH CARON
    '\u0111'   #  0xF0 -> LATIN SMALL LETTER D WITH STROKE
    '\u0144'   #  0xF1 -> LATIN SMALL LETTER N WITH ACUTE
    '\u0148'   #  0xF2 -> LATIN SMALL LETTER N WITH CARON
    '\xf3'     #  0xF3 -> LATIN SMALL LETTER O WITH ACUTE
    '\xf4'     #  0xF4 -> LATIN SMALL LETTER O WITH CIRCUMFLEX
    '\u0151'   #  0xF5 -> LATIN SMALL LETTER O WITH DOUBLE ACUTE
    '\xf6'     #  0xF6 -> LATIN SMALL LETTER O WITH DIAERESIS
    '\xf7'     #  0xF7 -> DIVISION SIGN
    '\u0159'   #  0xF8 -> LATIN SMALL LETTER R WITH CARON
    '\u016f'   #  0xF9 -> LATIN SMALL LETTER U WITH RING ABOVE
    '\xfa'     #  0xFA -> LATIN SMALL LETTER U WITH ACUTE
    '\u0171'   #  0xFB -> LATIN SMALL LETTER U WITH DOUBLE ACUTE
    '\xfc'     #  0xFC -> LATIN SMALL LETTER U WITH DIAERESIS
    '\xfd'     #  0xFD -> LATIN SMALL LETTER Y WITH ACUTE
    '\u0163'   #  0xFE -> LATIN SMALL LETTER T WITH CEDILLA
    '\u02d9'   #  0xFF -> DOT ABOVE
)

### Encoding table
encoding_table=codecs.charmap_build(decoding_table)
