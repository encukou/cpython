""" Python Character Mapping Codec iso8859_4 generated from 'MAPPINGS/ISO8859/8859-4.TXT' with gencodec.py.

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
        name='iso8859-4',
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
    '\u0138'   #  0xA2 -> LATIN SMALL LETTER KRA
    '\u0156'   #  0xA3 -> LATIN CAPITAL LETTER R WITH CEDILLA
    '\xa4'     #  0xA4 -> CURRENCY SIGN
    '\u0128'   #  0xA5 -> LATIN CAPITAL LETTER I WITH TILDE
    '\u013b'   #  0xA6 -> LATIN CAPITAL LETTER L WITH CEDILLA
    '\xa7'     #  0xA7 -> SECTION SIGN
    '\xa8'     #  0xA8 -> DIAERESIS
    '\u0160'   #  0xA9 -> LATIN CAPITAL LETTER S WITH CARON
    '\u0112'   #  0xAA -> LATIN CAPITAL LETTER E WITH MACRON
    '\u0122'   #  0xAB -> LATIN CAPITAL LETTER G WITH CEDILLA
    '\u0166'   #  0xAC -> LATIN CAPITAL LETTER T WITH STROKE
    '\xad'     #  0xAD -> SOFT HYPHEN
    '\u017d'   #  0xAE -> LATIN CAPITAL LETTER Z WITH CARON
    '\xaf'     #  0xAF -> MACRON
    '\xb0'     #  0xB0 -> DEGREE SIGN
    '\u0105'   #  0xB1 -> LATIN SMALL LETTER A WITH OGONEK
    '\u02db'   #  0xB2 -> OGONEK
    '\u0157'   #  0xB3 -> LATIN SMALL LETTER R WITH CEDILLA
    '\xb4'     #  0xB4 -> ACUTE ACCENT
    '\u0129'   #  0xB5 -> LATIN SMALL LETTER I WITH TILDE
    '\u013c'   #  0xB6 -> LATIN SMALL LETTER L WITH CEDILLA
    '\u02c7'   #  0xB7 -> CARON
    '\xb8'     #  0xB8 -> CEDILLA
    '\u0161'   #  0xB9 -> LATIN SMALL LETTER S WITH CARON
    '\u0113'   #  0xBA -> LATIN SMALL LETTER E WITH MACRON
    '\u0123'   #  0xBB -> LATIN SMALL LETTER G WITH CEDILLA
    '\u0167'   #  0xBC -> LATIN SMALL LETTER T WITH STROKE
    '\u014a'   #  0xBD -> LATIN CAPITAL LETTER ENG
    '\u017e'   #  0xBE -> LATIN SMALL LETTER Z WITH CARON
    '\u014b'   #  0xBF -> LATIN SMALL LETTER ENG
    '\u0100'   #  0xC0 -> LATIN CAPITAL LETTER A WITH MACRON
    '\xc1'     #  0xC1 -> LATIN CAPITAL LETTER A WITH ACUTE
    '\xc2'     #  0xC2 -> LATIN CAPITAL LETTER A WITH CIRCUMFLEX
    '\xc3'     #  0xC3 -> LATIN CAPITAL LETTER A WITH TILDE
    '\xc4'     #  0xC4 -> LATIN CAPITAL LETTER A WITH DIAERESIS
    '\xc5'     #  0xC5 -> LATIN CAPITAL LETTER A WITH RING ABOVE
    '\xc6'     #  0xC6 -> LATIN CAPITAL LETTER AE
    '\u012e'   #  0xC7 -> LATIN CAPITAL LETTER I WITH OGONEK
    '\u010c'   #  0xC8 -> LATIN CAPITAL LETTER C WITH CARON
    '\xc9'     #  0xC9 -> LATIN CAPITAL LETTER E WITH ACUTE
    '\u0118'   #  0xCA -> LATIN CAPITAL LETTER E WITH OGONEK
    '\xcb'     #  0xCB -> LATIN CAPITAL LETTER E WITH DIAERESIS
    '\u0116'   #  0xCC -> LATIN CAPITAL LETTER E WITH DOT ABOVE
    '\xcd'     #  0xCD -> LATIN CAPITAL LETTER I WITH ACUTE
    '\xce'     #  0xCE -> LATIN CAPITAL LETTER I WITH CIRCUMFLEX
    '\u012a'   #  0xCF -> LATIN CAPITAL LETTER I WITH MACRON
    '\u0110'   #  0xD0 -> LATIN CAPITAL LETTER D WITH STROKE
    '\u0145'   #  0xD1 -> LATIN CAPITAL LETTER N WITH CEDILLA
    '\u014c'   #  0xD2 -> LATIN CAPITAL LETTER O WITH MACRON
    '\u0136'   #  0xD3 -> LATIN CAPITAL LETTER K WITH CEDILLA
    '\xd4'     #  0xD4 -> LATIN CAPITAL LETTER O WITH CIRCUMFLEX
    '\xd5'     #  0xD5 -> LATIN CAPITAL LETTER O WITH TILDE
    '\xd6'     #  0xD6 -> LATIN CAPITAL LETTER O WITH DIAERESIS
    '\xd7'     #  0xD7 -> MULTIPLICATION SIGN
    '\xd8'     #  0xD8 -> LATIN CAPITAL LETTER O WITH STROKE
    '\u0172'   #  0xD9 -> LATIN CAPITAL LETTER U WITH OGONEK
    '\xda'     #  0xDA -> LATIN CAPITAL LETTER U WITH ACUTE
    '\xdb'     #  0xDB -> LATIN CAPITAL LETTER U WITH CIRCUMFLEX
    '\xdc'     #  0xDC -> LATIN CAPITAL LETTER U WITH DIAERESIS
    '\u0168'   #  0xDD -> LATIN CAPITAL LETTER U WITH TILDE
    '\u016a'   #  0xDE -> LATIN CAPITAL LETTER U WITH MACRON
    '\xdf'     #  0xDF -> LATIN SMALL LETTER SHARP S
    '\u0101'   #  0xE0 -> LATIN SMALL LETTER A WITH MACRON
    '\xe1'     #  0xE1 -> LATIN SMALL LETTER A WITH ACUTE
    '\xe2'     #  0xE2 -> LATIN SMALL LETTER A WITH CIRCUMFLEX
    '\xe3'     #  0xE3 -> LATIN SMALL LETTER A WITH TILDE
    '\xe4'     #  0xE4 -> LATIN SMALL LETTER A WITH DIAERESIS
    '\xe5'     #  0xE5 -> LATIN SMALL LETTER A WITH RING ABOVE
    '\xe6'     #  0xE6 -> LATIN SMALL LETTER AE
    '\u012f'   #  0xE7 -> LATIN SMALL LETTER I WITH OGONEK
    '\u010d'   #  0xE8 -> LATIN SMALL LETTER C WITH CARON
    '\xe9'     #  0xE9 -> LATIN SMALL LETTER E WITH ACUTE
    '\u0119'   #  0xEA -> LATIN SMALL LETTER E WITH OGONEK
    '\xeb'     #  0xEB -> LATIN SMALL LETTER E WITH DIAERESIS
    '\u0117'   #  0xEC -> LATIN SMALL LETTER E WITH DOT ABOVE
    '\xed'     #  0xED -> LATIN SMALL LETTER I WITH ACUTE
    '\xee'     #  0xEE -> LATIN SMALL LETTER I WITH CIRCUMFLEX
    '\u012b'   #  0xEF -> LATIN SMALL LETTER I WITH MACRON
    '\u0111'   #  0xF0 -> LATIN SMALL LETTER D WITH STROKE
    '\u0146'   #  0xF1 -> LATIN SMALL LETTER N WITH CEDILLA
    '\u014d'   #  0xF2 -> LATIN SMALL LETTER O WITH MACRON
    '\u0137'   #  0xF3 -> LATIN SMALL LETTER K WITH CEDILLA
    '\xf4'     #  0xF4 -> LATIN SMALL LETTER O WITH CIRCUMFLEX
    '\xf5'     #  0xF5 -> LATIN SMALL LETTER O WITH TILDE
    '\xf6'     #  0xF6 -> LATIN SMALL LETTER O WITH DIAERESIS
    '\xf7'     #  0xF7 -> DIVISION SIGN
    '\xf8'     #  0xF8 -> LATIN SMALL LETTER O WITH STROKE
    '\u0173'   #  0xF9 -> LATIN SMALL LETTER U WITH OGONEK
    '\xfa'     #  0xFA -> LATIN SMALL LETTER U WITH ACUTE
    '\xfb'     #  0xFB -> LATIN SMALL LETTER U WITH CIRCUMFLEX
    '\xfc'     #  0xFC -> LATIN SMALL LETTER U WITH DIAERESIS
    '\u0169'   #  0xFD -> LATIN SMALL LETTER U WITH TILDE
    '\u016b'   #  0xFE -> LATIN SMALL LETTER U WITH MACRON
    '\u02d9'   #  0xFF -> DOT ABOVE
)

### Encoding table
encoding_table=codecs.charmap_build(decoding_table)
