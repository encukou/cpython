""" Python Character Mapping Codec iso8859_10 generated from 'MAPPINGS/ISO8859/8859-10.TXT' with gencodec.py.

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
        name='iso8859-10',
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
    '\u0112'   #  0xA2 -> LATIN CAPITAL LETTER E WITH MACRON
    '\u0122'   #  0xA3 -> LATIN CAPITAL LETTER G WITH CEDILLA
    '\u012a'   #  0xA4 -> LATIN CAPITAL LETTER I WITH MACRON
    '\u0128'   #  0xA5 -> LATIN CAPITAL LETTER I WITH TILDE
    '\u0136'   #  0xA6 -> LATIN CAPITAL LETTER K WITH CEDILLA
    '\xa7'     #  0xA7 -> SECTION SIGN
    '\u013b'   #  0xA8 -> LATIN CAPITAL LETTER L WITH CEDILLA
    '\u0110'   #  0xA9 -> LATIN CAPITAL LETTER D WITH STROKE
    '\u0160'   #  0xAA -> LATIN CAPITAL LETTER S WITH CARON
    '\u0166'   #  0xAB -> LATIN CAPITAL LETTER T WITH STROKE
    '\u017d'   #  0xAC -> LATIN CAPITAL LETTER Z WITH CARON
    '\xad'     #  0xAD -> SOFT HYPHEN
    '\u016a'   #  0xAE -> LATIN CAPITAL LETTER U WITH MACRON
    '\u014a'   #  0xAF -> LATIN CAPITAL LETTER ENG
    '\xb0'     #  0xB0 -> DEGREE SIGN
    '\u0105'   #  0xB1 -> LATIN SMALL LETTER A WITH OGONEK
    '\u0113'   #  0xB2 -> LATIN SMALL LETTER E WITH MACRON
    '\u0123'   #  0xB3 -> LATIN SMALL LETTER G WITH CEDILLA
    '\u012b'   #  0xB4 -> LATIN SMALL LETTER I WITH MACRON
    '\u0129'   #  0xB5 -> LATIN SMALL LETTER I WITH TILDE
    '\u0137'   #  0xB6 -> LATIN SMALL LETTER K WITH CEDILLA
    '\xb7'     #  0xB7 -> MIDDLE DOT
    '\u013c'   #  0xB8 -> LATIN SMALL LETTER L WITH CEDILLA
    '\u0111'   #  0xB9 -> LATIN SMALL LETTER D WITH STROKE
    '\u0161'   #  0xBA -> LATIN SMALL LETTER S WITH CARON
    '\u0167'   #  0xBB -> LATIN SMALL LETTER T WITH STROKE
    '\u017e'   #  0xBC -> LATIN SMALL LETTER Z WITH CARON
    '\u2015'   #  0xBD -> HORIZONTAL BAR
    '\u016b'   #  0xBE -> LATIN SMALL LETTER U WITH MACRON
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
    '\xcf'     #  0xCF -> LATIN CAPITAL LETTER I WITH DIAERESIS
    '\xd0'     #  0xD0 -> LATIN CAPITAL LETTER ETH (Icelandic)
    '\u0145'   #  0xD1 -> LATIN CAPITAL LETTER N WITH CEDILLA
    '\u014c'   #  0xD2 -> LATIN CAPITAL LETTER O WITH MACRON
    '\xd3'     #  0xD3 -> LATIN CAPITAL LETTER O WITH ACUTE
    '\xd4'     #  0xD4 -> LATIN CAPITAL LETTER O WITH CIRCUMFLEX
    '\xd5'     #  0xD5 -> LATIN CAPITAL LETTER O WITH TILDE
    '\xd6'     #  0xD6 -> LATIN CAPITAL LETTER O WITH DIAERESIS
    '\u0168'   #  0xD7 -> LATIN CAPITAL LETTER U WITH TILDE
    '\xd8'     #  0xD8 -> LATIN CAPITAL LETTER O WITH STROKE
    '\u0172'   #  0xD9 -> LATIN CAPITAL LETTER U WITH OGONEK
    '\xda'     #  0xDA -> LATIN CAPITAL LETTER U WITH ACUTE
    '\xdb'     #  0xDB -> LATIN CAPITAL LETTER U WITH CIRCUMFLEX
    '\xdc'     #  0xDC -> LATIN CAPITAL LETTER U WITH DIAERESIS
    '\xdd'     #  0xDD -> LATIN CAPITAL LETTER Y WITH ACUTE
    '\xde'     #  0xDE -> LATIN CAPITAL LETTER THORN (Icelandic)
    '\xdf'     #  0xDF -> LATIN SMALL LETTER SHARP S (German)
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
    '\xef'     #  0xEF -> LATIN SMALL LETTER I WITH DIAERESIS
    '\xf0'     #  0xF0 -> LATIN SMALL LETTER ETH (Icelandic)
    '\u0146'   #  0xF1 -> LATIN SMALL LETTER N WITH CEDILLA
    '\u014d'   #  0xF2 -> LATIN SMALL LETTER O WITH MACRON
    '\xf3'     #  0xF3 -> LATIN SMALL LETTER O WITH ACUTE
    '\xf4'     #  0xF4 -> LATIN SMALL LETTER O WITH CIRCUMFLEX
    '\xf5'     #  0xF5 -> LATIN SMALL LETTER O WITH TILDE
    '\xf6'     #  0xF6 -> LATIN SMALL LETTER O WITH DIAERESIS
    '\u0169'   #  0xF7 -> LATIN SMALL LETTER U WITH TILDE
    '\xf8'     #  0xF8 -> LATIN SMALL LETTER O WITH STROKE
    '\u0173'   #  0xF9 -> LATIN SMALL LETTER U WITH OGONEK
    '\xfa'     #  0xFA -> LATIN SMALL LETTER U WITH ACUTE
    '\xfb'     #  0xFB -> LATIN SMALL LETTER U WITH CIRCUMFLEX
    '\xfc'     #  0xFC -> LATIN SMALL LETTER U WITH DIAERESIS
    '\xfd'     #  0xFD -> LATIN SMALL LETTER Y WITH ACUTE
    '\xfe'     #  0xFE -> LATIN SMALL LETTER THORN (Icelandic)
    '\u0138'   #  0xFF -> LATIN SMALL LETTER KRA
)

### Encoding table
encoding_table=codecs.charmap_build(decoding_table)
