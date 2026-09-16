""" Python Character Mapping Codec iso8859_14 generated from 'MAPPINGS/ISO8859/8859-14.TXT' with gencodec.py.

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
        name='iso8859-14',
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
    '\u1e02'   #  0xA1 -> LATIN CAPITAL LETTER B WITH DOT ABOVE
    '\u1e03'   #  0xA2 -> LATIN SMALL LETTER B WITH DOT ABOVE
    '\xa3'     #  0xA3 -> POUND SIGN
    '\u010a'   #  0xA4 -> LATIN CAPITAL LETTER C WITH DOT ABOVE
    '\u010b'   #  0xA5 -> LATIN SMALL LETTER C WITH DOT ABOVE
    '\u1e0a'   #  0xA6 -> LATIN CAPITAL LETTER D WITH DOT ABOVE
    '\xa7'     #  0xA7 -> SECTION SIGN
    '\u1e80'   #  0xA8 -> LATIN CAPITAL LETTER W WITH GRAVE
    '\xa9'     #  0xA9 -> COPYRIGHT SIGN
    '\u1e82'   #  0xAA -> LATIN CAPITAL LETTER W WITH ACUTE
    '\u1e0b'   #  0xAB -> LATIN SMALL LETTER D WITH DOT ABOVE
    '\u1ef2'   #  0xAC -> LATIN CAPITAL LETTER Y WITH GRAVE
    '\xad'     #  0xAD -> SOFT HYPHEN
    '\xae'     #  0xAE -> REGISTERED SIGN
    '\u0178'   #  0xAF -> LATIN CAPITAL LETTER Y WITH DIAERESIS
    '\u1e1e'   #  0xB0 -> LATIN CAPITAL LETTER F WITH DOT ABOVE
    '\u1e1f'   #  0xB1 -> LATIN SMALL LETTER F WITH DOT ABOVE
    '\u0120'   #  0xB2 -> LATIN CAPITAL LETTER G WITH DOT ABOVE
    '\u0121'   #  0xB3 -> LATIN SMALL LETTER G WITH DOT ABOVE
    '\u1e40'   #  0xB4 -> LATIN CAPITAL LETTER M WITH DOT ABOVE
    '\u1e41'   #  0xB5 -> LATIN SMALL LETTER M WITH DOT ABOVE
    '\xb6'     #  0xB6 -> PILCROW SIGN
    '\u1e56'   #  0xB7 -> LATIN CAPITAL LETTER P WITH DOT ABOVE
    '\u1e81'   #  0xB8 -> LATIN SMALL LETTER W WITH GRAVE
    '\u1e57'   #  0xB9 -> LATIN SMALL LETTER P WITH DOT ABOVE
    '\u1e83'   #  0xBA -> LATIN SMALL LETTER W WITH ACUTE
    '\u1e60'   #  0xBB -> LATIN CAPITAL LETTER S WITH DOT ABOVE
    '\u1ef3'   #  0xBC -> LATIN SMALL LETTER Y WITH GRAVE
    '\u1e84'   #  0xBD -> LATIN CAPITAL LETTER W WITH DIAERESIS
    '\u1e85'   #  0xBE -> LATIN SMALL LETTER W WITH DIAERESIS
    '\u1e61'   #  0xBF -> LATIN SMALL LETTER S WITH DOT ABOVE
    '\xc0'     #  0xC0 -> LATIN CAPITAL LETTER A WITH GRAVE
    '\xc1'     #  0xC1 -> LATIN CAPITAL LETTER A WITH ACUTE
    '\xc2'     #  0xC2 -> LATIN CAPITAL LETTER A WITH CIRCUMFLEX
    '\xc3'     #  0xC3 -> LATIN CAPITAL LETTER A WITH TILDE
    '\xc4'     #  0xC4 -> LATIN CAPITAL LETTER A WITH DIAERESIS
    '\xc5'     #  0xC5 -> LATIN CAPITAL LETTER A WITH RING ABOVE
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
    '\u0174'   #  0xD0 -> LATIN CAPITAL LETTER W WITH CIRCUMFLEX
    '\xd1'     #  0xD1 -> LATIN CAPITAL LETTER N WITH TILDE
    '\xd2'     #  0xD2 -> LATIN CAPITAL LETTER O WITH GRAVE
    '\xd3'     #  0xD3 -> LATIN CAPITAL LETTER O WITH ACUTE
    '\xd4'     #  0xD4 -> LATIN CAPITAL LETTER O WITH CIRCUMFLEX
    '\xd5'     #  0xD5 -> LATIN CAPITAL LETTER O WITH TILDE
    '\xd6'     #  0xD6 -> LATIN CAPITAL LETTER O WITH DIAERESIS
    '\u1e6a'   #  0xD7 -> LATIN CAPITAL LETTER T WITH DOT ABOVE
    '\xd8'     #  0xD8 -> LATIN CAPITAL LETTER O WITH STROKE
    '\xd9'     #  0xD9 -> LATIN CAPITAL LETTER U WITH GRAVE
    '\xda'     #  0xDA -> LATIN CAPITAL LETTER U WITH ACUTE
    '\xdb'     #  0xDB -> LATIN CAPITAL LETTER U WITH CIRCUMFLEX
    '\xdc'     #  0xDC -> LATIN CAPITAL LETTER U WITH DIAERESIS
    '\xdd'     #  0xDD -> LATIN CAPITAL LETTER Y WITH ACUTE
    '\u0176'   #  0xDE -> LATIN CAPITAL LETTER Y WITH CIRCUMFLEX
    '\xdf'     #  0xDF -> LATIN SMALL LETTER SHARP S
    '\xe0'     #  0xE0 -> LATIN SMALL LETTER A WITH GRAVE
    '\xe1'     #  0xE1 -> LATIN SMALL LETTER A WITH ACUTE
    '\xe2'     #  0xE2 -> LATIN SMALL LETTER A WITH CIRCUMFLEX
    '\xe3'     #  0xE3 -> LATIN SMALL LETTER A WITH TILDE
    '\xe4'     #  0xE4 -> LATIN SMALL LETTER A WITH DIAERESIS
    '\xe5'     #  0xE5 -> LATIN SMALL LETTER A WITH RING ABOVE
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
    '\u0175'   #  0xF0 -> LATIN SMALL LETTER W WITH CIRCUMFLEX
    '\xf1'     #  0xF1 -> LATIN SMALL LETTER N WITH TILDE
    '\xf2'     #  0xF2 -> LATIN SMALL LETTER O WITH GRAVE
    '\xf3'     #  0xF3 -> LATIN SMALL LETTER O WITH ACUTE
    '\xf4'     #  0xF4 -> LATIN SMALL LETTER O WITH CIRCUMFLEX
    '\xf5'     #  0xF5 -> LATIN SMALL LETTER O WITH TILDE
    '\xf6'     #  0xF6 -> LATIN SMALL LETTER O WITH DIAERESIS
    '\u1e6b'   #  0xF7 -> LATIN SMALL LETTER T WITH DOT ABOVE
    '\xf8'     #  0xF8 -> LATIN SMALL LETTER O WITH STROKE
    '\xf9'     #  0xF9 -> LATIN SMALL LETTER U WITH GRAVE
    '\xfa'     #  0xFA -> LATIN SMALL LETTER U WITH ACUTE
    '\xfb'     #  0xFB -> LATIN SMALL LETTER U WITH CIRCUMFLEX
    '\xfc'     #  0xFC -> LATIN SMALL LETTER U WITH DIAERESIS
    '\xfd'     #  0xFD -> LATIN SMALL LETTER Y WITH ACUTE
    '\u0177'   #  0xFE -> LATIN SMALL LETTER Y WITH CIRCUMFLEX
    '\xff'     #  0xFF -> LATIN SMALL LETTER Y WITH DIAERESIS
)

### Encoding table
encoding_table=codecs.charmap_build(decoding_table)
