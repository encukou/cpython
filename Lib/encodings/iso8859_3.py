""" Python Character Mapping Codec iso8859_3 generated from 'MAPPINGS/ISO8859/8859-3.TXT' with gencodec.py.

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
        name='iso8859-3',
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
    '\u0126'   #  0xA1 -> LATIN CAPITAL LETTER H WITH STROKE
    '\u02d8'   #  0xA2 -> BREVE
    '\xa3'     #  0xA3 -> POUND SIGN
    '\xa4'     #  0xA4 -> CURRENCY SIGN
    '\ufffe'
    '\u0124'   #  0xA6 -> LATIN CAPITAL LETTER H WITH CIRCUMFLEX
    '\xa7'     #  0xA7 -> SECTION SIGN
    '\xa8'     #  0xA8 -> DIAERESIS
    '\u0130'   #  0xA9 -> LATIN CAPITAL LETTER I WITH DOT ABOVE
    '\u015e'   #  0xAA -> LATIN CAPITAL LETTER S WITH CEDILLA
    '\u011e'   #  0xAB -> LATIN CAPITAL LETTER G WITH BREVE
    '\u0134'   #  0xAC -> LATIN CAPITAL LETTER J WITH CIRCUMFLEX
    '\xad'     #  0xAD -> SOFT HYPHEN
    '\ufffe'
    '\u017b'   #  0xAF -> LATIN CAPITAL LETTER Z WITH DOT ABOVE
    '\xb0'     #  0xB0 -> DEGREE SIGN
    '\u0127'   #  0xB1 -> LATIN SMALL LETTER H WITH STROKE
    '\xb2'     #  0xB2 -> SUPERSCRIPT TWO
    '\xb3'     #  0xB3 -> SUPERSCRIPT THREE
    '\xb4'     #  0xB4 -> ACUTE ACCENT
    '\xb5'     #  0xB5 -> MICRO SIGN
    '\u0125'   #  0xB6 -> LATIN SMALL LETTER H WITH CIRCUMFLEX
    '\xb7'     #  0xB7 -> MIDDLE DOT
    '\xb8'     #  0xB8 -> CEDILLA
    '\u0131'   #  0xB9 -> LATIN SMALL LETTER DOTLESS I
    '\u015f'   #  0xBA -> LATIN SMALL LETTER S WITH CEDILLA
    '\u011f'   #  0xBB -> LATIN SMALL LETTER G WITH BREVE
    '\u0135'   #  0xBC -> LATIN SMALL LETTER J WITH CIRCUMFLEX
    '\xbd'     #  0xBD -> VULGAR FRACTION ONE HALF
    '\ufffe'
    '\u017c'   #  0xBF -> LATIN SMALL LETTER Z WITH DOT ABOVE
    '\xc0'     #  0xC0 -> LATIN CAPITAL LETTER A WITH GRAVE
    '\xc1'     #  0xC1 -> LATIN CAPITAL LETTER A WITH ACUTE
    '\xc2'     #  0xC2 -> LATIN CAPITAL LETTER A WITH CIRCUMFLEX
    '\ufffe'
    '\xc4'     #  0xC4 -> LATIN CAPITAL LETTER A WITH DIAERESIS
    '\u010a'   #  0xC5 -> LATIN CAPITAL LETTER C WITH DOT ABOVE
    '\u0108'   #  0xC6 -> LATIN CAPITAL LETTER C WITH CIRCUMFLEX
    '\xc7'     #  0xC7 -> LATIN CAPITAL LETTER C WITH CEDILLA
    '\xc8'     #  0xC8 -> LATIN CAPITAL LETTER E WITH GRAVE
    '\xc9'     #  0xC9 -> LATIN CAPITAL LETTER E WITH ACUTE
    '\xca'     #  0xCA -> LATIN CAPITAL LETTER E WITH CIRCUMFLEX
    '\xcb'     #  0xCB -> LATIN CAPITAL LETTER E WITH DIAERESIS
    '\xcc'     #  0xCC -> LATIN CAPITAL LETTER I WITH GRAVE
    '\xcd'     #  0xCD -> LATIN CAPITAL LETTER I WITH ACUTE
    '\xce'     #  0xCE -> LATIN CAPITAL LETTER I WITH CIRCUMFLEX
    '\xcf'     #  0xCF -> LATIN CAPITAL LETTER I WITH DIAERESIS
    '\ufffe'
    '\xd1'     #  0xD1 -> LATIN CAPITAL LETTER N WITH TILDE
    '\xd2'     #  0xD2 -> LATIN CAPITAL LETTER O WITH GRAVE
    '\xd3'     #  0xD3 -> LATIN CAPITAL LETTER O WITH ACUTE
    '\xd4'     #  0xD4 -> LATIN CAPITAL LETTER O WITH CIRCUMFLEX
    '\u0120'   #  0xD5 -> LATIN CAPITAL LETTER G WITH DOT ABOVE
    '\xd6'     #  0xD6 -> LATIN CAPITAL LETTER O WITH DIAERESIS
    '\xd7'     #  0xD7 -> MULTIPLICATION SIGN
    '\u011c'   #  0xD8 -> LATIN CAPITAL LETTER G WITH CIRCUMFLEX
    '\xd9'     #  0xD9 -> LATIN CAPITAL LETTER U WITH GRAVE
    '\xda'     #  0xDA -> LATIN CAPITAL LETTER U WITH ACUTE
    '\xdb'     #  0xDB -> LATIN CAPITAL LETTER U WITH CIRCUMFLEX
    '\xdc'     #  0xDC -> LATIN CAPITAL LETTER U WITH DIAERESIS
    '\u016c'   #  0xDD -> LATIN CAPITAL LETTER U WITH BREVE
    '\u015c'   #  0xDE -> LATIN CAPITAL LETTER S WITH CIRCUMFLEX
    '\xdf'     #  0xDF -> LATIN SMALL LETTER SHARP S
    '\xe0'     #  0xE0 -> LATIN SMALL LETTER A WITH GRAVE
    '\xe1'     #  0xE1 -> LATIN SMALL LETTER A WITH ACUTE
    '\xe2'     #  0xE2 -> LATIN SMALL LETTER A WITH CIRCUMFLEX
    '\ufffe'
    '\xe4'     #  0xE4 -> LATIN SMALL LETTER A WITH DIAERESIS
    '\u010b'   #  0xE5 -> LATIN SMALL LETTER C WITH DOT ABOVE
    '\u0109'   #  0xE6 -> LATIN SMALL LETTER C WITH CIRCUMFLEX
    '\xe7'     #  0xE7 -> LATIN SMALL LETTER C WITH CEDILLA
    '\xe8'     #  0xE8 -> LATIN SMALL LETTER E WITH GRAVE
    '\xe9'     #  0xE9 -> LATIN SMALL LETTER E WITH ACUTE
    '\xea'     #  0xEA -> LATIN SMALL LETTER E WITH CIRCUMFLEX
    '\xeb'     #  0xEB -> LATIN SMALL LETTER E WITH DIAERESIS
    '\xec'     #  0xEC -> LATIN SMALL LETTER I WITH GRAVE
    '\xed'     #  0xED -> LATIN SMALL LETTER I WITH ACUTE
    '\xee'     #  0xEE -> LATIN SMALL LETTER I WITH CIRCUMFLEX
    '\xef'     #  0xEF -> LATIN SMALL LETTER I WITH DIAERESIS
    '\ufffe'
    '\xf1'     #  0xF1 -> LATIN SMALL LETTER N WITH TILDE
    '\xf2'     #  0xF2 -> LATIN SMALL LETTER O WITH GRAVE
    '\xf3'     #  0xF3 -> LATIN SMALL LETTER O WITH ACUTE
    '\xf4'     #  0xF4 -> LATIN SMALL LETTER O WITH CIRCUMFLEX
    '\u0121'   #  0xF5 -> LATIN SMALL LETTER G WITH DOT ABOVE
    '\xf6'     #  0xF6 -> LATIN SMALL LETTER O WITH DIAERESIS
    '\xf7'     #  0xF7 -> DIVISION SIGN
    '\u011d'   #  0xF8 -> LATIN SMALL LETTER G WITH CIRCUMFLEX
    '\xf9'     #  0xF9 -> LATIN SMALL LETTER U WITH GRAVE
    '\xfa'     #  0xFA -> LATIN SMALL LETTER U WITH ACUTE
    '\xfb'     #  0xFB -> LATIN SMALL LETTER U WITH CIRCUMFLEX
    '\xfc'     #  0xFC -> LATIN SMALL LETTER U WITH DIAERESIS
    '\u016d'   #  0xFD -> LATIN SMALL LETTER U WITH BREVE
    '\u015d'   #  0xFE -> LATIN SMALL LETTER S WITH CIRCUMFLEX
    '\u02d9'   #  0xFF -> DOT ABOVE
)

### Encoding table
encoding_table=codecs.charmap_build(decoding_table)
