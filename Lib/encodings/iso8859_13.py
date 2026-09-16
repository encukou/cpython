""" Python Character Mapping Codec iso8859_13 generated from 'MAPPINGS/ISO8859/8859-13.TXT' with gencodec.py.

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
        name='iso8859-13',
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
    '\u201d'   #  0xA1 -> RIGHT DOUBLE QUOTATION MARK
    '\xa2'     #  0xA2 -> CENT SIGN
    '\xa3'     #  0xA3 -> POUND SIGN
    '\xa4'     #  0xA4 -> CURRENCY SIGN
    '\u201e'   #  0xA5 -> DOUBLE LOW-9 QUOTATION MARK
    '\xa6'     #  0xA6 -> BROKEN BAR
    '\xa7'     #  0xA7 -> SECTION SIGN
    '\xd8'     #  0xA8 -> LATIN CAPITAL LETTER O WITH STROKE
    '\xa9'     #  0xA9 -> COPYRIGHT SIGN
    '\u0156'   #  0xAA -> LATIN CAPITAL LETTER R WITH CEDILLA
    '\xab'     #  0xAB -> LEFT-POINTING DOUBLE ANGLE QUOTATION MARK
    '\xac'     #  0xAC -> NOT SIGN
    '\xad'     #  0xAD -> SOFT HYPHEN
    '\xae'     #  0xAE -> REGISTERED SIGN
    '\xc6'     #  0xAF -> LATIN CAPITAL LETTER AE
    '\xb0'     #  0xB0 -> DEGREE SIGN
    '\xb1'     #  0xB1 -> PLUS-MINUS SIGN
    '\xb2'     #  0xB2 -> SUPERSCRIPT TWO
    '\xb3'     #  0xB3 -> SUPERSCRIPT THREE
    '\u201c'   #  0xB4 -> LEFT DOUBLE QUOTATION MARK
    '\xb5'     #  0xB5 -> MICRO SIGN
    '\xb6'     #  0xB6 -> PILCROW SIGN
    '\xb7'     #  0xB7 -> MIDDLE DOT
    '\xf8'     #  0xB8 -> LATIN SMALL LETTER O WITH STROKE
    '\xb9'     #  0xB9 -> SUPERSCRIPT ONE
    '\u0157'   #  0xBA -> LATIN SMALL LETTER R WITH CEDILLA
    '\xbb'     #  0xBB -> RIGHT-POINTING DOUBLE ANGLE QUOTATION MARK
    '\xbc'     #  0xBC -> VULGAR FRACTION ONE QUARTER
    '\xbd'     #  0xBD -> VULGAR FRACTION ONE HALF
    '\xbe'     #  0xBE -> VULGAR FRACTION THREE QUARTERS
    '\xe6'     #  0xBF -> LATIN SMALL LETTER AE
    '\u0104'   #  0xC0 -> LATIN CAPITAL LETTER A WITH OGONEK
    '\u012e'   #  0xC1 -> LATIN CAPITAL LETTER I WITH OGONEK
    '\u0100'   #  0xC2 -> LATIN CAPITAL LETTER A WITH MACRON
    '\u0106'   #  0xC3 -> LATIN CAPITAL LETTER C WITH ACUTE
    '\xc4'     #  0xC4 -> LATIN CAPITAL LETTER A WITH DIAERESIS
    '\xc5'     #  0xC5 -> LATIN CAPITAL LETTER A WITH RING ABOVE
    '\u0118'   #  0xC6 -> LATIN CAPITAL LETTER E WITH OGONEK
    '\u0112'   #  0xC7 -> LATIN CAPITAL LETTER E WITH MACRON
    '\u010c'   #  0xC8 -> LATIN CAPITAL LETTER C WITH CARON
    '\xc9'     #  0xC9 -> LATIN CAPITAL LETTER E WITH ACUTE
    '\u0179'   #  0xCA -> LATIN CAPITAL LETTER Z WITH ACUTE
    '\u0116'   #  0xCB -> LATIN CAPITAL LETTER E WITH DOT ABOVE
    '\u0122'   #  0xCC -> LATIN CAPITAL LETTER G WITH CEDILLA
    '\u0136'   #  0xCD -> LATIN CAPITAL LETTER K WITH CEDILLA
    '\u012a'   #  0xCE -> LATIN CAPITAL LETTER I WITH MACRON
    '\u013b'   #  0xCF -> LATIN CAPITAL LETTER L WITH CEDILLA
    '\u0160'   #  0xD0 -> LATIN CAPITAL LETTER S WITH CARON
    '\u0143'   #  0xD1 -> LATIN CAPITAL LETTER N WITH ACUTE
    '\u0145'   #  0xD2 -> LATIN CAPITAL LETTER N WITH CEDILLA
    '\xd3'     #  0xD3 -> LATIN CAPITAL LETTER O WITH ACUTE
    '\u014c'   #  0xD4 -> LATIN CAPITAL LETTER O WITH MACRON
    '\xd5'     #  0xD5 -> LATIN CAPITAL LETTER O WITH TILDE
    '\xd6'     #  0xD6 -> LATIN CAPITAL LETTER O WITH DIAERESIS
    '\xd7'     #  0xD7 -> MULTIPLICATION SIGN
    '\u0172'   #  0xD8 -> LATIN CAPITAL LETTER U WITH OGONEK
    '\u0141'   #  0xD9 -> LATIN CAPITAL LETTER L WITH STROKE
    '\u015a'   #  0xDA -> LATIN CAPITAL LETTER S WITH ACUTE
    '\u016a'   #  0xDB -> LATIN CAPITAL LETTER U WITH MACRON
    '\xdc'     #  0xDC -> LATIN CAPITAL LETTER U WITH DIAERESIS
    '\u017b'   #  0xDD -> LATIN CAPITAL LETTER Z WITH DOT ABOVE
    '\u017d'   #  0xDE -> LATIN CAPITAL LETTER Z WITH CARON
    '\xdf'     #  0xDF -> LATIN SMALL LETTER SHARP S (German)
    '\u0105'   #  0xE0 -> LATIN SMALL LETTER A WITH OGONEK
    '\u012f'   #  0xE1 -> LATIN SMALL LETTER I WITH OGONEK
    '\u0101'   #  0xE2 -> LATIN SMALL LETTER A WITH MACRON
    '\u0107'   #  0xE3 -> LATIN SMALL LETTER C WITH ACUTE
    '\xe4'     #  0xE4 -> LATIN SMALL LETTER A WITH DIAERESIS
    '\xe5'     #  0xE5 -> LATIN SMALL LETTER A WITH RING ABOVE
    '\u0119'   #  0xE6 -> LATIN SMALL LETTER E WITH OGONEK
    '\u0113'   #  0xE7 -> LATIN SMALL LETTER E WITH MACRON
    '\u010d'   #  0xE8 -> LATIN SMALL LETTER C WITH CARON
    '\xe9'     #  0xE9 -> LATIN SMALL LETTER E WITH ACUTE
    '\u017a'   #  0xEA -> LATIN SMALL LETTER Z WITH ACUTE
    '\u0117'   #  0xEB -> LATIN SMALL LETTER E WITH DOT ABOVE
    '\u0123'   #  0xEC -> LATIN SMALL LETTER G WITH CEDILLA
    '\u0137'   #  0xED -> LATIN SMALL LETTER K WITH CEDILLA
    '\u012b'   #  0xEE -> LATIN SMALL LETTER I WITH MACRON
    '\u013c'   #  0xEF -> LATIN SMALL LETTER L WITH CEDILLA
    '\u0161'   #  0xF0 -> LATIN SMALL LETTER S WITH CARON
    '\u0144'   #  0xF1 -> LATIN SMALL LETTER N WITH ACUTE
    '\u0146'   #  0xF2 -> LATIN SMALL LETTER N WITH CEDILLA
    '\xf3'     #  0xF3 -> LATIN SMALL LETTER O WITH ACUTE
    '\u014d'   #  0xF4 -> LATIN SMALL LETTER O WITH MACRON
    '\xf5'     #  0xF5 -> LATIN SMALL LETTER O WITH TILDE
    '\xf6'     #  0xF6 -> LATIN SMALL LETTER O WITH DIAERESIS
    '\xf7'     #  0xF7 -> DIVISION SIGN
    '\u0173'   #  0xF8 -> LATIN SMALL LETTER U WITH OGONEK
    '\u0142'   #  0xF9 -> LATIN SMALL LETTER L WITH STROKE
    '\u015b'   #  0xFA -> LATIN SMALL LETTER S WITH ACUTE
    '\u016b'   #  0xFB -> LATIN SMALL LETTER U WITH MACRON
    '\xfc'     #  0xFC -> LATIN SMALL LETTER U WITH DIAERESIS
    '\u017c'   #  0xFD -> LATIN SMALL LETTER Z WITH DOT ABOVE
    '\u017e'   #  0xFE -> LATIN SMALL LETTER Z WITH CARON
    '\u2019'   #  0xFF -> RIGHT SINGLE QUOTATION MARK
)

### Encoding table
encoding_table=codecs.charmap_build(decoding_table)
