""" Python Character Mapping Codec mac_farsi generated from 'MAPPINGS/VENDORS/APPLE/FARSI.TXT' with gencodec.py.

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
        name='mac-farsi',
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
    '\xa0'     #  0x81 -> NO-BREAK SPACE, right-left
    '\xc7'     #  0x82 -> LATIN CAPITAL LETTER C WITH CEDILLA
    '\xc9'     #  0x83 -> LATIN CAPITAL LETTER E WITH ACUTE
    '\xd1'     #  0x84 -> LATIN CAPITAL LETTER N WITH TILDE
    '\xd6'     #  0x85 -> LATIN CAPITAL LETTER O WITH DIAERESIS
    '\xdc'     #  0x86 -> LATIN CAPITAL LETTER U WITH DIAERESIS
    '\xe1'     #  0x87 -> LATIN SMALL LETTER A WITH ACUTE
    '\xe0'     #  0x88 -> LATIN SMALL LETTER A WITH GRAVE
    '\xe2'     #  0x89 -> LATIN SMALL LETTER A WITH CIRCUMFLEX
    '\xe4'     #  0x8A -> LATIN SMALL LETTER A WITH DIAERESIS
    '\u06ba'   #  0x8B -> ARABIC LETTER NOON GHUNNA
    '\xab'     #  0x8C -> LEFT-POINTING DOUBLE ANGLE QUOTATION MARK, right-left
    '\xe7'     #  0x8D -> LATIN SMALL LETTER C WITH CEDILLA
    '\xe9'     #  0x8E -> LATIN SMALL LETTER E WITH ACUTE
    '\xe8'     #  0x8F -> LATIN SMALL LETTER E WITH GRAVE
    '\xea'     #  0x90 -> LATIN SMALL LETTER E WITH CIRCUMFLEX
    '\xeb'     #  0x91 -> LATIN SMALL LETTER E WITH DIAERESIS
    '\xed'     #  0x92 -> LATIN SMALL LETTER I WITH ACUTE
    '\u2026'   #  0x93 -> HORIZONTAL ELLIPSIS, right-left
    '\xee'     #  0x94 -> LATIN SMALL LETTER I WITH CIRCUMFLEX
    '\xef'     #  0x95 -> LATIN SMALL LETTER I WITH DIAERESIS
    '\xf1'     #  0x96 -> LATIN SMALL LETTER N WITH TILDE
    '\xf3'     #  0x97 -> LATIN SMALL LETTER O WITH ACUTE
    '\xbb'     #  0x98 -> RIGHT-POINTING DOUBLE ANGLE QUOTATION MARK, right-left
    '\xf4'     #  0x99 -> LATIN SMALL LETTER O WITH CIRCUMFLEX
    '\xf6'     #  0x9A -> LATIN SMALL LETTER O WITH DIAERESIS
    '\xf7'     #  0x9B -> DIVISION SIGN, right-left
    '\xfa'     #  0x9C -> LATIN SMALL LETTER U WITH ACUTE
    '\xf9'     #  0x9D -> LATIN SMALL LETTER U WITH GRAVE
    '\xfb'     #  0x9E -> LATIN SMALL LETTER U WITH CIRCUMFLEX
    '\xfc'     #  0x9F -> LATIN SMALL LETTER U WITH DIAERESIS
    ' '        #  0xA0 -> SPACE, right-left
    '!'        #  0xA1 -> EXCLAMATION MARK, right-left
    '"'        #  0xA2 -> QUOTATION MARK, right-left
    '#'        #  0xA3 -> NUMBER SIGN, right-left
    '$'        #  0xA4 -> DOLLAR SIGN, right-left
    '\u066a'   #  0xA5 -> ARABIC PERCENT SIGN
    '&'        #  0xA6 -> AMPERSAND, right-left
    "'"        #  0xA7 -> APOSTROPHE, right-left
    '('        #  0xA8 -> LEFT PARENTHESIS, right-left
    ')'        #  0xA9 -> RIGHT PARENTHESIS, right-left
    '*'        #  0xAA -> ASTERISK, right-left
    '+'        #  0xAB -> PLUS SIGN, right-left
    '\u060c'   #  0xAC -> ARABIC COMMA
    '-'        #  0xAD -> HYPHEN-MINUS, right-left
    '.'        #  0xAE -> FULL STOP, right-left
    '/'        #  0xAF -> SOLIDUS, right-left
    '\u06f0'   #  0xB0 -> EXTENDED ARABIC-INDIC DIGIT ZERO, right-left (need override)
    '\u06f1'   #  0xB1 -> EXTENDED ARABIC-INDIC DIGIT ONE, right-left (need override)
    '\u06f2'   #  0xB2 -> EXTENDED ARABIC-INDIC DIGIT TWO, right-left (need override)
    '\u06f3'   #  0xB3 -> EXTENDED ARABIC-INDIC DIGIT THREE, right-left (need override)
    '\u06f4'   #  0xB4 -> EXTENDED ARABIC-INDIC DIGIT FOUR, right-left (need override)
    '\u06f5'   #  0xB5 -> EXTENDED ARABIC-INDIC DIGIT FIVE, right-left (need override)
    '\u06f6'   #  0xB6 -> EXTENDED ARABIC-INDIC DIGIT SIX, right-left (need override)
    '\u06f7'   #  0xB7 -> EXTENDED ARABIC-INDIC DIGIT SEVEN, right-left (need override)
    '\u06f8'   #  0xB8 -> EXTENDED ARABIC-INDIC DIGIT EIGHT, right-left (need override)
    '\u06f9'   #  0xB9 -> EXTENDED ARABIC-INDIC DIGIT NINE, right-left (need override)
    ':'        #  0xBA -> COLON, right-left
    '\u061b'   #  0xBB -> ARABIC SEMICOLON
    '<'        #  0xBC -> LESS-THAN SIGN, right-left
    '='        #  0xBD -> EQUALS SIGN, right-left
    '>'        #  0xBE -> GREATER-THAN SIGN, right-left
    '\u061f'   #  0xBF -> ARABIC QUESTION MARK
    '\u274a'   #  0xC0 -> EIGHT TEARDROP-SPOKED PROPELLER ASTERISK, right-left
    '\u0621'   #  0xC1 -> ARABIC LETTER HAMZA
    '\u0622'   #  0xC2 -> ARABIC LETTER ALEF WITH MADDA ABOVE
    '\u0623'   #  0xC3 -> ARABIC LETTER ALEF WITH HAMZA ABOVE
    '\u0624'   #  0xC4 -> ARABIC LETTER WAW WITH HAMZA ABOVE
    '\u0625'   #  0xC5 -> ARABIC LETTER ALEF WITH HAMZA BELOW
    '\u0626'   #  0xC6 -> ARABIC LETTER YEH WITH HAMZA ABOVE
    '\u0627'   #  0xC7 -> ARABIC LETTER ALEF
    '\u0628'   #  0xC8 -> ARABIC LETTER BEH
    '\u0629'   #  0xC9 -> ARABIC LETTER TEH MARBUTA
    '\u062a'   #  0xCA -> ARABIC LETTER TEH
    '\u062b'   #  0xCB -> ARABIC LETTER THEH
    '\u062c'   #  0xCC -> ARABIC LETTER JEEM
    '\u062d'   #  0xCD -> ARABIC LETTER HAH
    '\u062e'   #  0xCE -> ARABIC LETTER KHAH
    '\u062f'   #  0xCF -> ARABIC LETTER DAL
    '\u0630'   #  0xD0 -> ARABIC LETTER THAL
    '\u0631'   #  0xD1 -> ARABIC LETTER REH
    '\u0632'   #  0xD2 -> ARABIC LETTER ZAIN
    '\u0633'   #  0xD3 -> ARABIC LETTER SEEN
    '\u0634'   #  0xD4 -> ARABIC LETTER SHEEN
    '\u0635'   #  0xD5 -> ARABIC LETTER SAD
    '\u0636'   #  0xD6 -> ARABIC LETTER DAD
    '\u0637'   #  0xD7 -> ARABIC LETTER TAH
    '\u0638'   #  0xD8 -> ARABIC LETTER ZAH
    '\u0639'   #  0xD9 -> ARABIC LETTER AIN
    '\u063a'   #  0xDA -> ARABIC LETTER GHAIN
    '['        #  0xDB -> LEFT SQUARE BRACKET, right-left
    '\\'       #  0xDC -> REVERSE SOLIDUS, right-left
    ']'        #  0xDD -> RIGHT SQUARE BRACKET, right-left
    '^'        #  0xDE -> CIRCUMFLEX ACCENT, right-left
    '_'        #  0xDF -> LOW LINE, right-left
    '\u0640'   #  0xE0 -> ARABIC TATWEEL
    '\u0641'   #  0xE1 -> ARABIC LETTER FEH
    '\u0642'   #  0xE2 -> ARABIC LETTER QAF
    '\u0643'   #  0xE3 -> ARABIC LETTER KAF
    '\u0644'   #  0xE4 -> ARABIC LETTER LAM
    '\u0645'   #  0xE5 -> ARABIC LETTER MEEM
    '\u0646'   #  0xE6 -> ARABIC LETTER NOON
    '\u0647'   #  0xE7 -> ARABIC LETTER HEH
    '\u0648'   #  0xE8 -> ARABIC LETTER WAW
    '\u0649'   #  0xE9 -> ARABIC LETTER ALEF MAKSURA
    '\u064a'   #  0xEA -> ARABIC LETTER YEH
    '\u064b'   #  0xEB -> ARABIC FATHATAN
    '\u064c'   #  0xEC -> ARABIC DAMMATAN
    '\u064d'   #  0xED -> ARABIC KASRATAN
    '\u064e'   #  0xEE -> ARABIC FATHA
    '\u064f'   #  0xEF -> ARABIC DAMMA
    '\u0650'   #  0xF0 -> ARABIC KASRA
    '\u0651'   #  0xF1 -> ARABIC SHADDA
    '\u0652'   #  0xF2 -> ARABIC SUKUN
    '\u067e'   #  0xF3 -> ARABIC LETTER PEH
    '\u0679'   #  0xF4 -> ARABIC LETTER TTEH
    '\u0686'   #  0xF5 -> ARABIC LETTER TCHEH
    '\u06d5'   #  0xF6 -> ARABIC LETTER AE
    '\u06a4'   #  0xF7 -> ARABIC LETTER VEH
    '\u06af'   #  0xF8 -> ARABIC LETTER GAF
    '\u0688'   #  0xF9 -> ARABIC LETTER DDAL
    '\u0691'   #  0xFA -> ARABIC LETTER RREH
    '{'        #  0xFB -> LEFT CURLY BRACKET, right-left
    '|'        #  0xFC -> VERTICAL LINE, right-left
    '}'        #  0xFD -> RIGHT CURLY BRACKET, right-left
    '\u0698'   #  0xFE -> ARABIC LETTER JEH
    '\u06d2'   #  0xFF -> ARABIC LETTER YEH BARREE
)

### Encoding table
encoding_table=codecs.charmap_build(decoding_table)
