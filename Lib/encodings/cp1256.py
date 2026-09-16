""" Python Character Mapping Codec cp1256 generated from 'MAPPINGS/VENDORS/MICSFT/WINDOWS/CP1256.TXT' with gencodec.py.

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
        name='cp1256',
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
    '\u067e'   #  0x81 -> ARABIC LETTER PEH
    '\u201a'   #  0x82 -> SINGLE LOW-9 QUOTATION MARK
    '\u0192'   #  0x83 -> LATIN SMALL LETTER F WITH HOOK
    '\u201e'   #  0x84 -> DOUBLE LOW-9 QUOTATION MARK
    '\u2026'   #  0x85 -> HORIZONTAL ELLIPSIS
    '\u2020'   #  0x86 -> DAGGER
    '\u2021'   #  0x87 -> DOUBLE DAGGER
    '\u02c6'   #  0x88 -> MODIFIER LETTER CIRCUMFLEX ACCENT
    '\u2030'   #  0x89 -> PER MILLE SIGN
    '\u0679'   #  0x8A -> ARABIC LETTER TTEH
    '\u2039'   #  0x8B -> SINGLE LEFT-POINTING ANGLE QUOTATION MARK
    '\u0152'   #  0x8C -> LATIN CAPITAL LIGATURE OE
    '\u0686'   #  0x8D -> ARABIC LETTER TCHEH
    '\u0698'   #  0x8E -> ARABIC LETTER JEH
    '\u0688'   #  0x8F -> ARABIC LETTER DDAL
    '\u06af'   #  0x90 -> ARABIC LETTER GAF
    '\u2018'   #  0x91 -> LEFT SINGLE QUOTATION MARK
    '\u2019'   #  0x92 -> RIGHT SINGLE QUOTATION MARK
    '\u201c'   #  0x93 -> LEFT DOUBLE QUOTATION MARK
    '\u201d'   #  0x94 -> RIGHT DOUBLE QUOTATION MARK
    '\u2022'   #  0x95 -> BULLET
    '\u2013'   #  0x96 -> EN DASH
    '\u2014'   #  0x97 -> EM DASH
    '\u06a9'   #  0x98 -> ARABIC LETTER KEHEH
    '\u2122'   #  0x99 -> TRADE MARK SIGN
    '\u0691'   #  0x9A -> ARABIC LETTER RREH
    '\u203a'   #  0x9B -> SINGLE RIGHT-POINTING ANGLE QUOTATION MARK
    '\u0153'   #  0x9C -> LATIN SMALL LIGATURE OE
    '\u200c'   #  0x9D -> ZERO WIDTH NON-JOINER
    '\u200d'   #  0x9E -> ZERO WIDTH JOINER
    '\u06ba'   #  0x9F -> ARABIC LETTER NOON GHUNNA
    '\xa0'     #  0xA0 -> NO-BREAK SPACE
    '\u060c'   #  0xA1 -> ARABIC COMMA
    '\xa2'     #  0xA2 -> CENT SIGN
    '\xa3'     #  0xA3 -> POUND SIGN
    '\xa4'     #  0xA4 -> CURRENCY SIGN
    '\xa5'     #  0xA5 -> YEN SIGN
    '\xa6'     #  0xA6 -> BROKEN BAR
    '\xa7'     #  0xA7 -> SECTION SIGN
    '\xa8'     #  0xA8 -> DIAERESIS
    '\xa9'     #  0xA9 -> COPYRIGHT SIGN
    '\u06be'   #  0xAA -> ARABIC LETTER HEH DOACHASHMEE
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
    '\u061b'   #  0xBA -> ARABIC SEMICOLON
    '\xbb'     #  0xBB -> RIGHT-POINTING DOUBLE ANGLE QUOTATION MARK
    '\xbc'     #  0xBC -> VULGAR FRACTION ONE QUARTER
    '\xbd'     #  0xBD -> VULGAR FRACTION ONE HALF
    '\xbe'     #  0xBE -> VULGAR FRACTION THREE QUARTERS
    '\u061f'   #  0xBF -> ARABIC QUESTION MARK
    '\u06c1'   #  0xC0 -> ARABIC LETTER HEH GOAL
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
    '\xd7'     #  0xD7 -> MULTIPLICATION SIGN
    '\u0637'   #  0xD8 -> ARABIC LETTER TAH
    '\u0638'   #  0xD9 -> ARABIC LETTER ZAH
    '\u0639'   #  0xDA -> ARABIC LETTER AIN
    '\u063a'   #  0xDB -> ARABIC LETTER GHAIN
    '\u0640'   #  0xDC -> ARABIC TATWEEL
    '\u0641'   #  0xDD -> ARABIC LETTER FEH
    '\u0642'   #  0xDE -> ARABIC LETTER QAF
    '\u0643'   #  0xDF -> ARABIC LETTER KAF
    '\xe0'     #  0xE0 -> LATIN SMALL LETTER A WITH GRAVE
    '\u0644'   #  0xE1 -> ARABIC LETTER LAM
    '\xe2'     #  0xE2 -> LATIN SMALL LETTER A WITH CIRCUMFLEX
    '\u0645'   #  0xE3 -> ARABIC LETTER MEEM
    '\u0646'   #  0xE4 -> ARABIC LETTER NOON
    '\u0647'   #  0xE5 -> ARABIC LETTER HEH
    '\u0648'   #  0xE6 -> ARABIC LETTER WAW
    '\xe7'     #  0xE7 -> LATIN SMALL LETTER C WITH CEDILLA
    '\xe8'     #  0xE8 -> LATIN SMALL LETTER E WITH GRAVE
    '\xe9'     #  0xE9 -> LATIN SMALL LETTER E WITH ACUTE
    '\xea'     #  0xEA -> LATIN SMALL LETTER E WITH CIRCUMFLEX
    '\xeb'     #  0xEB -> LATIN SMALL LETTER E WITH DIAERESIS
    '\u0649'   #  0xEC -> ARABIC LETTER ALEF MAKSURA
    '\u064a'   #  0xED -> ARABIC LETTER YEH
    '\xee'     #  0xEE -> LATIN SMALL LETTER I WITH CIRCUMFLEX
    '\xef'     #  0xEF -> LATIN SMALL LETTER I WITH DIAERESIS
    '\u064b'   #  0xF0 -> ARABIC FATHATAN
    '\u064c'   #  0xF1 -> ARABIC DAMMATAN
    '\u064d'   #  0xF2 -> ARABIC KASRATAN
    '\u064e'   #  0xF3 -> ARABIC FATHA
    '\xf4'     #  0xF4 -> LATIN SMALL LETTER O WITH CIRCUMFLEX
    '\u064f'   #  0xF5 -> ARABIC DAMMA
    '\u0650'   #  0xF6 -> ARABIC KASRA
    '\xf7'     #  0xF7 -> DIVISION SIGN
    '\u0651'   #  0xF8 -> ARABIC SHADDA
    '\xf9'     #  0xF9 -> LATIN SMALL LETTER U WITH GRAVE
    '\u0652'   #  0xFA -> ARABIC SUKUN
    '\xfb'     #  0xFB -> LATIN SMALL LETTER U WITH CIRCUMFLEX
    '\xfc'     #  0xFC -> LATIN SMALL LETTER U WITH DIAERESIS
    '\u200e'   #  0xFD -> LEFT-TO-RIGHT MARK
    '\u200f'   #  0xFE -> RIGHT-TO-LEFT MARK
    '\u06d2'   #  0xFF -> ARABIC LETTER YEH BARREE
)

### Encoding table
encoding_table=codecs.charmap_build(decoding_table)
