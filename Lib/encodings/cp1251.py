""" Python Character Mapping Codec cp1251 generated from 'MAPPINGS/VENDORS/MICSFT/WINDOWS/CP1251.TXT' with gencodec.py.

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
        name='cp1251',
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
    '\u0402'   #  0x80 -> CYRILLIC CAPITAL LETTER DJE
    '\u0403'   #  0x81 -> CYRILLIC CAPITAL LETTER GJE
    '\u201a'   #  0x82 -> SINGLE LOW-9 QUOTATION MARK
    '\u0453'   #  0x83 -> CYRILLIC SMALL LETTER GJE
    '\u201e'   #  0x84 -> DOUBLE LOW-9 QUOTATION MARK
    '\u2026'   #  0x85 -> HORIZONTAL ELLIPSIS
    '\u2020'   #  0x86 -> DAGGER
    '\u2021'   #  0x87 -> DOUBLE DAGGER
    '\u20ac'   #  0x88 -> EURO SIGN
    '\u2030'   #  0x89 -> PER MILLE SIGN
    '\u0409'   #  0x8A -> CYRILLIC CAPITAL LETTER LJE
    '\u2039'   #  0x8B -> SINGLE LEFT-POINTING ANGLE QUOTATION MARK
    '\u040a'   #  0x8C -> CYRILLIC CAPITAL LETTER NJE
    '\u040c'   #  0x8D -> CYRILLIC CAPITAL LETTER KJE
    '\u040b'   #  0x8E -> CYRILLIC CAPITAL LETTER TSHE
    '\u040f'   #  0x8F -> CYRILLIC CAPITAL LETTER DZHE
    '\u0452'   #  0x90 -> CYRILLIC SMALL LETTER DJE
    '\u2018'   #  0x91 -> LEFT SINGLE QUOTATION MARK
    '\u2019'   #  0x92 -> RIGHT SINGLE QUOTATION MARK
    '\u201c'   #  0x93 -> LEFT DOUBLE QUOTATION MARK
    '\u201d'   #  0x94 -> RIGHT DOUBLE QUOTATION MARK
    '\u2022'   #  0x95 -> BULLET
    '\u2013'   #  0x96 -> EN DASH
    '\u2014'   #  0x97 -> EM DASH
    '\ufffe'   #  0x98 -> UNDEFINED
    '\u2122'   #  0x99 -> TRADE MARK SIGN
    '\u0459'   #  0x9A -> CYRILLIC SMALL LETTER LJE
    '\u203a'   #  0x9B -> SINGLE RIGHT-POINTING ANGLE QUOTATION MARK
    '\u045a'   #  0x9C -> CYRILLIC SMALL LETTER NJE
    '\u045c'   #  0x9D -> CYRILLIC SMALL LETTER KJE
    '\u045b'   #  0x9E -> CYRILLIC SMALL LETTER TSHE
    '\u045f'   #  0x9F -> CYRILLIC SMALL LETTER DZHE
    '\xa0'     #  0xA0 -> NO-BREAK SPACE
    '\u040e'   #  0xA1 -> CYRILLIC CAPITAL LETTER SHORT U
    '\u045e'   #  0xA2 -> CYRILLIC SMALL LETTER SHORT U
    '\u0408'   #  0xA3 -> CYRILLIC CAPITAL LETTER JE
    '\xa4'     #  0xA4 -> CURRENCY SIGN
    '\u0490'   #  0xA5 -> CYRILLIC CAPITAL LETTER GHE WITH UPTURN
    '\xa6'     #  0xA6 -> BROKEN BAR
    '\xa7'     #  0xA7 -> SECTION SIGN
    '\u0401'   #  0xA8 -> CYRILLIC CAPITAL LETTER IO
    '\xa9'     #  0xA9 -> COPYRIGHT SIGN
    '\u0404'   #  0xAA -> CYRILLIC CAPITAL LETTER UKRAINIAN IE
    '\xab'     #  0xAB -> LEFT-POINTING DOUBLE ANGLE QUOTATION MARK
    '\xac'     #  0xAC -> NOT SIGN
    '\xad'     #  0xAD -> SOFT HYPHEN
    '\xae'     #  0xAE -> REGISTERED SIGN
    '\u0407'   #  0xAF -> CYRILLIC CAPITAL LETTER YI
    '\xb0'     #  0xB0 -> DEGREE SIGN
    '\xb1'     #  0xB1 -> PLUS-MINUS SIGN
    '\u0406'   #  0xB2 -> CYRILLIC CAPITAL LETTER BYELORUSSIAN-UKRAINIAN I
    '\u0456'   #  0xB3 -> CYRILLIC SMALL LETTER BYELORUSSIAN-UKRAINIAN I
    '\u0491'   #  0xB4 -> CYRILLIC SMALL LETTER GHE WITH UPTURN
    '\xb5'     #  0xB5 -> MICRO SIGN
    '\xb6'     #  0xB6 -> PILCROW SIGN
    '\xb7'     #  0xB7 -> MIDDLE DOT
    '\u0451'   #  0xB8 -> CYRILLIC SMALL LETTER IO
    '\u2116'   #  0xB9 -> NUMERO SIGN
    '\u0454'   #  0xBA -> CYRILLIC SMALL LETTER UKRAINIAN IE
    '\xbb'     #  0xBB -> RIGHT-POINTING DOUBLE ANGLE QUOTATION MARK
    '\u0458'   #  0xBC -> CYRILLIC SMALL LETTER JE
    '\u0405'   #  0xBD -> CYRILLIC CAPITAL LETTER DZE
    '\u0455'   #  0xBE -> CYRILLIC SMALL LETTER DZE
    '\u0457'   #  0xBF -> CYRILLIC SMALL LETTER YI
    '\u0410'   #  0xC0 -> CYRILLIC CAPITAL LETTER A
    '\u0411'   #  0xC1 -> CYRILLIC CAPITAL LETTER BE
    '\u0412'   #  0xC2 -> CYRILLIC CAPITAL LETTER VE
    '\u0413'   #  0xC3 -> CYRILLIC CAPITAL LETTER GHE
    '\u0414'   #  0xC4 -> CYRILLIC CAPITAL LETTER DE
    '\u0415'   #  0xC5 -> CYRILLIC CAPITAL LETTER IE
    '\u0416'   #  0xC6 -> CYRILLIC CAPITAL LETTER ZHE
    '\u0417'   #  0xC7 -> CYRILLIC CAPITAL LETTER ZE
    '\u0418'   #  0xC8 -> CYRILLIC CAPITAL LETTER I
    '\u0419'   #  0xC9 -> CYRILLIC CAPITAL LETTER SHORT I
    '\u041a'   #  0xCA -> CYRILLIC CAPITAL LETTER KA
    '\u041b'   #  0xCB -> CYRILLIC CAPITAL LETTER EL
    '\u041c'   #  0xCC -> CYRILLIC CAPITAL LETTER EM
    '\u041d'   #  0xCD -> CYRILLIC CAPITAL LETTER EN
    '\u041e'   #  0xCE -> CYRILLIC CAPITAL LETTER O
    '\u041f'   #  0xCF -> CYRILLIC CAPITAL LETTER PE
    '\u0420'   #  0xD0 -> CYRILLIC CAPITAL LETTER ER
    '\u0421'   #  0xD1 -> CYRILLIC CAPITAL LETTER ES
    '\u0422'   #  0xD2 -> CYRILLIC CAPITAL LETTER TE
    '\u0423'   #  0xD3 -> CYRILLIC CAPITAL LETTER U
    '\u0424'   #  0xD4 -> CYRILLIC CAPITAL LETTER EF
    '\u0425'   #  0xD5 -> CYRILLIC CAPITAL LETTER HA
    '\u0426'   #  0xD6 -> CYRILLIC CAPITAL LETTER TSE
    '\u0427'   #  0xD7 -> CYRILLIC CAPITAL LETTER CHE
    '\u0428'   #  0xD8 -> CYRILLIC CAPITAL LETTER SHA
    '\u0429'   #  0xD9 -> CYRILLIC CAPITAL LETTER SHCHA
    '\u042a'   #  0xDA -> CYRILLIC CAPITAL LETTER HARD SIGN
    '\u042b'   #  0xDB -> CYRILLIC CAPITAL LETTER YERU
    '\u042c'   #  0xDC -> CYRILLIC CAPITAL LETTER SOFT SIGN
    '\u042d'   #  0xDD -> CYRILLIC CAPITAL LETTER E
    '\u042e'   #  0xDE -> CYRILLIC CAPITAL LETTER YU
    '\u042f'   #  0xDF -> CYRILLIC CAPITAL LETTER YA
    '\u0430'   #  0xE0 -> CYRILLIC SMALL LETTER A
    '\u0431'   #  0xE1 -> CYRILLIC SMALL LETTER BE
    '\u0432'   #  0xE2 -> CYRILLIC SMALL LETTER VE
    '\u0433'   #  0xE3 -> CYRILLIC SMALL LETTER GHE
    '\u0434'   #  0xE4 -> CYRILLIC SMALL LETTER DE
    '\u0435'   #  0xE5 -> CYRILLIC SMALL LETTER IE
    '\u0436'   #  0xE6 -> CYRILLIC SMALL LETTER ZHE
    '\u0437'   #  0xE7 -> CYRILLIC SMALL LETTER ZE
    '\u0438'   #  0xE8 -> CYRILLIC SMALL LETTER I
    '\u0439'   #  0xE9 -> CYRILLIC SMALL LETTER SHORT I
    '\u043a'   #  0xEA -> CYRILLIC SMALL LETTER KA
    '\u043b'   #  0xEB -> CYRILLIC SMALL LETTER EL
    '\u043c'   #  0xEC -> CYRILLIC SMALL LETTER EM
    '\u043d'   #  0xED -> CYRILLIC SMALL LETTER EN
    '\u043e'   #  0xEE -> CYRILLIC SMALL LETTER O
    '\u043f'   #  0xEF -> CYRILLIC SMALL LETTER PE
    '\u0440'   #  0xF0 -> CYRILLIC SMALL LETTER ER
    '\u0441'   #  0xF1 -> CYRILLIC SMALL LETTER ES
    '\u0442'   #  0xF2 -> CYRILLIC SMALL LETTER TE
    '\u0443'   #  0xF3 -> CYRILLIC SMALL LETTER U
    '\u0444'   #  0xF4 -> CYRILLIC SMALL LETTER EF
    '\u0445'   #  0xF5 -> CYRILLIC SMALL LETTER HA
    '\u0446'   #  0xF6 -> CYRILLIC SMALL LETTER TSE
    '\u0447'   #  0xF7 -> CYRILLIC SMALL LETTER CHE
    '\u0448'   #  0xF8 -> CYRILLIC SMALL LETTER SHA
    '\u0449'   #  0xF9 -> CYRILLIC SMALL LETTER SHCHA
    '\u044a'   #  0xFA -> CYRILLIC SMALL LETTER HARD SIGN
    '\u044b'   #  0xFB -> CYRILLIC SMALL LETTER YERU
    '\u044c'   #  0xFC -> CYRILLIC SMALL LETTER SOFT SIGN
    '\u044d'   #  0xFD -> CYRILLIC SMALL LETTER E
    '\u044e'   #  0xFE -> CYRILLIC SMALL LETTER YU
    '\u044f'   #  0xFF -> CYRILLIC SMALL LETTER YA
)

### Encoding table
encoding_table=codecs.charmap_build(decoding_table)
