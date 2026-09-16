""" Python Character Mapping Codec mac_cyrillic generated from 'MAPPINGS/VENDORS/APPLE/CYRILLIC.TXT' with gencodec.py.

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
        name='mac-cyrillic',
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
    '\u0410'   #  0x80 -> CYRILLIC CAPITAL LETTER A
    '\u0411'   #  0x81 -> CYRILLIC CAPITAL LETTER BE
    '\u0412'   #  0x82 -> CYRILLIC CAPITAL LETTER VE
    '\u0413'   #  0x83 -> CYRILLIC CAPITAL LETTER GHE
    '\u0414'   #  0x84 -> CYRILLIC CAPITAL LETTER DE
    '\u0415'   #  0x85 -> CYRILLIC CAPITAL LETTER IE
    '\u0416'   #  0x86 -> CYRILLIC CAPITAL LETTER ZHE
    '\u0417'   #  0x87 -> CYRILLIC CAPITAL LETTER ZE
    '\u0418'   #  0x88 -> CYRILLIC CAPITAL LETTER I
    '\u0419'   #  0x89 -> CYRILLIC CAPITAL LETTER SHORT I
    '\u041a'   #  0x8A -> CYRILLIC CAPITAL LETTER KA
    '\u041b'   #  0x8B -> CYRILLIC CAPITAL LETTER EL
    '\u041c'   #  0x8C -> CYRILLIC CAPITAL LETTER EM
    '\u041d'   #  0x8D -> CYRILLIC CAPITAL LETTER EN
    '\u041e'   #  0x8E -> CYRILLIC CAPITAL LETTER O
    '\u041f'   #  0x8F -> CYRILLIC CAPITAL LETTER PE
    '\u0420'   #  0x90 -> CYRILLIC CAPITAL LETTER ER
    '\u0421'   #  0x91 -> CYRILLIC CAPITAL LETTER ES
    '\u0422'   #  0x92 -> CYRILLIC CAPITAL LETTER TE
    '\u0423'   #  0x93 -> CYRILLIC CAPITAL LETTER U
    '\u0424'   #  0x94 -> CYRILLIC CAPITAL LETTER EF
    '\u0425'   #  0x95 -> CYRILLIC CAPITAL LETTER HA
    '\u0426'   #  0x96 -> CYRILLIC CAPITAL LETTER TSE
    '\u0427'   #  0x97 -> CYRILLIC CAPITAL LETTER CHE
    '\u0428'   #  0x98 -> CYRILLIC CAPITAL LETTER SHA
    '\u0429'   #  0x99 -> CYRILLIC CAPITAL LETTER SHCHA
    '\u042a'   #  0x9A -> CYRILLIC CAPITAL LETTER HARD SIGN
    '\u042b'   #  0x9B -> CYRILLIC CAPITAL LETTER YERU
    '\u042c'   #  0x9C -> CYRILLIC CAPITAL LETTER SOFT SIGN
    '\u042d'   #  0x9D -> CYRILLIC CAPITAL LETTER E
    '\u042e'   #  0x9E -> CYRILLIC CAPITAL LETTER YU
    '\u042f'   #  0x9F -> CYRILLIC CAPITAL LETTER YA
    '\u2020'   #  0xA0 -> DAGGER
    '\xb0'     #  0xA1 -> DEGREE SIGN
    '\u0490'   #  0xA2 -> CYRILLIC CAPITAL LETTER GHE WITH UPTURN
    '\xa3'     #  0xA3 -> POUND SIGN
    '\xa7'     #  0xA4 -> SECTION SIGN
    '\u2022'   #  0xA5 -> BULLET
    '\xb6'     #  0xA6 -> PILCROW SIGN
    '\u0406'   #  0xA7 -> CYRILLIC CAPITAL LETTER BYELORUSSIAN-UKRAINIAN I
    '\xae'     #  0xA8 -> REGISTERED SIGN
    '\xa9'     #  0xA9 -> COPYRIGHT SIGN
    '\u2122'   #  0xAA -> TRADE MARK SIGN
    '\u0402'   #  0xAB -> CYRILLIC CAPITAL LETTER DJE
    '\u0452'   #  0xAC -> CYRILLIC SMALL LETTER DJE
    '\u2260'   #  0xAD -> NOT EQUAL TO
    '\u0403'   #  0xAE -> CYRILLIC CAPITAL LETTER GJE
    '\u0453'   #  0xAF -> CYRILLIC SMALL LETTER GJE
    '\u221e'   #  0xB0 -> INFINITY
    '\xb1'     #  0xB1 -> PLUS-MINUS SIGN
    '\u2264'   #  0xB2 -> LESS-THAN OR EQUAL TO
    '\u2265'   #  0xB3 -> GREATER-THAN OR EQUAL TO
    '\u0456'   #  0xB4 -> CYRILLIC SMALL LETTER BYELORUSSIAN-UKRAINIAN I
    '\xb5'     #  0xB5 -> MICRO SIGN
    '\u0491'   #  0xB6 -> CYRILLIC SMALL LETTER GHE WITH UPTURN
    '\u0408'   #  0xB7 -> CYRILLIC CAPITAL LETTER JE
    '\u0404'   #  0xB8 -> CYRILLIC CAPITAL LETTER UKRAINIAN IE
    '\u0454'   #  0xB9 -> CYRILLIC SMALL LETTER UKRAINIAN IE
    '\u0407'   #  0xBA -> CYRILLIC CAPITAL LETTER YI
    '\u0457'   #  0xBB -> CYRILLIC SMALL LETTER YI
    '\u0409'   #  0xBC -> CYRILLIC CAPITAL LETTER LJE
    '\u0459'   #  0xBD -> CYRILLIC SMALL LETTER LJE
    '\u040a'   #  0xBE -> CYRILLIC CAPITAL LETTER NJE
    '\u045a'   #  0xBF -> CYRILLIC SMALL LETTER NJE
    '\u0458'   #  0xC0 -> CYRILLIC SMALL LETTER JE
    '\u0405'   #  0xC1 -> CYRILLIC CAPITAL LETTER DZE
    '\xac'     #  0xC2 -> NOT SIGN
    '\u221a'   #  0xC3 -> SQUARE ROOT
    '\u0192'   #  0xC4 -> LATIN SMALL LETTER F WITH HOOK
    '\u2248'   #  0xC5 -> ALMOST EQUAL TO
    '\u2206'   #  0xC6 -> INCREMENT
    '\xab'     #  0xC7 -> LEFT-POINTING DOUBLE ANGLE QUOTATION MARK
    '\xbb'     #  0xC8 -> RIGHT-POINTING DOUBLE ANGLE QUOTATION MARK
    '\u2026'   #  0xC9 -> HORIZONTAL ELLIPSIS
    '\xa0'     #  0xCA -> NO-BREAK SPACE
    '\u040b'   #  0xCB -> CYRILLIC CAPITAL LETTER TSHE
    '\u045b'   #  0xCC -> CYRILLIC SMALL LETTER TSHE
    '\u040c'   #  0xCD -> CYRILLIC CAPITAL LETTER KJE
    '\u045c'   #  0xCE -> CYRILLIC SMALL LETTER KJE
    '\u0455'   #  0xCF -> CYRILLIC SMALL LETTER DZE
    '\u2013'   #  0xD0 -> EN DASH
    '\u2014'   #  0xD1 -> EM DASH
    '\u201c'   #  0xD2 -> LEFT DOUBLE QUOTATION MARK
    '\u201d'   #  0xD3 -> RIGHT DOUBLE QUOTATION MARK
    '\u2018'   #  0xD4 -> LEFT SINGLE QUOTATION MARK
    '\u2019'   #  0xD5 -> RIGHT SINGLE QUOTATION MARK
    '\xf7'     #  0xD6 -> DIVISION SIGN
    '\u201e'   #  0xD7 -> DOUBLE LOW-9 QUOTATION MARK
    '\u040e'   #  0xD8 -> CYRILLIC CAPITAL LETTER SHORT U
    '\u045e'   #  0xD9 -> CYRILLIC SMALL LETTER SHORT U
    '\u040f'   #  0xDA -> CYRILLIC CAPITAL LETTER DZHE
    '\u045f'   #  0xDB -> CYRILLIC SMALL LETTER DZHE
    '\u2116'   #  0xDC -> NUMERO SIGN
    '\u0401'   #  0xDD -> CYRILLIC CAPITAL LETTER IO
    '\u0451'   #  0xDE -> CYRILLIC SMALL LETTER IO
    '\u044f'   #  0xDF -> CYRILLIC SMALL LETTER YA
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
    '\u20ac'   #  0xFF -> EURO SIGN
)

### Encoding table
encoding_table=codecs.charmap_build(decoding_table)
