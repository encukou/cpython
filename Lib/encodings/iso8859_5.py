""" Python Character Mapping Codec iso8859_5 generated from 'MAPPINGS/ISO8859/8859-5.TXT' with gencodec.py.

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
        name='iso8859-5',
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
    '\u0401'   #  0xA1 -> CYRILLIC CAPITAL LETTER IO
    '\u0402'   #  0xA2 -> CYRILLIC CAPITAL LETTER DJE
    '\u0403'   #  0xA3 -> CYRILLIC CAPITAL LETTER GJE
    '\u0404'   #  0xA4 -> CYRILLIC CAPITAL LETTER UKRAINIAN IE
    '\u0405'   #  0xA5 -> CYRILLIC CAPITAL LETTER DZE
    '\u0406'   #  0xA6 -> CYRILLIC CAPITAL LETTER BYELORUSSIAN-UKRAINIAN I
    '\u0407'   #  0xA7 -> CYRILLIC CAPITAL LETTER YI
    '\u0408'   #  0xA8 -> CYRILLIC CAPITAL LETTER JE
    '\u0409'   #  0xA9 -> CYRILLIC CAPITAL LETTER LJE
    '\u040a'   #  0xAA -> CYRILLIC CAPITAL LETTER NJE
    '\u040b'   #  0xAB -> CYRILLIC CAPITAL LETTER TSHE
    '\u040c'   #  0xAC -> CYRILLIC CAPITAL LETTER KJE
    '\xad'     #  0xAD -> SOFT HYPHEN
    '\u040e'   #  0xAE -> CYRILLIC CAPITAL LETTER SHORT U
    '\u040f'   #  0xAF -> CYRILLIC CAPITAL LETTER DZHE
    '\u0410'   #  0xB0 -> CYRILLIC CAPITAL LETTER A
    '\u0411'   #  0xB1 -> CYRILLIC CAPITAL LETTER BE
    '\u0412'   #  0xB2 -> CYRILLIC CAPITAL LETTER VE
    '\u0413'   #  0xB3 -> CYRILLIC CAPITAL LETTER GHE
    '\u0414'   #  0xB4 -> CYRILLIC CAPITAL LETTER DE
    '\u0415'   #  0xB5 -> CYRILLIC CAPITAL LETTER IE
    '\u0416'   #  0xB6 -> CYRILLIC CAPITAL LETTER ZHE
    '\u0417'   #  0xB7 -> CYRILLIC CAPITAL LETTER ZE
    '\u0418'   #  0xB8 -> CYRILLIC CAPITAL LETTER I
    '\u0419'   #  0xB9 -> CYRILLIC CAPITAL LETTER SHORT I
    '\u041a'   #  0xBA -> CYRILLIC CAPITAL LETTER KA
    '\u041b'   #  0xBB -> CYRILLIC CAPITAL LETTER EL
    '\u041c'   #  0xBC -> CYRILLIC CAPITAL LETTER EM
    '\u041d'   #  0xBD -> CYRILLIC CAPITAL LETTER EN
    '\u041e'   #  0xBE -> CYRILLIC CAPITAL LETTER O
    '\u041f'   #  0xBF -> CYRILLIC CAPITAL LETTER PE
    '\u0420'   #  0xC0 -> CYRILLIC CAPITAL LETTER ER
    '\u0421'   #  0xC1 -> CYRILLIC CAPITAL LETTER ES
    '\u0422'   #  0xC2 -> CYRILLIC CAPITAL LETTER TE
    '\u0423'   #  0xC3 -> CYRILLIC CAPITAL LETTER U
    '\u0424'   #  0xC4 -> CYRILLIC CAPITAL LETTER EF
    '\u0425'   #  0xC5 -> CYRILLIC CAPITAL LETTER HA
    '\u0426'   #  0xC6 -> CYRILLIC CAPITAL LETTER TSE
    '\u0427'   #  0xC7 -> CYRILLIC CAPITAL LETTER CHE
    '\u0428'   #  0xC8 -> CYRILLIC CAPITAL LETTER SHA
    '\u0429'   #  0xC9 -> CYRILLIC CAPITAL LETTER SHCHA
    '\u042a'   #  0xCA -> CYRILLIC CAPITAL LETTER HARD SIGN
    '\u042b'   #  0xCB -> CYRILLIC CAPITAL LETTER YERU
    '\u042c'   #  0xCC -> CYRILLIC CAPITAL LETTER SOFT SIGN
    '\u042d'   #  0xCD -> CYRILLIC CAPITAL LETTER E
    '\u042e'   #  0xCE -> CYRILLIC CAPITAL LETTER YU
    '\u042f'   #  0xCF -> CYRILLIC CAPITAL LETTER YA
    '\u0430'   #  0xD0 -> CYRILLIC SMALL LETTER A
    '\u0431'   #  0xD1 -> CYRILLIC SMALL LETTER BE
    '\u0432'   #  0xD2 -> CYRILLIC SMALL LETTER VE
    '\u0433'   #  0xD3 -> CYRILLIC SMALL LETTER GHE
    '\u0434'   #  0xD4 -> CYRILLIC SMALL LETTER DE
    '\u0435'   #  0xD5 -> CYRILLIC SMALL LETTER IE
    '\u0436'   #  0xD6 -> CYRILLIC SMALL LETTER ZHE
    '\u0437'   #  0xD7 -> CYRILLIC SMALL LETTER ZE
    '\u0438'   #  0xD8 -> CYRILLIC SMALL LETTER I
    '\u0439'   #  0xD9 -> CYRILLIC SMALL LETTER SHORT I
    '\u043a'   #  0xDA -> CYRILLIC SMALL LETTER KA
    '\u043b'   #  0xDB -> CYRILLIC SMALL LETTER EL
    '\u043c'   #  0xDC -> CYRILLIC SMALL LETTER EM
    '\u043d'   #  0xDD -> CYRILLIC SMALL LETTER EN
    '\u043e'   #  0xDE -> CYRILLIC SMALL LETTER O
    '\u043f'   #  0xDF -> CYRILLIC SMALL LETTER PE
    '\u0440'   #  0xE0 -> CYRILLIC SMALL LETTER ER
    '\u0441'   #  0xE1 -> CYRILLIC SMALL LETTER ES
    '\u0442'   #  0xE2 -> CYRILLIC SMALL LETTER TE
    '\u0443'   #  0xE3 -> CYRILLIC SMALL LETTER U
    '\u0444'   #  0xE4 -> CYRILLIC SMALL LETTER EF
    '\u0445'   #  0xE5 -> CYRILLIC SMALL LETTER HA
    '\u0446'   #  0xE6 -> CYRILLIC SMALL LETTER TSE
    '\u0447'   #  0xE7 -> CYRILLIC SMALL LETTER CHE
    '\u0448'   #  0xE8 -> CYRILLIC SMALL LETTER SHA
    '\u0449'   #  0xE9 -> CYRILLIC SMALL LETTER SHCHA
    '\u044a'   #  0xEA -> CYRILLIC SMALL LETTER HARD SIGN
    '\u044b'   #  0xEB -> CYRILLIC SMALL LETTER YERU
    '\u044c'   #  0xEC -> CYRILLIC SMALL LETTER SOFT SIGN
    '\u044d'   #  0xED -> CYRILLIC SMALL LETTER E
    '\u044e'   #  0xEE -> CYRILLIC SMALL LETTER YU
    '\u044f'   #  0xEF -> CYRILLIC SMALL LETTER YA
    '\u2116'   #  0xF0 -> NUMERO SIGN
    '\u0451'   #  0xF1 -> CYRILLIC SMALL LETTER IO
    '\u0452'   #  0xF2 -> CYRILLIC SMALL LETTER DJE
    '\u0453'   #  0xF3 -> CYRILLIC SMALL LETTER GJE
    '\u0454'   #  0xF4 -> CYRILLIC SMALL LETTER UKRAINIAN IE
    '\u0455'   #  0xF5 -> CYRILLIC SMALL LETTER DZE
    '\u0456'   #  0xF6 -> CYRILLIC SMALL LETTER BYELORUSSIAN-UKRAINIAN I
    '\u0457'   #  0xF7 -> CYRILLIC SMALL LETTER YI
    '\u0458'   #  0xF8 -> CYRILLIC SMALL LETTER JE
    '\u0459'   #  0xF9 -> CYRILLIC SMALL LETTER LJE
    '\u045a'   #  0xFA -> CYRILLIC SMALL LETTER NJE
    '\u045b'   #  0xFB -> CYRILLIC SMALL LETTER TSHE
    '\u045c'   #  0xFC -> CYRILLIC SMALL LETTER KJE
    '\xa7'     #  0xFD -> SECTION SIGN
    '\u045e'   #  0xFE -> CYRILLIC SMALL LETTER SHORT U
    '\u045f'   #  0xFF -> CYRILLIC SMALL LETTER DZHE
)

### Encoding table
encoding_table=codecs.charmap_build(decoding_table)
