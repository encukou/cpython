""" Python Character Mapping Codec cp874 generated from 'MAPPINGS/VENDORS/MICSFT/WINDOWS/CP874.TXT' with gencodec.py.

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
        name='cp874',
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
    '\ufffe'   #  0x81 -> UNDEFINED
    '\ufffe'   #  0x82 -> UNDEFINED
    '\ufffe'   #  0x83 -> UNDEFINED
    '\ufffe'   #  0x84 -> UNDEFINED
    '\u2026'   #  0x85 -> HORIZONTAL ELLIPSIS
    '\ufffe'   #  0x86 -> UNDEFINED
    '\ufffe'   #  0x87 -> UNDEFINED
    '\ufffe'   #  0x88 -> UNDEFINED
    '\ufffe'   #  0x89 -> UNDEFINED
    '\ufffe'   #  0x8A -> UNDEFINED
    '\ufffe'   #  0x8B -> UNDEFINED
    '\ufffe'   #  0x8C -> UNDEFINED
    '\ufffe'   #  0x8D -> UNDEFINED
    '\ufffe'   #  0x8E -> UNDEFINED
    '\ufffe'   #  0x8F -> UNDEFINED
    '\ufffe'   #  0x90 -> UNDEFINED
    '\u2018'   #  0x91 -> LEFT SINGLE QUOTATION MARK
    '\u2019'   #  0x92 -> RIGHT SINGLE QUOTATION MARK
    '\u201c'   #  0x93 -> LEFT DOUBLE QUOTATION MARK
    '\u201d'   #  0x94 -> RIGHT DOUBLE QUOTATION MARK
    '\u2022'   #  0x95 -> BULLET
    '\u2013'   #  0x96 -> EN DASH
    '\u2014'   #  0x97 -> EM DASH
    '\ufffe'   #  0x98 -> UNDEFINED
    '\ufffe'   #  0x99 -> UNDEFINED
    '\ufffe'   #  0x9A -> UNDEFINED
    '\ufffe'   #  0x9B -> UNDEFINED
    '\ufffe'   #  0x9C -> UNDEFINED
    '\ufffe'   #  0x9D -> UNDEFINED
    '\ufffe'   #  0x9E -> UNDEFINED
    '\ufffe'   #  0x9F -> UNDEFINED
    '\xa0'     #  0xA0 -> NO-BREAK SPACE
    '\u0e01'   #  0xA1 -> THAI CHARACTER KO KAI
    '\u0e02'   #  0xA2 -> THAI CHARACTER KHO KHAI
    '\u0e03'   #  0xA3 -> THAI CHARACTER KHO KHUAT
    '\u0e04'   #  0xA4 -> THAI CHARACTER KHO KHWAI
    '\u0e05'   #  0xA5 -> THAI CHARACTER KHO KHON
    '\u0e06'   #  0xA6 -> THAI CHARACTER KHO RAKHANG
    '\u0e07'   #  0xA7 -> THAI CHARACTER NGO NGU
    '\u0e08'   #  0xA8 -> THAI CHARACTER CHO CHAN
    '\u0e09'   #  0xA9 -> THAI CHARACTER CHO CHING
    '\u0e0a'   #  0xAA -> THAI CHARACTER CHO CHANG
    '\u0e0b'   #  0xAB -> THAI CHARACTER SO SO
    '\u0e0c'   #  0xAC -> THAI CHARACTER CHO CHOE
    '\u0e0d'   #  0xAD -> THAI CHARACTER YO YING
    '\u0e0e'   #  0xAE -> THAI CHARACTER DO CHADA
    '\u0e0f'   #  0xAF -> THAI CHARACTER TO PATAK
    '\u0e10'   #  0xB0 -> THAI CHARACTER THO THAN
    '\u0e11'   #  0xB1 -> THAI CHARACTER THO NANGMONTHO
    '\u0e12'   #  0xB2 -> THAI CHARACTER THO PHUTHAO
    '\u0e13'   #  0xB3 -> THAI CHARACTER NO NEN
    '\u0e14'   #  0xB4 -> THAI CHARACTER DO DEK
    '\u0e15'   #  0xB5 -> THAI CHARACTER TO TAO
    '\u0e16'   #  0xB6 -> THAI CHARACTER THO THUNG
    '\u0e17'   #  0xB7 -> THAI CHARACTER THO THAHAN
    '\u0e18'   #  0xB8 -> THAI CHARACTER THO THONG
    '\u0e19'   #  0xB9 -> THAI CHARACTER NO NU
    '\u0e1a'   #  0xBA -> THAI CHARACTER BO BAIMAI
    '\u0e1b'   #  0xBB -> THAI CHARACTER PO PLA
    '\u0e1c'   #  0xBC -> THAI CHARACTER PHO PHUNG
    '\u0e1d'   #  0xBD -> THAI CHARACTER FO FA
    '\u0e1e'   #  0xBE -> THAI CHARACTER PHO PHAN
    '\u0e1f'   #  0xBF -> THAI CHARACTER FO FAN
    '\u0e20'   #  0xC0 -> THAI CHARACTER PHO SAMPHAO
    '\u0e21'   #  0xC1 -> THAI CHARACTER MO MA
    '\u0e22'   #  0xC2 -> THAI CHARACTER YO YAK
    '\u0e23'   #  0xC3 -> THAI CHARACTER RO RUA
    '\u0e24'   #  0xC4 -> THAI CHARACTER RU
    '\u0e25'   #  0xC5 -> THAI CHARACTER LO LING
    '\u0e26'   #  0xC6 -> THAI CHARACTER LU
    '\u0e27'   #  0xC7 -> THAI CHARACTER WO WAEN
    '\u0e28'   #  0xC8 -> THAI CHARACTER SO SALA
    '\u0e29'   #  0xC9 -> THAI CHARACTER SO RUSI
    '\u0e2a'   #  0xCA -> THAI CHARACTER SO SUA
    '\u0e2b'   #  0xCB -> THAI CHARACTER HO HIP
    '\u0e2c'   #  0xCC -> THAI CHARACTER LO CHULA
    '\u0e2d'   #  0xCD -> THAI CHARACTER O ANG
    '\u0e2e'   #  0xCE -> THAI CHARACTER HO NOKHUK
    '\u0e2f'   #  0xCF -> THAI CHARACTER PAIYANNOI
    '\u0e30'   #  0xD0 -> THAI CHARACTER SARA A
    '\u0e31'   #  0xD1 -> THAI CHARACTER MAI HAN-AKAT
    '\u0e32'   #  0xD2 -> THAI CHARACTER SARA AA
    '\u0e33'   #  0xD3 -> THAI CHARACTER SARA AM
    '\u0e34'   #  0xD4 -> THAI CHARACTER SARA I
    '\u0e35'   #  0xD5 -> THAI CHARACTER SARA II
    '\u0e36'   #  0xD6 -> THAI CHARACTER SARA UE
    '\u0e37'   #  0xD7 -> THAI CHARACTER SARA UEE
    '\u0e38'   #  0xD8 -> THAI CHARACTER SARA U
    '\u0e39'   #  0xD9 -> THAI CHARACTER SARA UU
    '\u0e3a'   #  0xDA -> THAI CHARACTER PHINTHU
    '\ufffe'   #  0xDB -> UNDEFINED
    '\ufffe'   #  0xDC -> UNDEFINED
    '\ufffe'   #  0xDD -> UNDEFINED
    '\ufffe'   #  0xDE -> UNDEFINED
    '\u0e3f'   #  0xDF -> THAI CURRENCY SYMBOL BAHT
    '\u0e40'   #  0xE0 -> THAI CHARACTER SARA E
    '\u0e41'   #  0xE1 -> THAI CHARACTER SARA AE
    '\u0e42'   #  0xE2 -> THAI CHARACTER SARA O
    '\u0e43'   #  0xE3 -> THAI CHARACTER SARA AI MAIMUAN
    '\u0e44'   #  0xE4 -> THAI CHARACTER SARA AI MAIMALAI
    '\u0e45'   #  0xE5 -> THAI CHARACTER LAKKHANGYAO
    '\u0e46'   #  0xE6 -> THAI CHARACTER MAIYAMOK
    '\u0e47'   #  0xE7 -> THAI CHARACTER MAITAIKHU
    '\u0e48'   #  0xE8 -> THAI CHARACTER MAI EK
    '\u0e49'   #  0xE9 -> THAI CHARACTER MAI THO
    '\u0e4a'   #  0xEA -> THAI CHARACTER MAI TRI
    '\u0e4b'   #  0xEB -> THAI CHARACTER MAI CHATTAWA
    '\u0e4c'   #  0xEC -> THAI CHARACTER THANTHAKHAT
    '\u0e4d'   #  0xED -> THAI CHARACTER NIKHAHIT
    '\u0e4e'   #  0xEE -> THAI CHARACTER YAMAKKAN
    '\u0e4f'   #  0xEF -> THAI CHARACTER FONGMAN
    '\u0e50'   #  0xF0 -> THAI DIGIT ZERO
    '\u0e51'   #  0xF1 -> THAI DIGIT ONE
    '\u0e52'   #  0xF2 -> THAI DIGIT TWO
    '\u0e53'   #  0xF3 -> THAI DIGIT THREE
    '\u0e54'   #  0xF4 -> THAI DIGIT FOUR
    '\u0e55'   #  0xF5 -> THAI DIGIT FIVE
    '\u0e56'   #  0xF6 -> THAI DIGIT SIX
    '\u0e57'   #  0xF7 -> THAI DIGIT SEVEN
    '\u0e58'   #  0xF8 -> THAI DIGIT EIGHT
    '\u0e59'   #  0xF9 -> THAI DIGIT NINE
    '\u0e5a'   #  0xFA -> THAI CHARACTER ANGKHANKHU
    '\u0e5b'   #  0xFB -> THAI CHARACTER KHOMUT
    '\ufffe'   #  0xFC -> UNDEFINED
    '\ufffe'   #  0xFD -> UNDEFINED
    '\ufffe'   #  0xFE -> UNDEFINED
    '\ufffe'   #  0xFF -> UNDEFINED
)

### Encoding table
encoding_table=codecs.charmap_build(decoding_table)
