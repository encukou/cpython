""" Python Character Mapping Codec iso8859_8 generated from 'MAPPINGS/ISO8859/8859-8.TXT' with gencodec.py.

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
        name='iso8859-8',
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
    '\ufffe'
    '\xa2'     #  0xA2 -> CENT SIGN
    '\xa3'     #  0xA3 -> POUND SIGN
    '\xa4'     #  0xA4 -> CURRENCY SIGN
    '\xa5'     #  0xA5 -> YEN SIGN
    '\xa6'     #  0xA6 -> BROKEN BAR
    '\xa7'     #  0xA7 -> SECTION SIGN
    '\xa8'     #  0xA8 -> DIAERESIS
    '\xa9'     #  0xA9 -> COPYRIGHT SIGN
    '\xd7'     #  0xAA -> MULTIPLICATION SIGN
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
    '\xf7'     #  0xBA -> DIVISION SIGN
    '\xbb'     #  0xBB -> RIGHT-POINTING DOUBLE ANGLE QUOTATION MARK
    '\xbc'     #  0xBC -> VULGAR FRACTION ONE QUARTER
    '\xbd'     #  0xBD -> VULGAR FRACTION ONE HALF
    '\xbe'     #  0xBE -> VULGAR FRACTION THREE QUARTERS
    '\ufffe'
    '\ufffe'
    '\ufffe'
    '\ufffe'
    '\ufffe'
    '\ufffe'
    '\ufffe'
    '\ufffe'
    '\ufffe'
    '\ufffe'
    '\ufffe'
    '\ufffe'
    '\ufffe'
    '\ufffe'
    '\ufffe'
    '\ufffe'
    '\ufffe'
    '\ufffe'
    '\ufffe'
    '\ufffe'
    '\ufffe'
    '\ufffe'
    '\ufffe'
    '\ufffe'
    '\ufffe'
    '\ufffe'
    '\ufffe'
    '\ufffe'
    '\ufffe'
    '\ufffe'
    '\ufffe'
    '\ufffe'
    '\u2017'   #  0xDF -> DOUBLE LOW LINE
    '\u05d0'   #  0xE0 -> HEBREW LETTER ALEF
    '\u05d1'   #  0xE1 -> HEBREW LETTER BET
    '\u05d2'   #  0xE2 -> HEBREW LETTER GIMEL
    '\u05d3'   #  0xE3 -> HEBREW LETTER DALET
    '\u05d4'   #  0xE4 -> HEBREW LETTER HE
    '\u05d5'   #  0xE5 -> HEBREW LETTER VAV
    '\u05d6'   #  0xE6 -> HEBREW LETTER ZAYIN
    '\u05d7'   #  0xE7 -> HEBREW LETTER HET
    '\u05d8'   #  0xE8 -> HEBREW LETTER TET
    '\u05d9'   #  0xE9 -> HEBREW LETTER YOD
    '\u05da'   #  0xEA -> HEBREW LETTER FINAL KAF
    '\u05db'   #  0xEB -> HEBREW LETTER KAF
    '\u05dc'   #  0xEC -> HEBREW LETTER LAMED
    '\u05dd'   #  0xED -> HEBREW LETTER FINAL MEM
    '\u05de'   #  0xEE -> HEBREW LETTER MEM
    '\u05df'   #  0xEF -> HEBREW LETTER FINAL NUN
    '\u05e0'   #  0xF0 -> HEBREW LETTER NUN
    '\u05e1'   #  0xF1 -> HEBREW LETTER SAMEKH
    '\u05e2'   #  0xF2 -> HEBREW LETTER AYIN
    '\u05e3'   #  0xF3 -> HEBREW LETTER FINAL PE
    '\u05e4'   #  0xF4 -> HEBREW LETTER PE
    '\u05e5'   #  0xF5 -> HEBREW LETTER FINAL TSADI
    '\u05e6'   #  0xF6 -> HEBREW LETTER TSADI
    '\u05e7'   #  0xF7 -> HEBREW LETTER QOF
    '\u05e8'   #  0xF8 -> HEBREW LETTER RESH
    '\u05e9'   #  0xF9 -> HEBREW LETTER SHIN
    '\u05ea'   #  0xFA -> HEBREW LETTER TAV
    '\ufffe'
    '\ufffe'
    '\u200e'   #  0xFD -> LEFT-TO-RIGHT MARK
    '\u200f'   #  0xFE -> RIGHT-TO-LEFT MARK
    '\ufffe'
)

### Encoding table
encoding_table=codecs.charmap_build(decoding_table)
