""" Python Character Mapping Codec cp1006 generated from 'MAPPINGS/VENDORS/MISC/CP1006.TXT' with gencodec.py.

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
        name='cp1006',
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
    '\u06f0'   #  0xA1 -> EXTENDED ARABIC-INDIC DIGIT ZERO
    '\u06f1'   #  0xA2 -> EXTENDED ARABIC-INDIC DIGIT ONE
    '\u06f2'   #  0xA3 -> EXTENDED ARABIC-INDIC DIGIT TWO
    '\u06f3'   #  0xA4 -> EXTENDED ARABIC-INDIC DIGIT THREE
    '\u06f4'   #  0xA5 -> EXTENDED ARABIC-INDIC DIGIT FOUR
    '\u06f5'   #  0xA6 -> EXTENDED ARABIC-INDIC DIGIT FIVE
    '\u06f6'   #  0xA7 -> EXTENDED ARABIC-INDIC DIGIT SIX
    '\u06f7'   #  0xA8 -> EXTENDED ARABIC-INDIC DIGIT SEVEN
    '\u06f8'   #  0xA9 -> EXTENDED ARABIC-INDIC DIGIT EIGHT
    '\u06f9'   #  0xAA -> EXTENDED ARABIC-INDIC DIGIT NINE
    '\u060c'   #  0xAB -> ARABIC COMMA
    '\u061b'   #  0xAC -> ARABIC SEMICOLON
    '\xad'     #  0xAD -> SOFT HYPHEN
    '\u061f'   #  0xAE -> ARABIC QUESTION MARK
    '\ufe81'   #  0xAF -> ARABIC LETTER ALEF WITH MADDA ABOVE ISOLATED FORM
    '\ufe8d'   #  0xB0 -> ARABIC LETTER ALEF ISOLATED FORM
    '\ufe8e'   #  0xB1 -> ARABIC LETTER ALEF FINAL FORM
    '\ufe8e'   #  0xB2 -> ARABIC LETTER ALEF FINAL FORM
    '\ufe8f'   #  0xB3 -> ARABIC LETTER BEH ISOLATED FORM
    '\ufe91'   #  0xB4 -> ARABIC LETTER BEH INITIAL FORM
    '\ufb56'   #  0xB5 -> ARABIC LETTER PEH ISOLATED FORM
    '\ufb58'   #  0xB6 -> ARABIC LETTER PEH INITIAL FORM
    '\ufe93'   #  0xB7 -> ARABIC LETTER TEH MARBUTA ISOLATED FORM
    '\ufe95'   #  0xB8 -> ARABIC LETTER TEH ISOLATED FORM
    '\ufe97'   #  0xB9 -> ARABIC LETTER TEH INITIAL FORM
    '\ufb66'   #  0xBA -> ARABIC LETTER TTEH ISOLATED FORM
    '\ufb68'   #  0xBB -> ARABIC LETTER TTEH INITIAL FORM
    '\ufe99'   #  0xBC -> ARABIC LETTER THEH ISOLATED FORM
    '\ufe9b'   #  0xBD -> ARABIC LETTER THEH INITIAL FORM
    '\ufe9d'   #  0xBE -> ARABIC LETTER JEEM ISOLATED FORM
    '\ufe9f'   #  0xBF -> ARABIC LETTER JEEM INITIAL FORM
    '\ufb7a'   #  0xC0 -> ARABIC LETTER TCHEH ISOLATED FORM
    '\ufb7c'   #  0xC1 -> ARABIC LETTER TCHEH INITIAL FORM
    '\ufea1'   #  0xC2 -> ARABIC LETTER HAH ISOLATED FORM
    '\ufea3'   #  0xC3 -> ARABIC LETTER HAH INITIAL FORM
    '\ufea5'   #  0xC4 -> ARABIC LETTER KHAH ISOLATED FORM
    '\ufea7'   #  0xC5 -> ARABIC LETTER KHAH INITIAL FORM
    '\ufea9'   #  0xC6 -> ARABIC LETTER DAL ISOLATED FORM
    '\ufb84'   #  0xC7 -> ARABIC LETTER DAHAL ISOLATED FORMN
    '\ufeab'   #  0xC8 -> ARABIC LETTER THAL ISOLATED FORM
    '\ufead'   #  0xC9 -> ARABIC LETTER REH ISOLATED FORM
    '\ufb8c'   #  0xCA -> ARABIC LETTER RREH ISOLATED FORM
    '\ufeaf'   #  0xCB -> ARABIC LETTER ZAIN ISOLATED FORM
    '\ufb8a'   #  0xCC -> ARABIC LETTER JEH ISOLATED FORM
    '\ufeb1'   #  0xCD -> ARABIC LETTER SEEN ISOLATED FORM
    '\ufeb3'   #  0xCE -> ARABIC LETTER SEEN INITIAL FORM
    '\ufeb5'   #  0xCF -> ARABIC LETTER SHEEN ISOLATED FORM
    '\ufeb7'   #  0xD0 -> ARABIC LETTER SHEEN INITIAL FORM
    '\ufeb9'   #  0xD1 -> ARABIC LETTER SAD ISOLATED FORM
    '\ufebb'   #  0xD2 -> ARABIC LETTER SAD INITIAL FORM
    '\ufebd'   #  0xD3 -> ARABIC LETTER DAD ISOLATED FORM
    '\ufebf'   #  0xD4 -> ARABIC LETTER DAD INITIAL FORM
    '\ufec1'   #  0xD5 -> ARABIC LETTER TAH ISOLATED FORM
    '\ufec5'   #  0xD6 -> ARABIC LETTER ZAH ISOLATED FORM
    '\ufec9'   #  0xD7 -> ARABIC LETTER AIN ISOLATED FORM
    '\ufeca'   #  0xD8 -> ARABIC LETTER AIN FINAL FORM
    '\ufecb'   #  0xD9 -> ARABIC LETTER AIN INITIAL FORM
    '\ufecc'   #  0xDA -> ARABIC LETTER AIN MEDIAL FORM
    '\ufecd'   #  0xDB -> ARABIC LETTER GHAIN ISOLATED FORM
    '\ufece'   #  0xDC -> ARABIC LETTER GHAIN FINAL FORM
    '\ufecf'   #  0xDD -> ARABIC LETTER GHAIN INITIAL FORM
    '\ufed0'   #  0xDE -> ARABIC LETTER GHAIN MEDIAL FORM
    '\ufed1'   #  0xDF -> ARABIC LETTER FEH ISOLATED FORM
    '\ufed3'   #  0xE0 -> ARABIC LETTER FEH INITIAL FORM
    '\ufed5'   #  0xE1 -> ARABIC LETTER QAF ISOLATED FORM
    '\ufed7'   #  0xE2 -> ARABIC LETTER QAF INITIAL FORM
    '\ufed9'   #  0xE3 -> ARABIC LETTER KAF ISOLATED FORM
    '\ufedb'   #  0xE4 -> ARABIC LETTER KAF INITIAL FORM
    '\ufb92'   #  0xE5 -> ARABIC LETTER GAF ISOLATED FORM
    '\ufb94'   #  0xE6 -> ARABIC LETTER GAF INITIAL FORM
    '\ufedd'   #  0xE7 -> ARABIC LETTER LAM ISOLATED FORM
    '\ufedf'   #  0xE8 -> ARABIC LETTER LAM INITIAL FORM
    '\ufee0'   #  0xE9 -> ARABIC LETTER LAM MEDIAL FORM
    '\ufee1'   #  0xEA -> ARABIC LETTER MEEM ISOLATED FORM
    '\ufee3'   #  0xEB -> ARABIC LETTER MEEM INITIAL FORM
    '\ufb9e'   #  0xEC -> ARABIC LETTER NOON GHUNNA ISOLATED FORM
    '\ufee5'   #  0xED -> ARABIC LETTER NOON ISOLATED FORM
    '\ufee7'   #  0xEE -> ARABIC LETTER NOON INITIAL FORM
    '\ufe85'   #  0xEF -> ARABIC LETTER WAW WITH HAMZA ABOVE ISOLATED FORM
    '\ufeed'   #  0xF0 -> ARABIC LETTER WAW ISOLATED FORM
    '\ufba6'   #  0xF1 -> ARABIC LETTER HEH GOAL ISOLATED FORM
    '\ufba8'   #  0xF2 -> ARABIC LETTER HEH GOAL INITIAL FORM
    '\ufba9'   #  0xF3 -> ARABIC LETTER HEH GOAL MEDIAL FORM
    '\ufbaa'   #  0xF4 -> ARABIC LETTER HEH DOACHASHMEE ISOLATED FORM
    '\ufe80'   #  0xF5 -> ARABIC LETTER HAMZA ISOLATED FORM
    '\ufe89'   #  0xF6 -> ARABIC LETTER YEH WITH HAMZA ABOVE ISOLATED FORM
    '\ufe8a'   #  0xF7 -> ARABIC LETTER YEH WITH HAMZA ABOVE FINAL FORM
    '\ufe8b'   #  0xF8 -> ARABIC LETTER YEH WITH HAMZA ABOVE INITIAL FORM
    '\ufef1'   #  0xF9 -> ARABIC LETTER YEH ISOLATED FORM
    '\ufef2'   #  0xFA -> ARABIC LETTER YEH FINAL FORM
    '\ufef3'   #  0xFB -> ARABIC LETTER YEH INITIAL FORM
    '\ufbb0'   #  0xFC -> ARABIC LETTER YEH BARREE WITH HAMZA ABOVE ISOLATED FORM
    '\ufbae'   #  0xFD -> ARABIC LETTER YEH BARREE ISOLATED FORM
    '\ufe7c'   #  0xFE -> ARABIC SHADDA ISOLATED FORM
    '\ufe7d'   #  0xFF -> ARABIC SHADDA MEDIAL FORM
)

### Encoding table
encoding_table=codecs.charmap_build(decoding_table)
