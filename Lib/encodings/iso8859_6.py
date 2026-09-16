""" Python Character Mapping Codec iso8859_6 generated from 'MAPPINGS/ISO8859/8859-6.TXT' with gencodec.py.

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
        name='iso8859-6',
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
    '\ufffe'
    '\ufffe'
    '\xa4'     #  0xA4 -> CURRENCY SIGN
    '\ufffe'
    '\ufffe'
    '\ufffe'
    '\ufffe'
    '\ufffe'
    '\ufffe'
    '\ufffe'
    '\u060c'   #  0xAC -> ARABIC COMMA
    '\xad'     #  0xAD -> SOFT HYPHEN
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
    '\u061b'   #  0xBB -> ARABIC SEMICOLON
    '\ufffe'
    '\ufffe'
    '\ufffe'
    '\u061f'   #  0xBF -> ARABIC QUESTION MARK
    '\ufffe'
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
    '\ufffe'
    '\ufffe'
    '\ufffe'
    '\ufffe'
    '\ufffe'
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
)

### Encoding table
encoding_table=codecs.charmap_build(decoding_table)
