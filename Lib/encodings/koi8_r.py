""" Python Character Mapping Codec koi8_r generated from 'MAPPINGS/VENDORS/MISC/KOI8-R.TXT' with gencodec.py.

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
        name='koi8-r',
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
    '\u2500'   #  0x80 -> BOX DRAWINGS LIGHT HORIZONTAL
    '\u2502'   #  0x81 -> BOX DRAWINGS LIGHT VERTICAL
    '\u250c'   #  0x82 -> BOX DRAWINGS LIGHT DOWN AND RIGHT
    '\u2510'   #  0x83 -> BOX DRAWINGS LIGHT DOWN AND LEFT
    '\u2514'   #  0x84 -> BOX DRAWINGS LIGHT UP AND RIGHT
    '\u2518'   #  0x85 -> BOX DRAWINGS LIGHT UP AND LEFT
    '\u251c'   #  0x86 -> BOX DRAWINGS LIGHT VERTICAL AND RIGHT
    '\u2524'   #  0x87 -> BOX DRAWINGS LIGHT VERTICAL AND LEFT
    '\u252c'   #  0x88 -> BOX DRAWINGS LIGHT DOWN AND HORIZONTAL
    '\u2534'   #  0x89 -> BOX DRAWINGS LIGHT UP AND HORIZONTAL
    '\u253c'   #  0x8A -> BOX DRAWINGS LIGHT VERTICAL AND HORIZONTAL
    '\u2580'   #  0x8B -> UPPER HALF BLOCK
    '\u2584'   #  0x8C -> LOWER HALF BLOCK
    '\u2588'   #  0x8D -> FULL BLOCK
    '\u258c'   #  0x8E -> LEFT HALF BLOCK
    '\u2590'   #  0x8F -> RIGHT HALF BLOCK
    '\u2591'   #  0x90 -> LIGHT SHADE
    '\u2592'   #  0x91 -> MEDIUM SHADE
    '\u2593'   #  0x92 -> DARK SHADE
    '\u2320'   #  0x93 -> TOP HALF INTEGRAL
    '\u25a0'   #  0x94 -> BLACK SQUARE
    '\u2219'   #  0x95 -> BULLET OPERATOR
    '\u221a'   #  0x96 -> SQUARE ROOT
    '\u2248'   #  0x97 -> ALMOST EQUAL TO
    '\u2264'   #  0x98 -> LESS-THAN OR EQUAL TO
    '\u2265'   #  0x99 -> GREATER-THAN OR EQUAL TO
    '\xa0'     #  0x9A -> NO-BREAK SPACE
    '\u2321'   #  0x9B -> BOTTOM HALF INTEGRAL
    '\xb0'     #  0x9C -> DEGREE SIGN
    '\xb2'     #  0x9D -> SUPERSCRIPT TWO
    '\xb7'     #  0x9E -> MIDDLE DOT
    '\xf7'     #  0x9F -> DIVISION SIGN
    '\u2550'   #  0xA0 -> BOX DRAWINGS DOUBLE HORIZONTAL
    '\u2551'   #  0xA1 -> BOX DRAWINGS DOUBLE VERTICAL
    '\u2552'   #  0xA2 -> BOX DRAWINGS DOWN SINGLE AND RIGHT DOUBLE
    '\u0451'   #  0xA3 -> CYRILLIC SMALL LETTER IO
    '\u2553'   #  0xA4 -> BOX DRAWINGS DOWN DOUBLE AND RIGHT SINGLE
    '\u2554'   #  0xA5 -> BOX DRAWINGS DOUBLE DOWN AND RIGHT
    '\u2555'   #  0xA6 -> BOX DRAWINGS DOWN SINGLE AND LEFT DOUBLE
    '\u2556'   #  0xA7 -> BOX DRAWINGS DOWN DOUBLE AND LEFT SINGLE
    '\u2557'   #  0xA8 -> BOX DRAWINGS DOUBLE DOWN AND LEFT
    '\u2558'   #  0xA9 -> BOX DRAWINGS UP SINGLE AND RIGHT DOUBLE
    '\u2559'   #  0xAA -> BOX DRAWINGS UP DOUBLE AND RIGHT SINGLE
    '\u255a'   #  0xAB -> BOX DRAWINGS DOUBLE UP AND RIGHT
    '\u255b'   #  0xAC -> BOX DRAWINGS UP SINGLE AND LEFT DOUBLE
    '\u255c'   #  0xAD -> BOX DRAWINGS UP DOUBLE AND LEFT SINGLE
    '\u255d'   #  0xAE -> BOX DRAWINGS DOUBLE UP AND LEFT
    '\u255e'   #  0xAF -> BOX DRAWINGS VERTICAL SINGLE AND RIGHT DOUBLE
    '\u255f'   #  0xB0 -> BOX DRAWINGS VERTICAL DOUBLE AND RIGHT SINGLE
    '\u2560'   #  0xB1 -> BOX DRAWINGS DOUBLE VERTICAL AND RIGHT
    '\u2561'   #  0xB2 -> BOX DRAWINGS VERTICAL SINGLE AND LEFT DOUBLE
    '\u0401'   #  0xB3 -> CYRILLIC CAPITAL LETTER IO
    '\u2562'   #  0xB4 -> BOX DRAWINGS VERTICAL DOUBLE AND LEFT SINGLE
    '\u2563'   #  0xB5 -> BOX DRAWINGS DOUBLE VERTICAL AND LEFT
    '\u2564'   #  0xB6 -> BOX DRAWINGS DOWN SINGLE AND HORIZONTAL DOUBLE
    '\u2565'   #  0xB7 -> BOX DRAWINGS DOWN DOUBLE AND HORIZONTAL SINGLE
    '\u2566'   #  0xB8 -> BOX DRAWINGS DOUBLE DOWN AND HORIZONTAL
    '\u2567'   #  0xB9 -> BOX DRAWINGS UP SINGLE AND HORIZONTAL DOUBLE
    '\u2568'   #  0xBA -> BOX DRAWINGS UP DOUBLE AND HORIZONTAL SINGLE
    '\u2569'   #  0xBB -> BOX DRAWINGS DOUBLE UP AND HORIZONTAL
    '\u256a'   #  0xBC -> BOX DRAWINGS VERTICAL SINGLE AND HORIZONTAL DOUBLE
    '\u256b'   #  0xBD -> BOX DRAWINGS VERTICAL DOUBLE AND HORIZONTAL SINGLE
    '\u256c'   #  0xBE -> BOX DRAWINGS DOUBLE VERTICAL AND HORIZONTAL
    '\xa9'     #  0xBF -> COPYRIGHT SIGN
    '\u044e'   #  0xC0 -> CYRILLIC SMALL LETTER YU
    '\u0430'   #  0xC1 -> CYRILLIC SMALL LETTER A
    '\u0431'   #  0xC2 -> CYRILLIC SMALL LETTER BE
    '\u0446'   #  0xC3 -> CYRILLIC SMALL LETTER TSE
    '\u0434'   #  0xC4 -> CYRILLIC SMALL LETTER DE
    '\u0435'   #  0xC5 -> CYRILLIC SMALL LETTER IE
    '\u0444'   #  0xC6 -> CYRILLIC SMALL LETTER EF
    '\u0433'   #  0xC7 -> CYRILLIC SMALL LETTER GHE
    '\u0445'   #  0xC8 -> CYRILLIC SMALL LETTER HA
    '\u0438'   #  0xC9 -> CYRILLIC SMALL LETTER I
    '\u0439'   #  0xCA -> CYRILLIC SMALL LETTER SHORT I
    '\u043a'   #  0xCB -> CYRILLIC SMALL LETTER KA
    '\u043b'   #  0xCC -> CYRILLIC SMALL LETTER EL
    '\u043c'   #  0xCD -> CYRILLIC SMALL LETTER EM
    '\u043d'   #  0xCE -> CYRILLIC SMALL LETTER EN
    '\u043e'   #  0xCF -> CYRILLIC SMALL LETTER O
    '\u043f'   #  0xD0 -> CYRILLIC SMALL LETTER PE
    '\u044f'   #  0xD1 -> CYRILLIC SMALL LETTER YA
    '\u0440'   #  0xD2 -> CYRILLIC SMALL LETTER ER
    '\u0441'   #  0xD3 -> CYRILLIC SMALL LETTER ES
    '\u0442'   #  0xD4 -> CYRILLIC SMALL LETTER TE
    '\u0443'   #  0xD5 -> CYRILLIC SMALL LETTER U
    '\u0436'   #  0xD6 -> CYRILLIC SMALL LETTER ZHE
    '\u0432'   #  0xD7 -> CYRILLIC SMALL LETTER VE
    '\u044c'   #  0xD8 -> CYRILLIC SMALL LETTER SOFT SIGN
    '\u044b'   #  0xD9 -> CYRILLIC SMALL LETTER YERU
    '\u0437'   #  0xDA -> CYRILLIC SMALL LETTER ZE
    '\u0448'   #  0xDB -> CYRILLIC SMALL LETTER SHA
    '\u044d'   #  0xDC -> CYRILLIC SMALL LETTER E
    '\u0449'   #  0xDD -> CYRILLIC SMALL LETTER SHCHA
    '\u0447'   #  0xDE -> CYRILLIC SMALL LETTER CHE
    '\u044a'   #  0xDF -> CYRILLIC SMALL LETTER HARD SIGN
    '\u042e'   #  0xE0 -> CYRILLIC CAPITAL LETTER YU
    '\u0410'   #  0xE1 -> CYRILLIC CAPITAL LETTER A
    '\u0411'   #  0xE2 -> CYRILLIC CAPITAL LETTER BE
    '\u0426'   #  0xE3 -> CYRILLIC CAPITAL LETTER TSE
    '\u0414'   #  0xE4 -> CYRILLIC CAPITAL LETTER DE
    '\u0415'   #  0xE5 -> CYRILLIC CAPITAL LETTER IE
    '\u0424'   #  0xE6 -> CYRILLIC CAPITAL LETTER EF
    '\u0413'   #  0xE7 -> CYRILLIC CAPITAL LETTER GHE
    '\u0425'   #  0xE8 -> CYRILLIC CAPITAL LETTER HA
    '\u0418'   #  0xE9 -> CYRILLIC CAPITAL LETTER I
    '\u0419'   #  0xEA -> CYRILLIC CAPITAL LETTER SHORT I
    '\u041a'   #  0xEB -> CYRILLIC CAPITAL LETTER KA
    '\u041b'   #  0xEC -> CYRILLIC CAPITAL LETTER EL
    '\u041c'   #  0xED -> CYRILLIC CAPITAL LETTER EM
    '\u041d'   #  0xEE -> CYRILLIC CAPITAL LETTER EN
    '\u041e'   #  0xEF -> CYRILLIC CAPITAL LETTER O
    '\u041f'   #  0xF0 -> CYRILLIC CAPITAL LETTER PE
    '\u042f'   #  0xF1 -> CYRILLIC CAPITAL LETTER YA
    '\u0420'   #  0xF2 -> CYRILLIC CAPITAL LETTER ER
    '\u0421'   #  0xF3 -> CYRILLIC CAPITAL LETTER ES
    '\u0422'   #  0xF4 -> CYRILLIC CAPITAL LETTER TE
    '\u0423'   #  0xF5 -> CYRILLIC CAPITAL LETTER U
    '\u0416'   #  0xF6 -> CYRILLIC CAPITAL LETTER ZHE
    '\u0412'   #  0xF7 -> CYRILLIC CAPITAL LETTER VE
    '\u042c'   #  0xF8 -> CYRILLIC CAPITAL LETTER SOFT SIGN
    '\u042b'   #  0xF9 -> CYRILLIC CAPITAL LETTER YERU
    '\u0417'   #  0xFA -> CYRILLIC CAPITAL LETTER ZE
    '\u0428'   #  0xFB -> CYRILLIC CAPITAL LETTER SHA
    '\u042d'   #  0xFC -> CYRILLIC CAPITAL LETTER E
    '\u0429'   #  0xFD -> CYRILLIC CAPITAL LETTER SHCHA
    '\u0427'   #  0xFE -> CYRILLIC CAPITAL LETTER CHE
    '\u042a'   #  0xFF -> CYRILLIC CAPITAL LETTER HARD SIGN
)

### Encoding table
encoding_table=codecs.charmap_build(decoding_table)
