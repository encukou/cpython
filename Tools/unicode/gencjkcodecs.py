import os, string

codecs = {
    'cn': ('gb2312', 'gbk', 'gb18030', 'hz'),
    'tw': ('big5', 'cp950'),
    'hk': ('big5hkscs',),
    'jp': ('cp932', 'shift_jis', 'euc_jp', 'euc_jisx0213', 'shift_jisx0213',
           'euc_jis_2004', 'shift_jis_2004'),
    'kr': ('cp949', 'euc_kr', 'johab'),
    'iso2022': ('iso2022_jp', 'iso2022_jp_1', 'iso2022_jp_2',
                'iso2022_jp_2004', 'iso2022_jp_3', 'iso2022_jp_ext',
                'iso2022_kr'),
}

TEMPLATE = string.Template("""\
#
# $encoding.py: Python Unicode Codec for $ENCODING
#
# Written by Hye-Shik Chang <perky@FreeBSD.org>
#

import _codecs_$owner, codecs
import _multibytecodec as mbc

codec = _codecs_$owner.getcodec('$encoding')

class Codec(codecs.Codec):
    encode = codec.encode
    decode = codec.decode

class IncrementalEncoder(mbc.MultibyteIncrementalEncoder,
                         codecs.IncrementalEncoder):
    codec = codec

class IncrementalDecoder(mbc.MultibyteIncrementalDecoder,
                         codecs.IncrementalDecoder):
    codec = codec

class StreamReader(Codec, mbc.MultibyteStreamReader, codecs.StreamReader):
    codec = codec

class StreamWriter(Codec, mbc.MultibyteStreamWriter, codecs.StreamWriter):
    codec = codec

def getregentry():
    return codecs.CodecInfo(
        name='$encoding',
        encode=Codec().encode,
        decode=Codec().decode,
        incrementalencoder=IncrementalEncoder,
        incrementaldecoder=IncrementalDecoder,
        streamreader=StreamReader,
        streamwriter=StreamWriter,
""")

END = """\
    )
"""

def gencodecs(prefix):
    for loc, encodings in codecs.items():
        module = __import__('_codecs_' + loc)
        for enc in encodings:
            codec = module.getcodec(enc)
            code = TEMPLATE.substitute(ENCODING=enc.upper(),
                                       encoding=enc.lower(),
                                       owner=loc)
            codecpath = os.path.join(prefix, enc + '.py')
            with open(codecpath, 'w') as f:
                f.write(code)
                write_expat_table(f, codec)
                f.write(END)


###### EXPAT TABLE GENERATION ######

# ASCII characters that can appear in a well-formed XML document
# except the characters "$@\^`{}~".
EXPAT_COMPULSORY_CHARS = {*b'\t\n\r', *range(32, 127)} - set(br'$@\^`{}~')

def info(msg, *args):
    print('expat table: ' + msg % args, file=sys.stderr)

class ExpatIncompatibility(ValueError):
    """The expat table can't be generated"""

def fail(msg, *args):
    info(msg, *args)
    raise ExpatIncompatibility()

def write_expat_table(f, codec):
    try:
        mapping = create_expat_table(codec)
    except ExpatIncompatibility:
        print(' '*8 + '_expat_decoding_table=False,', file=f)
    else:
        print_expat_table(f, mapping)


def create_expat_table(codec):
    for i in EXPAT_COMPULSORY_CHARS:
        c = chr(i)
        b = bytes([i])
        try:
            decoded, nb = codec.decode(b)
            assert nb == 1
        except UnicodeDecodeError:
            fail('Cannot decode byte %#04x (%a)', i, c)
        if decoded != c:
            fail('Incompatible encoding: %#04x (%a) -> %a', i, c, decoded)

    mapping = [-1] * 256

    non_bmp = None
    for i in range(0x110000):
        char = chr(i)
        try:
            encoded, nb = codec.encode(char)
            assert nb == 1
        except UnicodeEncodeError:
            continue
        if i >= 0x10000 and non_bmp is None:
            non_bmp = i
            info('Non-BMP character: %r (U+%04X)', char, i)
        length = len(encoded)
        k = encoded[0]
        v = mapping[k]
        if length == 1:
            if v == -1:
                mapping[k] = i
            elif v < 0:
                fail('Ambiguous mapping for %#04x: '
                    '%r (U+%04X) and %d-byte sequence',
                    k, c, i, -v)
            else:
                info('Ambiguous mapping for %#04x: '
                    '%r (U+%04X) and %r (U+%04X)',
                    k, chr(v), v, char, i)
        else:
            if length > 4:
                fail('Too long encoding for %r (U+%04X): %r',
                    char, i, encoded)
            if v == -1:
                mapping[k] = -length
            elif v != -length:
                if v < 0:
                    fail('Ambiguous mapping for %#04x: '
                        '%d-byte sequence and %d-byte sequence %r',
                        k, -v, length, encoded)
                else:
                    fail('Ambiguous mapping for %#04x: '
                        '%r (U+%04X) and %d-byte sequence %r',
                        k, chr(v), v, length, encoded)
    return mapping

def print_expat_table(f, mapping):
    print(' '*8 + '_expat_decoding_table=(', end='', file=f)
    if mapping[:128] == list(range(128)):
        print('*range(128),', file=f)
        start = 128
    # Special case for shift_jis
    elif mapping[:128] == [*range(0x5c), 0xa5, *range(0x5d, 0x7e), 0x203e, 0x7f]:
        print('\n            *range(0x5c), 0xa5, *range(0x5d, 0x7e), 0x203e, 0x7f,', file=f)
        start = 128
    else:
        print(file=f)
        start = 0
    line = ''
    pos = 0
    i = start
    while True:
        if i % 8 == 0:
            if len(line) > 68:
                print(' '*12 + line[:pos].rstrip(), file=f)
                line = line[pos:]
            pos = len(line)
        if i == 256:
            break
        v = mapping[i]
        if v >= 0:
            v = hex(v)
        line += f'{v}, '
        i += 1
    print(' '*12 + line.rstrip().rstrip(',') + '),', file=f)


if __name__ == '__main__':
    import sys
    gencodecs(sys.argv[1])
