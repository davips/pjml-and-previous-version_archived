import hashlib
import json

import _pickle as pickle
import blosc
import lz4.frame as lz
import zstd as zs

# Disabling profiling when not needed.
try:
    import builtins
    profile = builtins.__dict__['profile']
except KeyError:
    # No line profiler, provide a pass-through version
    def profile(func): return func


@profile
def enc(big, alphabet='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
                      'abcdefghijklmnopqrstuvwxyzÀÂÃÄÅÆÇÈÊË'
                      'ÌÎÏÑÒÔÕÖØÙÛÜÝÞßàâãäåæçèêëìîïðñòóôõöøùûüýþ'):
    """
    Encode an integer to base-X.
    The default is base113 since it is enough to represent MD5 as 19 chars.
    The selected alphabet contains only numbers and letters. Similar letters
    were arbitrarily removed.
    This alphabet is intended to be printable and seen as part of a single
    word by most linux terminals and editors.
    I would call this subset as a subset of the double_click_friendly chars.

    The following list shows how the alphabet size relates to the number of
    necessary digits to represent the biggest MD5 number (2^128).
    The hexdigest alredy uses 32 digits, so we want less than that.
    Good choices for the alphabet size would be in the range 85-185, since
    values higher than 256 are outside latin1 range.

    alphabet-size number-of-digits comments
    2 128 # crude md5 as binary string
    16 32 # hexdigest as string
    24 28
    41 24 # reducing from 32 to 24 is kind of a improvement
    48 23
    57 22
    69 21
    85 20 # base64 library provides base85, but it is not double_click_friendly
    107 19 # super friendly (our default choice)
    139 18 # not terminator/intellij friendly
    185 17 # not double_click_friendly
    256 16 # would include lots of unprintable characters
    371 15 # 371 and beyond is outside a single byte and latin1
    566 14
    16-bit 4
    32-bit 2 # UTF-8?

    147 is the size of the largest subset of latin1 that is
    double_click_friendly. Latin1 is compatible with UTF-8 and extends ASCII.

    Example alphabets are given below:

    gnome-terminal friendly (base147)
#%&+,-./0123456789=?@ABCDEFGHIJKLMNOPQRSTUVWXYZ\_abcdefghijklmnopqrstuvwxyz~ª²³µ·¹º¼½¾ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝÞßàáâãäåæçèéêëìíîïðñòóôõöøùúûüýþ

    gnome-terminal/terminator/intellij friendly (base125)
#0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcdefghijklmnopqrstuvwxyzÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝÞßàáâãäåæçèéêëìíîïðñòóôõöøùúûüýþ

    :param alphabet: string with allowed digits
    :param big: an integer, usually a big MD5-like one
    :return: string representing a base-113 number
    """
    l = len(alphabet)
    res = []
    while (True):
        res.append(alphabet[big % l])
        big = big // l
        if big == 0:
            break
    return ''.join(res)[::-1]


@profile
def dec(digest, alphabet='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
                         'abcdefghijklmnopqrstuvwxyzÀÂÃÄÅÆÇÈÊË'
                         'ÌÎÏÑÒÔÕÖØÙÛÜÝÞßàâãäåæçèêëìîïðñòóôõöøùûüýþ'):
    """
    Decode digest from base-len(alphabet).
    See enc() for more info.
    :param digest:
    :param alphabet:
    :return:
    """
    res = 0
    last = len(digest) - 1
    base = len(alphabet)
    for i, d in enumerate(digest):
        res += alphabet.index(d) * pow(base, last - i)
    return res


@profile
def tiny_md5(hexdigest):
    """
    Converts hex MD5 representation (32 digits in base-16) to a friendly
    shorter one (19 digits in base-113).
    :param hexdigest:
    :return: string with 19 digits
    """
    return enc(int(hexdigest, 16))


@profile
def uuid(packed_content):
    """
    Generates a UUID for any reasonable finite universe.
    It is preferred to generate such MD5 on compressed data,
    since MD5 is much slower for bigger data than the compression itself.
    :param packed_content: packed Data of Xy... or a JSON dump of Component args
    :return: currently a MD5 hash in hex format
    """
    return tiny_md5(hashlib.md5(packed_content).hexdigest())


@profile
def pack_comp(obj):
    """
    Nondeterministic (fast) parallel compression!
    :param obj:
    :return:
    """
    pickled = pickle.dumps(obj)
    fast_reduced = lz.compress(pickled, compression_level=1)
    return blosc.compress(fast_reduced,
                          shuffle=blosc.NOSHUFFLE, cname='zstd', clevel=3)


@profile
def pack_data(obj):
    pickled = pickle.dumps(obj)
    fast_reduced = lz.compress(pickled, compression_level=1)
    return zs.compress(fast_reduced)


@profile
def unpack_comp(dump):
    decompressed = blosc.decompress(dump)
    fast_decompressed = lz.decompress(decompressed)
    return pickle.loads(fast_decompressed)


@profile
def unpack_data(dump):
    decompressed = zs.decompress(dump)
    fast_decompressed = lz.decompress(decompressed)
    return pickle.loads(fast_decompressed)


def zip_array(X):
    """
    WARNING, blosc size limits: bytesobj cannot be larger than 2147483631 bytes
    Attempt to zip faster than with pack, but benchmarks are still needed.
    ps. 1
    Parameters optimized for digits dataset. 115008 rows, 64 attrs
    ps. 2
    The hope of speed gains with this method is probably not worth the
    trouble of applying it to individual parts of Data.
    ps. 3
    Due to multithreading, blosc in nondeterministic and useless for UUIDs
    :param X:
    :return:
    """
    return blosc.compress(X.reshape(1, 115008), cname='blosclz',
                          shuffle=blosc.BITSHUFFLE)


def unzip_array(zipped):
    return blosc.decompress(zipped)


@profile
def uuid_enumerated_dic(l):
    return {uuid(x.encode()): x for x in l}


@profile
def json_pack(obj):
    dump = json.dumps(obj, sort_keys=True)
    return dump


@profile
def json_unpack(dump):
    obj = json.loads(dump)
    if obj == 'null':
        return None
    return obj
