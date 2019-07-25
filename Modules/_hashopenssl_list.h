/* Call the _HASH macro with all the hashes exported by OpenSSL,
 * at compile time.
 *
 * This file is meant to be included multiple times, with different values of
 * _HASH.
 */

_HASH(md5, "md5")
_HASH(sha1, "sha1")
_HASH(sha224, "sha224")
_HASH(sha256, "sha256")
_HASH(sha384, "sha384")
_HASH(sha512, "sha512")
_HASH(blake2b, "blake2b512")
_HASH(blake2s, "blake2s256")
_HASH(sha3_224, "sha3-224")
_HASH(sha3_256, "sha3-256")
_HASH(sha3_384, "sha3-384")
_HASH(sha3_512, "sha3-512")
_HASH(shake_128, "shake128")
_HASH(shake_256, "shake256")
