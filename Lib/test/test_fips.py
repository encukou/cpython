import unittest
import hmac, _hmacopenssl
import hashlib, _hashlib



class HashlibFipsTests(unittest.TestCase):

    @unittest.skipUnless(hashlib.get_fips_mode(), "Test only when FIPS is enabled")
    def test_fips_imports(self):
        """blake2s and blake2b should fail to import in FIPS mode
        """
        with self.assertRaises(ValueError, msg='blake2s not available in FIPS'):
            m = hashlib.blake2s()
        with self.assertRaises(ValueError, msg='blake2b not available in FIPS'):
            m = hashlib.blake2b()

    def compare_hashes(self, python_hash, openssl_hash):
        """
        Compare between the python implementation and the openssl one that the digests
        are the same
        """
        if python_hash.name.startswith('shake_128'):
            m = python_hash.hexdigest(16)
        elif python_hash.name.startswith('shake_256'):
            m = python_hash.hexdigest(32)
        else:
            m = python_hash.hexdigest()
        h = openssl_hash.hexdigest()

        self.assertEqual(m, h)

    @unittest.skipIf(hashlib.get_fips_mode(), "blake2 hashes are not available under FIPS")
    def test_blake2_hashes(self):
        self.compare_hashes(hashlib.blake2b(b'abc'), _hashlib.openssl_blake2b(b'abc'))
        self.compare_hashes(hashlib.blake2s(b'abc'), _hashlib.openssl_blake2s(b'abc'))

    def test_sha3_hashes(self):
        self.compare_hashes(hashlib.sha3_224(b'abc'), _hashlib.openssl_sha3_224(b'abc'))
        self.compare_hashes(hashlib.sha3_256(b'abc'), _hashlib.openssl_sha3_256(b'abc'))
        self.compare_hashes(hashlib.sha3_384(b'abc'), _hashlib.openssl_sha3_384(b'abc'))
        self.compare_hashes(hashlib.sha3_512(b'abc'), _hashlib.openssl_sha3_512(b'abc'))

    @unittest.skipIf(hashlib.get_fips_mode(), "shake hashes are not available under FIPS")
    def test_shake_hashes(self):
        self.compare_hashes(hashlib.shake_128(b'abc'), _hashlib.openssl_shake_128(b'abc'))
        self.compare_hashes(hashlib.shake_256(b'abc'), _hashlib.openssl_shake_256(b'abc'))

    def test_sha(self):
        self.compare_hashes(hashlib.sha1(b'abc'), _hashlib.openssl_sha1(b'abc'))
        self.compare_hashes(hashlib.sha224(b'abc'), _hashlib.openssl_sha224(b'abc'))
        self.compare_hashes(hashlib.sha256(b'abc'), _hashlib.openssl_sha256(b'abc'))
        self.compare_hashes(hashlib.sha384(b'abc'), _hashlib.openssl_sha384(b'abc'))
        self.compare_hashes(hashlib.sha512(b'abc'), _hashlib.openssl_sha512(b'abc'))

    def test_hmac_digests(self):
        self.compare_hashes(_hmacopenssl.new(b'My hovercraft is full of eels', digestmod='sha384'),
                            hmac.new(b'My hovercraft is full of eels', digestmod='sha384'))




if __name__ == "__main__":
    unittest.main()
