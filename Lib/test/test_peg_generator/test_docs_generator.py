import unittest
from Tools.peg_generator.docs_generator import TokenSet, LiteralToken, SymbolicToken


class TestTokenset(unittest.TestCase):
    def test_token_set_underscode_is_name(self):
        """The underscore is a NAME token."""
        self.assertTrue(
            TokenSet([LiteralToken('"_"')])
            .issubset(
                TokenSet([SymbolicToken('NAME')])
            )
        )


if __name__ == '__main__':
    unittest.main()
