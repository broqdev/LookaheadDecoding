import unittest
from triede import Trie

class TestTrie(unittest.TestCase):
    def test_record_and_gen(self):
        t = Trie(token=None)
        t.record([1, 2, 3])
        t.record([1, 2, 4, 5])
        t.record([1, 4, 5])

        token_seqs = t.gen_by_prefix([1])
        self.assertTrue([2, 3] in token_seqs)
        self.assertTrue([2, 4, 5] in token_seqs)
        self.assertTrue([4, 5] in token_seqs)
