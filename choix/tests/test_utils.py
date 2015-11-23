from unittest import TestCase
from choicelib import utils

class TestUtils(TestCase):
    def test_bla(self):
        s = utils.bla()
        self.assertTrue(isinstance(s, basestring))
