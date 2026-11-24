import unittest
from src.session import ConversationSession
class SessionTests(unittest.TestCase):
    def test_add_clear(self):
        s=ConversationSession();self.assertEqual(len(s.add("q","a",1,{})),1);self.assertEqual(s.clear(),[])
