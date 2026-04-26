import importlib
import os
import tempfile
import unittest


class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["MEMORY_BACKEND"] = "sqlite"
        os.environ["MEMORY_DB"] = os.path.join(self.tmp.name, "memory.sqlite3")
        os.environ["MAX_TURNS"] = "2"

        import memory
        self.memory = importlib.reload(memory)

    def tearDown(self):
        self.tmp.cleanup()

    def test_save_load_clear(self):
        phone = "+15551234567"
        turns = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "book me"},
            {"role": "assistant", "content": "what time?"},
            {"role": "user", "content": "tomorrow"},
            {"role": "assistant", "content": "noted"},
        ]

        self.memory.save(phone, turns)
        loaded = self.memory.load(phone)

        self.assertEqual(len(loaded), 4)
        self.assertEqual(loaded[-1]["content"], "noted")
        self.assertIn(phone, self.memory.all_callers())

        self.memory.clear(phone)
        self.assertEqual(self.memory.load(phone), [])

    def test_events(self):
        phone = "+15550000000"
        self.memory.log_event(phone, "call_started", {"room": "test"})
        events = self.memory.recent_events(limit=5)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["phone"], phone)
        self.assertEqual(events[0]["event_type"], "call_started")


if __name__ == "__main__":
    unittest.main()
