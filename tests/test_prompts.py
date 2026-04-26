import unittest

import prompts


class PromptTests(unittest.TestCase):
    def test_build_messages_includes_system_and_user(self):
        messages = prompts.build_messages([], "Hello", is_first_turn=True)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[-1]["role"], "user")
        self.assertEqual(messages[-1]["content"], "Hello")

    def test_clean_reply_preserves_normal_text(self):
        text = "Of course, please hold for a moment."
        self.assertEqual(prompts.clean_reply(text), text)


if __name__ == "__main__":
    unittest.main()
