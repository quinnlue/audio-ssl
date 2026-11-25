import unittest


class SmokeTests(unittest.TestCase):
    def test_import(self) -> None:
        import audiossl

        self.assertEqual(audiossl.__version__, "0.1.0")
        self.assertIs(audiossl.validate_config(audiossl.RuntimeConfig()), True)


if __name__ == "__main__":
    unittest.main()
