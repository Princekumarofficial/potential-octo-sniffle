import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data import parse_parameters


class ParameterTests(unittest.TestCase):
    def test_absent_candidates(self):
        for value in (None, float('nan'), '', '  ', 'null'):
            self.assertIsNone(parse_parameters(value, 'MSFT 2026-01 VWAP'))

    def test_valid_candidate(self):
        self.assertEqual(parse_parameters('{"stop_loss": 0.01}', 'test'), {'stop_loss': .01})

    def test_corrupt_parameters_are_not_silently_cash(self):
        for value in ('broken', '[]', '42', 42.):
            with self.assertRaisesRegex(ValueError, 'MSFT'):
                parse_parameters(value, 'MSFT 2026-01 VWAP')


if __name__ == '__main__':
    unittest.main()
