import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from surrogate.generate import main
from surrogate.params import NAMES


class GenerationShardsTest(unittest.TestCase):
    def test_shards_are_disjoint_and_retain_full_manifest(self):
        assignments = ([0, 1], [2, 3], [4], [5, 6], [7, 8, 9])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / 'params.csv'
            with manifest.open('w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['case', 'geom', 'split', *NAMES])
                for i in range(20):
                    writer.writerow([f'case_{i}', 'mean', 'train', *([1] * len(NAMES))])
            seen = set()
            for worker, shards in enumerate(assignments):
                out = root / f'worker_{worker}'
                with patch('surrogate.generate.one', return_value='ok') as solve:
                    main([str(manifest), str(out), '--shards', '10',
                          '--shard', *map(str, shards)])
                selected = {call.args[0]['case'] for call in solve.call_args_list}
                self.assertFalse(seen & selected)
                seen |= selected
                self.assertEqual((out / 'params.csv').read_bytes(), manifest.read_bytes())
            self.assertEqual(seen, {f'case_{i}' for i in range(20)})


if __name__ == '__main__':
    unittest.main()
