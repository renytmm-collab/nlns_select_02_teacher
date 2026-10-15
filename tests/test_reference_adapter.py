"""HGS interface regressions; mocked processes are not solver runtime proof."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from teacher_select.core import ROOT, Journal, sha
from teacher_select.reference import encode, solve, route_cost


class ReferenceAdapter(unittest.TestCase):
    def setUp(self):
        base = ROOT / 'tests/_reference_tmp'
        base.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=base)
        self.addCleanup(self.temp.cleanup)
        self.d = Path(self.temp.name)
        xy = [[i / 101, i / 103] for i in range(101)]
        self.state = {'original_locations': xy, 'demand': [0] + [1] * 100,
                      'capacity': 50}
        self.exe = self.d / 'mock-hgs'
        self.exe.write_bytes(b'mocked executable identity')
        self.config = {'solver_executable': str(self.exe),
                       'solver_sha256': sha(self.exe), 'solver_version': 'fixture',
                       'solver_seeds': [271828, 314159, 161803]}
        self.journal = Journal(self.d, 15)
        self.addCleanup(self.journal.close)

    def events(self):
        return [json.loads(s) for s in (self.d / 'calls.jsonl').read_text().splitlines()]

    def solution(self):
        rr = [list(range(1, 51)), list(range(51, 101))]
        return ''.join(f'Route #{j+1}: '+ ' '.join(map(str, r))+'\n'
                       for j, r in enumerate(rr)) + f'Cost {route_cost(self.state, rr):.8f}\n'

    def test_header_matches_hgs_skip_three_lines(self):
        text = encode(self.state)
        lines = text.splitlines()
        self.assertEqual([s.split()[0] for s in lines[:4]],
                         ['NAME', 'COMMENT', 'TYPE', 'DIMENSION'])
        # Reproduce the pinned reader's header skip and token-based field parsing.
        tokens = iter('\n'.join(lines[3:]).split())
        fields = {}
        for token in tokens:
            if token == 'NODE_COORD_SECTION':
                break
            self.assertEqual(next(tokens), ':')
            fields[token] = next(tokens)
        self.assertEqual(fields['DIMENSION'], '101')
        self.assertEqual(fields['CAPACITY'], '50')
        for i, xy in enumerate(self.state['original_locations'], 1):
            self.assertEqual(int(next(tokens)), i)
            self.assertEqual([float(next(tokens)), float(next(tokens))], xy)
        self.assertEqual(next(tokens), 'DEMAND_SECTION')

    def test_zero_exit_exception_stops_and_remains_charged(self):
        def process(cmd, **kw):
            kw['stdout'].write('EXCEPTION | The node numbering is not in order.\n')
            return SimpleNamespace(returncode=0)
        with patch('teacher_select.reference.subprocess.run', side_effect=process) as run:
            with self.assertRaisesRegex(ValueError, 'HGS exception'):
                solve(self.state, self.d/'solver', self.config, self.journal, 0)
        self.assertEqual(run.call_count, 1)
        self.assertEqual(self.journal.used, 5)
        event = self.events()[-1]
        self.assertEqual(event['status'], 'invalid')
        self.assertEqual(event['exit_code'], 0)
        self.assertIn('log_sha256', event)

    def test_exception_with_solution_still_invalid(self):
        def process(cmd, **kw):
            Path(cmd[2]).write_text(self.solution())
            kw['stdout'].write('EXCEPTION | runtime failure\n')
            return SimpleNamespace(returncode=0)
        with patch('teacher_select.reference.subprocess.run', side_effect=process):
            with self.assertRaisesRegex(ValueError, 'HGS exception'):
                solve(self.state, self.d/'solver', self.config, self.journal, 0)

    def test_nonzero_exit_is_engineering_failure(self):
        with patch('teacher_select.reference.subprocess.run', return_value=SimpleNamespace(returncode=2)) as run:
            with self.assertRaisesRegex(ValueError, 'exit code 2'):
                solve(self.state, self.d/'solver', self.config, self.journal, 0)
        self.assertEqual(run.call_count, 1)
        self.assertEqual(self.events()[-1]['exit_code'], 2)

    def test_no_solution_without_exception_retains_fallback(self):
        with patch('teacher_select.reference.subprocess.run', return_value=SimpleNamespace(returncode=0)):
            result = solve(self.state, self.d/'solver', self.config, self.journal, 0)
        self.assertIsNone(result['primary'])
        self.assertTrue(result['no_feasible_reference'])
        self.assertEqual(self.journal.used, 15)
        self.assertEqual([o['status'] for o in result['outcomes']], ['failed'] * 3)

    def test_successful_outputs_parse_and_select(self):
        def process(cmd, **kw):
            Path(cmd[2]).write_text(self.solution())
            return SimpleNamespace(returncode=0)
        with patch('teacher_select.reference.subprocess.run', side_effect=process):
            result = solve(self.state, self.d/'solver', self.config, self.journal, 0)
        self.assertFalse(result['no_feasible_reference'])
        self.assertEqual(result['primary']['seed_index'], 0)
        self.assertEqual([o['status'] for o in result['outcomes']], ['ok'] * 3)


if __name__ == '__main__':
    unittest.main()
