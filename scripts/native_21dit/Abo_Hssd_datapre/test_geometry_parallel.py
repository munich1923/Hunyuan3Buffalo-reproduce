"""CPU tests for the bounded geometry scheduler; no sampling or model work."""
import concurrent.futures as futures
import json, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
import prepare_geometry_parallel as scheduler
from core import ROOT, setup, readlines, cfg, dirs, atomic, now, fp

class SchedulerTests(unittest.TestCase):
    def test_unique_partition(self):
        rows=[{'uid':str(i)} for i in range(8)]
        good,bad,pending=scheduler.terminal_records(rows,[{'uid':'0'}],[{'uid':'1'}])
        self.assertEqual(list(good),['0']);self.assertEqual(list(bad),['1'])
        self.assertEqual([r['uid'] for r in pending],[str(i) for i in range(2,8)])
        for a,b in [([{'uid':'0'},{'uid':'0'}],[]),([{'uid':'0'}],[{'uid':'0'}]),([{'uid':'missing'}],[])]:
            with self.assertRaises(RuntimeError):scheduler.terminal_records(rows,a,b)
    def test_no_worker_global_rng_use(self):
        import ast
        tree=ast.parse(Path(scheduler.__file__).read_text())
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)]
        self.assertFalse(any(isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='random' for n in calls))
    def test_worker_exception_becomes_terminal_failure(self):
        row={'uid':'test_geometry_scheduler'}
        with patch.object(scheduler.geometry,'prepare_one',side_effect=ValueError('expected')):
            ok,event=scheduler.process_one(row,cfg(),'inventory','blender','python',{})
        self.assertFalse(ok);self.assertEqual(event['uid'],row['uid']);self.assertIn('expected',event['error'])
    def test_resource_append_contention(self):
        scheduler.worker_init()
        folder=ROOT/'tmp/abo_hssd_parallel_tests';folder.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=folder) as name:
            output=Path(name)/'parallel.jsonl'
            with futures.ThreadPoolExecutor(max_workers=2) as pool:
                list(pool.map(lambda i:scheduler.geometry.append(output,{'test_index':i}),range(100)))
            rows=readlines(output)
            self.assertEqual(len(rows),100);self.assertEqual({r['test_index'] for r in rows},set(range(100)))

if __name__=='__main__':
    setup();result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(SchedulerTests))
    atomic('results/abo_hssd_v1/parallel_switch_20261101/scheduler_cpu_tests.json',{'tests':result.testsRun,'passed':result.wasSuccessful(),'scheduler_sha256':__import__('core').sha(Path(scheduler.__file__)),'sampling_or_gpu_executed':False,**now()})
    raise SystemExit(0 if result.wasSuccessful() else 1)
