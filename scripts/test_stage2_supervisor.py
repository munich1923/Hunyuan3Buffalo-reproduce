"""CPU-only supervisor failure/retry contracts."""
from stage2_runtime import *
import unittest
from unittest.mock import patch
from prepare_stage2_runner import supervise

class Supervision(unittest.TestCase):
    def run_codes(self,codes):
        with tempfile.TemporaryDirectory(dir=path("environment/tmp/stage2")) as folder:
            folder=Path(folder)
            processes=[]
            class Worker:
                def __init__(self,*args,**kwargs):
                    self.code=codes[len(processes)];processes.append(self)
                def wait(self):return self.code
            with patch("prepare_stage2_runner.subprocess.Popen",Worker), patch("prepare_stage2_runner.time.sleep") as wait:
                code=supervise(folder,["fake-cpu-worker"])
            return code,read_json(folder/"dataset_status.json"),[c.args[0] for c in wait.call_args_list]
    def test_network_retries_more_than_old_limit(self):
        code,state,delays=self.run_codes([75]*12+[0])
        self.assertEqual(code,0);self.assertEqual(state["state"],"ready")
        self.assertEqual(state["attempt"],13)
        self.assertEqual(delays[:4],[60,120,240,300])
        self.assertEqual(delays[-1],300)
    def test_integrity_error_does_not_silently_retry(self):
        code,state,delays=self.run_codes([1])
        self.assertEqual(code,1);self.assertEqual(state["state"],"failed")
        self.assertEqual(delays,[])

if __name__=="__main__":unittest.main()
