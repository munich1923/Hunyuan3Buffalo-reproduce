"""CPU-only tests of shared GPU admission and allocator limits."""
from stage2_runtime import *
import unittest
from unittest.mock import patch,MagicMock

class Admission(unittest.TestCase):
    def setUp(self):
        mock=patch("stage2_runtime.lock",side_effect=lambda _:contextlib.nullcontext())
        mock.start();self.addCleanup(mock.stop)
    def replies(self,args,**kwargs):
        return "GPU-test,22075,81559,59005,99" if "--id=2" in args else "GPU-test,123"
    def test_memory_budget(self):
        self.assertEqual(shared_memory_limit(81559,59005),53104)
        with self.assertRaises(ValueError):shared_memory_limit(10,11)
    def test_occupied_default_denied(self):
        with patch.dict(os.environ,{"STAGE2_ALLOW_SHARED_GPU":""}),patch("subprocess.check_output",side_effect=self.replies):
            with self.assertRaises(RuntimeError):
                with gpu_lease(2):pass
    def test_shared_admission_caps_existing_free(self):
        fake=MagicMock();fake.cuda.is_initialized.return_value=False
        with patch.dict(sys.modules,{"torch":fake}),patch.dict(os.environ,{"STAGE2_ALLOW_SHARED_GPU":"2"}),patch("subprocess.check_output",side_effect=self.replies):
            with gpu_lease(2) as info:
                self.assertTrue(info["shared"])
                self.assertEqual(info["process_memory_limit_mib"],53104)
            fake.cuda.set_per_process_memory_fraction.assert_called_once_with(53104/81559,device=0)
    def test_other_gpu_not_authorized(self):
        with patch.dict(os.environ,{"STAGE2_ALLOW_SHARED_GPU":"3"}),patch("subprocess.check_output",side_effect=self.replies):
            with self.assertRaises(RuntimeError):
                with gpu_lease(2):pass

if __name__=="__main__":unittest.main(verbosity=2)
