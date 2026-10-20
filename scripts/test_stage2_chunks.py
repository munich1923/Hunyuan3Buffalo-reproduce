from stage2_runtime import *
from stage2_download import append_range
import unittest
class RangeChecks(unittest.TestCase):
    def test_interrupted_response_keeps_verified_prefix(self):
        with tempfile.TemporaryDirectory(dir=path("environment/tmp/stage2")) as tmp:
            piece=Path(tmp)/"part";payload=Path(tmp)/"response";payload.write_bytes(b"abc")
            self.assertEqual(append_range(piece,payload,"Content-Range: bytes 10-19/100",10,19,100),3)
            self.assertEqual(piece.read_bytes(),b"abc")
    def test_wrong_range_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=path("environment/tmp/stage2")) as tmp:
            piece=Path(tmp)/"part";payload=Path(tmp)/"response";payload.write_bytes(b"abc")
            with self.assertRaises(RuntimeError):append_range(piece,payload,"Content-Range: bytes 0-9/100",10,19,100)
            self.assertFalse(piece.exists())
if __name__=="__main__":unittest.main()
