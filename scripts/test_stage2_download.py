"""CPU-only download recovery contracts; no network or GPU calls."""
from stage2_runtime import *
import unittest
from unittest.mock import patch
from prepare_stage2 import download_command, download_archive, download_rate

class Recovery(unittest.TestCase):
    def test_resumable_serial_ipv4_command(self):
        command=download_command(Path("/tmp/example.partial"),"https://example.invalid/file")
        for flag in ["--continue=true","--disable-ipv6=true","--split=1",
                     "--max-connection-per-server=1","--auto-file-renaming=false","--lowest-speed-limit=64K"]:
            self.assertIn(flag,command)

    def test_rate_from_current_attempt_not_file_size(self):
        with tempfile.TemporaryDirectory(dir=path("environment/tmp/stage2")) as folder:
            log=Path(folder)/"transfer.log"
            old=b"[old DL:9MiB]\\n"
            log.write_bytes(old+b"[current DL:8.5KiB]\\n")
            self.assertEqual(download_rate(log,len(old)),8.5*1024)
            log.write_bytes(old)
            self.assertIsNone(download_rate(log,len(old)))

    def test_sustained_low_speed_restarts_and_resumes(self):
        with tempfile.TemporaryDirectory(dir=path("environment/tmp/stage2")) as folder:
            folder=Path(folder);partial=folder/"example.partial";partial.write_bytes(b"old")
            class Transfer:
                calls=0
                def __init__(self,*args,**kwargs):
                    Transfer.calls+=1;self.first=Transfer.calls==1
                    self.pid=12345;self.stopped=False
                    if not self.first:
                        assert partial.read_bytes()==b"old"
                        partial.write_bytes(b"old-new")
                def wait(self,timeout=None):
                    if self.first and not self.stopped:
                        raise subprocess.TimeoutExpired("fake",timeout)
                    return -15 if self.first else 0
                def poll(self):return -15 if self.stopped else (None if self.first else 0)
                def terminate(self):self.stopped=True
            cfg={"dataset":{"repo":"example/repo","revision":"fixed"}}
            with (patch("prepare_stage2.subprocess.Popen",Transfer),
                  patch("prepare_stage2.time.sleep"),
                  patch("prepare_stage2.time.monotonic",side_effect=[0,30,60,90,120]),
                  patch("prepare_stage2.download_rate",return_value=8192)):
                download_archive(cfg,"example",partial,7,folder)
            events=[json.loads(line) for line in (folder/"events.jsonl").read_text().splitlines()]
            self.assertEqual(sum(e["event"]=="download_slow_restart" for e in events),1)
            self.assertEqual(Transfer.calls,2)

    def test_retry_preserves_partial_and_control(self):
        with tempfile.TemporaryDirectory(dir=path("environment/tmp/stage2")) as folder:
            folder=Path(folder)
            partial=folder/"example.partial";partial.write_bytes(b"old")
            control=folder/"example.partial.aria2";control.write_bytes(b"resume-state")
            class Transfer:
                calls=0
                def __init__(self,*args,**kwargs):
                    Transfer.calls+=1
                    self.code=22 if Transfer.calls==1 else 0
                    if Transfer.calls==2:
                        assert partial.read_bytes()==b"old"
                        assert control.read_bytes()==b"resume-state"
                        partial.write_bytes(b"old-new")
                def wait(self,timeout=None):return self.code
                def poll(self):return self.code
            cfg={"dataset":{"repo":"example/repo","revision":"fixed"}}
            with patch("prepare_stage2.subprocess.Popen",Transfer), patch("prepare_stage2.time.sleep") as sleep:
                download_archive(cfg,"example",partial,7,folder)
                sleep.assert_called_once_with(60)
            self.assertEqual(partial.read_bytes(),b"old-new")
            events=[json.loads(line) for line in (folder/"events.jsonl").read_text().splitlines()]
            self.assertEqual([x["event"] for x in events],["download_attempt","download_backoff","download_attempt"])
            self.assertTrue(all(x.get("endpoint")=="https://hf-mirror.com" for x in events))

if __name__=="__main__":unittest.main()
