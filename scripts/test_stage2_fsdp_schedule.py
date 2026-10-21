"""CPU regressions for the new FSDP sampler, streams and persistent budget."""
from stage2_runtime import *
import unittest,tempfile,unittest.mock as mock
from stage2_schedule import Ledger,Prefetch,cpu_sampler_checks


class Tests(unittest.TestCase):
    def test_full_epoch_coverage_tail_resume(self):
        result=cpu_sampler_checks({"seed":20261019});self.assertTrue(result["passed"])
    def test_budget_shared_runs_and_failure(self):
        with tempfile.TemporaryDirectory(dir=path("environment/tmp/stage2")) as temp:
            file=Path(temp)/"ledger.json"
            with mock.patch("stage2_schedule.time.time",return_value=100.):
                book=Ledger(file,57600);item=book.start("verify","dual",2)
            with mock.patch("stage2_schedule.time.time",return_value=110.):
                book.end(item,1)
                other=Ledger(file,57600);self.assertEqual(other.used(),20)
                item=other.start("formal","decode",1)
            with mock.patch("stage2_schedule.time.time",return_value=120.):
                other.end(item,0);self.assertEqual(Ledger(file,57600).used(),30)
                with self.assertRaises(TimeoutError): other.start("test","fail",2,minimum=30000)
    def test_live_prior_owner_rejected(self):
        with tempfile.TemporaryDirectory(dir=path("environment/tmp/stage2")) as temp:
            file=Path(temp)/"ledger.json";book=Ledger(file,100);book.start("run","one",1)
            with self.assertRaises(RuntimeError): Ledger(file,100)
    def test_prefetch_order_and_bound(self):
        stream=((i,[i*2,i*2+1]) for i in range(4))
        with Prefetch(stream,lambda x:x*x) as queue:
            values=list(queue)
        self.assertEqual(values,[(i,[(2*i,(2*i)**2),(2*i+1,(2*i+1)**2)]) for i in range(4)])
    def test_snapshot_frames(self):
        import io,torch
        from stage2_fsdp_io import write_record,read_record
        stream=io.BytesIO()
        values=[{"tensor":torch.randn(53,19),"name":"one"},{"tensor":torch.empty(0),"name":"empty"},{"end":True}]
        for value in values: write_record(stream,value)
        stream.seek(0)
        for value in values:
            actual=read_record(stream)
            self.assertEqual(actual.keys(),value.keys())
            if "tensor" in value: self.assertTrue(torch.equal(actual["tensor"],value["tensor"]))
            else: self.assertEqual(actual,value)

    def test_incomplete_checkpoint_preserved(self):
        from run_stage2_fsdp_training import archive_incomplete
        with tempfile.TemporaryDirectory(dir=path("environment/tmp/stage2")) as temp:
            run=Path(temp);folder=run/"checkpoints/step_000500"
            folder.mkdir(parents=True);(folder/"partial").write_bytes(b"first")
            pending=folder.with_name(folder.name+".pending")
            pending.mkdir();(pending/"partial").write_bytes(b"second")
            archived=archive_incomplete(run,folder)
            self.assertEqual(len(archived),2)
            self.assertEqual(sorted((Path(a)/"partial").read_bytes() for a in archived),[b"first",b"second"])
            self.assertFalse(folder.exists());self.assertFalse(pending.exists())
            folder.mkdir();(folder/"complete.json").write_text("committed")
            self.assertEqual(archive_incomplete(run,folder),[])
            self.assertEqual((folder/"complete.json").read_text(),"committed")
            with self.assertRaises(ValueError): archive_incomplete(run,run/"wrong/step")

    def test_interrupted_logs_preserved_and_replayed_once(self):
        from run_stage2_fsdp_training import rewind_logs
        with tempfile.TemporaryDirectory(dir=path("environment/tmp/stage2")) as temp:
            run=Path(temp);file=run/"updates_rank_0.jsonl"
            original=bytes([10]).join([json.dumps({"update":500}).encode(),json.dumps({"update":501}).encode(),b"{broken"])
            file.write_bytes(original)
            archived=rewind_logs(run,500)
            self.assertEqual(len(archived),1);self.assertEqual(Path(archived[0]).read_bytes(),original)
            self.assertEqual(read_lines(file),[{"update":500}])
            self.assertEqual(rewind_logs(run,500),[])
            file.write_bytes(b"{broken}"+bytes([10]))
            with self.assertRaises(json.JSONDecodeError): rewind_logs(run,500)

    def test_formal_phase_order_and_initial_restart(self):
        import run_stage2_fsdp_training as parent
        from types import SimpleNamespace
        cfg={"orchestration":{"save_updates":[500,1000,1500,1565]}}
        with tempfile.TemporaryDirectory(dir=path("environment/tmp/stage2")) as temp:
            run=Path(temp);budget=run/"budget.json";report=run/"ready.json"
            contract={"test":True};atomic_json(report,{"passed":True,"contract":contract})
            atomic_json(budget,{"formal_start_authorized":True,"verification_report":str(report)})
            args=SimpleNamespace(budget=str(budget),resume="auto")
            events=[]
            def suite(*args,**kwargs): events.append(("suite",args[5]))
            def child(*args,**kwargs):
                job=args[5];target=job["target_update"]
                events.append(("train",target,job["checkpoint"]))
                name=f"step_{target:06d}"+("_final" if target==1565 else "")
                atomic_json(run/"checkpoints"/name/"complete.json",{"update":target})
            ledger=SimpleNamespace(used=lambda:0.)
            with mock.patch.object(parent,"suite",side_effect=suite),mock.patch.object(parent,"child",side_effect=child):
                parent.formal(cfg,args,run,ledger,{"contract":contract})
            self.assertEqual([e[:2] for e in events],[("suite","initial"),("train",500),("suite","step_000500"),
                ("train",1000),("suite","step_001000"),("train",1500),("suite","step_001500"),
                ("train",1565),("suite","step_001565_final")])
            self.assertIsNone(events[1][2]);self.assertTrue(events[3][2].endswith("step_000500"))
            args.resume=str(run/"checkpoints/step_000500")
            with self.assertRaisesRegex(RuntimeError,"latest committed"):
                parent.formal(cfg,args,run,ledger,{"contract":contract})

    def test_budget_reserves_graceful_shutdown(self):
        import run_stage2_fsdp_training as parent
        from types import SimpleNamespace
        for cards in [1,2]:
            self.assertGreaterEqual(parent.shutdown_reserve(cards),cards*(15+2+3))
        with tempfile.TemporaryDirectory(dir=path("environment/tmp/stage2")) as temp:
            run=Path(temp);args=SimpleNamespace(config="configs/stage2_nano10k_fsdp.yaml")
            cfg={"fsdp":{"allocator":"max_split_size_mb:256"},
                 "training":{"cublas_workspace_config":":4096:8"},
                 "orchestration":{"phase_minimum_seconds":{"train":120,"sample":240}}}
            ledger=SimpleNamespace(file=run/"ledger.json",limit=57600,remaining=lambda:539.)
            with mock.patch.object(parent,"admission",return_value=[{"uuid":"a"},{"uuid":"b"}]),                 mock.patch.object(parent.subprocess,"Popen") as spawn:
                with self.assertRaisesRegex(TimeoutError,"shutdown reserve"):
                    parent.child(cfg,args,run,ledger,"sample",{"phase":"sample"})
                spawn.assert_not_called()

    def test_supervisor_interrupt_reaps_only_its_child(self):
        import run_stage2_fsdp_training as parent
        from types import SimpleNamespace
        with self.assertRaises(InterruptedError): parent.supervisor_signal(parent.signal.SIGTERM,None)
        with tempfile.TemporaryDirectory(dir=path("environment/tmp/stage2")) as temp:
            run=Path(temp);args=SimpleNamespace(config="configs/stage2_nano10k_fsdp.yaml")
            cfg={"fsdp":{"allocator":"max_split_size_mb:256"},
                 "training":{"cublas_workspace_config":":4096:8"},
                 "orchestration":{"phase_minimum_seconds":{}}}
            ledger=SimpleNamespace(file=run/"ledger.json",limit=57600,used=lambda:0.,remaining=lambda:1000.)
            process=SimpleNamespace(pid=7654321,returncode=None)
            process.poll=lambda:process.returncode
            def waited(**kwargs): process.returncode=-15;return -15
            process.wait=mock.Mock(side_effect=waited)
            with mock.patch.object(parent.subprocess,"Popen",return_value=process),                 mock.patch.object(parent.os,"killpg") as terminate,                 mock.patch.object(parent.time,"sleep",side_effect=InterruptedError("signal")):
                with self.assertRaises(InterruptedError):
                    parent.child(cfg,args,run,ledger,"cpu",{"phase":"cpu"},cpu=True)
            terminate.assert_called_once_with(process.pid,parent.signal.SIGTERM)
            process.wait.assert_called_once_with(timeout=15)
            self.assertFalse(read_json(run/"status.json")["formal_training_started"])
            atomic_json(run/"formal_started.json",{"test":True});parent.state(run,"sample")
            self.assertTrue(read_json(run/"status.json")["formal_training_started"])

    def test_unlimited_budget_keeps_historical_usage(self):
        import math
        with tempfile.TemporaryDirectory(dir=path("environment/tmp/stage2")) as temp:
            file=Path(temp)/"ledger.json"
            with mock.patch("stage2_schedule.time.time",return_value=100.):
                book=Ledger(file,57600);item=book.start("verify","dual",2)
            with mock.patch("stage2_schedule.time.time",return_value=110.):
                book.end(item,0)
                unlimited=Ledger(file,None)
                self.assertEqual(unlimited.used(),20.)
                self.assertTrue(math.isinf(unlimited.remaining()))
                self.assertIsNone(unlimited.data["limit_gpu_seconds"])
                self.assertEqual(unlimited.data["limit_amendments"][0]["old_limit_gpu_seconds"],57600)
                item=unlimited.start("formal","train",2,minimum=999999)
            with mock.patch("stage2_schedule.time.time",return_value=100000.):
                unlimited.end(item,0)
                restored=Ledger(file,None)
                self.assertGreater(restored.used(),57600)
                self.assertTrue(math.isinf(restored.remaining()))
                self.assertEqual(len(restored.data["limit_amendments"]),1)

    def test_independent_flow_streams(self):
        import torch
        from stage2_model import flow
        def streams(): return {k:torch.Generator().manual_seed(i+31) for i,k in enumerate(["posterior","noise","time"])}
        a=streams();b=streams();m=torch.ones(1,16,4);l=torch.full_like(m,-2.)
        x,t,v=flow(m,l,1.0039506158752403,a)
        y,u,w=flow(m,l,1.0039506158752403,b)
        self.assertTrue(torch.equal(x,y) and torch.equal(t,u) and torch.equal(v,w))
        self.assertEqual(tuple(x.shape),(1,16,4))
        # Recover z1 and z0 to confirm the prescribed noise->data direction.
        z0=x-t[:,None,None]*v;z1=z0+v
        self.assertTrue(torch.allclose(x,(1-t[:,None,None])*z0+t[:,None,None]*z1))
        self.assertFalse(torch.cuda.is_initialized())


if __name__=="__main__": unittest.main()

