"""Meaningful CPU contract tests; no pretrained payload loading or CUDA initialization."""
from stage2_runtime import *
import copy
import tempfile
import unittest
import random
import numpy as np
import torch
from stage2_data import quotas,stratify,normalize,surface,geometry_hash
from stage2_model import Connector,Stage2Generator,flow,lr_factor
from stage2_optimizer import CPUOffloadAdamW
from stage2_checkpoint import save_checkpoint,load_checkpoint,restore_rng,rng_state


class Contracts(unittest.TestCase):
    def test_quota_and_disjoint_splits(self):
        counts={"a":55,"b":30,"c":15}
        q=quotas(counts,17)
        self.assertEqual(sum(q.values()),17)
        rows=[{"uid":f"{c}{i}","category":c} for c in counts for i in range(30)]
        splits,_=stratify(rows,counts,{"train":30,"val":6,"test":6},12)
        seen=set()
        for split,size in [("train",30),("val",6),("test",6)]:
            self.assertEqual(len(splits[split]),size)
            ids={r["uid"] for r in splits[split]}
            self.assertFalse(ids & seen);seen|=ids

    def test_geometry_transform_sampling(self):
        import trimesh
        mesh=trimesh.creation.box(extents=[2,4,1])
        mesh.apply_translation([2,3,4])
        result,center,scale=normalize(mesh)
        self.assertAlmostEqual(np.ptp(result.vertices,axis=0).max(),1.5)
        np.testing.assert_allclose(result.bounds.mean(axis=0),0)
        self.assertEqual(geometry_hash(mesh),geometry_hash(result))
        a=surface(result,128,45);b=surface(result,128,45)
        np.testing.assert_array_equal(a,b)
        self.assertEqual(a.shape,(128,7));self.assertTrue((a[:,-1]==0).all())

    def test_connector_length_and_gradient(self):
        c=Connector(12,8)
        for n in [1,9,37]:
            h=torch.randn(1,n,12,requires_grad=True)
            y=c(h)
            self.assertEqual(y.shape,(1,n,8))
            y.square().mean().backward()
            self.assertTrue(torch.isfinite(h.grad).all())

    def test_flow_path_and_scaling(self):
        mean=torch.randn(1,8,4);logvar=torch.full_like(mean,-2.)
        g=torch.Generator().manual_seed(99)
        noisy,t,velocity=flow(mean,logvar,1.003,g)
        g=torch.Generator().manual_seed(99)
        target=1.003*(mean+torch.exp(.5*logvar)*torch.randn(mean.shape,generator=g))
        noise=torch.randn(mean.shape,generator=g)
        torch.testing.assert_close(velocity,target-noise)
        torch.testing.assert_close(noisy,(1-t[:,None,None])*noise+t[:,None,None]*target)

    def test_scheduler_official_grid(self):
        from hy3dshape.schedulers import FlowMatchEulerDiscreteScheduler
        scheduler=FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000)
        scheduler.set_timesteps(sigmas=np.linspace(0,1,50),device="cpu")
        value=torch.zeros(1,8,4)
        for t in scheduler.timesteps:
            value=scheduler.step(torch.ones_like(value),t,value).prev_sample
        torch.testing.assert_close(value,torch.ones_like(value))
        self.assertEqual(float(scheduler.sigmas[-2]),1.)
        self.assertEqual(float(scheduler.sigmas[-1]),1.)

    def test_lr_and_tail(self):
        self.assertAlmostEqual(lr_factor(80,1565,80,.1),1)
        self.assertAlmostEqual(lr_factor(1565,1565,80,.1),.1)
        # Actual 16-sample tail gradient equals mean, rather than an erroneous division by 32.
        a=torch.tensor(2.,requires_grad=True)
        losses=[(a-i)**2 for i in range(16)]
        sum(loss/16 for loss in losses).backward()
        expected=2*(2-np.arange(16)).mean()
        self.assertAlmostEqual(float(a.grad),expected)

    def tiny(self):
        from hy3dshape.models.denoisers.hunyuandit import HunYuanDiTPlain
        return HunYuanDiTPlain(input_size=8,in_channels=4,hidden_size=32,context_dim=8,
            depth=3,num_heads=4,mlp_ratio=2,qk_norm=True,use_attention_pooling=False,
            num_moe_layers=2,num_experts=3,moe_top_k=2,qkv_bias=False)

    def test_native_parity_aux_and_checkpoint_gradient(self):
        torch.manual_seed(22)
        native=self.tiny().train()
        wrapped=Stage2Generator(copy.deepcopy(native),12).train()
        h=torch.randn(1,7,12)
        x=torch.randn(1,8,4)
        t=torch.rand(1)
        cond=wrapped.connector(h).detach()
        expected=native(x,t,{"main":cond})
        actual,aux=wrapped(x,t,h,False)
        torch.testing.assert_close(actual,expected)
        expected.square().mean().backward()
        (actual.square().mean()+aux).backward()
        for name,param in native.named_parameters():
            other=dict(wrapped.dit.named_parameters())[name]
            if param.grad is not None:
                torch.testing.assert_close(param.grad,other.grad,rtol=2e-4,atol=2e-6,msg=name)
        plain=copy.deepcopy(wrapped)
        checked=copy.deepcopy(wrapped)
        plain.zero_grad();checked.zero_grad()
        yp,ap=plain(x,t,h,False);yc,ac=checked(x,t,h,True)
        (yp.square().mean()+ap).backward();(yc.square().mean()+ac).backward()
        for (name,a),(_,b) in zip(plain.named_parameters(),checked.named_parameters()):
            if a.grad is not None:
                torch.testing.assert_close(a.grad,b.grad,rtol=2e-4,atol=2e-6,msg=name)
        self.assertTrue(all(p.requires_grad for p in wrapped.parameters()))

    def test_explicit_aux_accumulation(self):
        from hy3dshape.models.denoisers.moe_layers import MoEBlock
        from stage2_model import explicit_moe_forward
        import types
        torch.manual_seed(55)
        a=MoEBlock(8,num_experts=3,moe_top_k=2,ff_inner_dim=16)
        a.forward=types.MethodType(explicit_moe_forward,a)
        b=copy.deepcopy(a)
        xs=[torch.randn(1,5,8),torch.randn(1,5,8)]
        for x in xs:
            y=a(x);aux=a._stage2_aux;a._stage2_aux=None
            ((y.square().mean()+aux)/len(xs)).backward()
        terms=[]
        for x in xs:
            y=b(x);aux=b._stage2_aux;b._stage2_aux=None
            terms.append(y.square().mean()+aux)
        torch.stack(terms).mean().backward()
        for p,q in zip(a.parameters(),b.parameters()):
            if p.grad is not None: torch.testing.assert_close(p.grad,q.grad)

    def test_cpu_offload_adamw_parity_and_restore(self):
        torch.manual_seed(7)
        a=torch.nn.Linear(4,3);b=copy.deepcopy(a)
        oa=torch.optim.AdamW(a.parameters(),lr=1e-3,foreach=False)
        ob=CPUOffloadAdamW([{"params":list(b.parameters()),"lr":1e-3}])
        x=torch.randn(5,4)
        for _ in range(3):
            oa.zero_grad();ob.zero_grad()
            a(x).square().mean().backward();b(x).square().mean().backward()
            oa.step();ob.step()
        for p,q in zip(a.parameters(),b.parameters()): torch.testing.assert_close(p,q)
        state=ob.state_dict()
        oc=CPUOffloadAdamW([{"params":list(b.parameters()),"lr":1e-3}])
        oc.load_state_dict(state)
        self.assertEqual(oc.state_dict()["backend"],"cpu_adamw")

    def test_checkpoint_resume_rng_and_no_overwrite(self):
        with tempfile.TemporaryDirectory(dir=path("environment/tmp/stage2")) as folder:
            model=torch.nn.Linear(4,3);opt=torch.optim.AdamW(model.parameters())
            g=torch.Generator().manual_seed(3)
            model(torch.ones(1,4)).sum().backward();opt.step()
            file=Path(folder)/"step_000500"
            contract={"test":True}
            save_checkpoint(file,model,opt,{"update":500,"epoch":1,"offset":0},
                            contract,g,{"interval":{"samples":0}})
            expected=torch.rand(5)
            state=load_checkpoint(file,contract)
            restore_rng(state["rng"])
            torch.testing.assert_close(torch.rand(5),expected)
            clone=torch.nn.Linear(4,3);clone.load_state_dict(state["model"])
            for p,q in zip(model.parameters(),clone.parameters()):torch.testing.assert_close(p,q)
            with self.assertRaises(FileExistsError):
                save_checkpoint(file,model,opt,{},contract,g,{})
            with self.assertRaises(RuntimeError):
                load_checkpoint(file,{"test":False})

    def test_checkpoint_continuous_vs_resume_trajectory(self):
        with tempfile.TemporaryDirectory(dir=path("environment/tmp/stage2")) as folder:
            torch.manual_seed(72)
            original=torch.nn.Linear(4,3)
            optimizer=torch.optim.AdamW(original.parameters(),lr=1e-3)
            def update(model,opt):
                opt.zero_grad()
                data=torch.randn(5,4)
                model(data).square().mean().backward();opt.step()
            update(original,optimizer)
            save_checkpoint(Path(folder)/"checkpoint",original,optimizer,
                            {"update":1,"epoch":0,"offset":5},{"trajectory":True},
                            torch.Generator().manual_seed(7),{})
            update(original,optimizer)
            expected={k:v.clone() for k,v in original.state_dict().items()}
            loaded=load_checkpoint(Path(folder)/"checkpoint",{"trajectory":True})
            resumed=torch.nn.Linear(4,3);resumed.load_state_dict(loaded["model"])
            opt=torch.optim.AdamW(resumed.parameters(),lr=1e-3)
            opt.load_state_dict(loaded["optimizer"]);restore_rng(loaded["rng"])
            update(resumed,opt)
            for key,value in resumed.state_dict().items():
                torch.testing.assert_close(value,expected[key],rtol=0,atol=0)

    def test_protocol_and_no_cross_split_objects(self):
        from stage2_data import protocol
        cfg=config("configs/stage2_nano10k.yaml")
        rows={split:[{"uid":f"{split}_{category}_{i}","category":category,
                     "prompt":"A shape"} for category in ["a","b","c"] for i in range(80)]
              for split in ["train","val","test"]}
        result=protocol(rows,cfg)
        self.assertEqual(len(result["vae"]),30)
        self.assertEqual(len(result["val_heldout"]),100)
        for split in ["val","test"]:
            self.assertEqual(len(result[split+"_pairs"]),20)
            ids={row["uid"] for row in rows[split]}
            self.assertTrue(set(result[split+"_heldout"])<=ids)

    def test_cache_fingerprint_rejects_changed_prompt(self):
        from stage2_cache import text_key,shape_key,load_cached
        cfg={"cache":{"hidden_dim":12,"latent_shape":[8,4]}}
        row={"uid":"object","prompt":"A chair","mesh_sha256":"meshhash",
             "normalization":{"scale":1},"surface_seed":1}
        with tempfile.TemporaryDirectory(dir=path("environment/tmp/stage2")) as folder:
            base=Path(folder)
            bases={kind:(base/kind,kind+"contract") for kind in ["text","shape"]}
            save_tensor(base/"text/train/object.pt",{"hidden":torch.randn(7,12),
                        "fingerprint":text_key(row,"textcontract")})
            save_tensor(base/"shape/train/object.pt",{"mean":torch.randn(8,4),
                        "logvar":torch.randn(8,4),"fingerprint":shape_key(row,"shapecontract")})
            hidden,mean,logvar=load_cached(cfg,row,"train",bases)
            self.assertEqual(hidden.shape,(1,7,12))
            with self.assertRaises(RuntimeError):
                load_cached(cfg,{**row,"prompt":"A table"},"train",bases)

    def test_cpu_fixed_view_render_and_tensorboard_event(self):
        from stage2_evaluation import render_cpu
        from torch.utils.tensorboard import SummaryWriter
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        import trimesh
        with tempfile.TemporaryDirectory(dir=path("environment/tmp/stage2")) as folder:
            folder=Path(folder)
            render_cpu(trimesh.creation.box(),folder/"views")
            self.assertEqual(len(list((folder/"views").glob("*.png"))),4)
            writer=SummaryWriter(str(folder/"events"))
            writer.add_scalar("test/flow_mse",.25,100);writer.close()
            events=EventAccumulator(str(folder/"events"));events.Reload()
            self.assertEqual(events.Scalars("test/flow_mse")[0].step,100)

    def test_budget_and_occupied_gpu_admission(self):
        from train_stage2 import validate_budget
        from unittest.mock import patch
        cfg=config("configs/stage2_nano10k.yaml")
        with tempfile.TemporaryDirectory(dir=path("environment/tmp/stage2")) as folder:
            approval=Path(folder)/"unapproved.json"
            atomic_json(approval,{"approved":False})
            with self.assertRaisesRegex(RuntimeError,"not been approved"):
                validate_budget(cfg,Path(folder),{},approval)
        responses=["GPU-test, 39575, 81559, 0", "GPU-test, 1234"]
        with patch("stage2_runtime.subprocess.check_output",side_effect=responses):
            with self.assertRaisesRegex(RuntimeError,"occupied"):
                with gpu_lease(999): self.fail("Occupied device was admitted")

    def test_vae_decoder_only_matches_native_without_encoder(self):
        from hy3dshape.models.autoencoders import ShapeVAE
        from stage2_vae import GeometryDecoder,PosteriorEncoder
        params={"num_latents":8,"embed_dim":4,"width":32,"heads":4,
                "num_decoder_layers":1,"num_encoder_layers":1,"pc_size":128,
                "pc_sharpedge_size":0,"point_feats":4,"qkv_bias":False,
                "qk_norm":True,"num_freqs":2,"include_pi":False}
        native=ShapeVAE(**params).eval()
        decoder=GeometryDecoder(params).eval()
        encoder=PosteriorEncoder(params).eval()
        state=native.state_dict()
        decoder.load_state_dict({k:v for k,v in state.items() if k in decoder.state_dict()})
        encoder.load_state_dict({k:v for k,v in state.items() if k in encoder.state_dict()})
        self.assertFalse(hasattr(decoder,"encoder"))
        self.assertFalse(hasattr(decoder,"pre_kl"))
        self.assertFalse(hasattr(encoder,"geo_decoder"))
        latent=torch.randn(1,8,4)
        torch.testing.assert_close(decoder.decode(latent),native.decode(latent))

    def test_userdata_guard(self):
        with self.assertRaises(ValueError): path("/tmp/stage2")
        for name in ["HF_HOME","TORCH_HOME","TMPDIR","PIP_CACHE_DIR","TRITON_CACHE_DIR"]:
            self.assertTrue(Path(os.environ[name]).is_relative_to(ROOT))

    def test_real_tokenizer_selection(self):
        from transformers import AutoTokenizer
        from stage2_cache import tokenize_condition
        tokenizer=AutoTokenizer.from_pretrained(str(path("models/Qwen2.5-VL-7B-Instruct")),
                                                local_files_only=True)
        prompt="A chair with three legs and a tall back."
        encoded,indices,rendered=tokenize_condition(tokenizer,prompt)
        decoded=tokenizer.decode(encoded["input_ids"][0,indices[:-1]])
        self.assertEqual(decoded,prompt)
        self.assertEqual(indices[-1],encoded["input_ids"].shape[1]-1)
        self.assertNotIn("helpful assistant",decoded)
        _,empty,_=tokenize_condition(tokenizer,"")
        self.assertEqual(len(empty),1)


if __name__=="__main__":
    torch.set_num_threads(1)
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Contracts)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    report={"tests":result.testsRun,"failures":len(result.failures),"errors":len(result.errors),
            "passed":result.wasSuccessful(),"cuda_initialized":torch.cuda.is_initialized(),**clock()}
    atomic_json("results/summaries/stage2_cpu_tests.json",report)
    raise SystemExit(0 if result.wasSuccessful() else 1)
