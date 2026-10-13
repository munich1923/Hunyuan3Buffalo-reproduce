"""CPU adapter loading contract; detects incomplete or wrong-revision checkpoints."""
import tempfile,torch
from pathlib import Path
from torch import nn
from edit_model import Editor
SHA="82920d643c0dc2f7bfd7255f45f62d386edfe60c"
class Tiny(Editor):
    def __init__(self,lora=False):
        nn.Module.__init__(self);self.connector=nn.Linear(2,3);self.lora_names=["test"] if lora else []
        if lora:
            self.dit=nn.Module();self.dit.attn=nn.Module();self.dit.attn.a=nn.Linear(2,1,bias=False);self.dit.attn.b=nn.Linear(1,2,bias=False)
def main():
    root=Path(__file__).resolve().parents[1]/"tmp"
    with tempfile.TemporaryDirectory(dir=root) as tmp:
        path=Path(tmp)/"adapter.pt";plain=Tiny();lora=Tiny(True)
        state={"adapter":plain.adapter_state(),"stage":"connector","source_commit":SHA}
        torch.save(state,path);plain.load_adapter(path);lora.load_adapter(path)
        full={"adapter":lora.adapter_state(),"stage":"lora","source_commit":SHA}
        torch.save(full,path);lora.load_adapter(path)
        for bad in [dict(full,adapter={"connector.weight":full["adapter"]["connector.weight"]}),dict(full,source_commit="wrong")]:
            torch.save(bad,path)
            try:lora.load_adapter(path)
            except RuntimeError:continue
            raise AssertionError("Bad adapter accepted")
    print("PASS full adapters, connector-to-LoRA initialization, missing tensors and source revision")
if __name__=="__main__":main()
