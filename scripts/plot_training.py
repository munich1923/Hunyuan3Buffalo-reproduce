"""Standalone scientific figure from recorded velocity loss probes."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from remote_runtime import ROOT
def main():
    reports=[]
    for stage in ["overfit","connector","lora"]:
        p=ROOT/"environment"/("training_"+stage+".json")
        if p.exists():reports.append(json.loads(p.read_text()))
    if not reports:return
    fig,axes=plt.subplots(1,len(reports),figsize=(5*len(reports),4),squeeze=False)
    for ax,report in zip(axes[0],reports):
        h=report["history"];x=[0]+[r["step"] for r in h]
        ax.plot(x,[report["initial_train"]]+[r["train_probe"] for r in h],label="Fixed train probe")
        ax.plot([r["step"] for r in h],[r["validation"] for r in h],label="Validation probe (16)")
        ax.scatter([0,report["steps"]],[report["initial_val"],report["final_val"]],c="black",marker="x",label="Initial/final validation")
        ax.set(title=report["stage"],xlabel="Optimizer steps",ylabel="Velocity MSE")
        ax.grid(alpha=.25);ax.legend(fontsize=8)
    fig.suptitle("Fitting diagnostics; loss reduction does not prove geometry editing")
    fig.tight_layout();folder=ROOT/"results/figures";folder.mkdir(exist_ok=True)
    fig.savefig(folder/"training_velocity_losses.png",dpi=160);plt.close(fig)
if __name__=="__main__":main()
