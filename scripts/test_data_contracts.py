"""Check shared transforms, scene transforms, deterministic QC and geometry metric direction."""
import json,pathlib,tempfile
import numpy as np,trimesh
from prepare_dataset import mesh,normalize_pair,paired_qc,geometry_hash
from metrics import compare
from remote_runtime import ROOT
def main():
    source=trimesh.creation.box(extents=[1,1,1]);source.apply_translation([2,3,4])
    addition=trimesh.creation.box(extents=[0.2,0.2,0.2]);addition.apply_translation([2,3.4,4])
    target=trimesh.util.concatenate([source,addition])
    a,b,center,scale=normalize_pair(source,target)
    assert np.allclose(b.vertices[:len(source.vertices)],a.vertices),"Common vertices must remain aligned"
    assert np.allclose(center,[2,3,4])
    assert np.isclose(np.ptp(a.vertices,axis=0).max(),1.5)
    report1=paired_qc(source,target,3);report2=paired_qc(source,target,3)
    assert report1==report2,"QC must be deterministic"
    assert geometry_hash(source)==geometry_hash(source.copy())
    transformed=source.copy();transformed.apply_scale(2.7);transformed.apply_translation([9,-1,2])
    assert geometry_hash(source)==geometry_hash(transformed),"Identity grouping must tolerate translation and scale"
    scene=trimesh.Scene();scene.add_geometry(trimesh.creation.box(),transform=trimesh.transformations.translation_matrix([2,0,0]))
    with tempfile.TemporaryDirectory(dir=ROOT/"tmp") as d:
        path=pathlib.Path(d)/"scene.glb";scene.export(path)
        loaded=mesh(path)
        assert np.allclose(loaded.bounds.mean(axis=0),[2,0,0]),"Scene graph transforms must be applied"
    displaced=a.copy();displaced.apply_translation([0.5,0,0])
    same=compare(a,a,n=4096);wrong=compare(displaced,a,n=4096)
    assert same["cd_squared_sum"]<wrong["cd_squared_sum"]
    assert same["fscore_005"]>wrong["fscore_005"]
    print("PASS shared normalization, deterministic QC, scene graph transforms and metric direction")
if __name__=="__main__":main()
