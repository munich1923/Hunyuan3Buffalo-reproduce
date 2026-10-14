"""Render one exported mesh in a fresh EGL process."""
import argparse,trimesh
from render_views import render
p=argparse.ArgumentParser();p.add_argument('mesh');p.add_argument('output');a=p.parse_args()
render(trimesh.load(a.mesh,force='mesh'),a.output)
