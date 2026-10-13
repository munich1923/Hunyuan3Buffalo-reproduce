from run_pilot import *
lock=(ROOT/"environment/pilot.lock").open("a")
fcntl.flock(lock,fcntl.LOCK_EX)
state=readstate();state.pop("error",None)
run(state,"vae_precision_probe",ROOT/"environment/venvs/shape/bin/python","probe_vae_precision.py",gpu=True,cap=900)
state.update(status="precision_probe_checked",updated_utc=utc());atomic(state)
