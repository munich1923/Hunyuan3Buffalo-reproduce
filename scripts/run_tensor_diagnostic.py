from run_pilot import *
state=readstate()
lock=(ROOT/"environment/pilot.lock").open("a")
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
run(state,"tensor_contract",ROOT/"environment/venvs/shape/bin/python","test_edit_model.py",gpu=True,cap=300)
state.update(status="tensor_contract_checked",updated_utc=utc());atomic(state)
