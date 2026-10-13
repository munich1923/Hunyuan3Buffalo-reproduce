import sys
from run_pilot import ROOT,fcntl,main
lock=(ROOT/"environment/pilot.lock").open("a")
fcntl.flock(lock,fcntl.LOCK_EX);fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
sys.argv=[sys.argv[0],"--prepare-only"];main()
