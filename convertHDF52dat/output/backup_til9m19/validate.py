from pathlib import Path
import numpy as np
import sys

argv = sys.argv

p = Path("./"+argv[1])

for dat in list( p.glob("*.dat") ):
    data = np.loadtxt(str(dat))
    print( str(dat), np.max(data, axis=0))
