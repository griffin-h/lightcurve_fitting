import numpy as np
import json
from glob import glob

for filename in glob('*'):
    print(filename)
    with open(filename) as f:
        data = json.load(f)
    np.savetxt(data[0][0] + '.txt', data[1:], '%.1f %.9f')
