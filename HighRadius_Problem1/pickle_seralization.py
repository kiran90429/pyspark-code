import pickle
import numpy as np;

b = [[1,2,3],[2,3,4],[5,6,7]];

serialize = pickle.dumps(b,protocol=0);

print b

b = np.matrix(b);

print b.shape

deserialize = pickle.loads(serialize);

print deserialize;

print deserialize[2][2];