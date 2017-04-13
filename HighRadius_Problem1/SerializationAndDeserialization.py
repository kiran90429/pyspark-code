#reference
#http://stackoverflow.com/questions/30698004/how-can-i-serialize-a-numpy-array-while-preserving-matrix-dimensions

import pickle
import StringIO;
import numpy as np;
import json;

a = [[1,2,3],[4,5,6]];

print a;

#serialize
serialized = pickle.dumps(a,protocol=0);

memfile = StringIO.StringIO();

np.save(memfile,a);

memfile.seek(0);

serialized = json.dumps(memfile.read().decode('latin-1'));

print serialized;

#deserialize


#memfile = StringIO.StringIO;

memfile.write(json.loads(serialized).encode('latin-1'));
memfile.seek(0)
a = np.load(memfile)

print a;

print a[1][1]

