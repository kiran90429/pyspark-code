import StringIO;
import json
import numpy
memfile = StringIO.StringIO;

memfile.write(json.loads(serialized).encode('latin-1'));
memfile.seek(0)
a = numpy.load(memfile)

