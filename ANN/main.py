import sys
sys.path.append("../x64/PythonLib/")
from ANN import *

n = ANN([3, (25 * 25), 1])
"""
samples = [
        Sample(Vector([1.0, 1.0, 1.0]), Vector([1.0])),
	Sample(Vector([0.0, 1.0, 0.0]), Vector([1.0])),
	Sample(Vector([0.0, 1.0, 1.0]), Vector([1.0])),
	Sample(Vector([0.0, 0.0, 0.0]), Vector([0.0])),
	Sample(Vector([1.0, 0.0, 1.0]), Vector([1.0])),
	Sample(Vector([1.0, 1.0, 0.0]), Vector([1.0]))
]

for i in range(100000):
    n.learn(0.2, samples)
    if i % 1000 == 0:
        print()
        print()
        print("1 1 1", n.feedForward([1.0, 1.0, 1.0]))
        print("0 1 1", n.feedForward([0.0, 1.0, 1.0]))
        print("1 1 0", n.feedForward([1.0, 1.0, 0.0]))
        print("1 0 1", n.feedForward([1.0, 0.0, 1.0]))
        print("0 0 0", n.feedForward([0.0, 0.0, 0.0]))


input()
"""
