import random
import sys
sys.path.append("../x64/PythonLib/")
from ANN import *

import loader

print(loader.to_percent([1, 0.5, 0.1, 0.3]))

print("OpenCL initialized!!!")


batch_size = 10
epoch_count = 30

n = ANN([784, 100, 10])
print("Network initialized")



training_samples = []
validation_samples = []
test_samples = []

loader.load(training_samples, validation_samples, test_samples)

test_samples = test_samples[0:8]

random.shuffle(training_samples)

for test_sample in test_samples:
	print(test_sample[1], ": ", loader.to_percent(n.feedForward(test_sample[0])))

for k in range(epoch_count):
    for i in range(0, len(training_samples) - batch_size, batch_size):
        n.learn(3.0 / batch_size, training_samples[i:i+batch_size])
    print()
    print()
    for test_sample in test_samples:
        print(test_sample[1], ": ", loader.to_percent(n.feedForward(test_sample[0])))
        

for test_sample in test_samples:
    print(test_sample[1], ": ", loader.to_percent(n.feedForward(test_sample[0])))

input()
