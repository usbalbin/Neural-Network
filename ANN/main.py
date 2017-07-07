import random
import sys
sys.path.append("../x64/PythonLibDebug/")
from ANN import *

import loader


print("OpenCL initialized!!!")


batch_size = 10
epoch_count = 5             #30
learning_rate = 0.5

n = ANN("network.save")     #ANN([784, 100, 10])
print("Network initialized")



training_samples = []
validation_samples = []
test_samples = []

loader.load(training_samples, validation_samples, test_samples)

test_samples = test_samples[:100]

random.shuffle(training_samples)

print(loader.accuracy(n, test_samples), "of", len(test_samples))

for k in range(epoch_count):
    for i in range(0, len(training_samples) - batch_size, batch_size):
        n.learn(learning_rate / batch_size, training_samples[i:i+batch_size])
    print()
    print()
    print(loader.accuracy(n, test_samples), "of", len(test_samples))
        
print("\n\n\n-------------------FINAL RESULTS-------------------")
print(loader.accuracy(n, test_samples), "of", len(test_samples))
"""for test_sample in test_samples:
    print(test_sample[1], ": ", loader.to_percent(n.feedForward(test_sample[0])))
"""
n.writeToFile("network.save")                                   #Save network to file
n2 = ANN("network.save")                                         #Reopen
print(loader.accuracy(n2, test_samples), "of", len(test_samples))#Check if result is same
input()
