import classi_softmax as c
import matplotlib.pyplot as plt
import itertools
from collections import defaultdict
colors = ['red', 'green', 'blue', 'orange', 'purple', 'black']
best =  defaultdict(dict)
best_rate = 0
for j in [x*0.1 for x in range(1,10)]:
    c.setup_network(0,3,30,j)
    for t in range(1,1000):
        c.train_soft(10)
        best[j][t*10] = c.test_soft()
        if c.test_soft()>best_rate:
            best_rate = c.test_soft()

for data, c in zip(list(best.values()),itertools.cycle(colors)):
    x = list(data.keys())
    y = list(data.values())
    plt.plot(x,y,color=c)

plt.savefig('bilchen')
plt.show
print(best)
print(best_rate)
    
