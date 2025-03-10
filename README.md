1) first few moves are hard coded
2) Spend that time building tree
3) Keep building tree

3 layers 25 epocs 512 hidden:
Error rate: 0.42674694086519155. Average evals = 2.4762191934413784
Total time to test: 0.6545290946960449. Average time: 7.84055833674871e-05

3 layers 25 epocs 1024 hidden:
Error rate: 0.5425329121131816. Average evals = 2.5062917306268195
Total time to test: 0.7934219837188721. Average time: 9.504347596947633e-05

4 layers 25 epocs 512 hidden:
Error rate: 0.409876380645438. Average evals = 2.360135581118787
Total time to test: 0.7709460258483887. Average time: 9.23510710474516e-05

6 layers 25 epocs 512 hidden:
Error rate: 0.4388190658748643. Average evals = 2.4056823855567417
Total time to test: 0.9262411594390869. Average time: 0.00011095377492973207

6 layers 25 epocs 1024 hidden
Error rate: 0.3278641817644046. Average evals = 2.5201715651101235
Total time to test: 4.03649115562439. Average time: 0.0004835280437588292

etc.
My findings from data.csv:
Maximizing epocs is ideal for error rate and doesn't affect time
More hidden dimensions is better for error rate at the cost of time
Less layers is better for both?

Ideal models: always 100+ epocs
6 layers, 1024 hidden dimensions
10 layers, 1024+ hd
3 layers, 512 hidden dimensions