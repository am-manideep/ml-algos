import numpy
import scipy.stats as stats

def question2(N, M):
    X1 = [0] * N
    X2 = [0] * N
    for i in range(1, N+1):
        X1[i-1] = i
        X2[i-1] = i
    X3 = [(i, j) for i in X1 for j in X2]
    X4 = [X3[i] for i in numpy.random.choice(len(X3), size=M, replace=False)]
    X = [max(i, j) - i for (i, j) in X4]
    X5 = []
    X6 = []
    for (i, j) in X4:
        X5.append(max(i, j) - i)
        X6.append(i)
    X_map = {}
    for i in X:
        X_map[i] = 0
    for i in X:
        X_map[i] += 1
    X_prob_array = []
    for key in X_map:
        X_prob_array.append(X_map[key]/M)

    return [numpy.mean(X), numpy.var(X), numpy.cov(X5, X6)[0][1], stats.entropy(X_prob_array, base=2)]
