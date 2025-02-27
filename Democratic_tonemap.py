import numpy as np

def solvedynaprog(D, B, h, H, x, N, K):
    for i in range(K):
        for j in range(N):
            D[i, j] = 0
            B[i, j] = 0
    
    for k in range(K):
        mean_x1 = x[0]
        for i in range(1, N):
            if k == 0:
                D[k, i] = D[k, i - 1] + (H[i - 1] / H[i]) * ((x[i] - mean_x1) ** 2) * h[i]
                mean_x1 = (H[i - 1] * mean_x1 + h[i] * x[i]) / H[i]
                B[k, i] = 0
            else:
                D[k, i] = -1
                d = 0
                mean_xj = 0
                for j in range(i, -1, -1):
                    if j > 0:
                        d += ((H[i] - H[j]) / (H[i] - H[j - 1])) * ((x[j] - mean_xj) ** 2) * h[j]
                        mean_xj = (h[j] * x[j] + (H[i] - H[j]) * mean_xj) / (H[i] - H[j - 1])
                    else:
                        d += ((H[i] - H[j]) / H[i]) * ((x[j] - mean_xj) ** 2) * h[j]
                        mean_xj = (h[j] * x[j] + (H[i] - H[j]) * mean_xj) / H[i]
                    
                    if D[k, i] == -1 or (j == 0 and d <= D[k, i]) or (j > 0 and d + D[k - 1, j - 1] < D[k, i]):
                        D[k, i] = d if j == 0 else d + D[k - 1, j - 1]
                        B[k, i] = j

def backtrack(c, h, H, x, B, N, K):
    cluster_right = N - 1
    for k in range(K - 1, -1, -1):
        cluster_left = int(B[k, cluster_right])
        sumo = sum(h[a] * x[a] for a in range(cluster_left, cluster_right + 1))
        c[k] = sumo / (H[cluster_right] - (H[cluster_left - 1] if cluster_left > 0 else 0))
        if k > 0:
            cluster_right = cluster_left - 1

def hist_init(x, h, im, N, M, nrbins, maxy):
    x[:] = np.linspace(0, maxy, nrbins)
    h.fill(0)
    nrbins2 = 0
    
    for i in range(N):
        for j in range(M):
            id = int((nrbins - 1) * im[i, j] / maxy)
            if h[id] == 0:
                nrbins2 += 1
            h[id] += 1
    
    return nrbins2

def hist(x, h, nrbins):
    x2, h2, H2 = [], [], []
    sumo = 0
    for i in range(nrbins):
        if h[i] > 0:
            sumo += h[i]
            h2.append(h[i])
            x2.append(x[i])
            H2.append(sumo)
    return np.array(x2), np.array(h2), np.array(H2)

def toneim(im, c, maxy, nrbins, K, N, M):
    id_map = np.zeros(nrbins, dtype=int)
    
    for i in range(nrbins):
        x = i * maxy / (nrbins - 1)
        id_map[i] = np.argmin(np.abs(c - x))
    
    for i in range(N):
        for j in range(M):
            for k in range(3):
                idde = idde = min(max(int((nrbins - 1) * im[i, j, k] / maxy), 0), nrbins - 1)
                im[i, j, k] = c[id_map[idde]]

def tonemap(im, K):
    N, M, _ = im.shape
    maxfact = 5_000_000
    nrbins = 5000
    
    MM = np.max(im)
    mm = np.min(im)
    imgr = np.max(np.log1p(maxfact * (im - mm) / (MM - mm)), axis=2)
    
    x = np.zeros(nrbins)
    h = np.zeros(nrbins)
    nrbins2 = hist_init(x, h, imgr, N, M, nrbins, np.log1p(maxfact))
    
    if K < nrbins2:
        x2, h2, H2 = hist(x, h, nrbins)
        D = np.zeros((K, nrbins2))
        B = np.zeros((K, nrbins2))
        c = np.zeros(K)
        
        solvedynaprog(D, B, h2, H2, x2, nrbins2, K)
        backtrack(c, h2, H2, x2, B, nrbins2, K)
        toneim(im, c, np.log1p(maxfact), nrbins, K, N, M)
    else:
        toneim(im, x, np.log1p(maxfact), nrbins, K, N, M)
    
    return im
