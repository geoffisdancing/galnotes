# Code questions


# array.shape[0]
def bootstrap(arr, iterations=10000):
    if type(arr) != np.ndarray:
        arr = np.array(arr)

    if len(arr.shape) < 2:
        arr = arr[:, np.newaxis]

    nrows = arr.shape[0]
    boot_samples = []
    for _ in xrange(iterations):
        row_inds = np.random.randint(nrows, size=nrows)
        boot_sample = arr[row_inds, :]
        boot_samples.append(boot_sample)
    return boot_samples
