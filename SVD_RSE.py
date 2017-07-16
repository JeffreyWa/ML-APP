#coding=utf-8

# Run some recommendation experiments using MovieLens 100K
import pandas
import numpy
import scipy.sparse
import scipy.sparse.linalg
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sparsesvd import sparsesvd
import codecs

def moive_recomm():
    data_dir = "data/ml-100k/"
    data_shape = (943, 1682)

    df = pandas.read_csv(data_dir + "ua.base", sep="\t", header=-1)
    values = df.values
    values[:, 0:2] -= 1
    X_train = scipy.sparse.csr_matrix((values[:, 2], (values[:, 0], values[:, 1])), dtype=numpy.float, shape=data_shape)

    df = pandas.read_csv(data_dir + "ua.test", sep="\t", header=-1)
    values = df.values
    values[:, 0:2] -= 1
    X_test = scipy.sparse.csr_matrix((values[:, 2], (values[:, 0], values[:, 1])), dtype=numpy.float, shape=data_shape)
    
    moive_data = {}
    with codecs.open(data_dir + "u.item.txt") as f:
        for line in f:
            moive = line.strip().split("|")[:2]
            moive_data[int(moive[0])] = moive[1]

    # Compute means of nonzero elements
    train_rows, train_cols = X_train.nonzero()
    
    X_row_mean = numpy.zeros(data_shape[0])
    X_row_sum = numpy.zeros(data_shape[0])
    # Iterate through nonzero elements to compute sums and counts of rows elements
    for i in range(train_rows.shape[0]):
        X_row_mean[train_rows[i]] += X_train[train_rows[i], train_cols[i]]
        X_row_sum[train_rows[i]] += 1

    # Note that (X_row_sum == 0) is required to prevent divide by zero
    X_row_mean /= X_row_sum + (X_row_sum == 0)
    
    # Subtract mean rating for each user
    for i in range(train_rows.shape[0]):
        X_train[train_rows[i], train_cols[i]] -= X_row_mean[train_rows[i]]
    
    ########substract#################
    ##train_mat = X_train - np.asarray([(np.mean(X_train, 1))]).T

    test_rows, test_cols = X_test.nonzero()

    for i in range(test_rows.shape[0]):
        X_test[test_rows[i], test_cols[i]] -= X_row_mean[test_rows[i]]

    X_train = numpy.array(X_train.toarray())
    X_test = numpy.array(X_test.toarray())
    '''
    mask = numpy.isnan(X_train)
    masked_arr=numpy.ma.masked_array(X_train, mask)
    item_means=numpy.mean(masked_arr, axis=0)
    X_train = masked_arr.filled(item_means)
    x = numpy.tile(item_means, (X_train.shape[0],1))
    X_train = X_train - x
    '''
    ks = numpy.arange(2, 50)
    train_mae = numpy.zeros(ks.shape[0])
    test_mae = numpy.zeros(ks.shape[0])
    train_scores = X_train[(train_rows, train_cols)]
    test_scores = X_test[(test_rows, test_cols)]

	# Now take SVD of X_train
    U, s, Vt = numpy.linalg.svd(X_train, full_matrices=False)

    for j, k in enumerate(ks):
#        U, s, Vt = sparsesvd(X_train, k)
        X_pred = U[:, 0:k].dot(numpy.diag(s[0:k])).dot(Vt[0:k, :])

        pred_train_scores = X_pred[(train_rows, train_cols)]
        pred_test_scores = X_pred[(test_rows, test_cols)]

        train_mae[j] = mean_absolute_error(train_scores, pred_train_scores)
        test_mae[j] = mean_absolute_error(test_scores, pred_test_scores)

        print(k,  train_mae[j], test_mae[j])

    print('recommendation for ',moive_data[1])
    idx = test(1, 50, Vt)
    print(idx)
    for id in idx + 1:
        print(moive_data[id])

    plt.plot(ks, train_mae, 'k', label="Train")
    #plt.plot(ks, test_mae, 'r', label="Test")
    plt.xlabel("k")
    plt.ylabel("MAE")
    plt.legend()
    plt.show()

def top_cosine_similarity(data, movie_id, top_n=10):
    index = movie_id - 1 # Movie id starts from 1
    movie_row = data[index, :]
    magnitude = numpy.sqrt(numpy.einsum('ij, ij -> i', data, data))
    similarity = numpy.dot(movie_row, data.T) / (magnitude[index] * magnitude)
    sort_indexes = numpy.argsort(-similarity)
    return sort_indexes[:top_n]

def test(moiveid, k, V):
    sliced = V.T[:, :k] # representative data
    return top_cosine_similarity(sliced,moiveid)


if __name__ == "__main__":
    moive_recomm()

