import time
import numpy as np
import tensorflow as tf
import os
import time
import sklearn
import math



from models import SpGAT
from utils import process
from utils import KMeanspp
from utils.KMeanspp import SimilarityType
from utils import Evaluator

config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))

datasetName = 'pubmed'

checkpt_file = 'pre_trained/ckpt/' + datasetName + str(time.monotonic()) + '.ckpt'

from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering



# training params
batch_size = 1
init_seekIters = 5
init_epochs = 0
epochs = 0
epoch_increment = 0.2
episodes = 1000
patience = 100
lr = 0.005  # learning rate
l2_coef = 0.001  # weight decay
hid_units = [8] # numbers of hidden units per each attention head in each layer
n_heads = [8, 8] # additional entry for the output layer
residual = False
nonlinearity = tf.nn.elu
model = SpGAT
clusterRefreshIter = 200000
records = {'evl':[], 'acc':[], 'kmeansIter':[], 'clusterItemCount':[]}

print('Dataset: ' + datasetName)
print('----- Opt. hyperparams -----')
print('lr: ' + str(lr))
print('l2_coef: ' + str(l2_coef))
print('----- Archi. hyperparams -----')
print('nb. layers: ' + str(len(hid_units)))
print('nb. units per layer: ' + str(hid_units))
print('nb. attention heads: ' + str(n_heads))
print('residual: ' + str(residual))
print('nonlinearity: ' + str(nonlinearity))
print('model: ' + str(model))

adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, allY = process.load_data(datasetName)

# features = np.load('data/pubmed/features.npy')
features, spars = process.preprocess_features(features)

dataDistribute = [train_mask.argmin(), 500, 1000]
# features = np.array(features)
# for i in range(features.shape[0]):
#     for j in range(features.shape[1]):
#         if features[i][j] != 0:
#             features[i][j] = 1
# np.save('features', features)
# print("dadasdasdasdasd")

featuresCopy = np.array(features)

featuresCopy = featuresCopy.tolist()

nb_nodes = features.shape[0]
ft_size = features.shape[1]
nb_classes = y_train.shape[1]

# batch_num = int(nb_nodes/batch_size)+1

# features = np.array_split(features,batch_num)
# y_train = np.array_split(y_train,batch_num)
# y_val = np.array_split(y_val,batch_num)
# y_test = np.array_split(y_test,batch_num)
# train_mask = np.array_split(train_mask,batch_num)
# val_mask = np.array_split(val_mask,batch_num)
# test_mask = np.array_split(test_mask,batch_num)



# 先做一次聚类
NMI_pre = 0
init_pred = []
for i in range(init_seekIters):
    SCA = SpectralClustering(n_clusters=nb_classes)
    KMA = KMeans(n_clusters=nb_classes)
    # init_pred = SCA.fit_predict(featuresCopy)
    init_pred_temp = KMA.fit_predict(featuresCopy)
    evl = Evaluator.EvlClustering(featuresCopy, allY, init_pred_temp, labelIsNum=True)
    if evl[0] > NMI_pre:
        init_pred = init_pred_temp
        NMI_pre = evl[0]



# 扩维
features = features[np.newaxis]
y_train = y_train[np.newaxis]
y_val = y_val[np.newaxis]
y_test = y_test[np.newaxis]
train_mask = train_mask[np.newaxis]
val_mask = val_mask[np.newaxis]
test_mask = test_mask[np.newaxis]


biases = process.preprocess_adj_bias(adj)

with tf.Graph().as_default():
    with tf.name_scope('input'):
        ftr_in = tf.placeholder(dtype=tf.float32, shape=(batch_size, nb_nodes, ft_size))
        bias_in = tf.sparse_placeholder(dtype=tf.float32)
        lbl_in = tf.placeholder(dtype=tf.int32, shape=(batch_size, nb_nodes, nb_classes))
        msk_in = tf.placeholder(dtype=tf.int32, shape=(batch_size, nb_nodes))
        attn_drop = tf.placeholder(dtype=tf.float32, shape=())
        ffd_drop = tf.placeholder(dtype=tf.float32, shape=())
        is_train = tf.placeholder(dtype=tf.bool, shape=())

    logits = model.inference(ftr_in, nb_classes, nb_nodes, is_train,
                                attn_drop, ffd_drop,
                                bias_mat=bias_in,
                                hid_units=hid_units, n_heads=n_heads,
                                residual=residual, activation=nonlinearity)
    log_resh = tf.reshape(logits, [-1, nb_classes])
    lab_resh = tf.reshape(lbl_in, [-1, nb_classes])
    msk_resh = tf.reshape(msk_in, [-1])
    loss = model.masked_softmax_cross_entropy(log_resh, lab_resh, msk_resh)
    accuracy = model.masked_accuracy(log_resh, lab_resh, msk_resh)

    train_op = model.training(loss, lr, l2_coef)

    saver = tf.train.Saver()

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())




    formerClusters = []
    with tf.Session(config=config) as sess:
        for episode in range(episodes):
            sess.run(init_op)
            kmeansIterCount = 0
            pseudoAllY = [0 for i in range(len(allY))]
            vlss_mn = np.inf
            vacc_mx = 0.0
            curr_step = 0

            if os.path.exists(checkpt_file + ".index"):
                saver.restore(sess, checkpt_file)

            ts_size = features.shape[0]
            ts_step = 0
            ts_loss = 0.0
            ts_acc = 0.0

            while ts_step * batch_size < ts_size:
                logits_output, loss_value_ts, acc_ts = sess.run([logits, loss, accuracy],
                    feed_dict={
                        ftr_in: features[ts_step * batch_size:(ts_step + 1) * batch_size],
                        bias_in: biases,
                        lbl_in: y_test[ts_step * batch_size:(ts_step + 1) * batch_size],
                        msk_in: test_mask[ts_step * batch_size:(ts_step + 1) * batch_size],
                        is_train: False,
                        attn_drop: 0.0, ffd_drop: 0.0})
                ts_loss += loss_value_ts
                ts_acc += acc_ts
                ts_step += 1
            # print('Test loss:', ts_loss / ts_step, '; Test accuracy:', ts_acc / ts_step)

            # Clustering
            logits_output = logits_output[-1].tolist()

            dataSet = []
            if episode == 0:
                nb_epochs = init_epochs
                classes = np.diag([1 for i in range(nb_classes)]).tolist()
                for i in range(len(init_pred)):
                    pseudoAllY[i] = classes[init_pred[i]]
                    if i < dataDistribute[0]:
                        y_train[0][i] = classes[init_pred[i]]
                    if (i >= dataDistribute[0]) and (i < dataDistribute[0] + dataDistribute[1]):
                        y_val[0][i] = classes[init_pred[i]]
                    if i >= len(y_test[0]) - dataDistribute[2]:
                        y_test[0][i] = classes[init_pred[i]]

            if episode == 1:
                nb_epochs = epochs
                nb_epochs += epoch_increment
                for i in range(len(logits_output)):
                    logits_output[i].insert(0, i)
                    dataSet.append(logits_output[i])
                formerClusters = [[] for i in range(nb_classes)]
                for i in range(nb_nodes):
                    formerClusters[init_pred[i]].append(dataSet[i])

                clusters, kmeansIterCount = \
                    KMeanspp.KMeansClustering(dataSet, nb_classes, 1000, SimilarityType.Cosine, False, formerClusters)
                formerClusters = clusters

                # Allocate pseudo-label
                classes = np.diag([1 for i in range(nb_classes)]).tolist()
                for i in range(len(clusters)):
                    for item in clusters[i]:
                        pseudoAllY[item[0]] = classes[i]
                        if item[0] < dataDistribute[0]:
                            y_train[0][item[0]] = classes[i]
                        if (item[0] >= dataDistribute[0]) and (item[0] < dataDistribute[0] + dataDistribute[1]):
                            y_val[0][item[0]] = classes[i]
                        if item[0] >= len(y_test[0]) - dataDistribute[2]:
                            y_test[0][item[0]] = classes[i]

            elif episode > 1:
                nb_epochs += epoch_increment
                if episode % clusterRefreshIter == 0:
                    nb_epochs = 0
                    print("refresh cluster")
                    # for i in range(len(logits_output)):
                    #     # logits_output[i].insert(0, i)
                    #     dataSet.append(logits_output[i])
                    #
                    # KMA = KMeans(n_clusters=nb_classes)
                    # temp_pred = KMA.fit_predict(dataSet)
                    #
                    # classes = np.diag([1 for i in range(nb_classes)]).tolist()
                    # for i in range(len(temp_pred)):
                    #     pseudoAllY[i] = classes[temp_pred[i]]
                    #     if i < dataDistribute[0]:
                    #         y_train[0][i] = classes[temp_pred[i]]
                    #     if (i >= dataDistribute[0]) and (i < dataDistribute[0] + dataDistribute[1]):
                    #         y_val[0][i] = classes[temp_pred[i]]
                    #     if i >= len(y_test[0]) - dataDistribute[2]:
                    #         y_test[0][i] = classes[temp_pred[i]]
                    # formerClusters = [[] for i in range(nb_classes)]
                    # for i in range(nb_nodes):
                    #     formerClusters[temp_pred[i]].append([i] + dataSet[i])

                    for i in range(len(logits_output)):
                        logits_output[i].insert(0, i)
                        dataSet.append(logits_output[i])

                    clusters, kmeansIterCount = \
                        KMeanspp.KMeansClustering(dataSet, nb_classes, 1000, SSimilarityType.Cosine, True,
                                                  None)
                    formerClusters = clusters

                    # Allocate pseudo-label
                    classes = np.diag([1 for i in range(nb_classes)]).tolist()
                    for i in range(len(clusters)):
                        for item in clusters[i]:
                            pseudoAllY[item[0]] = classes[i]
                            if item[0] < dataDistribute[0]:
                                y_train[0][item[0]] = classes[i]
                            if (item[0] >= dataDistribute[0]) and (item[0] < dataDistribute[0] + dataDistribute[1]):
                                y_val[0][item[0]] = classes[i]
                            if item[0] >= len(y_test[0]) - dataDistribute[2]:
                                y_test[0][item[0]] = classes[i]

                else:
                    for i in range(len(logits_output)):
                        logits_output[i].insert(0, i)
                        dataSet.append(logits_output[i])
                    for i in range(len(formerClusters)):
                        for j in range(len(formerClusters[i])):
                            formerClusters[i][j][1:nb_classes+1]=dataSet[formerClusters[i][j][0]][1:nb_classes+1]
                    clusters, kmeansIterCount = \
                        KMeanspp.KMeansClustering(dataSet, nb_classes, 1000, SimilarityType.Cosine, False, formerClusters)
                    formerClusters = clusters

                    # Allocate pseudo-label
                    classes = np.diag([1 for i in range(nb_classes)]).tolist()
                    for i in range(len(clusters)):
                        for item in clusters[i]:
                            pseudoAllY[item[0]] = classes[i]
                            if item[0] < dataDistribute[0]:
                                y_train[0][item[0]] = classes[i]
                            if (item[0] >= dataDistribute[0]) and (item[0] < dataDistribute[0] + dataDistribute[1]):
                                y_val[0][item[0]] = classes[i]
                            if item[0] >= len(y_test[0]) - dataDistribute[2]:
                                y_test[0][item[0]] = classes[i]



            # Evaluation
            # evl zips [ARI, NMI, HOMO, COMP, VMeasure, FMS, SC]
            evl = Evaluator.EvlClustering(featuresCopy, allY, pseudoAllY)
            records['evl'].append(evl)

            if episode > 0:
                clusterItemCount = [len(clusters[i]) for i in range(len(clusters))]
                records['clusterItemCount'].append(clusterItemCount)
                print(np.array(records['clusterItemCount']))
                records['kmeansIter'].append(kmeansIterCount)
            print(np.array(records['evl']))
            # records['acc'].append([ts_loss / ts_step, ts_acc / ts_step])



            train_loss_avg = 0
            train_acc_avg = 0
            val_loss_avg = 0
            val_acc_avg = 0


            for epoch in range(math.ceil(nb_epochs)):
                tr_step = 0
                tr_size = features.shape[0]

                while tr_step * batch_size < tr_size:
                    _, loss_value_tr, acc_tr = sess.run([train_op, loss, accuracy],
                        feed_dict={
                            ftr_in: features[tr_step*batch_size:(tr_step+1)*batch_size],
                            bias_in: biases,
                            lbl_in: y_train[tr_step*batch_size:(tr_step+1)*batch_size],
                            msk_in: train_mask[tr_step*batch_size:(tr_step+1)*batch_size],
                            is_train: True,
                            attn_drop: 0.6, ffd_drop: 0.6})
                    train_loss_avg += loss_value_tr
                    train_acc_avg += acc_tr
                    tr_step += 1

                vl_step = 0
                vl_size = features.shape[0]

                while vl_step * batch_size < vl_size:
                    loss_value_vl, acc_vl = sess.run([loss, accuracy],
                        feed_dict={
                            ftr_in: features[vl_step*batch_size:(vl_step+1)*batch_size],
                            bias_in: biases,
                            lbl_in: y_val[vl_step*batch_size:(vl_step+1)*batch_size],
                            msk_in: val_mask[vl_step*batch_size:(vl_step+1)*batch_size],
                            is_train: False,
                            attn_drop: 0.0, ffd_drop: 0.0})
                    val_loss_avg += loss_value_vl
                    val_acc_avg += acc_vl
                    vl_step += 1

                print('Episode: %d, epoch: %d, training: loss = %.5f, acc = %.5f | Val: loss = %.5f, acc = %.5f' %
                        (episode, epoch, train_loss_avg/tr_step, train_acc_avg/tr_step,
                        val_loss_avg/vl_step, val_acc_avg/vl_step))

                if val_acc_avg/vl_step >= vacc_mx or val_loss_avg/vl_step <= vlss_mn:
                    if val_acc_avg/vl_step >= vacc_mx and val_loss_avg/vl_step <= vlss_mn:
                        vacc_early_model = val_acc_avg/vl_step
                        vlss_early_model = val_loss_avg/vl_step
                        saver.save(sess, checkpt_file)
                    vacc_mx = np.max((val_acc_avg/vl_step, vacc_mx))
                    vlss_mn = np.min((val_loss_avg/vl_step, vlss_mn))
                    curr_step = 0
                else:
                    curr_step += 1
                    if curr_step == patience:
                        print('Early stop! Min loss: ', vlss_mn, ', Max accuracy: ', vacc_mx)
                        print('Early stop model validation loss: ', vlss_early_model, ', accuracy: ', vacc_early_model)
                        break

                train_loss_avg = 0
                train_acc_avg = 0
                val_loss_avg = 0
                val_acc_avg = 0

        sess.close()
        # Save records
        fileName = 'records\ClusterRecord_' + time.asctime(time.localtime(time.time())).replace(" ","_").replace(":","-")
        np.savetxt(fileName + '_evl.csv', records['evl'], fmt='%f', delimiter=',')
        # np.savetxt(fileName + '_acc.csv', records['acc'], fmt='%f', delimiter=',')
        np.savetxt(fileName + '_kmeansIter.csv', records['kmeansIter'], fmt='%f', delimiter=',')
        np.savetxt(fileName + '_clusterItemCount.csv', records['clusterItemCount'], fmt='%f', delimiter=',')
