import numpy as np
from sklearn import svm


class MulticlassSVM:

    def __init__(self, mode):
        if mode != 'ovr' and mode != 'ovo' and mode != 'crammer-singer':
            raise ValueError('mode must be ovr or ovo or crammer-singer')
        self.mode = mode

    def fit(self, X, y):
        if self.mode == 'ovr':
            self.fit_ovr(X, y)
        elif self.mode == 'ovo':
            self.fit_ovo(X, y)
        elif self.mode == 'crammer-singer':
            self.fit_cs(X, y)

    def fit_ovr(self, X, y):
        self.labels = np.unique(y)
        self.binary_svm = self.bsvm_ovr_student(X, y)

    def fit_ovo(self, X, y):
        self.labels = np.unique(y)
        self.binary_svm = self.bsvm_ovo_student(X, y)

    def fit_cs(self, X, y):
        self.labels = np.unique(y)
        X_intercept = np.hstack([X, np.ones((len(X), 1))])

        N, d = X_intercept.shape
        K = len(self.labels)

        W = np.zeros((K, d))

        n_iter = 1500
        #n_iter = 3
        learning_rate = 1e-8
        print("X_intercept shape:",np.shape(X_intercept))
        #a = self.grad_student(W,X_intercept,y)
        for i in range(n_iter):
            W -= learning_rate * self.grad_student(W, X_intercept, y)
            #print(i)

        self.W = W

    def predict(self, X):
        if self.mode == 'ovr':
            return self.predict_ovr(X)
        elif self.mode == 'ovo':
            return self.predict_ovo(X)
        else:
            return self.predict_cs(X)

    def predict_ovr(self, X):
        scores = self.scores_ovr_student(X)
        return self.labels[np.argmax(scores, axis=1)]

    def predict_ovo(self, X):
        scores = self.scores_ovo_student(X)
        return self.labels[np.argmax(scores, axis=1)]

    def predict_cs(self, X):
        X_intercept = np.hstack([X, np.ones((len(X), 1))])
        return np.argmax(self.W.dot(X_intercept.T), axis=0)

    def bsvm_ovr_student(self, X, y):
        '''
        Train OVR binary classfiers.

        Arguments:
            X, y: training features and labels.

        Returns:
            binary_svm: a dictionary with labels as keys,
                        and binary SVM models as values.
        '''
        #pass
        #clf = svm.LinearSVC(random_state = 12345)
        binary_svm = {}
        N_samples = np.shape(X)[0]
        n_labels = self.labels.shape[0]
        for i in range(n_labels):
            clf = svm.LinearSVC(random_state = 12345)
            b_label = np.zeros((N_samples,))
            label_idx = np.where(y == i)
            print("label_idx shape", np.shape(label_idx))
            for idx in label_idx[0]:
                b_label[idx] = 1  #b_label is new y_label
            clf.fit(X,b_label)
            binary_svm.update({i:clf})
            #print("clf_ovr:",clf)
        return binary_svm


    def bsvm_ovo_student(self, X, y):
        '''
        Train OVO binary classfiers.

        Arguments:
            X, y: training features and labels.

        Returns:
            binary_svm: a dictionary with label pairs as keys,
                        and binary SVM models as values.
        '''
        pair_list = []
        binary_svm = {}
        N_samples = np.shape(X)[0]
        n_labels = self.labels.shape[0]
        for i in range(n_labels-1):
            for j in range(i+1,n_labels):
                pair = (i,j)
                pair_list.append(pair)
        for p in pair_list:
            #print("p[0],p[1]:",p[0],p[1])
            label0_idx = np.where(y == p[0])
            label1_idx = np.where(y == p[1])
            #print("label0_idx:", label0_idx)
            #print("label1_idx:",label1_idx)
            #print("label shape", np.shape(label1_idx))
            #b_label = np.zeros((n_labels,))
            y_0 = np.zeros((np.shape(label0_idx)[1],1))
            y_1 = np.ones((np.shape(label1_idx)[1],1))
            #print("y_0 shape:", np.shape(y_0))
            #print("y_1 shape:",  np.shape(y_1))
            #print("y_0", y_0)
            #print("y_1:",y_1)
            x_0 = np.take(X,label0_idx,axis =0)
            x_0 = np.squeeze(x_0, axis  = 0)
            x_1 = np.take(X,label1_idx, axis = 0)
            x_1 = np.squeeze(x_1, axis = 0)
            #print("x_0 shape", np.shape(x_0))
            #print("x_1 shape", np.shape(x_1))
            x = np.concatenate([x_0,x_1])
            y_label = np.concatenate([y_0,y_1])
            #print("ylabel shape:",np.shape(y_label),"x shape",np.shape(x))
            clf = svm.LinearSVC(random_state = 12345)
            clf.fit(x,y_label)
            binary_svm.update({p:clf})
        return binary_svm


    def scores_ovr_student(self, X):
        '''
        Compute class scores for OVR.

        Arguments:
            X: Features to predict.

        Returns:
            scores: a numpy ndarray with scores.
        '''
        #return shape of (#samples, #labels)
        #confidence score
        N_samples = np.shape(X)[0]
        feature_len = np.shape(X)[1]
        n_labels = len(self.binary_svm)
        scores = np.zeros((N_samples,n_labels))
        for i in range(N_samples):
            for label in self.binary_svm.keys():
                x_feature = np.reshape(X[i],(1,feature_len))
                scores[i][label] = self.binary_svm[label].decision_function(x_feature)
        return scores

        #pass

    def scores_ovo_student(self, X):
        '''
        Compute class scores for OVO.

        Arguments:
            X: Features to predict.

        Returns:
            scores: a numpy ndarray with scores.
        '''
        #return shape of (#samples, #labels)
        #score(i,j) is # of votes of jth label received for ith sample in OVO
        N_samples = np.shape(X)[0]
        feature_len = np.shape(X)[1]
        n_labels = len(self.binary_svm)
        scores = np.zeros((N_samples,n_labels))
        for i in range(N_samples):
            for label_pair in self.binary_svm.keys():
                idx_0 = label_pair[0]
                idx_1 = label_pair[1]
                x_feature = np.reshape(X[i],(1,feature_len))
                pred = self.binary_svm[label_pair].predict(x_feature)
                if(pred == 0):
                    scores[i][idx_0] += 1
                else:
                    scores[i][idx_1] += 1
        return scores


    def loss_student(self, W, X, y, C=1.0):
        '''
        Compute loss function given W, X, y.

        For exact definitions, please check the MP document.

        Arugments:
            W: Weights. Numpy array of shape (K, d)
            X: Features. Numpy array of shape (N, d)
            y: Labels. Numpy array of shape N
            C: Penalty constant. Will always be 1 in the MP.

        Returns:
            The value of loss function given W, X and y.
        '''
        '''
        N_samples = np.shape(X)[0]
        n_labels = self.labels.shape[0]
        feature_len = np.shape(X)[1]
        #print("w shape:", np.shape(W))
        #print("X shape:", np.shape(X))
        #print("y shape:", np.shape(y))
        w_loss = 0
        for cl in self.W:
            w = [cl]
            wTw =np.matmul(w,w.T)
            w_loss += wTw
        w_loss = 1/2* w_loss
        slack_loss = 0
        for i in range(N_samples):
            find_max = []
            for j in range(n_labels):
                x = np.reshape(X[i],(1,feature_len))
                w = [self.W[j]]
                xwT = np.matmul(x,w.T)
                if(y[i] == j):
                    m = xwT
                    xTwyi = xwT
                else:
                    m = 1+xwT
                print(m)
                print(type(m[0][0]), "m[0][0]:",m[0][0])
                find_max.append(m)
            tempmax = max(find_max) - xTwyi
            slack_loss += tempmax
        slack_loss = slack_loss * C
        loss_sum = w_loss + slack_loss
        return loss_sum
        '''
        '''
        N_samples = np.shape(X)[0]
        #n_labels = self.labels.shape[0]
        feature_len = np.shape(X)[1]
        #w_loss = 0

        for cl in W:
            w = np.reshape(cl,(1,feature_len))
            wTw =np.matmul(w,w.T)
            w_loss += wTw
        w_loss = 1/2* w_loss

        scores = X.dot(W.T)
        yi_scores = scores[np.arange(scores.shape[0]),y]
        print("yi shape:", np.shape(yi_scores))
        loss = np.maximum(0,scores - np.matrix(yi_scores).T+1)
        print("loss shape", np.shape(loss))
        loss[np.arange(N_samples),y] = 0
        loss = np.sum(loss,axis=1) + 1/2*np.sum(W*W)
        return loss
        '''
        N_samples = np.shape(X)[0]
        scores = X.dot(W.T)
        scores[np.arange(N_samples),y] -= 1
        scores = scores+1
        max_score = np.amax(scores,axis = 1)
        yi_score = scores[np.arange(N_samples),y]
        loss = 1/2*np.sum(W*W)+C*(np.sum(max_score)-np.sum(yi_score))
        return loss

    def grad_student(self, W, X, y, C=1.0):
        '''
        Compute gradient function w.r.t. W given W, X, y.

        For exact definitions, please check the MP document.

        Arugments:
            W: Weights. Numpy array of shape (K, d)
            X: Features. Numpy array of shape (N, d)
            y: Labels. Numpy array of shape N
            C: Penalty constant. Will always be 1 in the MP.

        Returns:
            The graident of loss function w.r.t. W,
            in a numpy array of shape (K, d).
        '''
        #print("w shape g:", np.shape(W))
        #print("X shape g:", np.shape(X))
        #print("y shape g:", np.shape(y))
        #print(self.predict_cs(X))
        #print(np.shape(self.predict_cs(X)))
        #w_gradient = 0
        '''
        N_samples = np.shape(X)[0]
        n_labels = self.labels.shape[0]
        feature_len = np.shape(X)[1]
        grad_sum = np.zeros((n_labels,feature_len))
        for i in range(N_samples):
            find_max = []
            for j in range(n_labels):
                x = np.reshape(X[i],(1,feature_len))
                #print(x)
                #print("type of w[j]", type(W[j]))
                #print("shape of w[j]", np.shape(W[j]))
                w = np.reshape(W[j],(1,np.shape(W[j])[0]))
                #print(w)
                #print("shape of w", np.shape(w))
                xwT = np.matmul(x,w.T)
                if(y[i] == j):
                    m = xwT
                    #xTwyi = xwT
                else:
                    m = 1+xwT
                find_max.append(m[0][0])
                #print(m)
                #print(type(m[0][0]), "m[0][0]:",m[0][0])
                #print("shape of x,w", np.shape(x), np.shape(w))
            find_max = np.array(find_max)
            #print("findmax:", find_max)
            #print("findmax shape:", np.shape(find_max))
            label_max = np.argmax(find_max) #UNSURE
            #print("label_max:",label_max)
            for f in range(feature_len):
                W[label_max][f] += X[i][f]
                W[y[i]][f] -= X[i][f]
        print("done")
        grad_sum  = grad_sum*C
        grad_sum += W
        return grad_sum
        '''
        N_samples = np.shape(X)[0]
        scores = X.dot(W.T)
        scores[np.arange(N_samples),y] -= 1
        scores = scores+1
        max_cls = np.argmax(scores,axis = 1)
        binary_1 = np.zeros((np.shape(scores)))
        binary_2 = np.zeros((np.shape(scores)))
        binary_1[np.arange(N_samples),max_cls] = 1
        binary_2[np.arange(N_samples),y] = -1
        grad1 = binary_1.T.dot(X)
        grad2 = binary_2.T.dot(X)
        grad = C*(grad1 +grad2)
        grad += W
        return grad
