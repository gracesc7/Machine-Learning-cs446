from sklearn import multiclass, svm


def sklearn_multiclass_prediction(mode, X_train, y_train, X_test):
    '''
    Use Scikit Learn built-in functions multiclass.OneVsRestClassifier
    and multiclass.OneVsOneClassifier to perform multiclass classification.

    Arguments:
        mode: one of 'ovr', 'ovo' or 'crammer'.
        X_train, X_test: numpy ndarray of training and test features.
        y_train: labels of training data, from 0 to 9.

    Returns:
        y_pred_train, y_pred_test: a tuple of 2 numpy ndarrays,
                                   being your prediction of labels on
                                   training and test data, from 0 to 9.
    '''
    #pass
    # x_train dimension: (5000,784)
    #y_train dimension: (5000,)
    if(mode == "ovr"):
        clf = svm.LinearSVC(multi_class = "ovr", random_state = 12345)
        #print("clfclflalalalalal", clf)
        ovr_classifier = multiclass.OneVsRestClassifier(clf)
        ovr_classifier.fit(X_train,y_train)
        y_pred_train = ovr_classifier.predict(X_train)
        y_pred_test = ovr_classifier.predict(X_test)
    if(mode =="ovo"):
        clf = svm.LinearSVC(random_state = 12345)
        ovo_classifier = multiclass.OneVsOneClassifier(clf)
        ovo_classifier.fit(X_train,y_train)
        y_pred_train = ovo_classifier.predict(X_train)
        y_pred_test = ovo_classifier.predict(X_test)
    if(mode == "crammer"):
        clf = svm.LinearSVC(multi_class = "crammer_singer", random_state = 12345)
        clf.fit(X_train,y_train)
        y_pred_train = clf.predict(X_train)
        y_pred_test = clf.predict(X_test)
    return y_pred_train, y_pred_test










    #multiclass.OneVsOneClassifier(clf)

    #if(mode == "ovo" ):

    #if(mode == "crammer"):
    #pass
