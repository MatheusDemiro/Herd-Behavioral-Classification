from pre_processing.clearData import PreProcessing

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV, train_test_split


class KNN():
    def __init__(self):
        self._processing = PreProcessing()
        self._NUM_TRIALS = 30

    def algorithm(self):
        hyper = {"n_neighbors": range(1,31)}

        labels = self._processing.getLabels()
        features = self._processing.getData("KNN")

        # features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.20,
        #                                                                             random_state=0)

        clf = KNeighborsClassifier()

        # rs = GridSearchCV(clf, hyper, cv=6, n_jobs=-1, iid=False)
        #
        # rs.fit(features, labels)

        #y_pred = rs.predict(features_test)

        # recall = recall_score(labels_test, y_pred, average='macro')
        # precision = precision_score(labels_test, y_pred, average='macro')
        # fmeasure = f1_score(labels_test, y_pred, average='macro')
        # accuracy = accuracy_score(labels_test, y_pred)
        #
        # print(rs.best_params_)
        #
        # return "Recall: %.4f\nPrecision: %.4f\nF-measure: %.4f\nAccuracy: %.4f" % (recall, precision, fmeasure, accuracy)

        precision_scores = []
        recall_scores = []
        f1_scores = []
        accuracy = []

        sss = StratifiedShuffleSplit(n_splits=10, test_size=0.30, random_state=0)
        for train_index, test_index in sss.split(features, labels):
            x_train, x_test = features[train_index], features[test_index]
            y_train, y_test = labels[train_index], labels[test_index]

            rs = GridSearchCV(clf, hyper, cv=6, n_jobs=-1, iid=False)

            rs.fit(x_train, y_train)
            y_pred = rs.predict(x_test)

            print(rs.best_params_)
            print()
            print("Recall: %.4f" % recall_score(y_test, y_pred, average='macro'))
            print("Precision: %.4f" % precision_score(y_test, y_pred, average='macro'))
            print("F-measure: %.4f" % f1_score(y_test, y_pred, average='macro'))
            print("Accuracy: %.4f" % accuracy_score(y_test, y_pred))
            print()

            precision_scores.append(precision_score(y_test, y_pred, average='macro'))
            recall_scores.append(recall_score(y_test, y_pred, average='macro'))
            f1_scores.append(f1_score(y_test, y_pred, average='macro'))
            accuracy.append(accuracy_score(y_test, y_pred))

        return "Recall: %.4f\nPrecision: %.4f\nF-measure: %.4f\nAccuracy: %.4f"%(sum(recall_scores)/len(recall_scores),
                                                                 sum(precision_scores)/len(precision_scores),
                                                                 sum(f1_scores) / len(f1_scores),
                                                                 sum(accuracy) / len(accuracy))

    def execution(self):
        return self.algorithm()

KNN = KNN()

print(KNN.execution())