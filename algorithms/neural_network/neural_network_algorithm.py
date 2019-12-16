from pre_processing.clearData import PreProcessing

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV


class KNN():
    def __init__(self):
        self._processing = PreProcessing()
        self._NUM_TRIALS = 30

    def algorithm(self):
        hyper = {
            'hidden_layer_sizes': [eval('(' + str(i) + ',)') for i in range(1, 101)],
            'momentum': [0.8],
            'max_iter': [500],
            'activation': ['tanh'],
            'solver': ['sgd'],
            'validation_fraction': [0.1],
            'learning_rate': ['constant', 'adaptive'],
            'early_stopping': [True]
        }

        labels = self._processing.getLabels()
        features = self._processing.getData("ANN")

        features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.30, random_state=0)

        clf = MLPClassifier(random_state=0)

        rs = GridSearchCV(clf, hyper, cv=6, n_jobs=-1, iid=False)

        rs.fit(features_train, labels_train)

        y_pred = rs.predict(features_test)

        recall = recall_score(labels_test, y_pred, average='micro')
        precision = precision_score(labels_test, y_pred, average='micro')
        fmeasure = f1_score(labels_test, y_pred, average='micro')
        accuracy = accuracy_score(labels_test, y_pred)

        print(rs.best_params_)

        return "Recall: %.4f\nPrecision: %.4f\nF-measure: %.4f\nAccuracy: %.4f" % (
            recall, precision, fmeasure, accuracy)

    def execution(self):
        return self.algorithm()

KNN = KNN()

print(KNN.execution())