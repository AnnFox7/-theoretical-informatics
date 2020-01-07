import csv
import numpy as np
import warnings

from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")


def unique(list1):
    return np.unique(np.array(list1)).tolist()


if __name__ == "__main__":
    news_data_train = list(csv.reader(open('news_train.txt', 'rt',
                                           encoding="utf8"), delimiter='\t'))
    news_data_test = list(csv.reader(open('news_test.txt', 'rt',
                                          encoding="utf8"), delimiter='\t'))

    X_train = []
    for news in news_data_train:
        X_train.append(news[2])

    Y_train = []
    for target in news_data_train:
        Y_train.append(target[0])

    X_test = []
    for news in news_data_test:
        X_test.append(news[1])

    class_names = unique(Y_train)
    le = preprocessing.LabelEncoder()
    le.fit(class_names)
    Y_train = le.transform(Y_train)

    print("Data is loaded!")
    print('Length train data is', len(X_train))
    print('Length test data is', len(X_test))
    print("Creating pipeline from CountVectorizer, TFidf, SGDClassifier")
    pipeline_models = Pipeline([('count_vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('SGD', SGDClassifier(loss='modified_huber', penalty='l2',
                                           alpha=1e-4, random_state=150)),
                         ])
    print("Start validation model on 33% of train data...")
    x_tr, x_valid, y_tr, y_valid = train_test_split(X_train, Y_train,
                                                    test_size=0.33,
                                                    random_state=42)
    val_pipeline = pipeline_models.fit(x_tr, y_tr)
    predicted_tr = val_pipeline.predict_proba(x_tr)
    predicted_val = val_pipeline.predict_proba(x_valid)

    pr_tr = []
    for x in predicted_tr:
        result = np.argmax(x)
        pr_tr.append(result)

    pr_val = []
    for x in predicted_val:
        result = np.argmax(x)
        pr_val.append(result)

    print("Accuracy score on train data is ", round(accuracy_score(y_tr, pr_tr), 2))
    print("Accuracy score on valid data is ", round(accuracy_score(y_valid, pr_val), 2))
    print("Start training model and predict classes")
    model = pipeline_models.fit(X_train, Y_train)
    predicted = model.predict(X_test)
    predicted = le.inverse_transform(predicted)
    print(f"Writing {predicted.shape[0]} predicted samples in news_output.txt")
    with open('news_output.txt', 'w') as f:
        for item in predicted:
            f.write("%s\n" % item)