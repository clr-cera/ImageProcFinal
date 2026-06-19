import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier


def evaluate(model, X_test, y_test):
    pred = model.predict(np.asarray(X_test))
    y_test = np.asarray(y_test)
    return {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
    }


def svm_classifier(
    X_train,
    y_train,
    X_test,
    y_test,
    C=1.0,
    max_iter=20000,
    class_weight="balanced",
    cv=3,
):
    svc = LinearSVC(C=C, max_iter=max_iter, dual="auto", class_weight=class_weight)
    clf = make_pipeline(
        StandardScaler(),
        CalibratedClassifierCV(svc, cv=cv),
    )
    clf.fit(np.asarray(X_train), np.asarray(y_train))
    return clf, evaluate(clf, X_test, y_test)


def logreg_classifier(
    X_train,
    y_train,
    X_test,
    y_test,
    C=1.0,
    max_iter=1000,
    class_weight="balanced",
):
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=C, max_iter=max_iter, class_weight=class_weight),
    )
    clf.fit(np.asarray(X_train), np.asarray(y_train))
    return clf, evaluate(clf, X_test, y_test)


def knn_classifier(
    X_train,
    y_train,
    X_test,
    y_test,
    n_neighbors=5,
    weights="distance",
):
    clf = make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights),
    )
    clf.fit(np.asarray(X_train), np.asarray(y_train))
    return clf, evaluate(clf, X_test, y_test)


def xgboost_classifier(
    X_train,
    y_train,
    X_test,
    y_test,
    n_estimators=200,
    max_depth=4,
    learning_rate=0.1,
):
    y_train = np.asarray(y_train)
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    scale_pos_weight = (n_neg / n_pos) if n_pos else 1.0
    clf = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
    )
    clf.fit(np.asarray(X_train), y_train)
    return clf, evaluate(clf, X_test, y_test)
