import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def build_rf_classifier(x, y, test_size=0.2, random_state=42, n_estimators=1000):
    # convert to numpy if needed
    if isinstance(x, torch.Tensor):
        x = x.numpy()
    if isinstance(y, torch.Tensor):
        y = y.numpy()

    # split
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y)

    # build and train model
    clf = RandomForestClassifier(n_estimators=5000, random_state=random_state)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print("=== report ===")
    print(classification_report(y_test, y_pred))

    return clf  # return a classifier
