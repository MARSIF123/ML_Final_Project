from sklearn.svm import SVC
import pickle

def svm_Architecture():
    # Hyperparameters
    C = 1.0
    kernel = "linear"
    model = SVC(C=C, kernel=kernel)
    return model

def svm_Training(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

def saving_Model(model, path):
    with open(path, "wb") as f:
        pickle.dump(model, f)
