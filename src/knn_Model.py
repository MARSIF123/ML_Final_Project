from sklearn.neighbors import KNeighborsClassifier
import pickle

def knn_Architecture():
    # Hyperparameter
    k = 5
    model = KNeighborsClassifier(n_neighbors=k)
    return model

def knn_Training(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

def saving_Model(model, path):
    with open(path, "wb") as f:
        pickle.dump(model, f)
