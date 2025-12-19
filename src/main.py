from data_Processing import load_and_process_data
from knn_Model import knn_Architecture, knn_Training, saving_Model as save_knn
from svm_Model import svm_Architecture, svm_Training, saving_Model as save_svm
from model_Evaluation import evaluate_model, compare_results

# Load and process data
X_train, X_test, y_train, y_test = load_and_process_data(
    "/content/dataset.csv"
)

# ----- KNN -----
knn_model = knn_Architecture()
knn_model = knn_Training(knn_model, X_train, y_train)
knn_acc = evaluate_model(knn_model, X_test, y_test)
save_knn(knn_model, "/content/ML_Final_Project/result/knn_model.pkl")

# ----- SVM -----
svm_model = svm_Architecture()
svm_model = svm_Training(svm_model, X_train, y_train)
svm_acc = evaluate_model(svm_model, X_test, y_test)
save_svm(svm_model, "/content/ML_Final_Project/result/svm_Model.pkl")

# Compare results
compare_results(svm_acc, knn_acc)
