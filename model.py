from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from imblearn.over_sampling import SMOTE
from sklearn import svm, metrics
from xgboost import XGBClassifier
import matplotlib.pyplot as pt
import pandas as pd
import numpy as np
import joblib


# Load the data
data = pd.read_csv('Datasets/main.csv')
x = data[['sleep_interval','appetite','going_out','friendship_status','family_relationships','relationship','hobbies_and_interests','self_perception','bullying_experience','social_media_impact','academic_pressure','intoxication','_and_ambitions','age','grade']].values
y = data[['depressionLevel']].values


# Encoding categorical features
le = LabelEncoder()
for i in range(x.shape[1]):
    x[:,i]= le.fit_transform(x[:, i])

y=le.fit_transform(y.ravel())
print(y[:20])

# Splitting data into training and test data
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.45)

# Standardize the features
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#Apply Smote to training data
smote = SMOTE(random_state=25)
x_train, y_train = smote.fit_resample(x_train, y_train)

#Ensure a balanced test set
smote_test = SMOTE(random_state=25, sampling_strategy='minority')
x_test, y_test = smote_test.fit_resample(x_test, y_test)


class Models:

    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test


    def svm_model(self):
        classify = svm.SVC(probability=True)
        classify.fit(self.x_train, self.y_train)
        predict = classify.predict(self.x_test)
        accuracy = metrics.accuracy_score(predict, self.y_test)
        score = metrics.f1_score(self.y_test, predict, average='weighted')
        precision = metrics.precision_score(self.y_test, predict, average='weighted')
        recall = metrics.recall_score(self.y_test, predict, average='weighted')
        roc_auc = metrics.roc_auc_score(self.y_test, classify.predict_proba(self.x_test), multi_class='ovr') if hasattr(classify,'predict_proba') else 'N/A'
        return roc_auc, predict, classify, accuracy, score, precision, recall

    def knn_model(self):
        classify = KNeighborsClassifier(weights='uniform', n_neighbors=42)
        classify.fit(self.x_train, self.y_train)
        predict = classify.predict(self.x_test)
        accuracy = metrics.accuracy_score(predict, self.y_test)
        score = metrics.f1_score(self.y_test, predict, average='weighted')
        precision = metrics.precision_score(self.y_test, predict, average='weighted')
        recall = metrics.recall_score(self.y_test, predict, average='weighted')
        return accuracy, score, precision, recall

    def random_model(self):
        classify = RandomForestClassifier(n_jobs=4, random_state=0)
        classify.fit(self.x_train, self.y_train)
        predict = classify.predict(self.x_test)
        accuracy = metrics.accuracy_score(predict, self.y_test)
        score = metrics.f1_score(self.y_test, predict, average='weighted')
        precision = metrics.precision_score(self.y_test, predict, average='weighted')
        recall = metrics.recall_score(self.y_test, predict, average='weighted')
        return accuracy, score, precision, recall

    def decision_model(self):
        classify = DecisionTreeClassifier(criterion='gini', random_state=0)
        classify.fit(self.x_train, self.y_train)
        predict = classify.predict(self.x_test)
        accuracy = metrics.accuracy_score(predict, self.y_test)
        score = metrics.f1_score(self.y_test, predict, average='weighted')
        precision = metrics.precision_score(self.y_test, predict, average='weighted')
        recall = metrics.recall_score(self.y_test, predict, average='weighted')
        return accuracy, score, precision, recall

    def gaussian_model(self):
        classify = GaussianNB()
        classify.fit(self.x_train, self.y_train)
        predict = classify.predict(self.x_test)
        accuracy = metrics.accuracy_score(predict, self.y_test)
        score = metrics.f1_score(self.y_test, predict, average='weighted')
        precision = metrics.precision_score(self.y_test, predict, average='weighted')
        recall = metrics.recall_score(self.y_test, predict, average='weighted')
        return accuracy, score, precision, recall

    def xgboost_model(self):
        classify = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss',
                                 scale_pos_weight=len(y_train[y_train == 0]) / len(
                                     y_train[y_train == 1]))
        classify.fit(self.x_train, self.y_train)
        predict = classify.predict(self.x_test)
        accuracy = metrics.accuracy_score(self.y_test, predict)
        score = metrics.f1_score(self.y_test, predict, average='weighted')
        precision = metrics.precision_score(self.y_test, predict, average='weighted')
        recall = metrics.recall_score(self.y_test, predict, average='weighted')
        return accuracy, score, precision, recall



if __name__ == '__main__':

    # Train all models and get results
     model = Models(x_train, x_test, y_train, y_test)
     random_acc, random_f1, random_pre, random_rec = model.random_model()
     decision_acc, decision_f1, decision_pre, decision_rec = model.decision_model()
     xgb_acc, xgb_f1, xgb_pre, xgb_rec = model.xgboost_model()


    #If there are missing values
     if data.isnull().any().any():

         #Output results for all models
         print(f"Random Forest Model: \n1)Accuracy: {random_acc * 100}% \n2)F1_score: {random_f1 * 100}% \n3)Precision: {random_pre * 100}% \n4)Recall: {random_rec * 100}%")
         print(f"Decision Tree Model: \n1)Accuracy: {decision_acc * 100}% \n2)F1_score: {decision_f1 * 100}% \n3)Precision: {decision_pre * 100}% \n4)Recall: {decision_rec * 100}%")

         #Plotting the graphs
         x_graph = np.arange(4)
         random = [random_acc * 100, random_f1 * 100, random_pre * 100, random_rec * 100]
         decision = [decision_acc * 100, decision_f1 * 100, decision_pre * 100, decision_rec * 100]
         width = 0.1
         pt.bar(x_graph, random, width, color='yellow')
         pt.bar(x_graph + 0.1, decision, width, color='green')
         pt.xticks(x_graph, ['Accuracy', 'F1_Score', 'Precision', 'Recall'])
         pt.xlabel('Evaluation of ML Models')
         pt.ylabel('Percentage')
         pt.legend(['Random Forest', "Decision Tree"])

     #If there are no missing values
     else:
         # Train all models and get results
         roc, svm_predict, trained_model, svm_acc, svm_f1, svm_pre, svm_rec = model.svm_model()
         knn_acc, knn_f1, knn_pre, knn_rec = model.knn_model()
         gaussian_acc, gaussian_f1, gaussian_pre, gaussian_rec = model.gaussian_model()

         #Print the correct instance
         print(f"Confusion Matrix (for best performing-model): \n {confusion_matrix(y_test, svm_predict)}")

         ##Output results for all models
         print(f"SVM Model: \n1)Accuracy: {svm_acc * 100:.2f}% \n2)F1_score: {svm_f1 * 100:.2f}% \n3)Precision: {svm_pre * 100:.2f}% \n4)Recall: {svm_rec * 100:.2f}%\n{roc}")
         print(f"KNeighbours Model: \n1)Accuracy: {knn_acc * 100:.2f}% \n2)F1_score: {knn_f1 * 100:.2f}% \n3)Precision: {knn_pre * 100:.2f}% \n4)Recall: {knn_rec * 100:.2f}%")
         print(f"Random Forest Model: \n1)Accuracy: {random_acc * 100:.2f}% \n2)F1_score: {random_f1 * 100:.2f}% \n3)Precision: {random_pre * 100:.2f}% \n4)Recall: {random_rec * 100:.2f}%")
         print(f"Decision Tree Model: \n1)Accuracy: {decision_acc * 100:.2f}% \n2)F1_score: {decision_f1 * 100:.2f}%\n3)Precision: {decision_pre * 100:.2f}% \n4)Recall: {decision_rec * 100:.2f}%")
         print(f"Gaussian Model: \n1)Accuracy: {gaussian_acc * 100:.2f}% \n2)F1_score: {gaussian_f1 * 100:.2f}% \n3)Precision: {gaussian_pre * 100:.2f}% \n4)Recall: {gaussian_rec * 100:.2f}%")
         print(f"XGBoost Model: \n1)Accuracy: {xgb_acc * 100:.2f}% \n2)F1_score: {xgb_f1 * 100:.2f}% \n3)Precision: {xgb_pre * 100:.2f}% \n4)Recall: {xgb_rec * 100:.2f}%")

         #Plotting the graphs
         x_graph = np.arange(4)
         svm = [svm_acc * 100, svm_f1 * 100, svm_pre * 100, svm_rec * 100]
         knn = [knn_acc * 100, knn_f1 * 100, knn_pre * 100, knn_rec * 100]
         random = [random_acc * 100, random_f1 * 100, random_pre * 100, random_rec * 100]
         decision = [decision_acc * 100, decision_f1 * 100, decision_pre * 100, decision_rec * 100]
         gaussian = [gaussian_acc * 100, gaussian_f1 * 100, gaussian_pre * 100, gaussian_rec * 100]
         xgboost = [xgb_acc*100, xgb_f1*100, xgb_pre*100, xgb_rec*100]
         width = 0.1

         pt.bar(x_graph - 0.2, svm, width, color='cyan')
         pt.bar(x_graph - 0.1, knn, width, color='orange')
         pt.bar(x_graph, random, width, color='yellow')
         pt.bar(x_graph + 0.1, decision, width, color='green')
         pt.bar(x_graph + 0.2, gaussian, width, color='red')
         pt.bar(x_graph+0.3, xgboost, width, color='black')


         pt.xticks(x_graph, ['Accuracy', 'F1_Score', 'Precision', 'Recall'])
         pt.xlabel('Evaluation of ML Models')
         pt.ylabel('Percentage')
         pt.legend(['SVM', 'KNeighbour', 'Random Forest', "Decision Tree", 'Gaussian', 'XGBoost'])


     pt.show()

     #Save the training model
     joblib.dump(trained_model, 'model.pkl')

     #Testing the model
     model=joblib.load('model.pkl')
     print(type(model))



