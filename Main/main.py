import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import joblib
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

df = pd.read_csv(r'D:\HUIT\Học Máy\NhomEEE_DoAnCuoiKy\Demo\Dataset\heart.csv')
df.head(10)

print("Số dòng của Dataset là:", df.shape[0])
print("\nSố cột của Dataset là:", df.shape[1])

df.isna().sum()

df.info()

df.describe()

figure, axis = plt.subplots(3,5, figsize = (20,10))

axis[0, 0].hist(df['age'])
axis[0, 0].set_title('Histogram of age')

axis[0, 1].hist(df['sex'])
axis[0, 1].set_title('Histogram of sex')

axis[0, 2].hist(df['cp'])
axis[0, 2].set_title('Histogram of cp')

axis[0, 3].hist(df['trestbps'])
axis[0, 3].set_title('Histogram of trestbps')

axis[0, 4].hist(df['chol'])
axis[0, 4].set_title('Histogram of chol')

axis[1, 0].hist(df['fbs'])
axis[1, 0].set_title('Histogram of fbs')

axis[1, 1].hist(df['restecg'])
axis[1, 1].set_title('Histogram of restecg')

axis[1, 2].hist(df['thalach'])
axis[1, 2].set_title('Histogram of thalach')

axis[1, 3].hist(df['exang'])
axis[1, 3].set_title('Histogram of exang')

axis[1, 4].hist(df['oldpeak'])
axis[1, 4].set_title('Histogram of oldpeak')

axis[2, 0].hist(df['slope'])
axis[2, 0].set_title('Histogram of slope')

axis[2, 1].hist(df['ca'])
axis[2, 1].set_title('Histogram of ca')

axis[2, 2].hist(df['thal'])
axis[2, 2].set_title('Histogram of thal')

axis[2, 3].hist(df['target'])
axis[2, 3].set_title('Histogram of target')

plt.show()


figure, axis = plt.subplots(1, figsize=(20, 10))
heatmap = sns.heatmap(df.corr(), annot=True, ax=axis)
axis.set_title('Heatmap')

plt.show()

df['target'].value_counts()

df_class_0 = df[df['target'] == 0]
df_class_1 = df[df['target'] == 1]

df_class_0_under = df_class_0.sample(499)
concat_df = pd.concat([df_class_1, df_class_0_under])

concat_df.head()


labels = concat_df['target']
values = concat_df[['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']]

best_features = SelectKBest(score_func=chi2, k='all')
fit = best_features.fit(values, labels)

df_scores = pd.DataFrame(fit.scores_, columns=["Scores"])
df_columns = pd.DataFrame(values.columns, columns=["Feature"])

feature_scores = pd.concat([df_columns, df_scores], axis=1)
feature_scores['Scores'] = feature_scores['Scores'].round(2)
feature_scores = feature_scores.sort_values(by='Scores', ascending=False)

print(feature_scores)

x = concat_df[['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']]
y = concat_df['target']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

logistic_model = LogisticRegression().fit(x_train, y_train)
decision_model = DecisionTreeClassifier().fit(x_train, y_train)
knn_model = KNeighborsClassifier().fit(x_train, y_train)
random_model = RandomForestClassifier().fit(x_train, y_train)

pre_y = logistic_model.predict(x_train)
target_names = ['class 0', 'class 1']
print(classification_report(y_train, pre_y, target_names=target_names))

def plot_confusion_matrix(model, x_data, y_data):
    y_pred = model.predict(x_data)
    labels = np.unique(y_data)
    cm = confusion_matrix(y_data, y_pred, labels=labels)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Nhẵn dự đoán')
    plt.ylabel('Nhẵn thực tế')
    plt.title('Ma trận nhầm lẫn')
    plt.show()

plot_confusion_matrix(logistic_model, x_train, y_train)

pre_y2 = decision_model.predict(x_train)
target_names = ['class 0', 'class 1']
print(classification_report(y_train, pre_y2, target_names=target_names))

def plot_confusion_matrix(model, x_data, y_data):
    y_pred = model.predict(x_data)
    labels = np.unique(y_data)
    cm = confusion_matrix(y_data, y_pred, labels=labels)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Nhẵn dự đoán')
    plt.ylabel('Nhẵn thực tế')
    plt.title('Ma trận nhầm lẫn')
    plt.show()

plot_confusion_matrix(decision_model, x_train, y_train)

pre_y3 = knn_model.predict(x_train)
target_names = ['class 0', 'class 1']
print(classification_report(y_train, pre_y3, target_names=target_names))

def plot_confusion_matrix(model, x_data, y_data):
    y_pred = model.predict(x_data)
    labels = np.unique(y_data)
    cm = confusion_matrix(y_data, y_pred, labels=labels)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Nhẵn dự đoán')
    plt.ylabel('Nhẵn thực tế')
    plt.title('Ma trận nhầm lẫn')
    plt.show()

plot_confusion_matrix(knn_model, x_train, y_train)

pre_y4 = random_model.predict(x_train)
target_names = ['class 0', 'class 1']
print(classification_report(y_train, pre_y4, target_names=target_names))

def plot_confusion_matrix(model, x_data, y_data):
    y_pred = model.predict(x_data)
    labels = np.unique(y_data)
    cm = confusion_matrix(y_data, y_pred, labels=labels)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Nhẵn dự đoán')
    plt.ylabel('Nhẵn thực tế')
    plt.title('Ma trận nhầm lẫn')
    plt.show()

plot_confusion_matrix(random_model, x_train, y_train)

pre_test1 = logistic_model.predict(x_test)
target_names = ['class 0', 'class 1']
print(classification_report(y_test, pre_test1, target_names=target_names))

def plot_confusion_matrix(model, x_data, y_data):
    y_pred = model.predict(x_data)
    labels = np.unique(y_data)
    cm = confusion_matrix(y_data, y_pred, labels=labels)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Nhẵn dự đoán')
    plt.ylabel('Nhẵn thực tế')
    plt.title('Ma trận nhầm lẫn')
    plt.show()

plot_confusion_matrix(logistic_model, x_test, y_test)

pre_test2 = decision_model.predict(x_test)
target_names = ['class 0', 'class 1']
print(classification_report(y_test, pre_test2, target_names=target_names))

def plot_confusion_matrix(model, x_data, y_data):
    y_pred = model.predict(x_data)
    labels = np.unique(y_data)
    cm = confusion_matrix(y_data, y_pred, labels=labels)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Nhẵn dự đoán')
    plt.ylabel('Nhẵn thực tế')
    plt.title('Ma trận nhầm lẫn')
    plt.show()

plot_confusion_matrix(decision_model, x_test, y_test)

pre_test3 = knn_model.predict(x_test)
target_names = ['class 0', 'class 1']
print(classification_report(y_test, pre_test3, target_names=target_names))

def plot_confusion_matrix(model, x_data, y_data):
    y_pred = model.predict(x_data)
    labels = np.unique(y_data)
    cm = confusion_matrix(y_data, y_pred, labels=labels)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Nhẵn dự đoán')
    plt.ylabel('Nhẵn thực tế')
    plt.title('Ma trận nhầm lẫn')
    plt.show()

plot_confusion_matrix(knn_model, x_test, y_test)

pre_test4 = random_model.predict(x_test)
target_names = ['class 0', 'class 1']
print(classification_report(y_test, pre_test4, target_names=target_names))

def plot_confusion_matrix(model, x_data, y_data):
    y_pred = model.predict(x_data)
    labels = np.unique(y_data)
    cm = confusion_matrix(y_data, y_pred, labels=labels)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Nhẵn dự đoán')
    plt.ylabel('Nhẵn thực tế')
    plt.title('Ma trận nhầm lẫn')
    plt.show()

plot_confusion_matrix(random_model, x_test, y_test)

joblib.dump(logistic_model, 'logistic_model.pkl')
joblib.dump(decision_model, 'decision_model.pkl')
joblib.dump(knn_model, 'knn_model.pkl')
joblib.dump(random_model, 'random_model.pkl')

# Tải lại các mô hình đã lưu
loaded_logistic_model = joblib.load('logistic_model.pkl')
loaded_decision_model = joblib.load('decision_model.pkl')
loaded_knn_model = joblib.load('knn_model.pkl')
loaded_random_model = joblib.load('random_model.pkl')