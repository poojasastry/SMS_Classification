
# coding: utf-8

# # Team Trini Machine Learning Project: SMS Spam Classification

# ## 1. Import Libraries

# In[225]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')
import warnings
warnings.filterwarnings('ignore')
get_ipython().magic("config InlineBackend.figure_format = 'retina'")


# ## 2. Load Data

# In[226]:

messages = pd.read_csv("./spam.csv",encoding='latin-1')
messages.head()


# ## 2.1. Data Preprocessing 

# ## 2.1.1. Drop the unnamed columns and change column 'v1' to 'label' and 'v2' to 'text'

# In[227]:

messages = messages.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
messages = messages.rename(columns={"v1":"label", "v2":"text"})
messages.tail()


# ## 2.1.2. Convert the column 'label' to have numerical values for 'ham' and 'spam'

# In[228]:

messages['label_num'] = messages.label.map({'ham':0, 'spam':1})
messages.head()


# ## 2.1.3. Drop rows of dataframe whose value is 'NaN'

# In[229]:

messages.dropna(subset=['text'])
#Count the number of observations for each label 'ham' and 'spam'
ham_count,spam_count = messages.label.value_counts()
print(messages.label.value_counts())


# ## 2.2. Explore Data

# In[230]:

sns.countplot(x="label",data=messages)


# In[231]:

messages.groupby('label').describe()


# #### Check if there is any connection between the lengths of spam and ham messages 

# In[232]:

messages['length'] = messages['text'].map(lambda text: len(text))
messages.hist(column='length', by='label', bins=25) 


# ## 2.3. Data Visualization

# In[233]:

import nltk
from nltk.corpus import stopwords
ham_words = ''
spam_words = ''
spam = messages[messages.label_num == 1]
ham = messages[messages.label_num ==0]


# In[234]:

for val in spam.text:
    text = val.lower()
    tokens = nltk.word_tokenize(text)
    for words in tokens:
        spam_words = spam_words + words + ' '
        
for val in ham.text:
    text = val.lower()
    tokens = nltk.word_tokenize(text)
    for words in tokens:
        ham_words = ham_words + words + ' '


# ### Generate word cloud images

# In[235]:

from wordcloud import WordCloud
spam_wordcloud = WordCloud(width=600, height=400).generate(spam_words)
ham_wordcloud = WordCloud(width=600, height=400).generate(ham_words)


# ### Spam Word Cloud:

# In[236]:

plt.figure( figsize=(8,6), facecolor='k')
plt.imshow(spam_wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()


# ### Ham Word Cloud:

# In[237]:

plt.figure( figsize=(8,6), facecolor='k')
plt.imshow(ham_wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()


# ## 3. Split data into Train and Test datasets

# In[238]:

from sklearn.model_selection import KFold
#Construct a k-folds object
X = messages["text"]
y = messages["label_num"]
kf = KFold(n_splits=10,shuffle = True)
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train,y_test = y[train_index], y[test_index]
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# ## 3.1. Text transformation

# In[239]:

from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()
vect.fit(X_train)


# In[240]:

print(vect.get_feature_names()[0:20])
print(vect.get_feature_names()[-20:])


# In[241]:

X_train_df = vect.transform(X_train)
X_test_df = vect.transform(X_test)
type(X_test_df)


# ## 4. Apply Different Classification Models

# In[242]:

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn import metrics
prediction = dict()
conf_mat = dict()
fpr = dict()
tpr = dict()
roc_auc = dict()
accuracy = dict()


# ## 4.1. Multinomial Naive Bayes

# In[243]:

from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(X_train_df,y_train)


# In[244]:

prediction["Multinomial_Naive_Bayes"] = nb.predict(X_test_df)


# In[245]:

accuracy["Multinomial_Naive_Bayes"] = accuracy_score(y_test,prediction["Multinomial_Naive_Bayes"])
print(accuracy["Multinomial_Naive_Bayes"])


# In[246]:

print(classification_report(y_test, prediction['Multinomial_Naive_Bayes'], target_names = ["Ham", "Spam"]))


# In[247]:

conf_mat["Multinomial_Naive_Bayes"] = confusion_matrix(y_test, prediction['Multinomial_Naive_Bayes'])
conf_mat_normalized = conf_mat["Multinomial_Naive_Bayes"].astype('float') / conf_mat["Multinomial_Naive_Bayes"].sum(axis=1)[:, np.newaxis]


# In[248]:

sns.heatmap(conf_mat_normalized)
plt.ylabel('True label')
plt.xlabel('Predicted label')


# In[249]:

print(conf_mat["Multinomial_Naive_Bayes"])


# In[250]:

print(roc_auc_score(y_test,prediction["Multinomial_Naive_Bayes"]))


# In[251]:

fpr["Multinomial_Naive_Bayes"], tpr["Multinomial_Naive_Bayes"], threshold = metrics.roc_curve(y_test,prediction["Multinomial_Naive_Bayes"])
roc_auc["Multinomial_Naive_Bayes"] = metrics.auc(fpr["Multinomial_Naive_Bayes"], tpr["Multinomial_Naive_Bayes"])


# ## 4.2. Random Forest Classifier

# In[252]:

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train_df,y_train)


# In[253]:

prediction["Random_Forest"] = rf.predict(X_test_df)


# In[254]:

accuracy["Random_Forest"] = accuracy_score(y_test,prediction["Random_Forest"])
print(accuracy["Random_Forest"])


# In[255]:

print(classification_report(y_test, prediction['Random_Forest'], target_names = ["Ham", "Spam"]))


# In[256]:

conf_mat["Random_Forest"] = confusion_matrix(y_test, prediction['Random_Forest'])
conf_mat_normalized = conf_mat["Random_Forest"].astype('float') / conf_mat["Random_Forest"].sum(axis=1)[:, np.newaxis]


# In[257]:

sns.heatmap(conf_mat_normalized)
plt.ylabel('True label')
plt.xlabel('Predicted label')


# In[258]:

print(conf_mat["Random_Forest"])


# In[259]:

print(roc_auc_score(y_test,prediction["Random_Forest"]))


# In[260]:

fpr["Random_Forest"], tpr["Random_Forest"], threshold = metrics.roc_curve(y_test,prediction["Random_Forest"])
roc_auc["Random_Forest"]  = metrics.auc(fpr["Random_Forest"], tpr["Random_Forest"])


# ## 4.3. Logistic Regression

# In[261]:

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train_df,y_train)


# In[262]:

prediction["Logistic_Regression"] = lr.predict(X_test_df)


# In[263]:

accuracy["Logistic_Regression"] = accuracy_score(y_test,prediction["Logistic_Regression"])
print(accuracy["Logistic_Regression"])


# In[264]:

print(classification_report(y_test, prediction['Logistic_Regression'], target_names = ["Ham", "Spam"]))


# In[265]:

conf_mat["Logistic_Regression"] = confusion_matrix(y_test, prediction['Logistic_Regression'])
conf_mat_normalized = conf_mat["Logistic_Regression"].astype('float') / conf_mat["Logistic_Regression"].sum(axis=1)[:, np.newaxis]


# In[266]:

sns.heatmap(conf_mat_normalized)
plt.ylabel('True label')
plt.xlabel('Predicted label')


# In[267]:

print(conf_mat["Logistic_Regression"])


# In[268]:

print(roc_auc_score(y_test,prediction["Logistic_Regression"]))


# In[269]:

fpr["Logistic_Regression"], tpr["Logistic_Regression"], threshold = metrics.roc_curve(y_test,prediction["Logistic_Regression"])
roc_auc["Logistic_Regression"]  = metrics.auc(fpr["Logistic_Regression"], tpr["Logistic_Regression"])


# ## 4.4. Support Vector Machine

# ### 4.4.1. Using kernel='polynomial' and C='1.25'

# In[270]:

from sklearn.svm import SVC
svm_compare = dict()
svc_poly = SVC(kernel='poly', C=1.25, gamma=0.825,class_weight='balanced')
svc_poly.fit(X_train_df,y_train)


# In[271]:

svm_compare["SVM_polynomial"] = accuracy_score(y_test,svc_poly.predict(X_test_df))
print(svm_compare["SVM_polynomial"])


# ### 4.4.2. Using kernel='rbf' and C='1.25'

# In[272]:

svc_rbf = SVC(kernel='rbf', C=1.25, gamma=0.825,class_weight='balanced')
svc_rbf.fit(X_train_df,y_train)


# In[273]:

svm_compare["SVM_RBF"] = accuracy_score(y_test,svc_rbf.predict(X_test_df))
print(svm_compare["SVM_RBF"])


# ### 4.4.3. Using kernel='linear' and C='1.25'

# In[274]:

svc_linear = SVC(kernel='linear', C=1.25, gamma=0.825,class_weight='balanced')
svc_linear.fit(X_train_df,y_train)


# In[275]:

prediction["SVM_linear"] = svc_linear.predict(X_test_df)


# In[276]:

svm_compare["SVM_linear"] = accuracy_score(y_test,svc_linear.predict(X_test_df))
print(svm_compare["SVM_linear"])


# ### Compare to find which kernel gives most accuracy

# In[277]:

bar_width = 0.3
plt.figure(figsize=(5,4))
plt.ylim((0.85,1.0))
plt.bar(range(len(svm_compare)),svm_compare.values(),bar_width,align='center')
plt.xticks(range(len(svm_compare)),svm_compare.keys(),fontweight='bold')
plt.xlabel('SVC Kernels',fontsize=12,fontweight='bold')
plt.ylabel('Accuracy Score',fontsize=12,fontweight='bold')
plt.title('Accuracy using different SVM Kernels',fontsize=15,fontweight='bold')
plt.show()


# #### Consider 'linear' kernel with C='0.10' which gives the best accuracy

# In[278]:

prediction["SVM"] = svc_linear.predict(X_test_df)


# In[279]:

accuracy["SVM"] = accuracy_score(y_test,prediction["SVM"])
print(accuracy["SVM"])


# In[280]:

print(classification_report(y_test, prediction['SVM'], target_names = ["Ham", "Spam"]))


# In[281]:

conf_mat["SVM"] = confusion_matrix(y_test, prediction['SVM'])
conf_mat_normalized = conf_mat["SVM"].astype('float') / conf_mat["SVM"].sum(axis=1)[:, np.newaxis]


# In[282]:

sns.heatmap(conf_mat_normalized)
plt.ylabel('True label')
plt.xlabel('Predicted label')


# In[283]:

print(conf_mat["SVM"])


# In[284]:

print(roc_auc_score(y_test,prediction["SVM"]))


# In[285]:

fpr["SVM"], tpr["SVM"], threshold = metrics.roc_curve(y_test,prediction["SVM"])
roc_auc["SVM"] = metrics.auc(fpr["SVM"], tpr["SVM"])


# ## 4.5. K-NN Classifier

# In[286]:

from sklearn.neighbors import KNeighborsClassifier
knnModel = KNeighborsClassifier(n_neighbors=5)
knnModel.fit(X_train_df,y_train)


# In[287]:

prediction["K_NN"] = knnModel.predict(X_test_df)


# In[288]:

accuracy["K_NN"] = accuracy_score(y_test,prediction["K_NN"])
print(accuracy["K_NN"])


# In[289]:

print(classification_report(y_test, prediction['K_NN'], target_names = ["Ham", "Spam"]))


# In[290]:

conf_mat["K_NN"] = confusion_matrix(y_test, prediction['K_NN'])
conf_mat_normalized = conf_mat["K_NN"].astype('float') / conf_mat["K_NN"].sum(axis=1)[:, np.newaxis]


# In[291]:

sns.heatmap(conf_mat_normalized)
plt.ylabel('True label')
plt.xlabel('Predicted label')


# In[292]:

print(conf_mat["K_NN"])


# ## 4.5.1. Parameter Tuning using GridSearchCV

# In[293]:

from sklearn.model_selection import GridSearchCV
k_range = np.arange(1,30)
param_grid = dict(n_neighbors=k_range)
print(param_grid)
model = KNeighborsClassifier()
grid = GridSearchCV(model,param_grid)
grid.fit(X_train_df,y_train)
print(grid.best_estimator_)


# In[294]:

prediction["K_NN"] = grid.predict(X_test_df)


# In[295]:

accuracy["K_NN"] = accuracy_score(y_test,prediction["K_NN"])
print(accuracy["K_NN"])


# In[296]:

print(classification_report(y_test, prediction['K_NN'], target_names = ["Ham", "Spam"]))


# In[297]:

conf_mat["K_NN"] = confusion_matrix(y_test, prediction['K_NN'])
conf_mat_normalized = conf_mat["K_NN"].astype('float') / conf_mat["K_NN"].sum(axis=1)[:, np.newaxis]


# In[298]:

sns.heatmap(conf_mat_normalized)
plt.ylabel('True label')
plt.xlabel('Predicted label')


# In[299]:

print(conf_mat["K_NN"])


# In[300]:

print(roc_auc_score(y_test,prediction["K_NN"]))


# In[301]:

fpr["K_NN"], tpr["K_NN"], threshold = metrics.roc_curve(y_test,prediction["K_NN"])
roc_auc["K_NN"] = metrics.auc(fpr["K_NN"], tpr["K_NN"])


# ## 4.6. Ada Boost Classifier

# In[302]:

from sklearn.ensemble import AdaBoostClassifier
adaModel = AdaBoostClassifier(n_estimators=80, random_state=1)
adaModel.fit(X_train_df,y_train)


# In[303]:

prediction["Ada_Boost"] = adaModel.predict(X_test_df)


# In[304]:

accuracy["Ada_Boost"] = accuracy_score(y_test,prediction["Ada_Boost"])
print(accuracy["Ada_Boost"])


# In[305]:

print(classification_report(y_test, prediction['Ada_Boost'], target_names = ["Ham", "Spam"]))


# In[306]:

conf_mat["Ada_Boost"] = confusion_matrix(y_test, prediction['Ada_Boost'])
conf_mat_normalized = conf_mat["Ada_Boost"].astype('float') / conf_mat["Ada_Boost"].sum(axis=1)[:, np.newaxis]


# In[307]:

sns.heatmap(conf_mat_normalized)
plt.ylabel('True label')
plt.xlabel('Predicted label')


# In[308]:

print(conf_mat["Ada_Boost"])


# In[309]:

print(roc_auc_score(y_test,prediction["Ada_Boost"]))


# In[310]:

fpr["Ada_Boost"], tpr["Ada_Boost"], threshold = metrics.roc_curve(y_test,prediction["Ada_Boost"])
roc_auc["Ada_Boost"] = metrics.auc(fpr["Ada_Boost"], tpr["Ada_Boost"])


# ## 4.7. Decision Tree Classifier

# In[311]:

from sklearn.tree import DecisionTreeClassifier
dtModel= DecisionTreeClassifier(min_samples_split=4, random_state=1,class_weight='balanced')
dtModel.fit(X_train_df,y_train)


# In[312]:

prediction["Decision_Tree"] = dtModel.predict(X_test_df)


# In[313]:

accuracy["Decision_Tree"] = accuracy_score(y_test,prediction["Decision_Tree"])
print(accuracy["Decision_Tree"])


# In[314]:

print(classification_report(y_test, prediction['Decision_Tree'], target_names = ["Ham", "Spam"]))


# In[315]:

conf_mat["Decision_Tree"] = confusion_matrix(y_test, prediction['Decision_Tree'])
conf_mat_normalized = conf_mat["Decision_Tree"].astype('float') / conf_mat["Decision_Tree"].sum(axis=1)[:, np.newaxis]


# In[316]:

sns.heatmap(conf_mat_normalized)
plt.ylabel('True label')
plt.xlabel('Predicted label')


# In[317]:

print(conf_mat["Decision_Tree"])


# In[318]:

print(roc_auc_score(y_test,prediction["Decision_Tree"]))


# In[319]:

fpr["Decision_Tree"], tpr["Decision_Tree"], threshold = metrics.roc_curve(y_test,prediction["Decision_Tree"])
roc_auc["Decision_Tree"] = metrics.auc(fpr["Decision_Tree"], tpr["Decision_Tree"])


# ## 4.8. Neural Network

# In[320]:

from sklearn.neural_network import MLPClassifier
nnModel = MLPClassifier(hidden_layer_sizes=(100, ), activation='relu', solver='adam', random_state=None, learning_rate_init=0.001, max_iter=100)
nnModel.fit(X_train_df,y_train)


# In[321]:

prediction["Neural_Network"] = nnModel.predict(X_test_df)


# In[322]:

accuracy["Neural_Network"] = accuracy_score(y_test,prediction["Neural_Network"])
print(accuracy["Neural_Network"])


# In[323]:

print(classification_report(y_test, prediction['Neural_Network'], target_names = ["Ham", "Spam"]))


# In[324]:

conf_mat["Neural_Network"] = confusion_matrix(y_test, prediction['Neural_Network'])
conf_mat_normalized = conf_mat["Neural_Network"].astype('float') / conf_mat["Neural_Network"].sum(axis=1)[:, np.newaxis]


# In[325]:

sns.heatmap(conf_mat_normalized)
plt.ylabel('True label')
plt.xlabel('Predicted label')


# In[326]:

print(conf_mat["Neural_Network"])


# In[327]:

print(roc_auc_score(y_test,prediction["Neural_Network"]))


# In[328]:

fpr["Neural_Network"], tpr["Neural_Network"], threshold = metrics.roc_curve(y_test,prediction["Neural_Network"])
roc_auc["Neural_Network"]  = metrics.auc(fpr["Neural_Network"], tpr["Neural_Network"])


# ## 5. Evaluation of Models

# ## 5.1. By Classification Accuracy

# In[329]:

bar_width = 0.5
plt.figure(figsize=(16,8))
plt.ylim((0.9,1.0))
plt.bar(range(len(accuracy)),accuracy.values(),bar_width,align='center')
plt.xticks(range(len(accuracy)), accuracy.keys(),fontweight='bold')
plt.xlabel('Classification Model',fontsize=16,fontweight='bold')
plt.ylabel('Accuracy Score',fontsize=16,fontweight='bold')
plt.title('Performance of Classifers by Accuracy',fontsize=20,fontweight='bold')
plt.show()


# ## 5.2. By Confusion Matrices

# In[330]:

import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title('Confusion matrix for ' + title,fontweight='bold')
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45,fontweight='bold')
    plt.yticks(tick_marks, classes,fontweight='bold')
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, round(cm[i, j],4),
                 horizontalalignment="center",fontweight='bold',
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout();
    plt.ylabel('True label',fontweight='bold');
    plt.xlabel('Predicted label',fontweight='bold')


# In[331]:

fig = plt.figure(figsize=(12,22))
i=1
for key,val in conf_mat.items():
    plt.subplot(5,2,i);
    i+=1
    plot_confusion_matrix(val,classes=['ham','spam'], normalize=False,title=key)


# ## 5.3. By ROC Curve

# In[332]:

def plot_roc_curve(cm, normalize=False,
                          #title='ROC curve',
                       title=key,
                          cmap=plt.cm.Blues):
    plt.title('ROC for ' + title,fontweight='bold')
    plt.plot(fpr[key], tpr[key], 'b', label = 'AUC = %0.2f' % roc_auc[key])
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate',fontweight='bold')
    plt.xlabel('False Positive Rate',fontweight='bold')
    plt.tight_layout();


# In[333]:

fig = plt.figure(figsize=(8,15))
i=1
for key,val in roc_auc.items():
    plt.subplot(4,2,i);
    i+=1
    plot_roc_curve(val,normalize=False,
                      title = key)


# ## 6. Comparing time for Multinomial NB and Neural Network

# ### Multinomial Naive Bayes

# In[334]:

import time
trainAndTestTime = {}
start = time.time()
nb.fit(X_train_df,y_train)
prediction["Multinomial_Naive_Bayes"] = nb.predict(X_test_df)
end = time.time()
trainAndTestTime["Multinomial_Naive_Bayes"] = end - start
print("{:7.7f}s".format(trainAndTestTime["Multinomial_Naive_Bayes"]))


# ### Neural Network

# In[335]:

start = time.time()
nnModel.fit(X_train_df,y_train)
prediction["Neural_Network"] = nnModel.predict(X_test_df)
end = time.time()
trainAndTestTime["Neural_Network"] = end - start
print("{:7.7f}s".format(trainAndTestTime["Neural_Network"]))


# ### Compare execution times

# In[336]:

bar_width = 0.2
plt.figure(figsize=(8,6))
plt.bar(range(len(trainAndTestTime)),trainAndTestTime.values(),bar_width,align='center')
plt.xticks(range(len(trainAndTestTime)), trainAndTestTime.keys(),fontweight='bold')
plt.xlabel('Classification Model',fontsize=16,fontweight='bold')
plt.ylabel('Training/Testing Time',fontsize=16,fontweight='bold')
plt.title('Training/Testing time of Multinomial NB and Neural Network',fontsize=20,fontweight='bold')
plt.show()

