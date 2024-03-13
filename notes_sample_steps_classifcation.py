# evaluate multinomial logistic regression model
import pandas as pd
from pandas.core.frame import DataFrame
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression



'''
data exploration
'''
import seaborn as sns
sns.histplot(test.col)

pd.read_csv('',header=None)
df.info
df.dtypes
len(df)
df.col.unique()
df.dropna(subset=[],axis=0)


'''
preprocessing
'''
pd.to_datetime(df.col)
df.col.fillna(np.median(col),inplace=True)

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stop = stopwords.words('english')

def preprocessing(data):
  data = data.str.lower()
  data = data.apply(lambda x:' '.join(re.findall(r'\w+', x))) # remove special characters
  data = data.apply(lambda x:re.sub(r' +',' ',x)) # remove extra white space
  data = data.apply(lambda x: ' '.join([word for word in x.split() if word not in stop]))
  return data

# stemming and lemmatization
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
list2 = nltk.word_tokenize(string)
lemmatized_string = ' '.join([lemmatizer.lemmatize(words) for words in list2])

# label encoding and one hot encoding
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
# label_encoder object knows how to understand word labels. 
label_encoder = preprocessing.LabelEncoder()
# Encode labels in column 'Country'. 
data['Country']= label_encoder.fit_transform(data['Country']) # it would have a rank
# creating one hot encoder object 
onehotencoder = OneHotEncoder()
#reshape the 1-D country array to 2-D as fit_transform expects 2-D and finally fit the object 
enc = OneHotEncoder(handle_unknown='ignore',drop='first')
enc.get_feature_names_out(['gender', 'group'])
enc.fit(df)
enc.tramsform(df).toarray()
#using pandas
one_hot_encoded_training_predictors = pd.get_dummies(train_predictors)
one_hot_encoded_test_predictors = pd.get_dummies(test_predictors)
final_train, final_test = one_hot_encoded_training_predictors.align(one_hot_encoded_test_predictors,join='left',axis=1)


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
cvec=CountVectorizer(min_df=10)
cvec.fit_transform(X_train)
cvec_df = pd.Dataframe(X_train_vec.todense(),cvec.get_feature_names_out()) # combined with numerical features
X_train_vec = cvec.fit_transform(X_train)
X_test_vec = cvec.transform(X_test)

vectorizer = TfidfVectorizer()
vectorizer.fit(text)


# oversampling
from imblearn.over_sampling import SMOTE  #synthetic minority class
sm=SMOTE(random_state=2)
X_train_res,y_train_res=sm.fit_resample(X_train,y_train)

from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(random_state=42, replacement=True)# fit predictor and target variable
x_rus, y_rus = rus.fit_resample(x, y)

'''
training
'''
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.25,stratify=y) #ensure similar proportion
scaler=StandardScaler()
X_train=pd.DataFrame(scaler.fit_transform(X_train))
X_test=pd.DataFrame(scaler.transform(X_test))
# define the multinomial logistic regression model 
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
model.fit(X,y)
lr= LogisticRegression(random_state=1)
lr.fit(X_train_vec,y_train)

# k-fold cross validation
# it's used to avoid overfitting, all parts will be able to be used as the testing data
# define the model evaluation procedure
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
cv = StratifiedKFold(n_splits=10, random_state=1)
# evaluate the model and collect the scores
n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
print(n_scores)
kf =KFold(n_splits=5, shuffle=True, random_state=42)
cnt = 1
# split()  method generate indices to split data into training and test set.
for train_index, test_index in kf.split(X, y):
    print(f'Fold:{cnt}, Train set: {len(train_index)}, Test set:{len(test_index)}')
    cnt += 1

'''
evaluation
'''
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix

accuracy_score(y_test,y_pred)
print(classification_report(y_test,y_pred))
plot_confusion_matrix(classifier, X_test, y_test)
plt.show()

Accuracy = (TP + TN) / (TP + FP + TN + FN)
from sklearn.metrics import precision_score
precision = tp/(tp+fp) # out of total predicted positive values how many were actually positive. spam missing important email
from sklearn.metrics import recall_score
recall = tp/(tp+fn) #eliminate fn, e.g. cancer
from sklearn.metrics import f1_score #when both are important