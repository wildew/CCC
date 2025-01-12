import warnings, matplotlib.pyplot as plt, seaborn as sns, pandas as pd, numpy as np, os, tensorflow as tf
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.metrics import classification_report, roc_auc_score, ConfusionMatrixDisplay, f1_score, accuracy_score
from sklearn.exceptions import NotFittedError
import time 
warnings.filterwarnings('ignore')

input_file = 'app/Loan.csv'
df = pd.read_csv(input_file)
print(df)

target = 'LoanApproved'

col_drop_list = []
col_drop_list.append('RiskScore')

print(df.describe())
print(df.info())

plt.figure(figsize=(15,6))
sns.heatmap(df.isna(),yticklabels=False)
plt.tight_layout()
plt.show() 

print(df.isna().sum())
print(df.columns)

df=df.drop(col_drop_list,axis=1)
print(df.info())

def feature_type_transform(df,col_list,col_type):
    for col in col_list:
        df[col]=df[col].astype(col_type)
    return df

def col_unique_val_check(df):
    for i,col in enumerate(df.columns):
        print(f"{col:40} ----> {df[col].nunique():10} unique values with dtype {str(df[col].dtype):10} at index {i}") 

def num_col_unique_val_check(df,target):
    for i,col in enumerate(df.select_dtypes(exclude='object')):
        if(col!=target):
            print(f"{col:40} ----> {df[col].nunique():10} unique values with dtype {str(df[col].dtype):10}")

def rem_col(df,col_list,target):
    rem_col=[]
    for i,col in enumerate(df.columns):
        if(col not in col_list and col!=target):
            rem_col.append(col)
    return rem_col

print(col_unique_val_check(df))
print(col_unique_val_check(df.select_dtypes(include='object')))
print(num_col_unique_val_check(df, target))

cat_col=[]
for col in df.select_dtypes(include='object'):
    if(df[col].nunique()<20 and col!=target):
        print(f"{col:30} ----> {df[col].nunique():10} unique values")
        cat_col.append(col)
print("="*100)
print(cat_col,len(cat_col))

num_cat_col=[]
for col in df.select_dtypes(exclude='object'):
    if(df[col].nunique()<5 and col!=target):
        print(f"{col:30} ----> {df[col].nunique():10} unique values")
        num_cat_col.append(col)
print("="*100)
print(num_cat_col,len(num_cat_col))

num_col=[]
for col in df.select_dtypes(exclude='object'):
    if(col not in (num_cat_col+cat_col) and col!=target):
        print(f"{col:30} ----> {df[col].nunique():10} unique values")
        num_col.append(col)
print("="*100)
print(num_col,len(num_col))

remcol = rem_col(df,cat_col+num_cat_col+num_col,target)
print(remcol,len(remcol))

print(df.drop(target,axis=1).shape[1],len(cat_col)+len(num_cat_col)+len(num_col)+len(remcol))

df['ApplicationDate']=pd.to_datetime(df['ApplicationDate'], infer_datetime_format=True)
df['ApplicationYear']=df['ApplicationDate'].dt.year
df = df.drop(['ApplicationDate'],axis=1)
print(df.info())

plt.figure(figsize=(10,4))
sns.countplot(x=target,data=df,palette='deep')
plt.tight_layout()
plt.show()

plt.figure(figsize=(20,18))
j=1
for i,columns in enumerate(cat_col):
    plt.subplot(5,1,i+1)
    sns.countplot(x=columns,data=df,palette='deep')
    j+=1
plt.tight_layout()
plt.show()

plt.figure(figsize=(20,18))
j=1
for i,columns in enumerate(cat_col):
    plt.subplot(5,1,i+1)
    sns.countplot(x=columns,data=df,palette='deep',hue=target)
    j+=1
plt.tight_layout()
plt.show()

plt.figure(figsize=(20,18))
j=1
for i,columns in enumerate(num_cat_col):
    plt.subplot(5,1,i+1)
    sns.countplot(x=columns,data=df,palette='deep',hue=target)
    j+=1
plt.tight_layout()
plt.show()

print(df.info())

oh = OneHotEncoder(handle_unknown='ignore')
ms, ss = MinMaxScaler(), StandardScaler()

ct = ColumnTransformer([('cat_encoder', oh, cat_col), ('num_encoder', ss, num_col)], remainder = 'passthrough', n_jobs = -1)
print(ct)

X=df.drop([target],axis=1)
y=df[[target]]
print(X.head(2))
print(y.head(2))

print(len(df))

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size = 0.15, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify = y_train, test_size = 0.15, random_state=42)

print(X_train.shape, y_train.shape)
print(X_val.shape, y_val.shape)
print(X_test.shape, y_test.shape)
print(X_test.head(2))
print(y_test.head(2))
print(X_train.info(verbose=True))

X_train = ct.fit_transform(X_train)
X_val = ct.fit_transform(X_val)
X_test = ct.transform(X_test)

print("-"*50)
for i in ct.transformers_:
    print(i)
    if(i[0]!='remainder'):
        print(i[1].get_feature_names_out())   
    print("-"*50)

print(ct)

tup1 = None; tup2 = None
try:
    tup1 = (X_test[0].toarray(), y_test.values[0]) 
    print("Sparse Matrix to Dense Array")
except:
    tup2 = (X_test[0], y_test.values[0]) 
    print("Normal Matrix to Dense Array")

print(tup1 if(tup1) else tup2)
print(X_test.shape, y_test.shape)

y_train_copy = y_train.values
y_val_copy = y_val.values
y_test_copy = y_test.values

batch_size = 1024

y_train_tf_copy = y_train_copy
y_val_tf_copy = y_val_copy
y_test_tf_copy = y_test_copy

data_tf_tr = tf.data.Dataset.from_tensor_slices((X_train, y_train_tf_copy))
data_train_batches = data_tf_tr.batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)

data_tf_val = tf.data.Dataset.from_tensor_slices((X_val, y_val_tf_copy))
data_val_batches = data_tf_val.batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)

data_tf_te = tf.data.Dataset.from_tensor_slices((X_test, y_test_tf_copy))
data_test_x_y_batches = data_tf_te.batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)

#Only for X_test
data_tf_tre = tf.data.Dataset.from_tensor_slices(X_test)
data_test_x_batches = data_tf_tre.batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Input,PReLU,LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau

l_relu=LeakyReLU()
para_relu = PReLU()
e = EarlyStopping(monitor='val_loss',patience=10,restore_best_weights=True,verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,patience=2, min_lr=0.001)
n = 1

model = Sequential()
model.add(Input(shape=(X_train.shape[1],)))
model.add(Dense(n*X_train.shape[1],activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics = [ 'accuracy',
                         tf.keras.metrics.AUC(name='AUC_ROC',curve='ROC',num_thresholds=10000) ,
                         tf.metrics.F1Score(name='F1_Score',average='macro',threshold=0.5)
                        ]
             )

model.summary()

start_time = time.time()
history = model.fit(
    data_train_batches,
    epochs=100,
    callbacks=[e],
    validation_data=data_val_batches,
    verbose=1
    )

end_time = time.time()

model.evaluate(data_test_x_y_batches)
score_dict={}
pred=(model.predict(X_test)>0.5).astype(int)
score_dict['model']={
        'roc_auc_score':roc_auc_score(y_test.values,pred),
        'f1_score':f1_score(y_test.values,pred),
        'accuracy_score':accuracy_score(y_test.values,pred)
    }

score_pd = pd.DataFrame(score_dict).transpose().sort_values('roc_auc_score',ascending=False)
print(score_pd)

print(classification_report(y_test, pred))

sns.set_style('white')
f, ax = plt.subplots(1, 1, figsize=(6, 6))
ConfusionMatrixDisplay.from_predictions(y_test, pred, ax = ax)
plt.tight_layout()
plt.show()

model.save('riskscore.h5')
