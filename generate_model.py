from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, FunctionTransformer
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score, cross_validate, train_test_split, GridSearchCV
from sklearn.metrics import f1_score, confusion_matrix, make_scorer


df = pd.read_csv("./Train.csv")
df['category'] = label_encoder.fit_transform(df['category'])

from sklearn.model_selection import train_test_split
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=df["spending_score"])
customer_segment = X_train.copy()


from sklearn.decomposition import PCA


num_pipeline = Pipeline([
    ("standardize", StandardScaler()),
])

num_pipeline

#def log_transform(x):
#    return np.log1p(x)

#family_size_pipeline = Pipeline([
#    ("log_transform", FunctionTransformer(log_transform, validate=True)),
#    ("standardize", StandardScaler())
#])

num_attribs = ["age", "work_experience", "family_size"]
cat_attribs = ["ever_married", "graduated", "profession", "var_1","spending_score"]


cat_pipeline = Pipeline([
    ("onehotencoder", OneHotEncoder(handle_unknown="ignore"))
])


preprocessing = ColumnTransformer([
    #("family_size",family_size_pipeline,["family_size"]),
    ("cat", cat_pipeline, cat_attribs),
    ("num", num_pipeline, num_attribs),
    
])
from sklearn.linear_model import LogisticRegression, RidgeClassifier
model = LogisticRegression(max_iter=1000, random_state=42, C=1, solver= 'lbfgs')

pipeline = Pipeline([
    ('preprocessing', preprocessing),   
    ('pca', PCA(n_components=18)),  
    ('classifier', model)  
])

print("fitting the model to training data")
pipeline.fit(customer_segment, y_train)
y_pred = pipeline.predict(X_test)

print("getting the score")

f1 = f1_score(y_test, y_pred, average="weighted")
print(f1)


import dill
with open('logr_v1.pkl', 'wb') as f:
    dill.dump(pipeline, f)

with open('logr_v1.pkl', 'rb') as f:
    reloaded_model = dill.load(f)

reloaded = f1_score(y_test, reloaded_model.predict(X_test), average= "weighted")
print("Reloaded Model:", reloaded)