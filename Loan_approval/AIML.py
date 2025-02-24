#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve

import warnings
warnings.filterwarnings("ignore")


# In[2]:


df=pd.read_csv("loan_data.csv")
df.head()


# In[3]:


df.info()


# In[4]:


df.shape


# In[5]:


df.columns


# In[6]:


df.describe()


# In[7]:


df.isnull().sum()


# In[8]:


df["person_gender"].unique()


# In[9]:


df["person_gender"].value_counts()


# In[10]:


fig = px.pie(df, names='person_gender')
fig.update_layout(width=600, height=400)
fig.show()


# In[11]:


df['person_gender'] = [ 1 if i=='male' else 0 for i in df['person_gender']]
df.head()


# In[12]:


df.loan_status.unique()


# In[13]:


df.loan_status.value_counts()


# In[14]:


fig = px.pie(df, names='loan_status')
fig.update_layout(width=600, height=400)
fig.show()


# In[15]:


df["person_education"].value_counts()


# In[16]:


df["person_education"].unique()


# In[17]:


fig = px.pie(df, names='person_education')
fig.update_layout(width=600, height=400)
fig.show()


# In[18]:


df['person_education'] = df['person_education'].replace('Master', 1, regex=True)
df['person_education'] = df['person_education'].replace('High School', 2, regex=True)
df['person_education'] = df['person_education'].replace('Bachelor', 3, regex=True)
df['person_education'] = df['person_education'].replace('Associate', 4, regex=True)
df['person_education'] = df['person_education'].replace('Doctorate', 5, regex=True)


# In[19]:


df['person_education'].unique()


# In[20]:


df.shape


# In[21]:


df["person_home_ownership"].unique()


# In[22]:


df["person_home_ownership"].value_counts()


# In[23]:


fig = px.pie(df, names='person_home_ownership')
fig.update_layout(width=600, height=400)
fig.show()


# In[24]:


df['person_home_ownership'] = df['person_home_ownership'].replace('RENT', 1, regex=True)
df['person_home_ownership'] = df['person_home_ownership'].replace('MORTGAGE', 2, regex=True)
df['person_home_ownership'] = df['person_home_ownership'].replace('OWN', 3, regex=True)
df['person_home_ownership'] = df['person_home_ownership'].replace('OTHER', 4, regex=True)


# In[25]:


df.previous_loan_defaults_on_file.value_counts()


# In[26]:


df.previous_loan_defaults_on_file.unique()


# In[27]:


fig = px.pie(df, names='previous_loan_defaults_on_file')
fig.update_layout(width=600, height=400)
fig.show()


# In[28]:


df['previous_loan_defaults_on_file'] = [ 1 if i=='Yes' else 0 for i in df['previous_loan_defaults_on_file']]
df.head()


# In[29]:


df.loan_intent.unique()


# In[30]:


df.loan_intent.value_counts()


# In[31]:


df['loan_intent'] = df['loan_intent'].replace('EDUCATION', 1, regex=True)
df['loan_intent'] = df['loan_intent'].replace('MEDICAL', 1, regex=True)
df['loan_intent'] = df['loan_intent'].replace('VENTURE', 1, regex=True)
df['loan_intent'] = df['loan_intent'].replace('PERSONAL', 1, regex=True)
df['loan_intent'] = df['loan_intent'].replace('DEBTCONSOLIDATION', 1, regex=True)
df['loan_intent'] = df['loan_intent'].replace('HOMEIMPROVEMENT', 1, regex=True)


# In[32]:


plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.show()


# In[33]:


threshold = 0.003
correlation_matrix = df.corr()
high_corr_features = correlation_matrix.index[abs(correlation_matrix["loan_status"]) > threshold].tolist()
high_corr_features.remove("loan_status")
print(high_corr_features)
X_selected = df[high_corr_features]
Y = df["loan_status"]


# In[34]:


X_selected.shape


# In[35]:


droped_cols=df.drop(X_selected,axis=1)
droped_cols.columns


# In[36]:


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)


# In[37]:


X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)


# In[38]:


from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier


# In[41]:


svm_clf = SVC(probability=True, kernel='rbf', random_state=42)  # SVM with RBF kernel
knn_clf = KNeighborsClassifier(n_neighbors=5)  # KNN with 5 neighbors
xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)  # XGBoost

# Combine the models into a Voting Classifier
voting_clf = VotingClassifier(
    estimators=[
        ('SVM', svm_clf),
        ('KNN', knn_clf),
        ('XGBoost', xgb_clf)
    ],
    voting='soft'  # Use 'hard' for majority voting
)
# Train and evaluate each model
models = {
    'SVM': svm_clf,
    'KNN': knn_clf,
    'XGBoost': xgb_clf,
    'Voting Classifier': voting_clf
}

for name, model in models.items():
    # Train the model
    model.fit(X_train, Y_train)
    # Make predictions
    y_pred = model.predict(X_test)
    # Evaluate the model
    print(f"{name} Accuracy: {accuracy_score(Y_test, y_pred):.2f}")
    print(f"Classification Report for {name}:\n{classification_report(Y_test, y_pred)}\n")


# In[42]:


# xgb_model = XGBClassifier(
#     n_estimators=100, 
#     learning_rate=0.1, 
#     max_depth=6, 
#     random_state=42, 
#     use_label_encoder=False,
#     eval_metric='logloss'
# )
# xgb_model.fit(X_train, Y_train)
# y_pred = xgb_model.predict(X_test)


# In[44]:


# Evaluate the model
print("Accuracy:", accuracy_score(Y_test, y_pred))
print("\nClassification Report:\n", classification_report(Y_test, y_pred))


# In[45]:


print("\nConfusion Matrix:\n", confusion_matrix(Y_test, y_pred))


# In[ ]:

import pickle
with open("loan_approval_model.pkl", "wb") as model_file:
    pickle.dump(voting_clf, model_file)


with open("scaler.pkl", "wb") as scaler_file:
    pickle.dump(scaler, scaler_file)


with open("feature_names.pkl", "wb") as feature_file:
    pickle.dump(high_corr_features, feature_file)
    
print("✅ Model, Scaler, and Feature Names saved successfully!")