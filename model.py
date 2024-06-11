# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# %%
df = pd.read_csv('./data/Maternal_Health_Risk_Data_Set.csv')

# %%
X_features = ['Age','SystolicBP','DiastolicBP','BS', 'BodyTemp', 'HeartRate']
df['RiskLevel'].value_counts()
df.drop(df[df['HeartRate'] < 50].index)
RiskLabels = ['low risk', 'mid risk', 'high risk']
df

# %%
fig,axes=plt.subplots(1, 6, figsize=(12, 3), sharey=True)
for i in range(len(axes)):
    sns.stripplot(data= df, x=X_features[i], y=df['RiskLevel'], ax=axes[i])

# %%
from sklearn import model_selection
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# %%
X = df.loc[: , df.columns!='RiskLevel']
y = df['RiskLevel']
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size= 0.2, random_state = 1)

print('X_train dimension= ', X_train.shape)
print('X_test dimension= ', X_test.shape)
print('y_train dimension= ', y_train.shape)
print('y_train dimension= ', y_test.shape)

# %%
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
print(rfc.score(X_test, y_test))

# %%
pickle.dump(rfc, open("model.pkl", "wb"))



# %%
