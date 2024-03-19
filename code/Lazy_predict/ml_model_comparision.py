
import pandas as pd
from lazypredict.Supervised import LazyRegressor
import category_encoders as ce
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

original = pd.read_csv('C:/Users/jax/Desktop/pythonProject/v2/data/original_preprocessed.csv')
additional = pd.read_csv('C:/Users/jax/Desktop/pythonProject/v2/data/additional_preprocessed.csv')
# positive = pd.read_csv('C:/Users/jax/Desktop/pythonProject/v2/data/positive_preprocessed.csv')
# test = pd.read_csv('C:/Users/jax/Desktop/pythonProject/v2/data/test_preprocessed.csv')
all = pd.concat([original,additional], axis=0)
all = all.reset_index(drop=True)
print(all.columns)
# for transformation of data
cat_col = ['cell line', 'test', 'organism', 'cell type', 'morphology', 'tissue', 'disease', ]
num_col = ['time (hr)', 'concentration (ug/ml)','Hydrodynamic diameter (nm)', 'Zeta potential (mV)', 'mcd',
           'electronegativity', 'rox', 'radii', 'Valance_electron', 'amw',
           'lipinskiHBA', 'lipinskiHBD', 'NumRotatableBonds', 'CrippenClogP',
           'chi0v', 'chi1v', 'chi2v', 'hallKierAlpha', 'kappa1']


# standard scaler and binary encoding
def transform(data):
    be = ce.BinaryEncoder()
    Xc = be.fit_transform(all[cat_col])
    Xct = be.transform(data[cat_col])  # put anything you want to transform

    sc = StandardScaler()
    X_all = sc.fit_transform(all[num_col])
    X_ss = sc.transform(data[num_col])  # put anything you want to transform
    X_sc = pd.DataFrame(X_ss, columns=list(num_col))
    join = pd.concat([Xct, X_sc], axis=1)
    return join

transfo = transform(all)

X = transfo
Y = all[['viability (%)']].copy()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Defines and builds the lazyclassifier
clf = LazyRegressor(verbose=2 ,ignore_warnings=False, custom_metric=None)
#sometime it can get stucked check for QuantileRegressor, it might be creating problem
train,test = clf.fit(X_train, X_test, Y_train, Y_test)

train_mod = train.iloc[: , :]
train_mod.to_csv('Model_comparision_train.csv')
print(train_mod)
test_mod = test.iloc[: , :]
test_mod.to_csv('Model_comparision_test.csv')
print(test_mod)


"""#Data Visualization

"""

# Bar plot of R-squared values
import matplotlib.pyplot as plt
import seaborn as sns

#train["R-Squared"] = [0 if i < 0 else i for i in train.iloc[:,0] ]

plt.figure(figsize=(5, 10))
sns.set_theme(style="whitegrid")
ax = sns.barplot(y=train_mod.index, x="R-Squared", data=train_mod)
ax.set(xlim=(0, 1))

# Bar plot of RMSE values
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(5, 10))
sns.set_theme(style="whitegrid")
ax = sns.barplot(y=train_mod.index, x="RMSE", data=train_mod)
ax.set(xlim=(0, 1))

# Bar plot of calculation time
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(5, 10))
sns.set_theme(style="whitegrid")
ax = sns.barplot(y=train.index, x="Time Taken", data=train)
ax.set(xlim=(0, 5))
plt.show()