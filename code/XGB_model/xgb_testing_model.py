import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pandas as pd
import numpy as np
import lightgbm
from sklearn import metrics
import category_encoders as ce
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
from xgboost import XGBRegressor



original = pd.read_csv('C:/Users/user/Desktop/project/v2/preprocessed/original_preprocessed.csv')
additional = pd.read_csv('C:/Users/user/Desktop/project/v2/preprocessed/additional_preprocessed.csv')
# positive = pd.read_csv('C:/Users/user/Desktop/project/v2/preprocessed/positive_preprocessed.csv')
# test = pd.read_csv('C:/Users/user/Desktop/project/v2/preprocessed/test_preprocessed.csv')
test = pd.read_csv('C:/Users/user/Desktop/project/v2/preprocessed/final_test.csv')
all = pd.concat([original,additional], axis=0)
all = all.reset_index(drop=True)

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
    X_sc = pd.DataFrame(X_ss, columns=num_col)
    join = pd.concat([Xct, X_sc], axis=1)
    return join

transfo = transform(all)

X = transfo
# print(X)
Y = all[['viability (%)']].copy()

test2 = transform(test)
X_val2 = test2
Y_val2 = test[['viability (%)']].copy()

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
#got 0.53
model = XGBRegressor(n_estimators=884,
                       learning_rate=0.035958915605443356,
                       max_depth=8,
                       min_child_weight= 3,
                       subsample= 0.8845398751174367,
                       colsample_bytree = 0.9852832952114663,
                       reg_lambda = 0.0013221099443591067,
                       reg_alpha = 3.16693837876956,
                     )

xgb_model = model.fit(X_train, Y_train)
# pickle.dump(model, open('lgbm_model_final.pkl', 'wb'))
train = model.predict(X_train)
validation = model.predict(X_test)

# testing = model.predict(X_val)
testing2 =model.predict(X_val2)
# testing3 =model.predict(X_t)

print('Train')
print('Mean Absolute Error:', metrics.mean_absolute_error(Y_train, train))
print('Mean Squared Error:', metrics.mean_squared_error(Y_train, train))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_train, train)))
print("Regressor R2-score: ", r2_score(Y_train, train))

print('validation')
print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, validation))
print('Mean Squared Error:', metrics.mean_squared_error(Y_test, validation))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, validation)))
print("Regressor R2-score: ", r2_score(Y_test, validation))
#
# print('testing')
# print('Mean Absolute Error:', metrics.mean_absolute_error(Y_val, testing))
# print('Mean Squared Error:', metrics.mean_squared_error(Y_val, testing))
# print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_val, testing)))
# print("Regressor R2-score: ", r2_score(Y_val, testing))

print('testing2')
print('Mean Absolute Error:', metrics.mean_absolute_error(Y_val2, testing2))
print('Mean Squared Error:', metrics.mean_squared_error(Y_val2, testing2))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_val2, testing2)))
print("Regressor R2-score: ", r2_score(Y_val2, testing2))

# print('testing3')
# print('Mean Absolute Error:', metrics.mean_absolute_error(Y_t, testing3))
# print('Mean Squared Error:', metrics.mean_squared_error(Y_t, testing3))
# print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_t, testing3)))
# print("Regressor R2-score: ", r2_score(Y_t, testing3))

#save predicted values
test['predicted'] = testing2
# test.to_csv('preprocessed/xgb_tested.csv')


"""#Visualization"""

custom_params = {"axes.spines.right": False, "axes.spines.top": False}

sns.set_theme(style="ticks", rc=custom_params)
sns.set(font_scale=2)
f, ax = plt.subplots(figsize=(13, 10))
plt.scatter(Y_train, train, color='#DD7059', s=70, label = 'train data')#, alpha=0.5)
plt.scatter(Y_test, validation , color='#569FC9',s=70, label = 'validation')#, alpha= 0.5)
# plt.scatter(Y_val, testing, color='#274E13',s=70, label = 'testing')
plt.scatter(Y_val2, testing2, color='#274E13',s=70, label = 'test')
plt.plot(Y_test, Y_test, color='#444444', linewidth=3)
plt.plot(Y_test, (Y_test - 2*np.sqrt(metrics.mean_squared_error(Y_test, validation))), color='#444444', linewidth = 1)
plt.plot(Y_test, (Y_test + 2*np.sqrt(metrics.mean_squared_error(Y_test, validation))), color='#444444', linewidth = 1)
# sns.regplot(x=Y_test, y=validation, ci=0.95, color='#274E13')
plt.title('LGBM Regressor')
plt.xlabel('actual data')
plt.ylabel('predicted data')
plt.legend()
plt.xlim(0, 135)
plt.ylim(0, 135)
plt.show()

# ax.figure.savefig("LGBM_regressor.png",transparent=True)

import shap
# shap.initjs()
X_importance = X_test
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_importance)
# print(shap_values, X_importance)
top_5_features = shap_values.abs().mean(0).sort_values().index[-5:]
shap.summary_plot(shap_values, X_importance, feature_order = top_5_features, show=False)
plt.figure(figsize= (15,10))
plt.show()
# plt.savefig('important_features.png')

