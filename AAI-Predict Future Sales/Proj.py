import pandas as pd
import numpy as np
import matplotlib.pylab as plt
plt.style.use('bmh')

from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder

df_test = pd.read_csv('test.csv')
df_ex = pd.read_csv('sample_submission.csv')
df_train = pd.read_csv('sales_train_v2.csv')
df_shop = pd.read_csv('shops.csv')
df_items = pd.read_csv('items.csv')
df_cat = pd.read_csv('item_categories.csv')

# Get item sells by month
grouped = df_train.groupby(['item_id', 'shop_id', 'date_block_num'])
month_stat = grouped['item_cnt_day'].sum()

# Create train table with "item_cnt_month"
X = np.array([*month_stat.index.values])
Y = np.array(month_stat.values)
df_train_m = pd.DataFrame({'date_block_num':X[:,2], 'item_id':X[:,0], 'shop_id':X[:,1], 'item_cnt_m':Y})
df_train_m = pd.merge(df_train_m, df_items[['item_id', 'item_category_id']], on='item_id')
df_train_m = df_train_m.sort_values(by='date_block_num')
df_train_m.head()

Y = df_train_m['item_cnt_m']
X0 = df_train_m[['item_id', 'shop_id', 'item_category_id', 'date_block_num']]
X0.head()




rgr_ridge = Ridge()
dtr = DecisionTreeRegressor()
ada = AdaBoostRegressor()
bag = BaggingRegressor(verbose=2)
random_forest = RandomForestRegressor(verbose=2)
gboost = GradientBoostingRegressor(verbose=2)

gboost.fit(X0, Y)

df_ex.head()

df_test.head()

df_test_a = pd.merge(df_test, df_items[['item_id', 'item_category_id']], on='item_id')
df_test_a = df_test_a.sort_values(by='ID')
df_test_a['date_block_num'] = [34]*len(df_test_a)
X0_test = df_test_a[['item_id', 'shop_id', 'item_category_id', 'date_block_num']]

Y_pred = gboost.predict(X0_test)
df_ex['item_cnt_month'] = Y_pred
df_ex = df_ex.set_index('ID')

df_ex.to_csv('pred.csv')