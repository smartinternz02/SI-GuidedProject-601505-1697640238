import pandas as pd
from sklearn.utils.validation import column_or_1d
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle
data = pd.read_csv('data.csv')


X = data.drop(['Resource_Allocation'], axis=1)
Y = data['Resource_Allocation'].values

y= column_or_1d(Y, warn=True)


x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)


random_forest_model = RandomForestRegressor(n_estimators=100, random_state=42)


random_forest_model.fit(x_train, y_train)

pickle.dump(random_forest_model,open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))


