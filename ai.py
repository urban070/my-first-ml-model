import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# df = pd.read_csv('https://raw.githubusercontent.com/urban070/ml-data/main/increment.csv')
# get data in csv format
df = pd.read_csv('./data/increment.csv')
print(df)

# prepare the data
y = df['lastNumber']
x = df.drop('lastNumber', axis=1)

# split the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)

# build and train the model
lr = LinearRegression()
lr.fit(x_train.values, y_train)

# make predictions
y_lr_train_pred = lr.predict(x_train.values)
y_lr_test_pred = lr.predict(x_test.values)

# evaluate the model
lr_train_mse = mean_squared_error(y_train, y_lr_train_pred)
lr_train_r2 = r2_score(y_train, y_lr_train_pred)

lr_test_mse = mean_squared_error(y_test, y_lr_test_pred)
lr_test_r2 = r2_score(y_test, y_lr_test_pred)

lr_results = pd.DataFrame(['Linear regression', lr_train_mse, lr_train_r2, lr_test_mse, lr_test_r2]).transpose()
lr_results.columns = ['Method', 'Training MSE', 'Training R2', 'Test MSE', 'Test R2']

print(lr_results)

user_input = input('Please enter your data, separated by commas, like so: 1,2,3,4. I will try to predict the next number.\n: ')
user_input_list = user_input.split(',')

user_input_list_numeric = [float(i) for i in user_input_list]
user_input_transformed = [user_input_list_numeric]
print(user_input_transformed)

user_input_prediction = lr.predict(user_input_transformed)


print('My prediction for the next number in this pattern is: ', user_input_prediction[0])