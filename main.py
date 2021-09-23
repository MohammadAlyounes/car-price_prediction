import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import streamlit as st


# Prepare simple interface using Streamlit

st.title('Mohammad Alyounes Apps Series: ')
st.header('Car Price Prediction App')
st.subheader("Use feature values like this: city_mpg:18, driven_wheels:all_wheel_drive, engine_cylinder:6 ")
st.subheader("engine_fuel_type:regular_unleaded, engine_hp:268.0, highway_mpg:25, make:toyota")
st.subheader("market_category:luxury, model: venza, number_of_doors:4.0, popularity:2031" )
st.subheader("transmission_type:automatic, vehicle_size: midsize, vehicle_style: sedan, year:2013")


# Load a dataframe and do some preprocessing data

df = pd.read_csv('data.csv')

df.columns = df.columns.str.lower().str.replace(' ', '_')
string_columns = list(df.dtypes[df.dtypes == 'object'].index)
for col in string_columns:
    df[col] = df[col].str.lower().str.replace(' ', '_')

# Split dataset into three parts: train, test, validation
n = len(df)
n_val = int(0.2*n)
n_test = int(0.2*n)
n_train = n - (n_val+n_test)

np.random.seed(2)
idx = np.arange(n)
np.random.shuffle(idx)

df_shuffled = df.iloc[idx]

df_train = df_shuffled.iloc[:n_train].copy()
df_val = df_shuffled.iloc[n_train:n_train+n_val].copy()
df_test = df_shuffled.iloc[n_train+n_val:].copy()

# make log transformation into target
y_train = np.log1p(df_train.msrp.values)
y_val = np.log1p(df_val.msrp.values)
y_test = np.log1p(df_test.msrp.values)

del df_train['msrp']
del df_val['msrp']
del df_test['msrp']


# linear regression function
def linear_regression(X, Y):
    # adding the dummy column
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])
    # normal equation formula
    XTX = X.T.dot(X)
    XTX_inv = np.linalg.inv(XTX)
    w = XTX_inv.dot(X.T).dot(Y)
    return w[0], w[1:]


base = ['engine_hp', 'engine_cylinders', 'highway_mpg', 'city_mpg', 'popularity']


def rmse(y, y_pred):
    error = y_pred - y
    mse = (error ** 2).mean()
    return np.sqrt(mse)


# convert a dataframe into matrix


def prepare_X(df):
    # Do one-hot-encoding on categorical feature
    df = df.copy()
    features = base.copy()

    df['age'] = 2017 - df.year
    features.append('age')

    for v in [2, 3, 4]:
        feature = 'num_doors_%s' % v
        df[feature] = (df['number_of_doors'] == v).astype(int)
        features.append(feature)

    for v in ['chevrolet', 'ford', 'volkswagen', 'toyota', 'dodge']:
        feature = 'is_make_%s' % v
        df[feature] = (df['make'] == v).astype(int)
        features.append(feature)

    for v in ['regular_unleaded', 'premium_unleaded_(required)',
'premium_unleaded_(recommended)',
'flex-fuel_(unleaded/e85)']:
        feature = 'is_type_%s' % v
        df[feature] = (df['engine_fuel_type'] == v).astype(int)
        features.append(feature)

    for v in ['automatic', 'manual', 'automated_manual']:
        feature = 'is_transmission_%s' % v
        df[feature] = (df['transmission_type'] == v).astype(int)
        features.append(feature)

    for v in ['front_wheel_drive', 'rear_wheel_drive',
              'all_wheel_drive', 'four_wheel_drive']:
        feature = 'is_driven_wheels_%s' % v
        df[feature] = (df['driven_wheels'] == v).astype(int)
        features.append(feature)

    for v in ['crossover', 'flex_fuel', 'luxury',
              'luxury,performance', 'hatchback']:
        feature = 'is_mc_%s' % v
        df[feature] = (df['market_category'] == v).astype(int)
        features.append(feature)

    for v in ['compact', 'midsize', 'large']:
        feature = 'is_size_%s' % v
        df[feature] = (df['vehicle_size'] == v).astype(int)
        features.append(feature)

    for v in ['sedan', '4dr_suv', 'coupe', 'convertible',
              '4dr_hatchback']:
        feature = 'is_style_%s' % v
        df[feature] = (df['vehicle_style'] == v).astype(int)
        features.append(feature)

    df_num = df[features]
    df_num = df_num.fillna(0)
    X = df_num.values
    return X


def linear_regression_reg(X, y, r=0.0):  # This we should use ( Linear regression with regularization )
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])

    XTX = X.T.dot(X)
    reg = r * np.eye(XTX.shape[0])
    XTX = XTX + reg

    XTX_inv = np.linalg.inv(XTX)
    w = XTX_inv.dot(X.T).dot(y)

    return w[0], w[1:]


X_train = prepare_X(df_train)
w_0, w = linear_regression_reg(X_train, y_train, r=0.001)

X_val = prepare_X(df_val)
y_pred = w_0 + X_val.dot(w)


# Get input from user using Streamlit UI
city_mpg = st.number_input('Please insert a city_mpg')
driven_wheels = st.selectbox('What type of driven wheel',
                             ['front_wheel_drive', 'rear_wheel_drive','all_wheel_drive', 'four_wheel_drive'])
engine_cylinders = st.number_input('Enter the number of cylinders')
engine_fuel_type = st.selectbox('Engine fuel type', ['regular_unleaded', 'premium_unleaded_(required)', 'premium_unleaded_(recommended)', 'flex-fuel_(unleaded/e85)'])
engine_hp = st.number_input('Enter the engine horse power')
highway_mpg = st.number_input('Enter highway mpg')
make = st.selectbox('Make', ['chevrolet', 'ford', 'volkswagen', 'toyota', 'dodge'])
market_category = st.selectbox('Choose a market category', ['crossover', 'flex_fuel', 'luxury', 'luxury,performance', 'hatchback'])
model = st.text_input('Enter the model of the car')
number_of_doors = st.selectbox('Choose a number of doors ', [2, 3, 4])
popularity = st.number_input('Enter the popularity')
transmission_type = st.selectbox('Choose the transmission type', ['automatic', 'manual', 'automated_manual'])
vehicle_size = st.selectbox('Choose the vehicle size', ['compact', 'midsize', 'large'])
vehicle_style = st.selectbox('Choose the vehicle style', ['sedan', '4dr_suv', 'coupe', 'convertible', '4dr_hatchback'])
year = st.number_input('Enter the year')

user_dict = {
    'city_mpg': city_mpg,
    'driven_wheels': driven_wheels,
    'engine_cylinders': engine_cylinders,
    'engine_fuel_type': engine_fuel_type,
    'engine_hp': engine_hp,
    'highway_mpg': highway_mpg,
    'make': make,
    'market_category': market_category,
    'model': model,
    'number_of_doors': number_of_doors,
    'popularity': popularity,
    'transmission_type': transmission_type,
    'vehicle_size': vehicle_size,
    'vehicle_style': vehicle_style,
    'year': year
}


# Apply model to the input data
user_input = pd.DataFrame([user_dict])
X_user = prepare_X(user_input)
y_pred = w_0 + X_user.dot(w)
price = np.expm1(y_pred)

st.write('The predicted price of your car is:', price)
