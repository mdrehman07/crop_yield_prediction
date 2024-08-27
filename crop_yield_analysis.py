
# Importing libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import geopandas as gpd
import matplotlib.patches as mpatches


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer
#from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from catboost import CatBoostRegressor


# Loading the crop_yield dataset

df = pd.read_csv('crop_yield.csv')
print(df.head())

# checking the dataset 

print("Shape of the dataset : ",df.shape) # checking the shape of the dataset

print(df.describe()) # checking the statistical measures of the dataset

print(df.info()) # checking the datatype of each column

print('duplicates',df.duplicated().sum())  #checking the duplicates

columns_to_check = ['Crop', 'Season', 'State'] # checking the unqiue values of column whose datatype is object

for col in columns_to_check:
    print(f'------------------------------- {col} -------------------------------')
    unique_values = df[col].unique().tolist()
    print(f'Total different values of {col}: {len(unique_values)}')
    print(f'Different values of {col}: {unique_values}')
    print("\n")

# removing extra spaces in season
df['Season'] = df['Season'].replace({'Whole Year ':'Whole Year','Kharif     ':'Kharif', 'Rabi       ':'Rabi', 'Autumn     ':'Autumn', 'Summer     ':'Summer', 'Winter     ':'Winter'})

df = df[df['Crop_Year']!=2020]  # data is incomplete for year 2020 so removing it

#  -------------------------------------------------------------   Analysis & Visualization of the dataset -------------------------------------------------

# ------------------------------------------------- Year wise analysis -------------------------------------------------



yield_per_year = df.groupby('Crop_Year').sum() # Grouping the data on basis of year and add it
#print(yield_per_year)

# plotting the total area under cultivation over the years
plt.figure(figsize = (12,5))
plt.plot(yield_per_year.index, yield_per_year['Area'],color='blue', linestyle='dashed', marker='o', markersize=10, markerfacecolor='blue')
plt.xlabel('Year')
plt.ylabel('Area (Hectares)')
plt.title('Area under cultivation over the years')
plt.savefig('area_under_cultivation_over_years.png')

# plotting the total fertilizer used over the years

plt.figure(figsize = (12,5))
plt.plot(yield_per_year.index, yield_per_year['Fertilizer'],color='blue', linestyle='dashed', marker='o', markersize=12, markerfacecolor='magenta')
plt.xlabel('Year')
plt.ylabel('Fertilizer (Kilogram)')
plt.title('Use of Fertilizer over the years')
plt.savefig('total_fertilizer_usage_over_years.png')

# plotting the total pesticide used over the years

plt.figure(figsize = (12,5))
plt.plot(yield_per_year.index, yield_per_year['Pesticide'],color='red', linestyle='dashed', marker='o', markersize=12, markerfacecolor='cyan')
plt.xlabel('Year')
plt.ylabel('Pesticide (Kilogram)')
plt.title('Use of Pesticide over the Years')
plt.savefig('total_pesticide_usage_over_years.png')

# plotting the total crop yield used over the years

plt.figure(figsize = (12,5))
plt.plot(yield_per_year.index, yield_per_year['Yield'],color='blue', linestyle='dashed', marker='o', markersize=12, markerfacecolor='green')
plt.xlabel('Year')
plt.ylabel('Yield (tons per hectare)')
plt.title('Measure of Yield over the years')
plt.savefig('crop_yield_over_years.png')



## ------------------------------------------------- State wise analysis -------------------------------------------------


# Load the GeoJSON file for Indian states
india_map = gpd.read_file('india_telengana.geojson') # loading the geojson file to get geodataframa of india map

# print(india_map)
india_ut = ['Andaman and Nicobar', 'Dadra and Nagar Haveli', 'Daman and Diu', 'Puducherry', 'Lakshadweep', 'Chandigarh']
india_states_original = india_map.drop(india_map[india_map['NAME_1'].isin(india_ut)].index)  # removing the union territories  of india

# updating the names of states as per the dataset
india_states_original['NAME_1'] = india_states_original['NAME_1'].replace('Orissa','Odisha')
india_states_original['NAME_1'] = india_states_original['NAME_1'].replace('Uttaranchal','Uttarakhand')
#print(india_states_original)


df_filtered = df[df['Crop_Year'].isin([1999, 2009, 2019])] # Taking the cultivated area for 3 years (1999, 2009, 2019)


df_area = df_filtered.groupby(['State', 'Crop_Year'])['Area'].sum().unstack() # grouping the data on basis of state and crop_year

#print(df_area)


# plotting are of land cultivated in year 1999,2009,2019

crop_year = [ 1999, 2009, 2019]


india_states_area = india_states_original.copy() # making a copy of geodataframe of indian map

india_states_area = india_states_area.merge(df_area, how='left', left_on='NAME_1', right_on='State') # merging geo-dataframe and df_area by left join
#print(india_states_copy.head())

fig, axes = plt.subplots(1, len(crop_year), figsize=(20, 20))

for i, var in enumerate(crop_year):
      ax = axes[i]
      india_states_area.plot(column=var, ax=ax, edgecolor='black', linewidth=0.4, legend=True, cmap='Oranges',
                            legend_kwds={"label": "Land area (hectare) in year "+ str(var), "orientation": "horizontal"}, # plotting the geodataframe
                            missing_kwds={"color": "lightgrey", "label": "Missing data"})

      missing_patch = mpatches.Patch(color='lightgrey', label='Missing data') # data is missing for the Rajasthan state
      ax.legend(handles=[missing_patch], loc='lower left')
      ax.set_title(f"Cultivated Area of Land by State in {crop_year[i]}")




plt.savefig('area_cultivated_3_different_years_indian_states.png')



#finding mean annual rainfall of all the states

df_unique_rainfall = df.drop_duplicates(subset=['State', 'Annual_Rainfall'])

mean_rainfall = df_unique_rainfall.groupby('State')['Annual_Rainfall'].mean().reset_index()

mean_rainfall.columns = ['state', 'mean_annual_rainfall']

# plotting mean annual rainfall of indian states

india_states_rainfall = india_states_original.copy()

india_states_rainfall = india_states_rainfall.merge(mean_rainfall, how='left', left_on='NAME_1', right_on='state') # merge geodata frame with mean rainfall

#print(india_states_rainfall)

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
india_states_rainfall.plot(column='mean_annual_rainfall', ax=ax, edgecolor='black',linewidth=0.4,legend=True, cmap='Blues', # plotting the geodataframe
                   legend_kwds={"label": "Mean Annual Rainfall (mm)",
                                "orientation": "horizontal"},
                   missing_kwds={"color": "lightgrey",
                                "label": "Missing data"}
                  )


missing_patch = mpatches.Patch(color='lightgrey', label='Missing data') # data is missing for the Rajasthan state
plt.legend(handles=[missing_patch], loc='lower left')

plt.title(" Mean Annual Rainfall in Indian States")
plt.savefig('mean_annual_rainfall_indian_states.png')


# finding overall crop yield in the indian states

df_state = df.groupby('State').sum()

india_states_crop_yield = india_states_original.copy()

india_states_crop_yield = india_states_crop_yield.merge(df_state, how='left', left_on='NAME_1', right_on='State')

#india_states.to_csv('india_states_crop_yield.csv')
#print(india_states)


fig, ax = plt.subplots(1, 1, figsize=(10, 10))
india_states_crop_yield.plot(column='Yield', ax=ax, edgecolor='black', linewidth=0.4,legend=True, cmap='Purples', # plotting the geodataframe
                   legend_kwds={'label': "Overall Crop Yield (tons per hectare)",
                                'orientation': "horizontal"},
                   missing_kwds={"color": "lightgrey", "label": "Missing data"})


missing_patch = mpatches.Patch(color='lightgrey', label='Missing data') # data is missing for the Rajasthan state
ax.legend(handles=[missing_patch], loc='lower left')


plt.title("Crop Yield in Indian States")
plt.savefig('overall_crop_yield_indian_states.png')



# ------------------------------------------------- Season wise analysis --------------------------------------------



# finding the area culivated for each season in year 2019

df_2019 = df[df['Crop_Year']==2019]
df_season_area = df_2019.groupby('Season').agg({'Area': 'sum'}).reset_index() 

plt.figure(figsize=(12, 5))
sns.barplot(x='Season', y='Area', data=df_season_area, palette='viridis')
plt.title('Area Under Cultivation by Each Season in 2019')
plt.ylabel('Total Area (hectares)')
plt.savefig('area_cultivated_seasons.png')

#  Crop yield for Rabi and Kharif season in indian states for the year 2019

seasons = ['Rabi', 'Kharif']

df_filtered = df[(df['Crop_Year'] == 2019) & (df['Season'].isin(seasons))] # filter the data for rabi and kharif season for year 2019

df_grouped = df_filtered.groupby(['State', 'Season'])['Yield'].sum().reset_index() # getting total yeild for each state


fig, axes = plt.subplots(1, len(seasons), figsize=(10, 10))

for i, name in enumerate(seasons):

      india_states_seasons = india_states_original.copy()

      india_states_seasons = india_states_seasons.merge(df_grouped[df_grouped['Season'] == name], how='left', left_on='NAME_1', right_on='State') 

      ax = axes[i]

      india_states_seasons.plot(column='Yield', ax=ax, edgecolor='black', linewidth=0.4, legend=True, cmap='Purples', # plotting the geodataframe
                            legend_kwds={"label": f"Crop Yield (tons per hectare)", "orientation": "horizontal"},
                            missing_kwds={"color": "lightgrey", "label": "Missing data"})

      missing_patch = mpatches.Patch(color='lightgrey', label='Missing data')
      ax.legend(handles=[missing_patch], loc='lower left')
      ax.set_title(f"Crop yield in {name} season in year 2019")



plt.tight_layout()
plt.savefig('crop_yield_seasons_indian_states.png')



# ------------------------------------------------- Crop wise Analysis -------------------------------------------------


# finding number of different crops grown in each indian state

df_crops_count = df.groupby('State')['Crop'].nunique().reset_index()
df_crops_count.columns = ['state', 'crop_count']
#df_crops_count

# plotting heatmap of number of diferent crops grown in each indian state

india_states_crop = india_states_original.copy()

india_states_crop = india_states_crop.merge(df_crops_count, how='left', left_on='NAME_1', right_on='state')


#print(india_states_crop)


fig, ax = plt.subplots(1, 1, figsize=(10, 10))
india_states_crop.plot(column='crop_count', ax=ax, edgecolor='black', linewidth=0.4,legend=True, cmap='Greens', # plotting the geodataframe
                   legend_kwds={'label': "No of different crops",
                                'orientation': "horizontal"},
                   missing_kwds={"color": "lightgrey", "label": "Missing data"})


missing_patch = mpatches.Patch(color='lightgrey', label='Missing data')
ax.legend(handles=[missing_patch], loc='lower left')
#ax.set_title(f" {titles[i]} in Indian States")

plt.title("Number Of Different Crops Grown In Indian States")

plt.savefig('different_crops_indian_states.png')


# filtering the data where yeild is more greater than 0

df_ynz = df[df['Yield']>0]  # where yield is more than zero
df_crop = df_ynz.groupby('Crop').sum() # group the rows as per the column crop
# print(df_crop)

# plotting a line graph to find which crop uses more fertilizer

plt.figure(figsize = (30,10))
plt.plot(df_crop.index, df_crop['Fertilizer'],color='red', linestyle='dashed', marker='o',
        markersize=12, markerfacecolor='cyan')
plt.xlabel('Crops')
plt.ylabel('Fertilizer (Kilogram)')
plt.title(' Use of Fertilizer in different Crops')
plt.xticks(rotation=30)

plt.savefig('crop_fertlizer.png')




# finding top 5 states for the crops: 'Rice', 'Wheat', 'Sugarcane','Banana','Tobacco','Cotton(lint)'

crops = ['Rice', 'Wheat', 'Sugarcane','Banana','Tobacco','Cotton(lint)']
fig, axes = plt.subplots(2, 3, figsize=(20,20))
axes = axes.flatten()


for i, crop in enumerate(crops):

    crop_data = df[df['Crop'] == crop].groupby('State').agg({'Production': 'sum'}).reset_index() # calculating total production of each crop
    #print(crop_data)
    #break
    crop_data = crop_data.set_index('State')
    india_states_corp = india_states_original.copy() # making copy of geodata of indian map
    india_states_corp = india_states_corp.set_index('NAME_1')

    merged_geo_crop_data = india_states_corp.join(crop_data, how='left') # merging the geodata with the crop data
    #print(merged)
    top_5_states = merged_geo_crop_data.nlargest(5, 'Production') # getting the top 5 states for each crop

    india_states_corp.plot(ax=axes[i], color='lightgrey', edgecolor='black')   # plotting the heatmap
    top_5_states.plot(ax=axes[i], color='gold', edgecolor='black', legend=True)


    axes[i].set_title(f'Top 5 {crop} Producing States')
    #break


plt.tight_layout()

plt.savefig('top_5_crops_indian_states.png')




# -------------------------------------------------  Crop Yield Prediction using ML models -------------------------------------------------





df_ml = df.copy()

# dropping year column as it will not help in the prediction,  pesticide column is being dropped because it shows same correlation as fertilizer
df_ml = df_ml.drop(['Crop_Year','Pesticide'], axis = 1)

category_columns = df_ml.select_dtypes(include = ['object']).columns


df_ml = pd.get_dummies(df_ml, columns = category_columns, drop_first=True) # doing one hot encoding for dummy columns

print(df_ml.shape)



# Removing the target column for the features
X = df_ml.drop(['Yield'], axis = 1)
y = df_ml[['Yield']]

print('shape of X', X.shape)
print('shape of y', y.shape) 



# -----------------  Splitting  the data set into train and test set -----------------



x_train, x_test, y_train,y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

print(x_train.shape, x_test.shape, y_train.shape,y_test.shape) 

# ----------------- Feature Engineering -----------------

# using power transformer to transform the data into gaussian distribution
pt = PowerTransformer(method='yeo-johnson')

x_train_pt_transform = pt.fit_transform(x_train)
x_test_pt_transform = pt.fit_transform(x_test)

df_pt_transform = pd.DataFrame(x_train_pt_transform, columns=x_train.columns)
df_pt_transform.head()


# to store accuracy values of each model
train_accuracy_list = []
test_accuracy_list= []


# ----------------- Linear Regression -----------------

# Apply linear regresssion on the original dataset

lr = LinearRegression() # loading the model
lr.fit(x_train,y_train) # training the model

print(' Running linear regression on original data\n')
y_pred_train = lr.predict(x_train) # getting the train predictions
print("Training Accuracy (r2_score): ",r2_score(y_train,y_pred_train))  # getting the training accuracy
print("Training Error (MSE):", mean_squared_error(y_train, y_pred_train)) # getting the training error

print()

y_pred_test = lr.predict(x_test) # getting the test predictions
print("Test Accuracy (r2_score) : ",r2_score(y_test,y_pred_test))  # getting the test accuracy
print("Test Error (MSE):", mean_squared_error(y_test, y_pred_test)) # getting the training error



# Using Linear Regression model on transformed data
lr.fit(x_train_pt_transform, y_train ) # training the model

# prediction 
y_pred_train_lr = lr.predict(x_train_pt_transform) 
y_pred_test_lr = lr.predict(x_test_pt_transform)

print(' Running linear regression on transformed data\n')

print("Training Accuracy (r2_score): ",r2_score(y_train,y_pred_train_lr))  # getting the training accuracy
print("Training Error (MSE):", mean_squared_error(y_train, y_pred_train_lr)) # getting the training error

print()
print("Test Accuracy (r2_score) : ",r2_score(y_test,y_pred_test_lr))  # getting the test accuracy
print("Test Error (MSE):", mean_squared_error(y_test, y_pred_test_lr)) # getting the training error

train_accuracy_list.append(r2_score(y_train,y_pred_train_lr))
test_accuracy_list.append(r2_score(y_test,y_pred_test_lr))

# ----------------- Random Forest Regressor -----------------

# Using Random Forest Regresssion
regr = RandomForestRegressor()

regr.fit(x_train_pt_transform, y_train) 

#prediction 

y_pred_train_rf= regr.predict(x_train_pt_transform)
y_pred_test_rf = regr.predict(x_test_pt_transform)

print(' Running Random forest on transformed data\n') 

print("Training Accuracy (r2_score): ",r2_score(y_train,y_pred_train_rf))  # getting the training accuracy
print("Training Error (MSE):", mean_squared_error(y_train, y_pred_train_rf)) # getting the training error

print()
print("Test Accuracy (r2_score) : ",r2_score(y_test,y_pred_test_rf))  # getting the test accuracy
print("Test Error (MSE):", mean_squared_error(y_test, y_pred_test_rf)) # getting the training error

train_accuracy_list.append(r2_score(y_train,y_pred_train_rf))
test_accuracy_list.append(r2_score(y_test,y_pred_test_rf))

# ----------------- Support Vector Regressor -----------------

# Using Support Vector Machine Regression model
svr = SVR()
svr.fit(x_train_pt_transform, y_train)


#prediction 

y_pred_train_svm= svr.predict(x_train_pt_transform)
y_pred_test_svm = svr.predict(x_test_pt_transform)

print(' Running Support vector machine on transformed data\n')

print("Training Accuracy (r2_score): ",r2_score(y_train,y_pred_train_svm))  # getting the training accuracy
print("Training Error (MSE):", mean_squared_error(y_train, y_pred_train_svm)) # getting the training error

print()
print("Test Accuracy (r2_score) : ",r2_score(y_test,y_pred_test_svm))  # getting the test accuracy
print("Test Error (MSE):", mean_squared_error(y_test, y_pred_test_svm)) # getting the training error

train_accuracy_list.append(r2_score(y_train,y_pred_train_svm))
test_accuracy_list.append(r2_score(y_test,y_pred_test_svm))

# ----------------- CatBoostRegressor -----------------

# using catboost regression model

cat = CatBoostRegressor(learning_rate=0.15)
cat.fit(x_train_pt_transform, y_train)

y_pred_train_cat = cat.predict(x_train_pt_transform)
y_pred_test_cat = cat.predict(x_test_pt_transform)

print(' Running Catboost on transformed data \n')

print("Training Accuracy (r2_score): ",r2_score(y_train,y_pred_train_cat))  # getting the training accuracy
print("Training Error (MSE):", mean_squared_error(y_train, y_pred_train_cat)) # getting the training error

print()
print("Test Accuracy (r2_score) : ",r2_score(y_test,y_pred_test_cat))  # getting the test accuracy
print("Test Error (MSE):", mean_squared_error(y_test, y_pred_test_cat)) # getting the training error

train_accuracy_list.append(r2_score(y_train,y_pred_train_cat))
test_accuracy_list.append(r2_score(y_test,y_pred_test_cat))

# Comparison of the models

algorithm = ['LinearRegression','RandomForestRegressor','SupprtVectorRegressor','CatBoostRegressor']
accu_data = {'Training Accuracy':train_accuracy_list,'Test Accuracy':test_accuracy_list}
results = pd.DataFrame(accu_data, index = algorithm)
print(results)

