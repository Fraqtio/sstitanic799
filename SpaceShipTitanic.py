import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import GridSearchCV


def unity_family(data):
    """
    Function to sort out passangers that travel with family and fill empty data in confident cases, like spends of
    people in CryoSleep or Number of Passangers Cabin that travel with family.
    :param data: pd.DataFrame
    :return: pd.DataFrame
    """
    data.sort_values(['NumOfGroup', 'Surname'], inplace=True)
    data.index = [i for i in range(data.shape[0])]
    last_surname, last_group = data['Surname'][0], data['NumOfGroup'][0]
    for i in range(1, data.shape[0]):
        if (last_surname, last_group) == (data['Surname'][i], data['NumOfGroup'][i]):
            data.loc[i, 'WithFamily'] = 1
            data.loc[i-1, 'WithFamily'] = 1
            for sign in lost_signs:
                if pd.isnull(data.loc[i, sign]):
                    data.loc[i, sign] = data.loc[i-1, sign]
        else:
            last_surname, last_group = data['Surname'][i], data['NumOfGroup'][i]
        if (pd.isnull(data['VIP'][i]) == True) and (data.loc[i, 'CryoSleep'] == True):
            data.loc[i, 'VIP'] = False
        elif ((pd.isnull(data['VIP'][i]) == True) and (data['Spends'][i] > 0.95)):
            data.loc[i, 'VIP'] = True
        if ((data['CryoSleep'][i]) == False) and (data['Spends'][i] == 0) and ((pd.isnull(data['Age'][i]) == True)):
            data.loc[i, 'Age'] = 10
        if ((data['Age'][i]) < 16) and (data['Spends'][i] == 0) and ((pd.isnull(data['CryoSleep'][i]) == True)):
            data.loc[i, 'CryoSleep'] = True
    return data
#-------------------------------------------------------------------------
# Loading Train and Test datas and announce a few helping lists
#-------------------------------------------------------------------------
train_data = pd.read_csv('C:/Users/invek/Desktop/Spaceship_Titanic/train.csv')
test_data = pd.read_csv('C:/Users/invek/Desktop/Spaceship_Titanic/test.csv')
lost_signs = ['Cabin', 'HomePlanet', 'Destination']
columns_to_drop = ['NumOfGroup', 'Name', 'NumInGroup', 'Surname', 'Cabin',]
spends = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Spends']
num_values = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', ]
bool_values = ['VIP', 'CryoSleep']
categor_values = ['HomePlanet', 'Destination', 'Deck']
datas = [train_data, test_data]
normalizer = Normalizer() # Tool to normilize spends values
#-------------------------------------------------------------------------
# Preprocessing of data or 'my answer to pipelines in small datasets'
#-------------------------------------------------------------------------
for data in datas:
    data['NumOfGroup'] = data['PassengerId'].str.split('_', expand=True)[0]
    data['NumInGroup'] = data['PassengerId'].str.split('_', expand=True)[1]
    data['Name'].fillna(method='ffill', inplace=True)
    data['Surname'] = data['Name'].str.split(expand=True)[1]
    data['WithFamily'] = 0
    data['Spends'] = data[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].sum(axis=1)
    data.loc[data['FoodCourt'] > 17000, 'FoodCourt'] = data.loc[
        data['FoodCourt'] < 17000, 'FoodCourt'].mean()
    data.loc[data['ShoppingMall'] > 9000, 'ShoppingMall'] = data.loc[
        data['ShoppingMall'] < 9000, 'ShoppingMall'].mean()
    data.loc[data['Spa'] > 15000, 'Spa'] = data.loc[data['Spa'] < 15000, 'Spa'].mean()
    data.loc[data['VRDeck'] > 13000, 'VRDeck'] = data.loc[data['VRDeck'] < 13000, 'VRDeck'].mean()
    data.loc[data['RoomService'] > 8500, 'RoomService'] = data.loc[
        data['RoomService'] < 8500, 'RoomService'].mean()
    for spend in spends:
        data[spend].fillna(value=0, inplace=True)
    data[spends] = normalizer.fit_transform(data[spends])
    data = unity_family(data)
    data['Deck'] = data['Cabin'].str.split('/', expand=True)[0]
    data['Deck'].fillna(value=data['Deck'].mode()[0], inplace=True)
    data['SideOfCabin'] = data['Cabin'].str.split('/', expand=True)[2]
    data['SideOfCabin'].fillna(value=data['SideOfCabin'].mode()[0], inplace=True)
    data['SideOfCabin'] = data['SideOfCabin'].apply(lambda x: int(x != 'S'))
    data['HomePlanet'].fillna(value=data['HomePlanet'].mode()[0], inplace=True)
    data['Destination'].fillna(value=data['Destination'].mode()[0], inplace=True)
    data['Age'].fillna(value=data['Age'].mean(), inplace=True)
    data['VIP'].fillna(value=False, inplace=True)
    data['CryoSleep'].fillna(value=False, inplace=True)
    for bool_value in bool_values:
        data[bool_value] = data[bool_value].apply(lambda x: int(x))
    for categor_value in categor_values:
        data[pd.get_dummies(data).columns] = pd.get_dummies(data[categor_value])
    data.drop(columns_to_drop, inplace=True, axis=1)
    data.drop(categor_values, inplace=True, axis=1)
    data.sort_values('PassengerId', inplace=True)
    data.set_index('PassengerId', inplace=True)

X = train_data.drop('Transported', axis=1)
y = train_data['Transported']

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=42)
#-------------------------------------------------
# Create and fir XGB Classifier model and find best parameters with GridSearch, then fit.
#-------------------------------------------------
model = XGBClassifier(learning_rate=0.05, n_estimators=90, max_depth=6, random_state=1, min_child_weight=3)
# params = {'learning_rate' : [i/100 for i in range(2, 10)],
#           'n_estimators' : [i*10 for i in range(30, 50)],
#           'max_depth' : [3, 4, 5, 6],
#           'min_child_weight' : [2, 3, 4, 5]}
# grid_search = GridSearchCV(model, params)
model.fit(X_train, y_train)
print('Score = ', model.score(X_valid, y_valid))
# print(grid_search.best_params_)
#----------------------------------
# Create prediction of test data and seve it ti csv file
#----------------------------------
# predictions = pd.read_csv("C:/Users/invek/Desktop/Spaceship_Titanic/sample_submission.csv")
# predictions.drop("Transported", axis=1, inplace=True)
# predictions["Transported"] = model.predict(test_data).astype(bool)
# predictions.to_csv("C:/Users/invek/Desktop/Spaceship_Titanic/sample_submission_1.csv", index=False)
