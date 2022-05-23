import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
pd.plotting.register_matplotlib_converters()
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns


def Catplot(df, x, y):
    plt.subplots(1, 2, figsize=(14, 5))
    plt.subplot(1, 2, 1)
    sns.countplot(x=df[x].dropna(), hue=df[y])

    plt.subplot(1, 2, 2)
    plt.ylim(0, 1)
    sns.lineplot(x=df[x], y=df[y], data=df, ci=None, linewidth=3, marker="o")
    plt.show()

def unity_family(data):
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


train_data = pd.read_csv('C:/Users/invek/Desktop/Spaceship_Titanic/train.csv')
test_data = pd.read_csv('C:/Users/invek/Desktop/Spaceship_Titanic/test.csv')
lost_signs = ['Cabin', 'HomePlanet', 'Destination']
columns_to_drop = ['NumOfGroup', 'Name', 'NumInGroup', 'Surname', 'Cabin',]# 'Deck_A', 'Deck_D', 'Deck_T'
spends = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Spends']
num_values = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', ]
bool_values = ['VIP', 'CryoSleep']
categor_values = ['HomePlanet', 'Destination', 'Deck']
datas = [train_data, test_data]
normalizer = Normalizer()

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
    train_data = pd.concat([train_data, pd.get_dummies(train_data[categor_value], prefix=categor_value)], axis=1)
train_data.drop(columns_to_drop, inplace=True, axis=1)
train_data.drop(categor_values, inplace=True, axis=1)
train_data.sort_values('PassengerId', inplace=True)
train_data.set_index('PassengerId', inplace=True)
for categor_value in categor_values:
    test_data = pd.concat([test_data, pd.get_dummies(test_data[categor_value], prefix=categor_value)], axis=1)
test_data.drop(columns_to_drop, inplace=True, axis=1)
test_data.drop(categor_values, inplace=True, axis=1)
test_data.sort_values('PassengerId', inplace=True)
test_data.set_index('PassengerId', inplace=True)


X = train_data.drop('Transported', axis=1)
y = train_data['Transported']

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.9, test_size=0.1, random_state=42)

params = {'learning_rate' : [i/100 for i in range(2, 10)],
          'n_estimators' : [i*10 for i in range(30, 50)]}

model = XGBClassifier(learning_rate=0.05, n_estimators=90, max_depth=6, random_state=1, min_child_weight=3)
grid_search = GridSearchCV(model, params)
model.fit(X_train, y_train)
print('Score = ', model.score(X_valid, y_valid))
# print(grid_search.best_params_)
#
# predictions = pd.read_csv("C:/Users/invek/Desktop/Spaceship_Titanic/sample_submission.csv")
# predictions.drop("Transported", axis=1, inplace=True)
# predictions["Transported"] = model.predict(test_data).astype(bool)
# predictions.to_csv("C:/Users/invek/Desktop/Spaceship_Titanic/sample_submission_1.csv", index=False)

