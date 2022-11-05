import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer,KNNImputer
import itertools

def feature_inputer(
    df:pd.DataFrame,
    to_mode = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP'],
    to_median = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
) -> pd.DataFrame:
    for col in to_mode:
        mode_inputer = SimpleImputer(strategy='most_frequent')
        df[col] = mode_inputer.fit_transform(df[[col]])

    
    for col in to_median:
        inputer = SimpleImputer(strategy='median')
        df[col] = inputer.fit_transform(df[[col]])
    return df

def dtype_memory_reducer(df: pd.DataFrame) -> pd.DataFrame:
    df['CryoSleep'] = df['CryoSleep'].astype(bool)
    df['VIP'] = df['VIP'].astype(bool)

    df['HomePlanet'] = df['HomePlanet'].astype('category')
    df['Destination'] = df['Destination'].astype('category')
    return df

def outliers_to_log(df:pd.DataFrame) -> pd.DataFrame:
    to_log = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    for col in to_log:
        df[col] = np.log(df[col] + 1)
    return df

def cabin_inputer(df:pd.DataFrame) -> pd.DataFrame:
    cabin_features = df['Cabin'].str.split("/",expand=True)[[0,2]].rename(columns={0:'Deck',2:'side'})
    df = pd.concat([df,cabin_features],axis=1)
    df['Deck'] = df['Deck'].fillna('G').replace("T","G")
    df = df.drop(['side','Cabin'],axis=1)
    return df

def vip_knn_input(df: pd.DataFrame) -> pd.DataFrame:
    inputer = KNNImputer(n_neighbors=5)
    df['VIP'] = inputer.fit_transform(df[['VIP','RoomService']])[:,0]
    return df

def apply_interactions(df:pd.DataFrame, target_col: str):
    for x in itertools.combinations(df.columns.drop(target_col), 2):
        df[f'{x[0]}_{x[1]}'] = df[x[0]]*df[x[1]]
    df = df.drop(df.columns[df.nunique() == 1].tolist(),axis=1)
    return df
