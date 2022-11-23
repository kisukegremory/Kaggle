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

def outliers_to_log(
    df:pd.DataFrame, 
    cols: list = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    ) -> pd.DataFrame:
    for col in cols:
        mask = df[col] > 0
        df.loc[mask,col] = np.log(df.loc[mask,col])
    return df

def cabin_inputer(df:pd.DataFrame) -> pd.DataFrame:
    cabin_features = df['Cabin'].str.split("/",expand=True).rename(columns={0:'Deck',1:'Num',2:'Side'})
    df = pd.concat([df,cabin_features],axis=1)
    df['Side'] = df['Side'].fillna("U")
    df['Deck'] = df['Deck'].fillna('G').replace("T","G")
    df['Num'] = df['Num'].fillna(df['Num'].median()).astype(int)
    df = df.drop(['Cabin'],axis=1)
    return df

def vip_knn_input(df: pd.DataFrame) -> pd.DataFrame:
    mask = (df['HomePlanet'] == 'Earth') & df['VIP'].isna()
    df.loc[mask,'VIP'] = False 
    inputer = KNNImputer(n_neighbors=5)
    df['VIP'] = inputer.fit_transform(df[['VIP','RoomService']])[:,0]
    return df

def apply_interactions(df:pd.DataFrame, target_col: str):
    for x in itertools.combinations(df.columns.drop(target_col), 2):
        df[f'{x[0]}_{x[1]}'] = df[x[0]]*df[x[1]]
    df = df.drop(df.columns[df.nunique() == 1].tolist(),axis=1)
    return df

def calculate_group_lastname_size(df_index: pd.core.indexes.base.Index, df: pd.DataFrame):
    '''
    usage example:
    df['GroupLastNameSize'] = calculate_group_lastname_size(
        df_train.index,
        pd.concat([df_train,df_test])
    )
    '''
    df[['first_name','last_name']] = df['Name'].str.split(" ",expand=True)
    df['group_id'] = df.index.to_frame()['PassengerId'].str.split("_",expand=True).astype(int)[0].to_frame('GroupID')
    df['GroupSize'] = df.groupby(['group_id'])['group_id'].transform('count')
    df['GroupLastNameSize'] = df.groupby(['group_id','last_name'])['group_id'].transform('count')
    df['GroupLastNameSize'] = df['GroupLastNameSize'].fillna(1).astype(int)
    return df['GroupLastNameSize'].loc[df_index]

def calculate_groupsize(df_index: pd.core.indexes.base.Index, full_index: pd.core.indexes.base.Index):
    full_index = full_index.to_frame()['PassengerId'].str.split("_",expand=True).astype(int)[0].to_frame('GroupID')
    full_index['GroupSize'] = full_index.groupby(['GroupID'])['GroupID'].transform('count')
    return full_index.loc[df_index,'GroupSize']

def calculate_seat_id(df_index: pd.core.indexes.base.Index):
    return df_index.to_frame()['PassengerId'].str.split("_",expand=True).astype(int)[1]


def fillna_homeplanet_and_destination(df: pd.DataFrame) -> pd.DataFrame:
    home = ['Earth','Europa', 'Mars']
    dest = ['PSO J318.5-22','55 Cancri e','TRAPPIST-1e']

    fill_home = dict(zip(dest,home))
    mask = df['HomePlanet'].isna() & df['Destination'].notnull()
    df.loc[mask,'HomePlanet'] = df.loc[mask,'Destination'].map(fill_home)

    fill_dest = dict(zip(home,dest))
    mask = df['HomePlanet'].notnull() & df['Destination'].isna()
    df.loc[mask,'Destination'] = df.loc[mask,'HomePlanet'].map(fill_dest)

    df['Destination'].fillna('TRAPPIST-1e', inplace=True)
    df['HomePlanet'].fillna('Earth', inplace=True)

    return df

def fill_with_0_people_with_no_other_wastes(df: pd.DataFrame) -> pd.DataFrame:
    waste_features = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    for col in waste_features:
        other_cols = [feat for feat in waste_features if feat != col]
        mask = (df[col].isna()) & (df[other_cols].notnull().all(axis=1)) & (df[other_cols].sum(axis=1) == 0)
        df.loc[mask,col] = 0
    return df

def fill_0_wastes_people_cryosleep(df:pd.DataFrame) -> pd.DataFrame:
    mask = df['CryoSleep'] == True
    waste_features = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    for col in waste_features:
        df.loc[mask,col] = 0
    return df

def fill_with_non_0_median(df):
    waste_features = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    for col in waste_features:
        mask = df[col].isna()
        df.loc[mask,col] = df.loc[df[col] > 0,col].median()
    return df

def fill_cryosleep(df):
    df['0_bills'] = df[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].sum(axis=1) == 0
    df.loc[df['CryoSleep'].isna(),'CryoSleep'] = df.loc[df['CryoSleep'].isna(),'0_bills']
    df['CryoSleep'] = df['CryoSleep'].astype(bool)
    df.drop(['0_bills'],axis=1,inplace=True)
    return df