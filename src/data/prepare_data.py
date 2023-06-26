from src.utils.paths import get_project_path

import os
import shutil
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from rich.progress import track

import warnings

warnings.filterwarnings('ignore')

def read_excel(path):
    df = pd.ExcelFile(path)
    data_2021 = pd.read_excel(df, sheet_name=df.sheet_names[1])
    data_2022 = pd.read_excel(df, sheet_name=df.sheet_names[0])
    
    data_dict = {
        'data_2021': data_2021,
        'data_2022': data_2022
    }
    
    return data_dict

def prepare_table():
    tables = read_excel(os.path.join(get_project_path(), 'data', 'raw', 'table_data.xlsx'))
    table = pd.concat([tables['data_2021'], tables['data_2022']], ignore_index=True)
    list_of_images = get_image_paths(os.path.join(get_project_path(), 'data', 'raw', 'image_data'))
    
    list_of_series = []
    
    for path in track(list_of_images, description='[yellow]Processing table...'):
        df = table.loc[table[table.columns[0]] == path.split(os.sep)[-1]]
        df['Year'] = path.split(os.sep)[-3]
        list_of_series.append(df)
        
    final_table = pd.concat(list_of_series, ignore_index=True)
    final_table.rename(columns={
        final_table.columns[0]: 'File name',
        final_table.columns[1]: 'Laundering degree',
        final_table.columns[2]: 'Magnification',
        final_table.columns[3]: 'Substance type'
    }, inplace=True)
    
    final_table.to_csv(os.path.join(get_project_path(), 'data', 'processed', 'tables', 'table.csv'), index=False)

def get_image_paths(path):
    list_of_paths = []
    
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.tif'):
                list_of_paths.append(os.path.join(root, file))
                
    return list_of_paths

def read_and_prepare_images(path):
    list_of_images = []
    
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.tif'):
                list_of_images.append(os.path.join(root, file))
        
    list_of_conveted_images = []
        
    for file in track(list_of_images, description="[blue]Converting images..."):
        img = Image.open(file)
        save_path = os.path.join(get_project_path(), 'tmp', os.sep.join(file.split(os.sep)[-3:]))
        
        if not os.path.exists(os.sep.join(save_path.split(os.sep)[0:-1])):
            os.makedirs(os.sep.join(save_path.split(os.sep)[0:-1]))
        
        img.save(save_path[:-4]+'.png', 'png', quality=100)
        list_of_conveted_images.append((save_path[:-4]+'.png'))
    
    for file in track(list_of_conveted_images, description="[green]Processing images..."):
        img = cv2.imread(file)
        img = cv2.resize(img, dsize=(448, 448))
        img = img[:-30]
        img = cv2.resize(img, dsize=(300, 300))
        edges = cv2.Canny(img, 1, 200)
        
        save_path = os.path.join(get_project_path(), 'data', 'processed', 'images')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
        cv2.imwrite(os.path.join(save_path, file.split(os.sep)[-1]), img)
        cv2.imwrite(os.path.join(save_path, (file.split(os.sep)[-1][:-4]+'_edges.png')), edges)

    shutil.rmtree(os.path.join(get_project_path(), 'tmp'))

def prepare_dataset():
    prepare_table()
    read_and_prepare_images(os.path.join(get_project_path(), 'data', 'raw', 'image_data'))

if __name__ == "__main__":
    prepare_dataset()