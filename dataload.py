#!/usr/bin/env python
# coding: utf-8

# In[22]:

import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader,random_split
from torchvision import transforms, utils
from PIL import Image
import numpy as np


# In[18]:


def dataGathering(path,field=[],value=0,number=0):
    data=pd.read_csv(path,index_col='image_id')
    data=data[field]
    return data

def ConcatenateSample(data):
    minsample=0
    first=True
    for i in data:
        num=i[i.columns[0]].count()
        if first:
            minsample=num
            first=False
        if minsample>num:
            minsample=num
    df=[]
    for i in data:
        df.append(i.sample(n=minsample, random_state=1))        
    df=pd.concat(df)
    df=df.sort_index()
    return df
def load_image( infilename ):
    img = Image.open('celeba-dataset/img_align_celeba/'+ infilename )
    return img    


# In[27]:


class ImageDataset(Dataset):
    def __init__(self,data,transform=None):
        self.transform=transform
        self.data=data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = load_image(self.data.index[idx])
        image=image.resize((64,64), Image.ANTIALIAS)
        image=np.array(image)        
        if self.transform:
            image = self.transform(image)
        Male =self.data.iloc[idx, 0]
        Male = np.array([Male])
        Glasses =self.data.iloc[idx, 1]
        Glasses = np.array([Glasses])
        sample = {'image': image, 'Male': Male,'Glasses':Glasses}    
        sample=list(sample.values())
        return sample
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, image):

        image = np.array(image)
        image=torch.from_numpy(image/255)
        
        image=image.permute(2, 1, 0)
        
        return  image


# In[30]:


def load_data():

    attribute_path='img_data.csv'
    targetfield=["Male","Eyeglasses"]
    attribute_Data=dataGathering(attribute_path,targetfield)
    Male_Glasses = attribute_Data.loc[(attribute_Data['Male'] == 1) & (attribute_Data['Eyeglasses']== 1)] 
    Female_Glasses = attribute_Data.loc[(attribute_Data['Male'] == -1) & (attribute_Data['Eyeglasses']== 1)] 
    Male_NonGlasses = attribute_Data.loc[(attribute_Data['Male'] == 1) & (attribute_Data['Eyeglasses']== -1)] 
    Female_NonGlasses = attribute_Data.loc[(attribute_Data['Male'] == -1)& (  attribute_Data['Eyeglasses']== -1)] 
    transform =transforms.Compose([transforms.ToPILImage(),
                               transforms.RandomHorizontalFlip(),                              
                               ToTensor()])  
    transformed_dataset=ImageDataset(attribute_Data,transform)
    
    return transformed_dataset

def batchfy(dataset=load_data(),batch_size=64,ratio=0.9,num_workers=0):
    
    train_size=size=int(ratio*len(dataset))
    test_size=len(dataset) - train_size
    train_dataset, test_dataset =random_split(dataset, [train_size, test_size])
    train_load = DataLoader(train_dataset, batch_size=batch_size,
                         num_workers=num_workers,shuffle=True)
    test_load = DataLoader(test_dataset, batch_size=batch_size,
                         num_workers=num_workers)
    return train_load,test_load
    


# In[23]:





# In[ ]:





# In[ ]:




