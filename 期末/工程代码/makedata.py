import numpy as np
import pandas as pd
import os
from PIL import Image


def luoData(path):
    lis = os.listdir('C:\\Users\\xuduoduo\\Desktop\\dog_divide\\data')
    print(lis)
    labels = []
    imgData = []
    for ele in lis:
        sample = os.listdir(path + '/' + ele)
        for i in sample:
            print(path + "/" + ele + '/' + i)
            if ele == "n000001-Shiba_Dog":
                img = Image.open(path + "/" + ele + '/' + i)
                img = img.resize((128, 128), Image.ANTIALIAS)
                img = np.array(img)
                img = img / 255.0
                imgData.append(img)
                labels.append(0)
            if ele == 'n000002-French_bulldog':
                img = Image.open(path + "/" + ele + '/' + i)
                img = img.resize((128, 128), Image.ANTIALIAS)
                img = np.array(img)
                img = img / 255.
                imgData.append(img)
                labels.append(1)
            if ele == 'n000003-Siberian_husky':
                img = Image.open(path + "/" + ele + '/' + i)
                img = img.resize((128, 128), Image.ANTIALIAS)
                img = np.array(img)
                img = img / 255.
                imgData.append(img)
                labels.append(2)
            if ele == 'n000004-Pomeranian':
                img = Image.open(path + "/" + ele + '/' + i)
                img = img.resize((128, 128), Image.ANTIALIAS)
                img = np.array(img)
                img = img / 255.
                imgData.append(img)
                labels.append(3)
            if ele == 'n000005-golden_retriever':
                img = Image.open(path + "/" + ele + '/' + i)
                img = img.resize((128, 128), Image.ANTIALIAS)
                img = np.array(img)
                img = img / 255.
                imgData.append(img)
                labels.append(4)

    imgData = np.array(imgData)
    labels = np.array(labels)

    return imgData, labels


# luoData('E:\parcharm_project\cat_dog\data')
