import csv

import pandas as pd
import glob
import os




#    IMPORTANT !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#    THIS PYTHON FOR MERGE TWO CSV.
#    YOU DON'T NEED TO USE THIS SCRIPT IF YOU HAVE TRAIN AND TEST DATASET.
#    IMPORTANT !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!





# def Merge(fileName1,encoding1,delimiter1,fileName2,encoding2,delimiter2):
#     with open(fileName2, encoding=encoding2) as file2:
#         reader = csv.reader(file2, delimiter=delimiter2)
#         for line in reader:
#             if firstFlag:
#                 mergeList.append(line)
#             firstFlag = True
#
#     df = pd.DataFrame(mergeList, columns=["questionID","answerText","label"])
#     df.to_csv("Deneme.csv")
#     return mergeList
#
# Merge("csvFiles/Answer.csv","utf-8",";","csvFiles/Employees_Answers_21.06.2022.csv","windows-1254",";")
#
# def drop1():
#     data = pd.read_csv("Answer.csv")
#     data.drop(["ID","answerID","userID","createdDate","userName"], axis=1, inplace=True)
#     data.to_csv("Data.csv", index=False, sep= ";")
# drop1()



#
# def convertUTF8():
#
#     mergeList = []
#     firstFlag=False
#
#     with open("Answers0.csv", encoding='windows-1254') as file1:
#         reader=csv.reader(file1,delimiter=";")
#         for line in reader:
#             if firstFlag:
#                 mergeList.append([line[0],line[1],line[2]])
#
#             firstFlag = True
#
#     df = pd.DataFrame(mergeList)
#     df.to_csv("Answers1.csv",encoding="utf-8",index=False)
#     pass
# convertUTF8()
#
# def merge():
#     files = os.path.join("C:\\Users\\PycharmProjects\\BertTrainModel\\", "Answers*.csv")
#
#     files = glob.glob(files)
#     data = pd.concat(map(pd.read_csv, files), ignore_index=True)
#     data['0'] = data['0'].astype('float').astype('Int64')
#     data['2'] = data['2'].astype('float').astype('Int64')
#     data.to_csv("Data.csv",encoding="utf-8",index=False,sep=";")
# merge()

