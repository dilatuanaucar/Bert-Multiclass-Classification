# Bert-Multiclass-Classification
Turkish Bert Model for Positive, Negative and Neural Classification.


mergedCSV.py -> If you don't want to merge your different CSV files, you don't have to run this file.

BertTrainModel.py -> Used pre-trained Turkish Bert Model for train positive, negative and neural labeled data. You have to run for every label seperately. 

    Write in line 254 and 255,
    
    X = negativeTextList
    y = negativeLabelList
    
    /////////////////////  
    
    X = positiveTextList
    y = positiveLabelList
    
    /////////////////////
    
    X = neuralTextList
    y = neuralLabelList
    
    You can chance your model name in line 297.
    
Test.py -> Test your model here.
