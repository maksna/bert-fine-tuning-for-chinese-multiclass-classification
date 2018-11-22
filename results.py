import numpy as np
data_dir1="./test_results.tsv"
data_dir="./test.tsv"
with open(file=data_dir,mode="r",encoding="utf-8") as f:
    text=f.readlines()
    y_true=[]
    for t in text:
        if  t.split("\t")!=0 and t!="\n":
            y_true.append(int(t.split("\t")[0]))

with open(file=data_dir1,mode="r",encoding="utf-8") as f:
    result=f.readlines()
    y_pred=[]
    for l in result:
        l=list(map(float,l.split("\t")))
        y_pred.append(np.argmax(l))

from sklearn import metrics
# 混淆矩阵
print("Confusion Matrix...")
cm = metrics.confusion_matrix(y_true, y_pred)
from sklearn.metrics import classification_report
target_names = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']
print(classification_report(y_true, y_pred, target_names=target_names))