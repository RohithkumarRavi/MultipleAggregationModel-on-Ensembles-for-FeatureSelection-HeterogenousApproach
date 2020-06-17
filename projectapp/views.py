from django.shortcuts import render
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
import pandas as pd
import numpy as np
import seaborn as sns
from pylab import savefig
import matplotlib.pyplot as plt
from skfeature.function.information_theoretical_based import MRMR
from skfeature.function.similarity_based import reliefF
from skfeature.function.information_theoretical_based import DISR
from skfeature.function.information_theoretical_based import MRMR
from skfeature.function.information_theoretical_based import FCBF
from skfeature.function.information_theoretical_based import MIFS
from skfeature.function.information_theoretical_based import ICAP
from skfeature.function.information_theoretical_based import JMI
from skfeature.function.similarity_based import SPEC
from skfeature.function.similarity_based import fisher_score
from sklearn import model_selection
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix,precision_recall_fscore_support
import sys,os
sys.path.append('E:/Anaconda/Lib/site-packages')
from pyrankagg.rankagg import FullListRankAggregator
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelBinarizer
from sklearn.multiclass import OneVsRestClassifier


def split_the_data_into_X_and_Y(data):
    y = data.shape[1]
    X = data.iloc[:,0:y-1]
    X1 = X.to_numpy()
    Y = data.iloc[:,-1]
    Y1 = Y.to_numpy()
    return X1,Y1

def feature_ranking(idx):
    idx = idx.tolist()
    length = len(idx)
    rank = [None]*length
    for i in range(length):
        a = idx[i]
        rank[a] = i+1
    return np.array(rank)
def score_to_rank(score):
    id_x = np.argsort(score, 0)
    idx = id_x[::-1]
    rank = feature_ranking(idx)
    return rank

def ranking_features(column_names,rank):
    rank = list(rank)
    column_names = column_names[:-1]
    features = dict(zip(column_names,rank))
    return features

def MRMR_featureSelection(x,y):
    idx = MRMR.mrmr(x,y)
    rank = feature_ranking(idx)
    return rank
def FBCF_featureSelection(x,y):
    idx = FCBF.fcbf(x,y)
    rank = feature_ranking(idx)
    return rank
def MIFS_featureSelection(x,y):
    idx = MIFS.mifs(x,y)
    rank = feature_ranking(idx)
    return rank
def DISR_featureSelection(x,y):
    idx = DISR.disr(x,y)
    rank = feature_ranking(idx)
    return rank
def ICAP_featureSelection(x,y):
    idx = ICAP.icap(x,y)
    rank = feature_ranking(idx)
    return rank
def JMI_featureSelection(x,y):
    idx = JMI.jmi(x,y)
    rank = feature_ranking(idx)
    return rank
def SPEC_featureSelection(x,y):
    score = SPEC.spec(x,y)
    rank = score_to_rank(score)
    return rank
def reliefF_featureSelection(x,y):
    score = reliefF.reliefF(x,y)
    rank = score_to_rank(score)
    return rank
def fischer_score_featureSelection(x,y):
    score = fisher_score.fisher_score(x,y)
    rank = score_to_rank(score)
    return rank
def feature_selection(ranklist,features,column_names):
    rank = np.array(ranklist)
    a,b = rank.shape
    rank = rank.reshape(b,a)
    ranks = pd.DataFrame(rank)
    ranks = ranks.values.tolist()
    attributes = ['attributes']
    attributes = attributes + column_names[0:-1]
    features = [features] + ranks
    feature_selection = dict(zip(attributes,features))
    return feature_selection



def handle_uploaded_file(f):   
    with open('C:/Users/Mr Ravi/Desktop/featureselection/projectapp/'+f.name, 'wb+') as destination:   
        for chunk in f.chunks(): 
            destination.write(chunk)   
def home(request):
    return render(request, "base.html") 

@csrf_exempt
def upload(request):
    if request.method == 'POST':
        file = request.FILES['fileupload']
        filename = file.name
        df = pd.read_csv(file)
        instances = df.shape[0]
        no_of_features = len(df.columns.tolist())-1
        col = df.columns[0:-1].tolist()
        num = list(np.arange(no_of_features))
        last = df.columns[-1]
        dict_names = dict(zip(num,col))
        X,Y = split_the_data_into_X_and_Y(df)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3,random_state = 42)
        data1_X = pd.DataFrame(X_train)
        data1_Y = pd.DataFrame(Y_train)
        data1_Y = data1_Y.rename(columns = {0:last})
        data1 = pd.concat([data1_X,data1_Y],axis=1)
        data1 = data1.rename(columns = dict_names)
        data2_X = pd.DataFrame(X_test)
        data2_Y = pd.DataFrame(Y_test)
        data2_Y = data2_Y.rename(columns = {0:last})
        data2 = pd.concat([data2_X,data2_Y],axis=1)
        data2 = data2.rename(columns = dict_names)
        column_names = df.columns.tolist()
        features = column_names[0:no_of_features]
        data1.to_csv('C:/Users/Mr Ravi/Desktop/featureselection/projectapp/file1.csv')
        data2.to_csv('C:/Users/Mr Ravi/Desktop/featureselection/projectapp/file2.csv')
        request.session['file'] = filename
    return render(request,'Uploaded.html',{'Filename':filename,'Features':no_of_features,'Instances':instances,'classes': len(df[df.columns[-1]].unique().tolist()),'missing_values':df.isnull().any().sum()})
def features(request):
    if request.method == 'POST':
        array = request.POST.getlist('FeatureSelection')
        threshold = request.POST.get('ThresholdValues')
        agg = request.POST.get('RAmethods')
        df = pd.read_csv('C:/Users/Mr Ravi/Desktop/featureselection/projectapp/file1.csv')
        data2 = pd.read_csv('C:/Users/Mr Ravi/Desktop/featureselection/projectapp/file2.csv')
        df = df.drop(df.columns[0],axis=1)
        data2 = data2.drop(data2.columns[0],axis=1)
        no_of_features = len(df.columns)-1
        column_names = df.columns.tolist()
        last = df.columns[-1]
        output = df[last]
        X1,Y1 = split_the_data_into_X_and_Y(df)
        scorelist = []
        ranklist = []
        for ele in array:
            if (ele == 'MRMR_featureSelection'):
                rank = MRMR_featureSelection(X1,Y1)
                feature = ranking_features(column_names,rank)
                
            elif (ele == 'FBCF_featureSelection'):
                rank = FBCF_featureSelection(X1,Y1)
                feature = ranking_features(column_names,rank)
                   
            elif (ele == 'MIFS_featureSelection'):
                rank = MIFS_featureSelection(X1,Y1)
                feature = ranking_features(column_names,rank)
                
            elif (ele =='DISR_featureSelection'):
                rank = DISR_featureSelection(X1,Y1)
                feature = ranking_features(column_names,rank)
                
            elif (ele == 'ICAP_featureSelection'):
                rank = ICAP_featureSelection(X1,Y1)
                feature = ranking_features(column_names,rank)

            elif (ele == 'JMI_featureSelection'):
                rank = JMI_featureSelection(X1,Y1)
                feature = ranking_features(column_names,rank)

            elif (ele =='reliefF_featureSelection'):
                rank = reliefF_featureSelection(X1,Y1)
                feature = ranking_features(column_names,rank)
        
            elif (ele == 'SPEC_featureSelection'):
                rank = SPEC_featureSelection(X1,Y1)
                feature = ranking_features(column_names,rank)
        
            elif (ele == 'fisher_score_featureSelection'):
                rank = fischer_score_featureSelection(X1,Y1)
                feature = ranking_features(column_names,rank)
            ranklist.append(rank)
            scorelist.append(feature)
        FLRA = FullListRankAggregator()
        if(agg == 'Borda'):
            aggRanks = FLRA.borda_aggregation(scorelist)
        elif(agg == 'Robust'):
            aggRanks = FLRA.robust_aggregation(scorelist)
        elif(agg == 'StabilitySelection'):
            aggRanks = FLRA.stability_selection(scorelist)
        elif(agg == 'Exponential weighting'):
            aggRanks = FLRA.exponential_weighting(scorelist)
        ranked_feature = list(aggRanks.keys())
        if(threshold == '25 percent features'):
            no_of_selected_features = int(no_of_features*(25/100))
        elif(threshold == '50 percent features'):
            no_of_selected_features = int(no_of_features*(50/100))
        elif(threshold == '75 percent features'):
            no_of_selected_features = int(no_of_features*(75/100))
        selected_feature = ranked_feature[:no_of_selected_features]
        new_data = data2[selected_feature].copy()
        new_data[last] = output
        new_data.to_csv('C:/Users/Mr Ravi/Desktop/featureselection/projectapp/newdata.csv')
    return render(request,'SelectedFeatures.html',{'Selectedfeatures':selected_feature,'feature_selection':feature_selection(ranklist,array,column_names),'AggRank':aggRanks,'no_of_features':no_of_features,'no_of_selected_features':no_of_selected_features})

def score(Y_test,Y_pred,label):
    a = precision_recall_fscore_support(Y_test, Y_pred, average=None,labels=label)
    a = pd.DataFrame(a)
    a = a.values.tolist()
    score = ['Precision','Recall','F_Score','Support']
    report = dict(zip(score,a))
    return report
def ROC_Curve(Y_test,Y_pred,output_labels,y_score):
    if (len(output_labels) == 2):
        false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_test, Y_pred)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        plt.figure(figsize=(5,5))
        plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC = %0.2f' % roc_auc)
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],linestyle='--')
        plt.axis('tight')
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.savefig('C:/Users/Mr Ravi/Desktop/featureselection/static/Images/ROC.png',dpi=100)
        plt.clf()   
    else:
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(len(output_labels)):
            fpr[i], tpr[i], _ = roc_curve(Y_test[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        for i in range(len(output_labels)):
            plt.figure()
            plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.legend(loc="lower right")
            plt.savefig('C:/Users/Mr Ravi/Desktop/featureselection/static/Images/ROC.png',dpi=100)
            plt.clf()   


      
    
def logisticRegression(X,Y,kfold,label,X1,Y1):
    model = LogisticRegression()
    results = model_selection.cross_val_score(model, X, Y, cv=kfold)
    full_data_result = model_selection.cross_val_score(model, X1, Y1, cv=kfold)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    model.fit(X_train,Y_train)
    Y_pred = model.predict(X_test)
    matrix = confusion_matrix(Y_test,Y_pred, labels=label)
    plot_confusion_matrix(conf_mat=matrix)
    plt.savefig('C:/Users/Mr Ravi/Desktop/featureselection/static/Images/confusion.png')
    #cla = OneVsRestClassifier(LogisticRegression(random_state=0))
    #y_score = cla.fit(X_train, Y_train).decision_function(X_test)
    #ROC_Curve(Y_test,Y_pred,label,y_score)
    report = score(Y_test,Y_pred,label)
    return results,report,full_data_result

def svm(X,Y,kfold,label,X1,Y1):
    model = SVC()
    results = model_selection.cross_val_score(model, X, Y, cv=kfold)
    full_data_result = model_selection.cross_val_score(model, X1, Y1, cv=kfold)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    model.fit(X_train,Y_train)
    Y_pred = model.predict(X_test)
    matrix = confusion_matrix(Y_test,Y_pred, labels=label)
    plot_confusion_matrix(conf_mat=matrix)
    plt.savefig('C:/Users/Mr Ravi/Desktop/featureselection/static/Images/confusion.png')
    #cla = OneVsRestClassifier(SVC(random_state=0))
    #y_score = cla.fit(X_train, Y_train).decision_function(X_test)
    #ROC_Curve(Y_test,Y_pred,label,y_score)
    report = score(Y_test,Y_pred,label)
    return results,report,full_data_result
def Bernoulli_NB(X,Y,kfold,label,X1,Y1):
    model = BernoulliNB()
    results = model_selection.cross_val_score(model, X, Y, cv=kfold)
    full_data_result = model_selection.cross_val_score(model, X1, Y1, cv=kfold)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    model.fit(X_train,Y_train)
    Y_pred = model.predict(X_test)
    matrix = confusion_matrix(Y_test,Y_pred, labels=label)
    plot_confusion_matrix(conf_mat=matrix)
    plt.savefig('C:/Users/Mr Ravi/Desktop/featureselection/static/Images/confusion.png')
    #cla = OneVsRestClassifier(BernoulliNB(random_state=0))
    #y_score = cla.fit(X_train, Y_train).decision_function(X_test)
    #ROC_Curve(Y_test,Y_pred,label,y_score)
    report = score(Y_test,Y_pred,label)
    return results,report,full_data_result
def Gaussian(X,Y,kfold,label,X1,Y1):
    model = GaussianNB()
    results = model_selection.cross_val_score(model, X, Y, cv=kfold)
    full_data_result = model_selection.cross_val_score(model, X1, Y1, cv=kfold)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    model.fit(X_train,Y_train)
    Y_pred = model.predict(X_test)
    matrix = confusion_matrix(Y_test,Y_pred, labels=label)
    plot_confusion_matrix(conf_mat=matrix)
    plt.savefig('C:/Users/Mr Ravi/Desktop/featureselection/static/Images/confusion.png')
    #cla = OneVsRestClassifier(GaussianNB(random_state=0))
    #y_score = cla.fit(X_train, Y_train).decision_function(X_test)
    #ROC_Curve(Y_test,Y_pred,label,y_score)
    report = score(Y_test,Y_pred,label)
    return results,report,full_data_result
def DecisionTree(X,Y,kfold,label,X1,Y1):
    model = DecisionTreeClassifier()
    results = model_selection.cross_val_score(model, X, Y, cv=kfold)
    full_data_result = model_selection.cross_val_score(model, X1, Y1, cv=kfold)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    model.fit(X_train,Y_train)
    Y_pred = model.predict(X_test)
    matrix = confusion_matrix(Y_test,Y_pred, labels=label)
    plot_confusion_matrix(conf_mat=matrix)
    plt.savefig('C:/Users/Mr Ravi/Desktop/featureselection/static/Images/confusion.png')
    #cla = OneVsRestClassifier(model(random_state=0))
    #y_score = cla.fit(X_train, Y_train).decision_function(X_test)
    #ROC_Curve(Y_test,Y_pred,label,y_score)
    report = score(Y_test,Y_pred,label)
    return results,report,full_data_result
def RandomForest(X,Y,kfold,label,X1,Y1):
    model = RandomForestClassifier() 
    results = model_selection.cross_val_score(model, X, Y, cv=kfold)
    full_data_result = model_selection.cross_val_score(model, X1, Y1, cv=kfold)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    model.fit(X_train,Y_train)
    Y_pred = model.predict(X_test)
    matrix = confusion_matrix(Y_test,Y_pred, labels=label)
    plot_confusion_matrix(conf_mat=matrix)
    plt.savefig('C:/Users/Mr Ravi/Desktop/featureselection/static/Images/confusion.png')
    #cla = OneVsRestClassifier(RandomForestClassifier(random_state=0))
    #y_score = cla.fit(X_train, Y_train).decision_function(X_test)
    #ROC_Curve(Y_test,Y_pred,label,y_score)
    report = score(Y_test,Y_pred,label)
    return results,report,full_data_result
def ExtraTree(X,Y,kfold,label,X1,Y1):
    model = ExtraTreesClassifier()
    results = model_selection.cross_val_score(model, X, Y, cv=kfold)
    full_data_result = model_selection.cross_val_score(model, X1, Y1, cv=kfold)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    model.fit(X_train,Y_train)
    Y_pred = model.predict(X_test)
    matrix = confusion_matrix(Y_test,Y_pred, labels=label)
    plot_confusion_matrix(conf_mat=matrix)
    plt.savefig('C:/Users/Mr Ravi/Desktop/featureselection/static/Images/confusion.png')
    #cla = OneVsRestClassifier(ExtraTreesClassifier(random_state=0))
    #y_score = cla.fit(X_train, Y_train).decision_function(X_test)
    #ROC_Curve(Y_test,Y_pred,label,y_score)
    report = score(Y_test,Y_pred,label)
    return results,report,full_data_result
def AdaBoost(X,Y,kfold,label,X1,Y1):
    model = AdaBoostClassifier()
    results = model_selection.cross_val_score(model, X, Y, cv=kfold)
    full_data_result = model_selection.cross_val_score(model, X1, Y1, cv=kfold)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    model.fit(X_train,Y_train)
    Y_pred = model.predict(X_test)
    matrix = confusion_matrix(Y_test,Y_pred, labels=label)
    plot_confusion_matrix(conf_mat=matrix)
    plt.savefig('C:/Users/Mr Ravi/Desktop/featureselection/static/Images/confusion.png')
    #cla = OneVsRestClassifier(AdaBoostClassifier(random_state=0))
    #y_score = cla.fit(X_train, Y_train).decision_function(X_test)
    #ROC_Curve(Y_test,Y_pred,label,y_score)
    report = score(Y_test,Y_pred,label)
    return results,report,full_data_result
def GradientBoosting(X,Y,kfold,label,X1,Y1):
    model = GradientBoostingClassifier()
    results = model_selection.cross_val_score(model, X, Y, cv=kfold)
    full_data_result = model_selection.cross_val_score(model, X1, Y1, cv=kfold)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    model.fit(X_train,Y_train)
    Y_pred = model.predict(X_test)
    matrix = confusion_matrix(Y_test,Y_pred, labels=label)
    plot_confusion_matrix(conf_mat=matrix)
    plt.savefig('C:/Users/Mr Ravi/Desktop/featureselection/static/Images/confusion.png')
    #cla = OneVsRestClassifier(GradientBoostingClassifier(random_state=0))
    #y_score = cla.fit(X_train, Y_train).decision_function(X_test)
    #ROC_Curve(Y_test,Y_pred,label,y_score)
    report = score(Y_test,Y_pred,label)
    return results,report,full_data_result
def Bagging(X,Y,kfold,label,X1,Y1):
    model = BaggingClassifier()
    results = model_selection.cross_val_score(model, X, Y, cv=kfold)
    full_data_result = model_selection.cross_val_score(model, X1, Y1, cv=kfold)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    model.fit(X_train,Y_train)
    Y_pred = model.predict(X_test)
    matrix = confusion_matrix(Y_test,Y_pred, labels=label)
    plot_confusion_matrix(conf_mat=matrix)
    plt.savefig('C:/Users/Mr Ravi/Desktop/featureselection/static/Images/confusion.png')
    #cla = OneVsRestClassifier(BaggingClassifier(random_state=0))
    #y_score = cla.fit(X_train, Y_train).decision_function(X_test)
    #ROC_Curve(Y_test,Y_pred,label,y_score)
    report = score(Y_test,Y_pred,label)
    return results,report,full_data_result


def classifier(request):
    if request.method == 'POST':
        classify = request.POST.get('Classifier')
        data = pd.read_csv('C:/Users/Mr Ravi/Desktop/featureselection/projectapp/newdata.csv')
        data = data.drop(data.columns[0],axis=1)
        full_data = pd.read_csv('C:/Users/Mr Ravi/Desktop/featureselection/projectapp/file2.csv')
        full_data = full_data.drop(full_data.columns[0],axis=1)
        last = data.columns[-1]
        output_labels = data[last].unique().tolist()
        X,Y = split_the_data_into_X_and_Y(data)
        X1,Y1 = split_the_data_into_X_and_Y(full_data)
        kfold = model_selection.KFold(n_splits=10, random_state= None)
        if (classify == 'Logistic Regression'):
            results, report,full_data_result = logisticRegression(X,Y,kfold,output_labels,X1,Y1)
        if (classify == 'SVM'):
            results, report, full_data_result = svm(X,Y,kfold,output_labels,X1,Y1)
        if (classify == 'BernoulliNB'):
            results, report, full_data_result = Bernoulli_NB(X,Y,kfold,output_labels,X1,Y1)
        if (classify == 'GaussianNB'):
            results, report, full_data_result = Gaussian(X,Y,kfold,output_labels,X1,Y1)
        if (classify == 'DecisionTree'):
            results, report, full_data_result = DecisionTree(X,Y,kfold,output_labels,X1,Y1)
        if (classify == 'RandomForest'):
            results, report, full_data_result = RandomForest(X,Y,kfold,output_labels,X1,Y1)
        if (classify == 'ExtraTreeClassifier'):
            results, report, full_data_result = ExtraTree(X,Y,kfold,output_labels,X1,Y1)
        if (classify == 'AdaBoost'):
            results, report, full_data_result = AdaBoost(X,Y,kfold,output_labels,X1,Y1)
        if (classify == 'GradientBoosting'):
            results, report,full_data_result = GradientBoosting(X,Y,kfold,output_labels,X1,Y1)
        if (classify == 'BaggingClassifier'):
            results, report, full_data_result = Bagging(X,Y,kfold,output_labels,X1,Y1)
    return render(request,'Classifier.html',{'Result':results.mean(),'FullDataResult':full_data_result.mean(),'Classifier':classify,'Score':report}) 


def scatter(request):
    if request.method == 'POST':
        df = pd.read_csv('C:/Users/Mr Ravi/Desktop/featureselection/projectapp/file1.csv')
        df = df.drop(df.columns[0],axis=1)
        features = df.columns[0:-1].tolist()
    return render(request,'Scatter.html',{'Features':features})
def box(request):
    if request.method == 'POST':
        df = pd.read_csv('C:/Users/Mr Ravi/Desktop/featureselection/projectapp/file1.csv')
        df = df.drop(df.columns[0],axis=1)
        features = df.columns[0:-1].tolist()
    return render(request,'Box.html',{'Features':features})
def violin(request):
    if request.method == 'POST':
        df = pd.read_csv('C:/Users/Mr Ravi/Desktop/featureselection/projectapp/file1.csv')
        df = df.drop(df.columns[0],axis=1)
        features = df.columns[0:-1].tolist()
    return render(request,'Violin.html',{'Features':features})
def dist(request):
    if request.method == 'POST':
        df = pd.read_csv('C:/Users/Mr Ravi/Desktop/featureselection/projectapp/file1.csv')
        df = df.drop(df.columns[0],axis=1)
        features = df.columns[0:-1].tolist()
    return render(request,'Histogram.html',{'Features':features})
def swarm(request):
    if request.method == 'POST':
        df = pd.read_csv('C:/Users/Mr Ravi/Desktop/featureselection/projectapp/file1.csv')
        df = df.drop(df.columns[0],axis=1)
        features = df.columns[0:-1].tolist()
    return render(request,'Swarm.html',{'Features':features})
    
def scatterplot(request):
    if request.method == 'POST':
        x_dim = request.POST.get('column1')
        y_dim = request.POST.get('column2')
        df = pd.read_csv('C:/Users/Mr Ravi/Desktop/featureselection/projectapp/file1.csv')
        df = df.drop(df.columns[0],axis=1)
        category = df.columns[-1]
        features = df.columns[0:-1].tolist()
        scatter = sns.scatterplot(x = x_dim, y = y_dim, hue = category, data = df)
        figure = scatter.get_figure()    
        figure.savefig('C:/Users/Mr Ravi/Desktop/featureselection/static/Images/plot.png', dpi=100)
        plt.clf()
    return render(request,'Scatterplot.html',{'Features':features})
def boxplot(request):
    if request.method == "POST":
        x_dim = request.POST['column1']
        df = pd.read_csv('C:/Users/Mr Ravi/Desktop/featureselection/projectapp/file1.csv')
        df = df.drop(df.columns[0],axis=1)
        category = df.columns[-1]
        features = df.columns[0:-1].tolist()
        box = sns.boxplot(x = x_dim , hue = category, data = df)
        figure = box.get_figure()    
        figure.savefig('C:/Users/Mr Ravi/Desktop/featureselection/static/Images/plot.png', dpi=100)
        plt.clf()
    return render(request,'Boxplot.html',{'Features':features})   

def violinplot(request):
    if request.method == "POST":
        x_dim = request.POST.get('column1')
        df = pd.read_csv('C:/Users/Mr Ravi/Desktop/featureselection/projectapp/file1.csv')
        df = df.drop(df.columns[0],axis=1)
        category = df.columns[-1]
        features = df.columns[0:-1].tolist()
        count = sns.violinplot(x = x_dim, data = df)
        figure = count.get_figure()    
        figure.savefig('C:/Users/Mr Ravi/Desktop/featureselection/static/Images/plot.png', dpi=100)
        plt.clf()
    return render(request,'Violinplot.html',{'Features':features})    
def distplot(request):
    if request.method == "POST":
        x_dim = request.POST.get('column1')
        df = pd.read_csv('C:/Users/Mr Ravi/Desktop/featureselection/projectapp/file1.csv')
        df = df.drop(df.columns[0],axis=1)
        features = df.columns[0:-1].tolist()
        dist = sns.distplot(df[x_dim])
        figure = dist.get_figure()    
        figure.savefig('C:/Users/Mr Ravi/Desktop/featureselection/static/Images/plot.png', dpi=100)
        plt.clf()
    return render(request,'Histogramplot.html',{'Features':features})   
def swarmplot(request):
    if request.method == 'POST':
        x_dim = request.POST.get('column1')
        df = pd.read_csv('C:/Users/Mr Ravi/Desktop/featureselection/projectapp/file1.csv')
        df = df.drop(df.columns[0],axis=1)
        category = df.columns[-1]
        features = df.columns[0:-1].tolist()
        swarm = sns.swarmplot(x = x_dim, data = df)
        figure = swarm.get_figure()    
        figure.savefig('C:/Users/Mr Ravi/Desktop/featureselection/static/Images/plot.png', dpi=100)
        plt.clf()
    return render(request,'Swarmplot.html',{'Features':features})

def aboutfeatures(request):
    return render(request,'AboutFeatures.html')             
def aboutclassifier(request):
    return render(request,'AboutClassifiers.html')
def homereturn(request):
    return render(request,'base.html')

        