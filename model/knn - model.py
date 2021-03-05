#Import necesaary packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def calculateDistance(dataSet, query_point):
    """
    [parameters]:
    dataset -> It takes entire training set or 2D Dataframe
    query_point -> It takes 1D list/ array
    
    [Objective]: -> Calculates the Euclidean distance between each row of training data w.r.t the query_point and finally returns a list of distances 
    """
    distance = []
    
    # Start iterations over training dataset
    for each_row in dataSet.values:
        
        #Calculate Euclidean distance 
        dis = np.add.reduce(np.square(each_row - query_point))
        distance.append(np.sqrt(dis))
    
    #Returns the entire list of distances
    return distance

def calculateManhattenDistances(dataSet, query_point):
    """
    [parameters]:
    dataSet -> It takes entire training set or 2D Dataframe
    query_point -> It takes 1D list/ array
    
    [Objective]: -> Calculates the Manhattan distance between each row of training data w.r.t the query_point and finally returns a list of distances 
    """
    distance = []
    
    # Start iterations over training dataset
    for each_row in dataSet.values:
        
        #Calculate Manhattan distance 
        dis = np.add.reduce(np.abs(each_row - query_point))
        distance.append(dis)
    
    #Returns the entire list of distances
    return distance

def minMaxNormalization(trainset, testset):
    """
    [parameters]:
    trainset -> It takes entire training set or nD Dataframe
    testset -> It takes entire testing dataset or a datapoint.
    
    [Objective]: -> It scales each feature of data to the minmax normalization ~[0,1]. Note it uses Min() and Max() of the training data and normalizes the testing data
                    to prevent Data Leaking issue.
    """
    #Initiate variables
    trainset_scaled = np.array([])
    testset_scaled = np.array([])
    
    #Convert datframe to numpy array
    trainset = trainset.values
    testset = testset.values
    
    #Applying Minmax normalization to training set
    trainset_scaled = (trainset - trainset.min(axis = 0)) / (trainset.max(axis = 0) - trainset.min(axis = 0))
    
    #Transoforming testing data to Minmax normalized data, using min() and max() of training data
    testset_scaled = (testset - trainset.min(axis = 0)) / (trainset.max(axis = 0) - trainset.min(axis = 0))
    return trainset_scaled, testset_scaled

def zNormalization(trainset, testset):
    """
    [parameters]:
    trainset -> It takes entire training set or nD Dataframe
    testset -> It takes entire testing dataset or a datapoint.
    
    [Objective]: -> It converts each feature of data to the standard normal distribution ~N(0,1). Note it uses Mean and Standard Deviation of the training data and normalizes the testing data
                    to prevent Data Leaking issue.
    """
    #Initiate variables
    trainset_scaled = np.array([])
    testset_scaled = np.array([])
    
    #Convert datframe to numpy array
    trainset = trainset.values
    testset = testset.values
    
    #Applying standard Normal Distribution to training set
    trainset_scaled = (trainset - trainset.mean(axis = 0)) / np.std(trainset, axis = 0 , dtype = np.float64)
    
    #Transoforming testing data to standard Normal Data, using mean and std of training data
    testset_scaled = (testset - trainset.mean(axis = 0)) / np.std(trainset, axis = 0 , dtype = np.float64)
    return trainset_scaled, testset_scaled

def handleScaling(X_train, X_test, scaling):
    """
    [parameters]:
    X_train -> It takes entire training set or 2D Dataframe
    X_test -> It takes entire testing dataset or a datapoint.
    scaling -> As per the input it uses scaling to the data
    
    [Objective]: -> It handles which type of scaling is required by user
    """
    cols = []
    cols = getFeatureSet(X_train).columns
    
    #Checks the condition
    if scaling == 'minmax':
        X_train, X_test = minMaxNormalization(X_train, X_test)
        X_train = pd.DataFrame(X_train, columns = cols) 
        X_test = pd.DataFrame(X_test, columns = cols) 
    elif scaling == 'znormal':
        X_train, X_test = zNormalization(X_train, X_test)
        X_train = pd.DataFrame(X_train, columns = cols) 
        X_test = pd.DataFrame(X_test, columns = cols)
    return X_train, X_test

def confusion_matrix(y_test, y_pred):
    """
    [parameters]:
    y_pred -> It takes the predicted classes by the model.
    y_test -> It takes the actual classes from the testing data.
    
    [Objective]: -> Creates confusion matrix to evaluate performance by the model.
    """
    # Changing data type, for consistency.
    y_test, y_pred = list(y_test), list(y_pred)
    cm_df = pd.DataFrame({
        'y_pred': y_pred,
        'y_true': y_test
    })
    confusion_matrix = pd.crosstab(cm_df['y_true'], cm_df['y_pred'], rownames = ['Actual'], colnames = ['Predicted'])
    return confusion_matrix
        
def getFeatureSet(data_frame):
    """
    [parameters]:
    data_frame -> It takes a dataframe
    
    [Objective]: -> It returns the column names of dataset, to prepare training data.
    """
    return data_frame[["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides","free sulfur dioxide",
                       "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"]]

def getLabelSet(data_frame):
    """
    [parameters]:
    data_frame -> It takes a dataframe
    
    [Objective]: -> It returns the column names of dataset, to prepare testing data.
    """
    return data_frame["Quality"]

def knn(k, isWeightedknn, dist_metric = None, scaling = None):
    """
    [parameters]:
    k -> Number of neighbours to build the model.
    isWeightedknn -> If Weightedknn is required. Boolean. Default = False
    dist_metric -> This distance metric is used to calculate the distance between a query point and its neighbours.
                    (Values) - By default, 'euclidean', other 'manhattan'.
    scaling -> It uses the given user input to scale the entire dataset.
               (values) - By default, no Scaling. Options, 'minmax' or 'znormal'. 
    
    [Objective]: -> Takes the training and testing dataset, apply Scaling if required. For each datapoint of testing data, it calculates the distance of the closest 'k' points and applies
                    majority voting amongst the classes of corresponding k points. Thus, each data point of testing data is classified as one of the classes. Further compares the classified
                    data points with the corresponding true class from testing data and accuracy metric, confusion matrix is generated.
    """
    #Initiate variables and constants
    PATH = 'dataset-p1/dataset-p1/' #Change as per location
    correctClassifications = 0
    correctClassifications_weighted = 0
    accuracy = 0
    y_pred = [] #Holds the predicted classes
    y_pred_weighted = []
    dis_array = []
    dist_metric = 'euclidean' if dist_metric is None else dist_metric
    
    #Load the data and make training and testing parts
    df_train = pd.read_csv(PATH + "wine-data-project-train.csv")
    df_test  = pd.read_csv(PATH + "wine-data-project-test.csv")
    X_train = getFeatureSet(df_train)
    X_test = getFeatureSet(df_test)
    y_train = getLabelSet(df_train)
    y_test = getLabelSet(df_test)
    
    #Check if feature scaling is required
    if scaling is not None:
        X_train, X_test = handleScaling(X_train, X_test, scaling)
    
    # Iter through each data point of testing set
    for idx , test_sample in X_test.iterrows():
        #Store the distances of query point from the training data
        dis_array = calculateManhattenDistances(X_train, test_sample) if dist_metric == 'manhattan' \
                    else calculateDistance(X_train, test_sample)

        #Store the index values of top k number(s) of least distances
        dis_idx = np.argsort(dis_array)[:k]
        
        #Fetch corresponding labels
        class_labels = y_train[dis_idx]
        class_labels = list(class_labels)
        
        #Store corresponding True Labels from testing dataset.
        actual_label = y_test[idx]

        #Apply majority voting amongs k classes to determine predicted class
        class_out = max(set(class_labels), key = class_labels.count)
        
        # If weightedknn is required, below function returns the predicted class
        weightknn_classount = Weightedknn(dis_array, k, y_train) if isWeightedknn else None
        
        y_pred.append(class_out)
        y_pred_weighted.append(weightknn_classount)
        
        #If predicted and actual classes match then adds 1 to the counter of correctClassifications
        correctClassifications += np.sum(actual_label == class_out)
        
        if weightknn_classount is not None:
            #If predicted and actual classes match then adds 1 to the counter of correctClassifications_weighted
            correctClassifications_weighted += np.sum(actual_label == weightknn_classount)
            
    #Store the confusion matrix between testing and predicted labels for model evaluation
    cm = confusion_matrix(y_test, y_pred)
    
    #Store the confusion matrix between testing and predicted labels for model evaluation
    weighted_cm = confusion_matrix(y_test, y_pred_weighted)
    
    #Store the final accuracy of the KNN model for a value of K
    accuracy = np.round((correctClassifications*100)/y_test.shape[0], 2)
    
    #Store the final accuracy of the weightedKNN model for a value of K
    accuracyWeightedModel = np.round((correctClassifications_weighted*100)/y_test.shape[0], 2)
    
    if isWeightedknn:
        return accuracy, cm, accuracyWeightedModel, weighted_cm
    else:
        return accuracy, cm


def Weightedknn(dis_array, k, y_train):
    """
    [parameters]:
    dis_array -> This distance array should contain the calculated distances between a query point and the entire training data.
    k -> Number of neighbours to build the model.
    y_train -> This array should contain the list of classes from the training data.
    
    [Objective]: -> For each datapoint of testing data, it calculates the distance of the closest 'k' points. Based on the classes it sums up the inverse distance squared
                    value and forms weight for each group of classes. The highest weight's corresponding class is sent as the predicted class for the query point.
                    
    """
    np.seterr(divide = 'ignore')
    #Store the index values of top k number(s) of least distances
    w1, w2, clss_1, clss_2 = 0, 0, 0, 0
    dis_idx = np.argsort(dis_array)[:k] 
    
    #Store the values of top k number(s) of least distances
    dis_sorted = np.sort(dis_array)[:k]
        
    #Fetch corresponding labels
    class_labels = y_train[dis_idx]
    class_labels = list(class_labels)
        
    #Check if the top K values have the same class labels 
    isPure_class = True if np.unique(class_labels).shape[0] == 1 else False
        
    #If all the labels are same then skip the weight calculation process
    if not isPure_class:
            
        #Calculate the weights of k neighbours with a factor of 1/d**2, where d is the distance between the query point and one of k points from trainig set
        for d, clss in zip(dis_sorted, class_labels):
              
            #Check for class -1
            if clss == np.unique(class_labels)[0]:
                # In case d = 0, add 1 in weights
                if np.isinf(1/d**2):
                    w1 += 1
                    clss_1 = clss
                else:
                    w1 += 1/d**2
                    clss_1 = clss
            else:
                if np.isinf(1/d**2):
                    w2 += 1
                    clss_2 = clss
                else:
                    w2 += 1/d**2
                    clss_2 = clss
                    
        #Based on weights assign the class
        class_out = clss_1 if w1 > w2 else clss_2
    else:
        class_out = class_labels[0]
        
    return class_out

def main(dist_metric = None, isWeightedknn = False, scaling = None):
    """
    [parameters]:
    dist_metric -> This distance metric is used to calculate the distance between a query point and its neighbours.
                    (Values) - By default, 'euclidean', other 'manhattan'.
    isWeightedknn -> If turned True then, weightedknn is measured alongside knn
                    (Values) - By default, False, other True.
    scaling -> It uses the given user input to scale the entire dataset.
               (values) - By default, no Scaling. Options, 'minmax' or 'znormal'. 
    
    [Objective]: -> This is the main function to run the knn and weightedknn for the odd values of k from 2 to 40. It plots the appropriate plots and confusion matrix to evaluate the classifier
    """
    
    #Initiate the variables to load scores
    allResults = {}
    allWeightedResults = {}
    
    #Run odd ks' for the series of values
    for k in range(2, 40, 2):
        if isWeightedknn:
            #Store the accuracy for each iterations, keeps a dummy placeholder for the confusion matrix.
            accuracy, _, accuracyWeighted, _ = knn(k+1, isWeightedknn, dist_metric, scaling)
        else:
            #Store the accuracy for each iterations, keeps a dummy placeholder for the confusion matrix.
            accuracy, _ = knn(k + 1, isWeightedknn, dist_metric, scaling) 
        allResults[k+1] = accuracy
        
        #Store the value k and corresponding accuracy within a dictionary, checks if accuracyWeighted is returned i.e., isWeightedknn = True
        allWeightedResults[k+1] = accuracyWeighted if isinstance(accuracyWeighted, float) else {}
    
    
    #Confusion matrix for best accuracy of KNN and WeightedKnn
    if isWeightedknn:
        _, knn_cm, _, _ = knn(max(allResults, key = allResults.get), isWeightedknn, dist_metric, scaling)
        _, _, _, weightedKnn_cm = knn(max(allWeightedResults, key = allWeightedResults.get), isWeightedknn, dist_metric, scaling)
    else:
        _, knn_cm, _, _ = knn(max(allResults, key = allResults.get), isWeightedknn, dist_metric, scaling)
    
    #plots weightedknn and knn
    sns.set_style("darkgrid")
    plt.figure(figsize = (15, 8))
    plt.plot( list(allResults.keys()), allResults.values(), color = 'red', label = "Knn Accuracy")
    plt.plot(list(allWeightedResults.keys()), allWeightedResults.values(),  label = "Weighted Knn Accuracy")
    plt.xlabel('K')
    plt.ylabel('Accuracy')
    plt.title('Accuracy performance on multiple k')
    plt.legend()
    
    #Plot confusion matrices
    plt.figure(figsize = (15 , 6))
    plt.subplot(121)
    plt.title(f'Knn confusion matrix at K = {max(allResults, key = allResults.get)}')
    sns.heatmap(data = knn_cm, annot = True)
    plt.subplot(122)
    plt.title(f'Weighted Knn confusion matrix at K = {max(allWeightedResults, key = allWeightedResults.get)}')
    sns.heatmap(data = weightedKnn_cm, annot = True)
    plt.show()
    
    #Conditionally prints the best k value and it's corresponding accuacy as per the other conditions
    if scaling is not None:
        print(f'Best knn evaluation score with {scaling} scaling, k = {max(allResults, key = allResults.get)} and Accuracy = {max(allResults.values()):.2f}%')
        if isWeightedknn:
            print(f'Best WeightedKnn evaluation score with {scaling} scaling, k = {max(allWeightedResults, key = allWeightedResults.get)} and Accuracy = {max(allWeightedResults.values()):.2f}%')
    else:
        print(f'Best knn evaluation score, k = {max(allResults, key = allResults.get)} and Accuracy = {max(allResults.values()):.2f}%')
        print(f'Best WeightedKnn evaluation score, k = {max(allWeightedResults, key = allWeightedResults.get)} and Accuracy = {max(allWeightedResults.values()):.2f}%')



if __name__ == "__main__":
    """
    main function arguments: ->
    
    If scaling is required, accepts scaling = 'minmax' or scaling = 'znormal', default 'None'
    If distance metric requires to change, accepts dist_metric = 'manhattan', default dist_metric = 'euclidean'
    If only knn is required, change isWeightedknn = False
    """
    main(isWeightedknn = True)