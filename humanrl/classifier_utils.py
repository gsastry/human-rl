from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.metrics import precision_score, recall_score, roc_auc_score
import numpy as np
import matplotlib.pyplot as plt


def zero_one_score(y, pred_y):
    total_errors =  np.sum( np.abs(y - pred_y) )
    return total_errors / len(y)


def run_predict_logistic_regression(X_train,Y_train,X_test,Y_test):
    clf = LogisticRegression()
    clf = clf.fit(X_train, Y_train)
    pred = clf.predict(X_test)
    print('Logistic 0-1 error. \n Training: ', zero_one_score(Y_train, clf.predict(X_train)),
          '\n Test:', zero_one_score(Y_test, pred))
    
    return clf


def show_im(im):
    plt.imshow(im, cmap='gray')
    plt.show()

    
def run_forests():    
    print('random forest: \n')   
    params = []
    scores = []
    
    for _ in range(5):
        max_features = np.random.randint(400,800)
        max_depth = np.random.choice([None, None, None, None, 30, 40, 60])
        forest = RandomForestClassifier(n_estimators=50,
                                        max_features=max_features,
                                        max_depth=max_depth)                                   
        forest_fit = forest.fit(X_train, Y_train)
        pred = forest_fit.predict(X_test)
        print('\n params:', dict(max_features=max_features, max_depth=max_depth))
        print('forest train: ',zero_one_score(Y_train, forest_fit.predict(X_train)), ' test: ',
                  zero_one_score(Y_test, pred))

        params.append( (max_features, max_depth) )
        scores.append( zero_one_score(Y_test, pred))

    print('best:', params[np.argmin(scores)])



class SKClassifier(object):
    def __init__(self, clf):
        self.clf = clf
        self.threshold = 0.0
        #tf.add_to_collection("threshold", self.threshold)
        
    def fit(self,X_train,y_train):
        self.clf = self.clf.fit(X_train,y_train)
        return self.clf
    
    def predict_proba(self,X):
        return self.clf.predict_proba(X)
    
    def predict_proba_with_loss(self, X, y):
        y_pred = self.predict_proba(X)
        loss = log_loss(y,y_pred)
        return y_pred, loss
        
    # smallest prob given to an actual catastrophe
    def threshold_from_data(self, X, y):
        y_bool = y == 1.   ## true if x is a catast
        y_pred = self.predict_proba(X) 
        if np.count_nonzero(y) == 0:
            return np.max(y_pred)
        return np.min(y_pred[y_bool][:,1])   # TODO CHANGED FROM WILL CODE

    def metrics(self, X, y):
        metrics = {}
        y_pred_pair, loss = self.predict_proba_with_loss(X, y)
        y_pred = y_pred_pair[:,1]  ## From softmax pair to prob of catastrophe
        
        metrics['loss'] = loss
        threshold = self.threshold_from_data(X, y)
        metrics['threshold'] = threshold
        metrics['np.std(y_pred)'] = np.std(y_pred)
        denom = np.count_nonzero(y == False)
        num = np.count_nonzero(np.logical_and(y == False, y_pred >= threshold))
        metrics['fpr'] = float(num) / float(denom)
        if any(y) and not all(y):
            metrics['auc'] = roc_auc_score(y, y_pred)
            y_pred_bool = y_pred >= threshold
            if (any(y_pred_bool) and not all(y_pred_bool)):
                metrics['precision'] = precision_score(np.array(y, dtype=np.float32), y_pred_bool)
                metrics['recall'] = recall_score(y, y_pred_bool)

        return metrics
    
    def get_gaps(self,X,y):
        return np.abs(y - self.predict_proba(X)[:,1])
    
    def show_mistakes(self,X,y,k):
        gaps = self.get_gaps(X,y)
        index_gap = np.array([[i,gap] for i,gap in enumerate(gaps)])
        sorted_index_gap = index_gap[index_gap[:,1].argsort()]
        tail_index_gap = sorted_index_gap[-k:,:]
               
        xs = []
        probs = []
        for row in range(k):
            index,gap = tail_index_gap[row,:]
            x = X[int(index),:]
            xs.append(x)
            show_im(x.reshape(40,40))
            print('Contains ball:', y[int(index)] == 1., '. Prob gap: ', gap, ' \n ')    
        return xs
   


def run_predict_random_forest(X_train,Y_train,X_test,Y_test, n_estimators=30, max_features=500, show_mistakes=False):
    forest = RandomForestClassifier(n_estimators=10, max_features=20, max_depth=10) 
    clf = SKClassifier(forest)
    forest_fit = clf.fit(X_train, Y_train)
    pred = forest_fit.predict(X_test)
    print('\n Random forest 0-1 error.  \n Train: ',zero_one_score(Y_train, forest_fit.predict(X_train)), '\n Test: ', 
      zero_one_score(Y_test, pred))
        
    
    met = clf.metrics(X_test,Y_test)
    if show_mistakes:
        mis = clf.show_mistakes(X_test,Y_test,10)
    print('Metrics:', met)
    return clf

