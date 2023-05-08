import argparse
import pickle

config = {
    "trainingdata": "/nfs/home/arafatj/master_project/trainingdata/326samples_trainingdata.pkl",
    "k": 5
}

def random_forest():
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import cross_val_score

    with open(config['trainingdata'], 'rb') as pkl:
        data = pickle.load(pkl)
    
    # split feature and ground truth arrays
    features = data[:, :-1]
    groundtruth = data[:, -1]

    # split into train and test sets
    features_train, features_test, groundtruth_train, groundtruth_test = train_test_split(features, groundtruth, test_size=0.2)

    # train random forest model
    rf_model = RandomForestClassifier(random_state=23)
    #rf_model.fit(features_train, groundtruth_train)

    # k-fold cross-validation
    scores = cross_val_score(rf_model, features, groundtruth, cv=config['k'])
    print(f'{config["k"]}-fold cross-validation: {"{:.2f}".format(scores.mean())} accuracy with a standard deviation of {"{:.2f}".format(scores.std())}')

    """
    # predict test set and measure accuracy
    predictions = rf_model.predict(features_test)
    accuracy = accuracy_score(groundtruth_test, predictions)
    print(f'Accuracy: {"{:.2f}".format(accuracy * 100)}%')
    """



def xgboost():
    from sklearn.model_selection import train_test_split
    from xgboost import XGBClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import cross_val_score

    with open(config['trainingdata'], 'rb') as pkl:
        data = pickle.load(pkl)
    
    # split feature and ground truth arrays
    features = data[:, :-1]
    groundtruth = data[:, -1]

    # split into train and test sets
    features_train, features_test, groundtruth_train, groundtruth_test = train_test_split(features, groundtruth, test_size=0.2, random_state=23)

    # train xgb model
    xgb_model = XGBClassifier(random_state=23, tree_method="gpu_hist")
    #xgb_model.fit(features_train, groundtruth_train)

    # k-fold cross-validation
    scores = cross_val_score(xgb_model, features, groundtruth, cv=config['k'])
    print(f'{config["k"]}-fold cross-validation: {"{:.2f}".format(scores.mean())} accuracy with a standard deviation of {"{:.2f}".format(scores.std())}')

    """
    # predict test set and measure accuracy
    predictions = xgb_model.predict(features_test)
    accuracy = accuracy_score(groundtruth_test, predictions)
    print(f'Accuracy: {"{:.2f}".format(accuracy * 100)}%')
    """




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Utility methods')
    parser.add_argument('--rf', action='store_true', help="Random Forest")
    parser.add_argument('--xgb', action='store_true', help="XGBoost")
    args = parser.parse_args()

    if args.rf:
        random_forest()
    
    if args.xgb:
        xgboost()