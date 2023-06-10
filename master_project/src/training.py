import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import yaml

with open('/nfs/home/arafatj/master_project/src/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# speaker role mapping hierarchy level (passed in args)
speaker_hierarchy_level = None

def training(args):
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import confusion_matrix

    if args.seg:
        with open(config['speaker_trainingdata_tagesschau_seg'].format(speaker_hierarchy_level=speaker_hierarchy_level), 'rb') as pkl:
            tagesschau_data = pickle.load(pkl)
        with open(config['speaker_trainingdata_bildtv_seg'].format(speaker_hierarchy_level=speaker_hierarchy_level), 'rb') as pkl:
            bildtv_data = pickle.load(pkl)
        with open(config['speaker_trainingdata_compacttv_seg'].format(speaker_hierarchy_level=speaker_hierarchy_level), 'rb') as pkl:
            compacttv_data = pickle.load(pkl)
    elif args.sw:
        with open(config['speaker_trainingdata_tagesschau_window'].format(speaker_hierarchy_level=speaker_hierarchy_level), 'rb') as pkl:
            tagesschau_data = pickle.load(pkl)
        with open(config['speaker_trainingdata_bildtv_window'].format(speaker_hierarchy_level=speaker_hierarchy_level), 'rb') as pkl:
            bildtv_data = pickle.load(pkl)
        with open(config['speaker_trainingdata_compacttv_window'].format(speaker_hierarchy_level=speaker_hierarchy_level), 'rb') as pkl:
            compacttv_data = pickle.load(pkl)
    else:
        print('ERROR: Please select a training data aggregation method (--seg for segment-wise feature aggregation or --sw for window-based feature aggregation)')
        return
    
    np.random.seed(23)
    # if no split of news sources is given the data will be composed of all three news sources
    if not args.trainset:
        data = np.concatenate((tagesschau_data, bildtv_data, compacttv_data), axis=0)
        # shuffle data since news data is in order of news sources (only needed for k-fold-cross-validation later on)
        np.random.shuffle(data)
        # split feature and ground truth arrays
        features = data[:, :-1]
        groundtruth = data[:, -1]
        # split into train and test sets
        train_features, test_features, train_groundtruth, test_groundtruth = train_test_split(features, groundtruth, test_size=0.2, random_state=22)
    elif 'tag' in args.trainset and 'bild' in args.trainset:
        data = np.concatenate((tagesschau_data, bildtv_data))
        train_features = data[:, :-1]
        train_groundtruth = data[:, -1]
        test_features = compacttv_data[:, :-1]
        test_groundtruth = compacttv_data[:, -1]
    elif 'tag' in args.trainset and 'com' in args.trainset:
        data = np.concatenate((tagesschau_data, compacttv_data))
        train_features = data[:, :-1]
        train_groundtruth = data[:, -1]
        test_features = bildtv_data[:, :-1]
        test_groundtruth = bildtv_data[:, -1]
    elif 'bild' in args.trainset and 'com' in args.trainset:
        data = np.concatenate((bildtv_data, compacttv_data))
        train_features = data[:, :-1]
        train_groundtruth = data[:, -1]
        test_features = tagesschau_data[:, :-1]
        test_groundtruth = tagesschau_data[:, -1]
    
    # store utilized model as string for file naming later on
    model_s = ''
    if args.rf:
        model = RandomForestClassifier(random_state=23)
        model_s = 'rf'
    elif args.xgb:
        model = XGBClassifier(random_state=23, tree_method="gpu_hist")
        model_s = 'xgb'
    else:
        print('Please choose between Random Forest or XGBoost by using either --rf or --xgb')
        return

    # k-fold cross-validation
    if args.kfold:
        scores = cross_val_score(model, features, groundtruth, cv=config['k'])
        print(f'{config["k"]}-Fold Cross-Validation: {"{:.2f}".format(scores.mean())} accuracy with a standard deviation of {"{:.2f}".format(scores.std())}')

    # fit model for later experiments
    model.fit(train_features, train_groundtruth)

    # single prediction on a train and test split (only print when requested)
    if args.traintest or args.confusion or args.importances:
        # predict test set and measure accuracy
        predictions = model.predict(test_features)
        accuracy = accuracy_score(test_groundtruth, predictions)
        print(f'Single Split Accuracy: {"{:.2f}".format(accuracy)}')

    # plot the confusion matrix with labels (matrix with true label being i-th class and predicted label being j-th class)
    if args.confusion:
        cm = confusion_matrix(test_groundtruth, predictions)
        print(cm)
        df_cm = pd.DataFrame(cm)

        row_sums = df_cm.sum(axis=1)
        df_percentage = df_cm.div(row_sums, axis=0).round(2)

        sns.set(font_scale=1.4)
        sns.heatmap(df_percentage, xticklabels=config[f'labels_{speaker_hierarchy_level}'], yticklabels=config[f'labels_{speaker_hierarchy_level}'], annot=True, annot_kws={'size':16}, cmap='binary')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plot_path = f'/nfs/home/arafatj/master_project/graphics/training/confusion_matrix_{model_s}.png'
        plt.savefig(plot_path)
        print(f'Confusion Matrix: {plot_path}')

    # plot feature importances (what features are most taken into consideration by the classifier when predicting)
    if args.importances:
        importances = model.feature_importances_
        indices = np.argsort(importances)
        plt.figure(figsize=(15,8))
        plt.title("Feature Importances")
        plt.barh(range(len(indices)), importances[indices], align='center')
        plt.yticks(range(len(indices)), [config['feature_names'][i] for i in indices])
        plt.xlabel('Relative Importance')
        plot_path = f'/nfs/home/arafatj/master_project/graphics/training/feature_importances_{model_s}.png'
        plt.savefig(plot_path)
        print(f'Feature Importances: {plot_path}')

    # correlation matrix between features and labels
    if args.correlation:
        features_with_labels = pd.DataFrame(np.column_stack((features, groundtruth)), columns=config['feature_names'] + ['predicted_label'])
        corr_matrix = features_with_labels.corr()
        plt.figure(figsize=(15,13))
        sns.heatmap(corr_matrix, cmap='coolwarm')

        plot_path = f'/nfs/home/arafatj/master_project/graphics/training/correlation_matrix.png'
        plt.savefig(plot_path)
        print(f'Correlation Matrix: {plot_path}')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Utility methods')
    parser.add_argument('--rf', action='store_true', help="Random Forest")
    parser.add_argument('--xgb', action='store_true', help="XGBoost")
    parser.add_argument('--seg', action='store_true', help="Training data based on speaker segments")
    parser.add_argument('--sw', action='store_true', help="Training data based on sliding windows")
    parser.add_argument('--hierarchy', action='store', type=int, choices=[0, 1], default=0, help="Speaker mapping hierarchy level")

    # Experiments
    parser.add_argument('--kfold', action='store_true', help="k-fold-cross-validation")
    parser.add_argument('--traintest', action='store_true', help="Split into train and test set and validate accuracy on one split")
    parser.add_argument('--trainset', nargs='+', choices=['tag', 'bild', 'com'], help="Split dataset by news sources. I.e. , the training set will be trainied on compact and tagesschau and tested on bild")
    parser.add_argument('--confusion', action='store_true', help="Plot confusion matrix")
    parser.add_argument('--importances', action='store_true', help="Plot feature importance")
    parser.add_argument('--correlation', action='store_true', help="Plot correlation between features and labels")
    
    args = parser.parse_args()

    speaker_hierarchy_level = args.hierarchy

    if args.rf or args.xgb:
        training(args)