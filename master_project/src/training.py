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
    from sklearn.model_selection import cross_val_score, cross_val_predict, KFold
    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import GridSearchCV

    # load the trainset based on passed arguments
    if args.seg and args.speaker:
        # segment based speaker classification
        with open(config['speaker_trainingdata_tagesschau_seg'].format(speaker_hierarchy_level=speaker_hierarchy_level), 'rb') as pkl:
            tagesschau_data = pickle.load(pkl)
        with open(config['speaker_trainingdata_bildtv_seg'].format(speaker_hierarchy_level=speaker_hierarchy_level), 'rb') as pkl:
            bildtv_data = pickle.load(pkl)
        with open(config['speaker_trainingdata_compacttv_seg'].format(speaker_hierarchy_level=speaker_hierarchy_level), 'rb') as pkl:
            compacttv_data = pickle.load(pkl)
    elif args.sw and args.speaker:
        # sliding window based speaker classification
        with open(config['speaker_trainingdata_tagesschau_window'].format(speaker_hierarchy_level=speaker_hierarchy_level), 'rb') as pkl:
            tagesschau_data = pickle.load(pkl)
        with open(config['speaker_trainingdata_bildtv_window'].format(speaker_hierarchy_level=speaker_hierarchy_level), 'rb') as pkl:
            bildtv_data = pickle.load(pkl)
        with open(config['speaker_trainingdata_compacttv_window'].format(speaker_hierarchy_level=speaker_hierarchy_level), 'rb') as pkl:
            compacttv_data = pickle.load(pkl)
    elif args.seg and args.situations:
        # segment based news situation classification
        with open(config['situations_trainingdata_tagesschau_seg'], 'rb') as pkl:
            tagesschau_data = pickle.load(pkl)
            # ariticial 'commenting' sample because only BildTV has commentings
            commenting_dummy = np.zeros(53)
            commenting_dummy[-1] = 3
            tagesschau_data = np.vstack((tagesschau_data, commenting_dummy))   
        with open(config['situations_trainingdata_bildtv_seg'], 'rb') as pkl:
            bildtv_data = pickle.load(pkl)
        with open(config['situations_trainingdata_compacttv_seg'], 'rb') as pkl:
            compacttv_data = pickle.load(pkl)
    elif args.sw and args.situations:
        # sliding window based news situation classification
        with open(config['situations_trainingdata_tagesschau_window'], 'rb') as pkl:
            tagesschau_data = pickle.load(pkl)
        with open(config['situations_trainingdata_bildtv_window'], 'rb') as pkl:
            bildtv_data = pickle.load(pkl)
        with open(config['situations_trainingdata_compacttv_window'], 'rb') as pkl:
            compacttv_data = pickle.load(pkl)
    else:
        print('ERROR: Please select a training data aggregation method (--seg for segment-wise feature aggregation or --sw for window-based feature aggregation)')
        return
    
    np.random.seed(config['seed'])
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
        model = RandomForestClassifier(random_state=config['seed'], n_estimators=110)
        model_s = 'rf'
    elif args.xgb:
        model = XGBClassifier(random_state=config['seed'], tree_method="gpu_hist", n_estimators=110)
        model_s = 'xgb'
    else:
        print('Please choose between Random Forest or XGBoost by using either --rf or --xgb')
        return

    # k-fold cross-validation
    if args.kfold:
        scores = cross_val_score(model, features, groundtruth, cv=config['k'])
        print(f'{config["k"]}-Fold Cross-Validation: {"{:.4f}".format(scores.mean())} accuracy with a standard deviation of {"{:.2f}".format(scores.std())}')

    # fit model for later experiments
    model.fit(train_features, train_groundtruth)

    # single prediction on a train and test split (only print when requested)
    if args.traintest: # or args.confusion or args.importances
        # predict test set and measure accuracy
        
        predictions = model.predict(test_features)
        accuracy = accuracy_score(test_groundtruth, predictions)
        print(f'Single Split Accuracy: {"{:.4f}".format(accuracy)}')
        
        """
        # get test vector with certain ground truth value to extract some qualitative examples
        np.set_printoptions(suppress=True)
        for i, feature_vec in enumerate(test_features):
            if test_groundtruth[i] == 3:
                print(feature_vec)
                print()
        print('prediction: ', model.predict([[0.03799942553536419, 0.005991228985116251, 0.001511286966406795, 2.7630266360717417e-06, 6.326894765829909e-06, 4.885013595412602e-06, 0.0008562736312502666, 0.00044543594755369295, 0.0014630865577297907, 0.00012724985215612048, 0.21641812254400813, 0.39569485888761635, 0.11829699385984271, 0.22117284787636177, 9.215292114580128e-06, 0.012857456472428405, 0.5624888475034752, 0.0036847912352623454, 0.008948381885192265, 0.41202052273586687, 0.9982874101283503, 0.0004884887663031216, 0.0012241007575272199, 0.7722245797137727, 25.29, 0.22935129246131206, 0.21508056275908732, 0.5555681606961621, 4, 0.0379746835443038, 0.10126582278481013, 0.12658227848101267, 0.05063291139240506, 0.012658227848101266, 0.17721518987341772, 0.0, 0.20253164556962025, 0.0, 0.0, 0.06329113924050633, 0.0379746835443038, 0.06329113924050633, 0.12658227848101267, 1.0, 1.0, 1.0, 0.0, 1.0, 0.75, 0.72998046875, 0.6391387581825256, 0.4834901690483093]]))
        # speaker misclassification ex: 0.04017149205319583, 0.4508280088504156, 0.0010132630556957641, 0.00010003007701016031, 0.0010307618495668672, 0.00109536349773407, 0.107949265340964, 0.013909763904909294, 0.16516335358222325, 0.0014063651416411935, 0.028134463102711987, 0.07907780134119094, 0.03497583265416324, 0.0723439773816305, 0.0028002465104994675, 0.36291042069594065, 0.42872194747130077, 0.00048466057536037017, 0.0005435606891599794, 0.20733939098815124, 0.560313614209493, 0.0029773110368599494, 0.43670908908049266, 0.8674701165471224, 7.491999999999962, 0.04235951416194439, 0.0553019642829895, 0.9023384749889374, 2, 0.02857142857142857, 0.11428571428571428, 0.17142857142857143, 0.02857142857142857, 0.08571428571428572, 0.08571428571428572, 0.0, 0.11428571428571428, 0.02857142857142857, 0.0, 0.05714285714285714, 0.02857142857142857, 0.11428571428571428, 0.14285714285714285, 0.0, 0.0, 1.0, 0.0, 0.0, 0.5537109375, 0.53427734375, 0.46963265538215637, 0.199191614985466
        """
        
    # path to output files
    out_basepath = ''
    if args.speaker:
        if args.seg:
            out_basepath = '/nfs/home/arafatj/master_project/graphics/training/speaker/seg/'
        elif args.sw:
            out_basepath = '/nfs/home/arafatj/master_project/graphics/training/speaker/sw/'
    elif args.situations:
        if args.seg:
            out_basepath = '/nfs/home/arafatj/master_project/graphics/training/situations/seg/'
        if args.sw:
            out_basepath = '/nfs/home/arafatj/master_project/graphics/training/situations/sw/'

    classification_target = 'speaker' if args.speaker else 'situations'

    # plot the confusion matrix with labels (matrix with true label being i-th class and predicted label being j-th class)
    if args.confusion:
        # confusion matrix on cross validation results
        kfold= KFold(n_splits=config['k'])
        predictions = cross_val_predict(model, features, groundtruth, cv=kfold)
        cm = confusion_matrix(groundtruth, predictions)
        df_cm = pd.DataFrame(cm)
        print(df_cm)
        
        """
        # One test split
        predictions = model.predict(test_features)
        cm = confusion_matrix(test_groundtruth, predictions)
        df_cm = pd.DataFrame(cm)
        print(df_cm)
        """
        
        # convert to relative values (normalize)
        row_sums = df_cm.sum(axis=1)
        df_percentage = df_cm.div(row_sums, axis=0).round(2)

        if args.speaker and args.hierarchy == 1:
            # drop 'other' since there are not many samples
            df_percentage = df_percentage.drop(2)
            df_percentage = df_percentage.drop(2, axis=1)

        plt.figure(figsize=(13,13))
        sns.set(font_scale=1.8)
        sns.heatmap(df_percentage, xticklabels=config[f'labels_{speaker_hierarchy_level}'], yticklabels=config[f'labels_{speaker_hierarchy_level}'], annot=True, annot_kws={'size':25}, cmap='Purples', fmt='.2f')
        plt.xlabel('Predicted Label', fontweight='bold')
        plt.ylabel('True Label', fontweight='bold')
        #plt.tight_layout()
        plot_path = f'{out_basepath}/confusion_matrix_{model_s}.png'
        plt.savefig(plot_path)
        print(f'Confusion Matrix: {plot_path}')

    # plot feature importances (what features are most taken into consideration by the classifier when predicting)
    if args.importances:
        importances = model.feature_importances_
        indices = np.argsort(importances)
        plt.figure(figsize=(26,20))
        plt.rcParams['font.size'] = 18
        if args.speaker:
            plt.title("Speaker Role Recognition - Feature Importances", fontweight='bold')
        elif args.situations:
            plt.title("News Situation Recognition - Feature Importances", fontweight='bold')
        plt.barh(range(len(indices)), importances[indices], align='center')
        plt.yticks(range(len(indices)), [config[f'feature_names_{classification_target}'][i] for i in indices])
        if model_s == 'rf':
            plt.xlabel('Mean Decrease in Impurity')
        elif model_s == 'xgb':
            plt.xlabel('Relative Information Gain')
        plot_path = f'{out_basepath}/feature_importances_{model_s}.png'
        plt.savefig(plot_path)
        print(f'Feature Importances: {plot_path}')

    # correlation matrix between features and labels
    if args.correlation:
        features_with_labels = pd.DataFrame(np.column_stack((features, groundtruth)), columns=config[f'feature_names_{classification_target}'] + ['predicted_label'])

        if args.speaker:
            labels = config['groundtruth_numerical_speaker']
        elif args.situations:
            labels = config['groundtruth_numerical_situations']

        label_feature_corrs = pd.DataFrame()
        for label_numerical in labels.values():
            # extract entries that match the label
            df_label = features_with_labels[features_with_labels['predicted_label'] == label_numerical]
            # set all predicted_label entries to 1 (indicating that the label is existent in these samples)
            df_label.loc[:, 'predicted_label'] = 1
            # extract all other entries that were not extracted in the step before
            df_without_label = features_with_labels[~features_with_labels.index.isin(df_label.index)]
            # set all predicted_label entries to 0 (indicating that the label is not existent in these samples)
            df_without_label.loc[:, 'predicted_label'] = 0
            # merge both dataframes again, so we have the distinction between samples where the label matches vs. non-matching samples
            merged_df = pd.concat([df_label, df_without_label])
            # create correlation matrix
            corr_matrix = merged_df.corr()
            # extract only the row which represents the feature-label correlation
            label_feature_corr = corr_matrix['predicted_label']

            # label numerical as string instead
            label_string = list(labels.keys())[label_numerical]
            # convert pandas series to dictionary
            label_feature_corr_dict = label_feature_corr.to_dict()
            # set label as string into the dictionary
            label_feature_corr_dict['label'] = label_string
            # convert the dictionary to a dataframe
            df_row = pd.DataFrame.from_records([label_feature_corr_dict])
            # add the correlation between features and labels for the current label to the dataframe
            label_feature_corrs = pd.concat([label_feature_corrs, df_row], ignore_index=True)

        label_feature_corrs.set_index('label', inplace=True)
        label_feature_corrs = label_feature_corrs.drop('predicted_label', axis=1)

        if args.speaker and args.hierarchy == 0:
            label_feature_corrs = label_feature_corrs.drop(['expert', 'layperson', 'politician'])
        elif args.speaker and args.hierarchy == 1:
            label_feature_corrs = label_feature_corrs.drop(['other'])

        """
        # top feature correlations
        label_feature_corrs_sorted = label_feature_corrs.apply(lambda row: sorted(row, key=lambda x: abs(x), reverse=True), axis=1)
        print(label_feature_corrs_sorted)
        label_feature_corrs_sorted.to_csv('/nfs/home/arafatj/master_project/graphics/training/top_correlations.csv')
        """

        # split dataframe, else heatmap is too wide
        split_index = label_feature_corrs.columns.get_loc('LengthOfSpeech')
        df1 = label_feature_corrs.iloc[:, :split_index + 1]
        df2 = label_feature_corrs.iloc[:, split_index + 1:]

        # split 1 plot
        plt.figure(figsize=(30,10))
        plt.rcParams['font.size'] = 22
        cmap = sns.diverging_palette(15, 120, as_cmap=True)
        sns.heatmap(df1, cmap=cmap, linewidths=0.5, square=True, cbar=False, annot=True, fmt=".2f")
        plt.xlabel(' ')
        plt.ylabel(' ')
        plot_path = f'{out_basepath}/correlation_matrix_1.png'
        plt.savefig(plot_path)
        print(f'Correlation Matrix 1: {plot_path}')

        # split 2 plot
        plt.clf()
        plt.figure(figsize=(30,13))
        sns.heatmap(df2, cmap=cmap, linewidths=0.5, square=True, cbar=False, annot=True, fmt=".2f")
        plot_path = f'{out_basepath}/correlation_matrix_2.png'
        plt.xlabel(' ')
        plt.ylabel(' ')
        plt.savefig(plot_path)
        print(f'Correlation Matrix 2: {plot_path}')

        """
        # for correlation matrix between features amongst themselves (not correlation between label and features)
        #corr_matrix = features_with_labels.corr() 
        #sns.heatmap(corr_matrix, cmap='coolwarm', linewidths=0.5, square=True, cbar=False)

        plot_path = f'{out_basepath}/correlation_matrix.png'
        plt.savefig(plot_path)
        print(f'Correlation Matrix: {plot_path}')
        """
        
    # hyperparameter tuning
    if args.tuning:
        # define the hyperparameter grid with range of values
        param_grid = {
            'n_estimators': [50, 100, 110, 120, 130, 140, 150],
            'max_depth': [None, 5, 10, 15, 20, 25, 30, 35],
            'min_samples_split': [2, 4, 8, 16],
            'min_samples_leaf': [1, 2, 4, 8]
        }

        # perform grid search with cross-validation
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
        grid_search.fit(train_features, train_groundtruth)
        print('Best hyperparameters:',  grid_search.best_params_)
        
        # evaluate the performance on the test set using the best model
        best_model = grid_search.best_estimator_
        accuracy = best_model.score(test_features, test_groundtruth)
        print("Accuracy:", accuracy)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Utility methods')
    parser.add_argument('--rf', action='store_true', help="Random Forest")
    parser.add_argument('--xgb', action='store_true', help="XGBoost")

    parser.add_argument('--speaker', action='store_true', help="Speaker classification")
    parser.add_argument('--situations', action='store_true', help="News situation classification (segment based by default)")
    parser.add_argument('--seg', action='store_true', help="Training data based on speaker segments (only with --speaker)")
    parser.add_argument('--sw', action='store_true', help="Training data based on sliding windows (only with --speaker)")

    # Experiments
    parser.add_argument('--hierarchy', action='store', type=int, choices=[0, 1], default=0, help="Speaker mapping hierarchy level")
    parser.add_argument('--kfold', action='store_true', help="k-fold-cross-validation")
    parser.add_argument('--traintest', action='store_true', help="Split into train and test set and validate accuracy on one split")
    parser.add_argument('--trainset', nargs='+', choices=['tag', 'bild', 'com'], help="Split dataset by news sources. I.e. , the training set will be trainied on compact and tagesschau and tested on bild")
    parser.add_argument('--confusion', action='store_true', help="Plot confusion matrix")
    parser.add_argument('--importances', action='store_true', help="Plot feature importance")
    parser.add_argument('--correlation', action='store_true', help="Plot correlation between features and labels")
    parser.add_argument('--tuning', action='store_true', help="Hyperparameter Tuning")

    
    args = parser.parse_args()

    if args.speaker:
        speaker_hierarchy_level = args.hierarchy
    elif args.situations:
        # news situations do not have a class hierarchy, just set a placeholder
        speaker_hierarchy_level = 'situations'
    else:
        print('ERROR: Please select the classifier target (--speaker for speaker classification or --situations for news situation classification)')
        quit()

    if args.rf or args.xgb:
        training(args)