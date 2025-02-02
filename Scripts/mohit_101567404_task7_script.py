import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, RFE, SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from lightgbm import LGBMClassifier
import argparse

# Feature Selection Methods
def cor_selector(X, y, num_feats):
    cor_list = []
    feature_name = X.columns.tolist()
    for i in X.columns.tolist():
        cor = np.corrcoef(X[i], y)[0, 1]
        cor_list.append(cor)
    cor_list = [0 if np.isnan(i) else i for i in cor_list]
    cor_feature = X.iloc[:, np.argsort(np.abs(cor_list))[-num_feats:]].columns.tolist()
    cor_support = [True if i in cor_feature else False for i in feature_name]
    return cor_support, cor_feature

def chi_squared_selector(X, y, num_feats):
    X_norm = MinMaxScaler().fit_transform(X)
    chi_selector = SelectKBest(chi2, k=num_feats)
    chi_selector.fit(X_norm, y)
    chi_support = chi_selector.get_support()
    chi_feature = X.loc[:, chi_support].columns.tolist()
    return chi_support, chi_feature

def rfe_selector(X, y, num_feats):
    rfe_selector = RFE(estimator=LogisticRegression(), n_features_to_select=num_feats, step=10, verbose=5)
    rfe_selector.fit(X, y)
    rfe_support = rfe_selector.get_support()
    rfe_feature = X.loc[:, rfe_support].columns.tolist()
    return rfe_support, rfe_feature

def embedded_log_reg_selector(X, y, num_feats):
    embedded_lr_selector = SelectFromModel(LogisticRegression(penalty="l2"), max_features=num_feats)
    embedded_lr_selector.fit(X, y)
    embedded_lr_support = embedded_lr_selector.get_support()
    embedded_lr_feature = X.loc[:, embedded_lr_support].columns.tolist()
    return embedded_lr_support, embedded_lr_feature

def embedded_rf_selector(X, y, num_feats):
    embedded_rf_selector = SelectFromModel(RandomForestClassifier(n_estimators=100), max_features=num_feats)
    embedded_rf_selector.fit(X, y)
    embedded_rf_support = embedded_rf_selector.get_support()
    embedded_rf_feature = X.loc[:, embedded_rf_support].columns.tolist()
    return embedded_rf_support, embedded_rf_feature

def embedded_lgbm_selector(X, y, num_feats):
    embedded_lgbm_selector = SelectFromModel(LGBMClassifier(n_estimators=100), max_features=num_feats)
    embedded_lgbm_selector.fit(X, y)
    embedded_lgbm_support = embedded_lgbm_selector.get_support()
    embedded_lgbm_feature = X.loc[:, embedded_lgbm_support].columns.tolist()
    return embedded_lgbm_support, embedded_lgbm_feature

# Main function to run feature selection
def auto_feature_selector(X, y, num_feats, methods):
    feature_name = list(X.columns)
    feature_selection_df = pd.DataFrame({'Feature': feature_name})

    # Apply selected methods
    if 'pearson' in methods:
        cor_support, _ = cor_selector(X, y, num_feats)
        feature_selection_df['Pearson'] = cor_support
    if 'chi2' in methods:
        chi_support, _ = chi_squared_selector(X, y, num_feats)
        feature_selection_df['Chi-2'] = chi_support
    if 'rfe' in methods:
        rfe_support, _ = rfe_selector(X, y, num_feats)
        feature_selection_df['RFE'] = rfe_support
    if 'log_reg' in methods:
        embedded_lr_support, _ = embedded_log_reg_selector(X, y, num_feats)
        feature_selection_df['Logistics'] = embedded_lr_support
    if 'random_forest' in methods:
        embedded_rf_support, _ = embedded_rf_selector(X, y, num_feats)
        feature_selection_df['Random Forest'] = embedded_rf_support
    if 'lightgbm' in methods:
        embedded_lgbm_support, _ = embedded_lgbm_selector(X, y, num_feats)
        feature_selection_df['LightGBM'] = embedded_lgbm_support

    # Calculate total votes for each feature
    feature_selection_df['Total'] = np.sum(feature_selection_df.iloc[:, 1:], axis=1)

    # Sort features by votes
    feature_selection_df = feature_selection_df.sort_values(['Total', 'Feature'], ascending=False)

    # Reset index
    feature_selection_df.index = range(1, len(feature_selection_df) + 1)

    return feature_selection_df.head(num_feats)

# Main execution block
if __name__ == "__main__":
    # Command line arguments parser
    parser = argparse.ArgumentParser(description="Auto Feature Selector")
    parser.add_argument('dataset', type=str, help="Path to the dataset (CSV format)")
    parser.add_argument('num_feats', type=int, help="Number of top features to select")
    parser.add_argument('methods', type=str, help="List of feature selection methods to apply", nargs='+')

    args = parser.parse_args()

    # Load dataset
    player_df = pd.read_csv(args.dataset)
    
    # Prepare the dataset (you can modify this part as needed)
    numcols = ['Overall', 'Crossing', 'Finishing', 'ShortPassing', 'Dribbling', 'LongPassing', 'BallControl', 'Acceleration', 'SprintSpeed', 'Agility', 'Stamina', 'Volleys', 'FKAccuracy', 'Reactions', 'Balance', 'ShotPower', 'Strength', 'LongShots', 'Aggression', 'Interceptions']
    catcols = ['Preferred Foot', 'Position', 'Body Type', 'Nationality', 'Weak Foot']
    
    player_df = player_df[numcols + catcols]
    traindf = pd.concat([player_df[numcols], pd.get_dummies(player_df[catcols])], axis=1)
    features = traindf.columns
    traindf = traindf.dropna()
    
    X = traindf.copy()
    y = (traindf['Overall'] >= 87).astype(int)
    del X['Overall']

    # Get the best features based on selected methods
    best_features_df = auto_feature_selector(X, y, args.num_feats, args.methods)

    # Print or save the selected features
    print(best_features_df)

    # Optionally, you could also save the results to a file
    best_features_df.to_csv('selected_features.csv', index=False)
