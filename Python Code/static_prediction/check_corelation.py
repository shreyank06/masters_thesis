import pandas as pd
import dcor
from mlxtend.feature_selection import ExhaustiveFeatureSelector
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_selection import SelectKBest
import matplotlib.pyplot as plt
from sklearn.feature_selection import chi2
import sys

class CorrelationChecker:
    def __init__(self, df):
        self.df = df

    def distance_correlation_df(self, target_column):
        distance_correlations = {}
        df_numeric = self.df.dropna().select_dtypes(include=['float64', 'int64'])  # Filter out non-numeric columns and drop missing values
        for column in df_numeric.columns:
            if column != target_column:
                distance_corr = dcor.distance_correlation(df_numeric[target_column], df_numeric[column])
                distance_correlations[column] = distance_corr
        return distance_correlations

    def preprocess_data(self):
        # Handle categorical variables if any
        cat_columns = self.df.select_dtypes(include=['object']).columns
        if not cat_columns.empty:  # Use .empty to check if the Index is empty or not
            self.df = pd.get_dummies(self.df, columns=cat_columns, drop_first=True)

        # Impute missing values if any
        self.df.fillna(self.df.median(), inplace=True)

    # def compute_correlations(self):
    #     target_column = "subscriber_count_Connected"
    #     corr_matrix = self.df.corr(method='spearman', numeric_only=True)
    #     corr = corr_matrix[target_column].sort_values(ascending=False)
        
    #     corr_matrix_2 = self.df.corr(method='pearson', numeric_only=True)
    #     corr_2 = corr_matrix_2[target_column].sort_values(ascending=False)
        
    #     distance_correlations = self.distance_correlation_df(target_column)
    #     corr_3 = pd.Series(distance_correlations).sort_values(ascending=False)

    #     corr_df = pd.DataFrame({"Spearman Correlation (corr)": corr.values}, index=corr.index)
    #     corr_2_df = pd.DataFrame({"Pearson Correlation (corr_2)": corr_2.values}, index=corr_2.index)
    #     corr_3_df = pd.DataFrame({"Distance Correlation (corr_3)": corr_3.values}, index=corr_3.index)

    #     correlation_df = pd.merge(corr_df, corr_2_df, left_index=True, right_index=True)
    #     correlation_df = pd.merge(correlation_df, corr_3_df, left_index=True, right_index=True)

    #     correlation_df.reset_index(inplace=True)
    #     correlation_df.rename(columns={"index": "Column Name"}, inplace=True)

    #     correlation_df.to_csv('correlation_data_2.csv', index=True, mode='w', header=True)

    def exhaustive_feature_selection(self, target_column):
        self.preprocess_data()  # Preprocess the data before feature selection

        X = self.df.drop(columns=[target_column])
        y = self.df[target_column]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestRegressor()
        model.fit(X,y)
        feature_importances = model.feature_importances_
        feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
        feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
        print(feature_importance_df)
        # efs = ExhaustiveFeatureSelector(model, min_features=1, max_features=len(X.columns), scoring='neg_mean_squared_error', cv=5)

        # efs = efs.fit(X_train, y_train)

        # selected_features = list(X.columns[list(efs.best_idx_)])
        # print("Selected features:", selected_features)

    def fisher_score_feature_selection(self, target_column):
        print(22)
        self.preprocess_data()  # Preprocess the data before feature selection

        X = self.df.drop(columns=[target_column])
        y = self.df[target_column]

        fisher_scores, _ = f_classif(X, y)

        selected_features = X.columns[fisher_scores.argsort()[::-1]]  # Sort features by Fisher Score
        print("Selected features:", selected_features)

    def chi_squared_feture_selection(self, target_column):
    
        X = self.df.drop(columns=[target_column])
        y = self.df[target_column]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        def prepare_inputs(X_train, X_test):
            oe = OrdinalEncoder()
            oe.fit(X_train)
            X_train_enc = oe.transform(X_train)
            X_test_enc = oe.transform(X_test)
            return X_train_enc, X_test_enc
        
        def prepare_targets(y_train, y_test):
            le = LabelEncoder()
            le.fit(y_train)
            y_train_enc = le.transform(y_train)
            y_test_enc = le.transform(y_test)
            return y_train_enc, y_test_enc

        def select_features(X_train, y_train, X_test):
            fs = SelectKBest(score_func=chi2, k='all')
            fs.fit(X_train, y_train)
            X_train_fs = fs.transform(X_train)
            X_test_fs = fs.transform(X_test)
            return X_train_fs, X_test_fs, fs
        
        X_train_enc, X_test_enc = prepare_inputs(X_train, X_test)
        y_train_enc, y_test_enc = prepare_targets(y_train, y_test)
        X_train_fs, X_test_fs, fs = select_features(X_train_enc, y_train_enc, X_test_enc)

        for i in range(len(fs.scores_)):
            print('Feature %d: %f' % (i, fs.scores_[i]))
        # plot the scores
        plt.figure(figsize = (12,6))
        plt.bar([i for i in range(len(fs.scores_))], fs.scores_)
        plt.title("Feature Importance Score", size = 20)
        plt.xlabel("Features/ Variables", size = 16, color = "black")
        plt.ylabel("Importance Score", size = 16, color = "black")
        plt.show()

        



# Example usage:
# Load your DataFrame
# df = pd.read_csv('your_data.csv')
# checker = CorrelationChecker(df)
# checker.compute_correlations()
# checker.exhaustive_feature_selection("phoenix_memory_used_cm_sessionP_smf")
