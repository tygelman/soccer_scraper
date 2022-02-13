from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import dill as pickle

class MultiBayes:
    def __init__(self, data, name_matches):
        print('Model initializing.')
        print('Reading data...')
        self.data = pd.read_csv(data)
        self.kmeans = self.recluster()
        self.name_matches = pd.read_csv(name_matches).iloc[:, 1:]
        print('Model loaded!')

    def recluster(self):
        print('Reclustering Model...')
        X = self.data[['longitude', 'latitude']].values
        kmeans = KMeans(n_clusters=12)
        kmeans.fit(X)
        self.data['cluster'] = kmeans.labels_
        print('Done clustering!')
        return kmeans

    def admit_rate(self, name):
        if name not in self.name_matches.d_names.values:
            print(f'College {name} was not found')
            return -1
        return self.name_matches[self.name_matches.d_names == name].admit_rate.values[0]


    def pTOKENS_H(self, df, college_name, tokens):
        college = df[df.College == college_name]
        n_players = len(college)

        pTOKENS_H = 1
        for token, value in tokens.items():
            n_token = len(college[college[token] == value])
            pTokenI_H = n_token / n_players
            pTOKENS_H *= pTokenI_H

        return pTOKENS_H


    def pTOKENS_notH(self, df, college_name, tokens):
        not_college = df[df.College != college_name]
        n_players_not_college = len(not_college)

        pTOKENS_notH = 1
        for token, value in tokens.items():
            n_not_token = len(not_college[not_college[token] == value])
            pTokenI_notH = n_not_token / n_players_not_college
            pTOKENS_notH *= pTokenI_notH

        return pTOKENS_notH


    def multiple_bayes(self, df, college_name, tokens):
        admissions_rate = self.admit_rate(college_name)
        pT_H = self.pTOKENS_H(df, college_name, tokens)
        pT_notH = self.pTOKENS_notH(df, college_name, tokens)
        numerator = admissions_rate * pT_H
        denominator_right_half = (1 - admissions_rate) * pT_notH
        denominator_full = numerator + denominator_right_half
        pH_T = numerator / denominator_full

        return pH_T, admissions_rate

    def predict(self, college_name, tokens, relative=False):
        df = self.data.copy()
        pred = self.multiple_bayes(df, college_name, tokens)
        if relative:
            pred /= self.admit_rate(college_name)
        return pred

class MultiBayesACT:
    def __init__(self, data, geo_clusters, act_clusters):
        self.data = data #should be a dataframe
        self.geo_clusters = geo_clusters
        self.act_clusters = act_clusters
        print('Model loaded!')

    def admit_rate(self, name):
        if name not in self.data.college_name.unique():
            print(f'College {name} was not found')
            return -1
        return self.data[self.data.college_name == name].admit_rate.values[0]


    def pTOKENS_H(self, df, college_name, tokens):
        college = df[df.college_name == college_name]
        n_players = len(college)

        pTOKENS_H = 1
        for token, value in tokens.items():
            n_token = len(college[college[token] == value])
            pTokenI_H = n_token / n_players
            pTOKENS_H *= pTokenI_H

        return pTOKENS_H


    def pTOKENS_notH(self, df, college_name, tokens):
        not_college = df[df.college_name != college_name]
        n_players_not_college = len(not_college)

        pTOKENS_notH = 1
        for token, value in tokens.items():
            n_not_token = len(not_college[not_college[token] == value])
            pTokenI_notH = n_not_token / n_players_not_college
            pTOKENS_notH *= pTokenI_notH

        return pTOKENS_notH


    def multiple_bayes(self, df, college_name, tokens):
        admissions_rate = self.admit_rate(college_name)
        pT_H = self.pTOKENS_H(df, college_name, tokens)
        pT_notH = self.pTOKENS_notH(df, college_name, tokens)
        numerator = admissions_rate * pT_H
        denominator_right_half = (1 - admissions_rate) * pT_notH
        denominator_full = numerator + denominator_right_half
        pH_T = numerator / denominator_full

        return pH_T, admissions_rate

    def predict(self, college_name, tokens, relative=False):
        df = self.data.copy()
        pred, admit_rate = self.multiple_bayes(df, college_name, tokens)
        if pred > 0:
            if relative:
                pred /= self.admit_rate(college_name)
            return pred, admit_rate
        else:
            return np.nan, admit_rate