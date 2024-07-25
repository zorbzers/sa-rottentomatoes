"""
Feature selection.
"""
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_extraction.text import CountVectorizer

import nltk
from nltk.corpus import wordnet as wn
nltk.download('wordnet', quiet=True)

class FeatureSelection:
    def __init__(self, method_selection, number_classes):
        self.METHOD = method_selection
        self.number_classes = number_classes

        # Set the optimal constants
        if self.number_classes == 3:
            self.chi_k = 1800
            self.tfidf_threshold = 0.7
        else:
            self.chi_k = 950
            self.tfidf_threshold = 4

        self.chi_features = None

    def select_features(self, all_phrases, sentiments=None):
        """
        Select the features to be used for the classification.

        Parameters:
        all_phrases (DataFrame): The data to be transformed.
        sentiments (DataFrame): The sentiments of the phrases.
        """
        if self.METHOD == None:
            return
        elif self.METHOD == 'TF-IDF':
            self.tfidf(all_phrases)
        elif self.METHOD == 'NVAR':
            self.nvar(all_phrases)
        elif self.METHOD == 'chi-square':
            self.chi_square(all_phrases, sentiments)

    def tfidf(self, all_phrases):
        """
        Select the features by selecting TF-IDF values greater than a certain threshold.

        Parameters:
        all_phrases (DataFrame): The data to be transformed.
        """
        for i in range (len(all_phrases)):
            phrase = all_phrases.at[i]
        
            # select all words with tfidf > threshold
            tokens = {k: v for k, v in phrase.items() if v > self.tfidf_threshold}

            # get a list of tokens 
            phrase = list(tokens.keys())

            # the phrase returned has the tfidf values now discarded
            all_phrases.at[i] = phrase

    def nvar(self, all_phrases):
        """
        Select the features using nouns, verbs, adjectives and adverbs as features.
        Uses words from the WordNet corpus as features.

        Parameters:
        all_phrases (DataFrame): The data to be transformed.
        """
        for i in range(len(all_phrases)):
            phrase = all_phrases.at[i]

            new_phrase = []

            for token in phrase:
                if wn.synsets(token):
                    new_phrase.append(token)

            all_phrases.at[i] = new_phrase

    def chi_square(self, all_phrases, sentiments=None):
        """
        Select the features using the Chi-Square method.

        Parameters:
        all_phrases (DataFrame): The data to be transformed.
        sentiments (DataFrame): The sentiments of the phrases.
        """

        # Determine the feature set
        if sentiments is not None:
            self.chi_features = self.chi_square_get_features(all_phrases, sentiments)

        # remove all tokens not in feature set
        for i in range (len(all_phrases)):
            phrase = all_phrases.at[i]

            new_phrase = []

            for token in phrase:
                if token in self.chi_features:
                    new_phrase.append(token)

            all_phrases.at[i] = new_phrase

    def chi_square_get_features(self, all_phrases, sentiments):
        """
        Determine the feature set using the Chi-Square method.

        Parameters:
        all_phrases (DataFrame): The data to be transformed.
        sentiments (DataFrame): The sentiments of the phrases.

        Returns:
        selected_feature_names (list): The selected features.
        """
        all_phrases = all_phrases.apply(lambda x: ' '.join(x))

        # Vectorize the data
        vectorizer = CountVectorizer()
        all_phrases_counted = vectorizer.fit_transform(all_phrases)

        # Select the k best features
        k = self.chi_k 
        chi2_selector = SelectKBest(chi2, k=k)
        chi2_selector.fit_transform(all_phrases_counted, sentiments)

        # Get the names of the selected features
        selected_feature_names = vectorizer.get_feature_names_out()[chi2_selector.get_support()]
        
        return selected_feature_names