"""
Feature extraction.
"""
import math

class FeatureExtraction:
    def __init__(self, method_extraction, number_classes):
        self.METHOD = method_extraction
        self.number_classes = number_classes

    def extract_features(self, all_phrases):
        """
        Extract the features to be used for the classification.

        Parameters:
        all_phrases (DataFrame): The data to be transformed, where each phrase is a list of tokens.
        """
        if self.METHOD == None:
            return
        elif self.METHOD == 'TF':
            self.tf(all_phrases)
        elif self.METHOD == 'TF-IDF':
            self.tfidf(all_phrases)

    def tf(self, all_phrases):
        """
        Extract the features using the TF method.

        Parameters:
        all_phrases (DataFrame): The data to be transformed, where each phrase is a list of tokens.
        """
        for i in range (len(all_phrases)):
            phrase = all_phrases.at[i]
            phrase_to_tf = {}

            for token in phrase:
                tf = phrase.count(token)
                phrase_to_tf[token] = tf

            all_phrases.at[i] = phrase_to_tf

    def tfidf(self, all_phrases):
        """
        Extract the features using the TF-IDF method.

        Parameters:
        all_phrases (DataFrame): The data to be transformed, where each phrase is a list of tokens.
        """   
        all_idfs = self.calc_idfs(all_phrases)

        for i in range (len(all_phrases)):
            phrase = all_phrases.at[i]
            phrase_to_tfidf = {}

            for token in phrase:
                tf = self.calc_tf(token, phrase)
                phrase_to_tfidf[token] = tf * all_idfs[token]

            all_phrases.at[i] = phrase_to_tfidf    

    def calc_tf(self, token, phrase): 
        """
        Calculate the term frequency (TF) of a token in a phrase.
        
        Parameters:
        token (str): The token to calculate the TF for.
        phrase (list): The phrase of the token.

        Returns:
        tf (int): The term frequency of the token in the phrase.
        """      
        tf = phrase.count(token)
        return tf

    def calc_idfs(self, all_phrases):
        """
        Calculate the inverse document frequency (IDF) of each token.

        Parameters:
        all_phrases (DataFrame): The data to be transformed, where each phrase is a list of tokens.

        Returns:
        idfs (dict): A dictionary of the IDF for each token {token: IDF}.
        """
        dnum = len(all_phrases)

        # Flatten the DataFrame and get a Series of all tokens
        all_tokens = all_phrases.explode()

        # Calculate the document frequency (DF) for each token
        token_df = all_tokens.groupby(all_tokens).size()

        # Calculate the IDF for each token
        idfs = token_df.apply(lambda df: math.log(dnum / df))

        return idfs.to_dict()
