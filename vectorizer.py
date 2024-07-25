'''
(DEV) Class to vectorise tfidf data.

Modelled loosely on the sklearn Vectorizer classes, to be compatible with
other sklearn functions, while having a custom vectorizer to make use of my
implemented TF-IDF.
'''

from scipy.sparse import csr_matrix, lil_matrix

class Vectorizer:
    def __init__(self):
        self.vector = None

    def transform(self, all_phrases):
        """
        Transform the data to be used for the classification.

        Parameters:
        all_phrases (DataFrame): The data to be transformed, where each phrase is a dictionary {tokens : tfidf value}.
        """
        # Create a set of all unique tokens
        all_tokens = set()
        for tokens_to_tfidf in all_phrases:
            all_tokens.update(tokens_to_tfidf.keys())

        # Create the vector
        vector = lil_matrix((len(all_phrases), len(all_tokens)), dtype=int)

        self.token_to_col = {token: col_index for col_index, token in enumerate(all_tokens)}

        # Fill the vector
        for i in range(len(all_phrases)):
            tokens_to_tfidf = all_phrases.at[i]

            for token, tfidf in tokens_to_tfidf.items():
                col_index = list(all_tokens).index(token)
                vector[i, col_index] = tfidf

        self.vector = csr_matrix(vector)
        return vector
    
    def get_feature_names(self, mask):
        """
        Get the names of the features given a mask.
        """
        # Get the indices of selected features from the mask
        selected_indices = [i for i, selected in enumerate(mask) if selected]

        # Create a list of feature names corresponding to the selected features
        selected_features = [feature_name for feature_name, col_index in self.token_to_col.items() if col_index in selected_indices]

        return selected_features
            
        