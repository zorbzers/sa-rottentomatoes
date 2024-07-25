"""
Utils class.
Contains functions to:
    - Process the datasets. 
    - Apply preprocessing techniques.
    - Calculate F1 scores.
    - Generate confusion matrices.
    - Save the predictions to files.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

from feature_selection import FeatureSelection
from feature_extraction import FeatureExtraction

class Utils:
    def __init__(self, method_extraction, method_selection, number_classes, features, training, dev, test):
        self.method_extraction = method_extraction
        self.method_selection = method_selection

        self.number_classes = number_classes
        self.features = features

        self.training = training
        self.dev = dev
        self.test = test

        # Maps the 5 class sentiments to 3 classes.
        self.LABEL_MAP = {0:0, 1:0, 2:1, 3:2, 4:2}

        # Set the optimal preprocessing steps for the selected configuration
        #                       LC     ABR   SL     NEG   BIN    LEM   STEM 
        if self.number_classes == 3 and self.features == 'features':
            self.prepro_combo = [True, True, False, True, False, True, False]
        elif self.number_classes == 5 and self.features == 'features':
            self.prepro_combo = [True, False, False, False, False, True, False]
        elif self.number_classes == 3 and self.features == 'all_words':
            self.prepro_combo = [False, True, True, True, True, True, True]
        elif self.number_classes == 5 and self.features == 'all_words':
            self.prepro_combo = [False, True, True, True, False, True, False]

        self.lc, self.abr, self.sl, self.neg, self.bin, self.lem, self.stem = self.prepro_combo


    def process_data(self):
        """
        Process the data from the tsv files, map the labels to the correct values 
        when necessary, apply preprocessing steps, and extract and select features.

        Returns:
        training_data (DataFrame): The processed training data.
        dev_data (DataFrame): The processed dev data.
        test_data (DataFrame): The processed test data.
        """
        training_data = pd.read_csv(self.training, sep='\t')
        dev_data = pd.read_csv(self.dev, sep='\t')
        test_data = pd.read_csv(self.test, sep='\t')

        if self.number_classes == 3:
            training_data['Sentiment'] = training_data['Sentiment'].map(self.LABEL_MAP)
            dev_data['Sentiment'] = dev_data['Sentiment'].map(self.LABEL_MAP) 

        training_data['Phrase'] = training_data['Phrase'].apply(self.preprocessing)
        dev_data['Phrase'] = dev_data['Phrase'].apply(self.preprocessing)
        test_data['Phrase'] = test_data['Phrase'].apply(self.preprocessing)

        if self.features == 'features':
            if self.method_extraction != None:
                feature_extraction = FeatureExtraction(self.method_extraction, self.number_classes)
                feature_extraction.extract_features(training_data['Phrase'])
                feature_extraction.extract_features(dev_data['Phrase'])
                feature_extraction.extract_features(test_data['Phrase'])

            feature_selection = FeatureSelection(self.method_selection, self.number_classes)
            feature_selection.select_features(training_data['Phrase'], training_data['Sentiment'])
            feature_selection.select_features(dev_data['Phrase'])
            feature_selection.select_features(test_data['Phrase'])

        return (training_data, dev_data, test_data)
    
    def preprocessing(self, phrase):
        """
        Apply preprocessing steps to the phrases:
            - Convert to lowercase.
            - Replace abbreviations.
            - Remove stopwords.
            - Apply stemming.
            - Apply lemmatisation.
            - Apply negation.
            - Apply binarisation.

        Parameters:
        phrase (str): The phrase to preprocess.

        Returns:
        tokens (list): The list of tokens after preprocessing.
        """
        phrase = phrase.lower() if self.lc else phrase # Convert to lowercase
        tokens = phrase.split() # Array for easier modification
        tokens = self.replace_abbr(tokens) if self.abr else tokens # Replace abbreviations
        tokens = self.apply_stoplist(tokens) if self.sl else tokens # Remove stopwords
        tokens = self.apply_stemming(tokens) if self.lem else tokens # Apply stemming
        tokens = self.apply_lemmatisation(tokens) if self.stem else tokens # Apply lemmatisation
        tokens = self.apply_negtation(tokens) if self.neg else tokens # Apply negation
        tokens = self.apply_binarisation(tokens) if self.bin else tokens # Apply binarisation
        
        return tokens
    
    def replace_abbr(self, tokens):
        """
        Replace abbreviations with their full words.

        Parameters:
        tokens (list): The list of tokens to replace abbreviations in.

        Returns:
        tokens (list): The list of tokens with abbreviations replaced.
        """
        map = {
            "n't": "not",
            "'s": "is",
            "'m": "am",
            "'re": "are",
            "'ll": "will",
            "'ve": "have",
            "'d": "would",
            "ca": "can",
            "wo": "will"
        }

        for i in range(len(tokens)):
            if tokens[i] in map:
                tokens[i] = map[tokens[i]]
        return tokens
    
    def apply_stoplist(self, tokens):
        """
        Remove stopwords from the tokens.

        Parameters:
        tokens (list): The list of tokens to remove stopwords from.

        Returns:
        tokens (list): The list of tokens with stopwords removed.
        """
        with open('stoplist.txt', 'r') as file:
            stoplist = file.read().split()

        tokens = [word for word in tokens if word not in stoplist]

        return tokens
    
    def apply_negtation(self, tokens):
        """
        Apply negation to the tokens.

        Parameters:
        tokens (list): The list of tokens to apply negation to.

        Returns:
        tokens (list): The list of tokens with negation applied.
        """
        negation_words = ['not', 'no', 'never', 'n\'t', 'nor']

        for i in range(len(tokens)):
            if tokens[i] in negation_words:
                if i+1 < len(tokens):
                    tokens[i] = 'not_' + tokens[i+1]
        return tokens
    
    def apply_binarisation(self, tokens):
        """
        Apply binarisation to the tokens.

        Parameters:
        tokens (list): The list of tokens to apply binarisation to.

        Returns:
        tokens (list): The list of tokens with binarisation applied.
        """
        seen = set()
        result = []

        for token in tokens:
            if token not in seen:
                result.append(token)
                seen.add(token)
        return result

    def apply_stemming(self, tokens):
        """
        Apply stemming to the tokens.

        Parameters:
        tokens (list): The list of tokens to apply stemming to.

        Returns:
        tokens (list): The list of tokens with stemming applied.
        """
        ps = PorterStemmer()
        for i in range(len(tokens)):
            tokens[i] = ps.stem(tokens[i])
        return tokens

    def apply_lemmatisation(self, tokens):
        """
        Apply lemmatisation to the tokens.

        Parameters:
        tokens (list): The list of tokens to apply lemmatisation to.

        Returns:
        tokens (list): The list of tokens with lemmatisation applied.
        """
        lemmatizer = WordNetLemmatizer()
        for i in range(len(tokens)):
            tokens[i] = lemmatizer.lemmatize(tokens[i])
        return tokens

    def calculate_macro_f1(self, dev_data, dev_predictions):
        """
        Calculate the macro-F1 score for the dev set.

        Parameters:
        dev_data (DataFrame): The dev data.
        dev_predictions (dict): The predictions for the dev set.

        Returns:
        macro_f1 (float): The macro-F1 score.
        """
        f1s_scores = self.calculate_f1s(dev_data, dev_predictions)
        return sum(f1s_scores) / len(f1s_scores)
    
    def calculate_f1s(self, dev_data, dev_predictions):
        """
        Calculate the F1 score for each class.

        Parameters:
        dev_data (DataFrame): The dev data.
        dev_predictions (dict): The predictions for the dev set.

        Returns:
        f1s_scores (list): The F1 scores for each class.
        """
        f1s_scores = []

        for i in range(self.number_classes):
            tp = 0
            tn = 0
            fp = 0
            fn = 0 

            for _, row in dev_data.iterrows():
                if row['Sentiment'] == i:
                    if dev_predictions[row['SentenceId']] == i:
                        tp += 1
                    else:
                        fn += 1
                else:
                    if dev_predictions[row['SentenceId']] == i:
                        fp += 1
                    else:
                        tn += 1

            top = 2 * tp
            bot = (2 * tp) + fp + fn

            f1s_scores.append(top / bot)

        return f1s_scores

    def generate_heatmap(self, dev_data, dev_predictions):
        """
        Generate a heatmap of the confusion matrix.

        Parameters:
        dev_data (DataFrame): The dev data.
        dev_predictions (dict): The predictions for the dev set.
        """
        confusion_matrix = self.generate_confusion_matrix(dev_data, dev_predictions)

        visual = sns.heatmap(confusion_matrix, annot=True, fmt='g')
        visual.invert_yaxis()
        plt.xlabel('Predicted Labels')
        plt.ylabel('Actual Labels')
        plt.show()

    def generate_confusion_matrix(self, dev_data, dev_predictions):
        """
        Generate the confusion matrix.

        Parameters:
        dev_data (DataFrame): The dev data.
        dev_predictions (dict): The predictions for the dev set.

        Returns:
        confusion_matrix (list): The confusion matrix.
        """
        confusion_matrix = [[0 for _ in range(self.number_classes)] for _ in range(self.number_classes)]

        for _, row in dev_data.iterrows():
            actual_label = row['Sentiment']
            predicted_label = dev_predictions[row['SentenceId']]

            confusion_matrix[actual_label][predicted_label] += 1

        return confusion_matrix

    def save_predictions(self, dev_predictions, test_predictions):
        """
        Save the predictions to files.

        Parameters:
        dev_predictions (dict): The predictions for the dev set.
        test_predictions (dict): The predictions for the test set.
        """
        dev_predictions = pd.DataFrame.from_dict(dev_predictions, orient='index', columns=['Sentiment'])
        test_predictions = pd.DataFrame.from_dict(test_predictions, orient='index', columns=['Sentiment'])

        dev_predictions.to_csv(f'./dev_predictions_{self.number_classes}classes.tsv', sep='\t', index_label='SentenceId')
        test_predictions.to_csv(f'./test_predictions_{self.number_classes}classes.tsv', sep='\t', index_label='SentenceId')
