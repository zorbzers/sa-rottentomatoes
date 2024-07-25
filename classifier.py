"""
Naive Bayes Classifier with Laplace smoothing.
"""
from collections import Counter

class Classifier:
    def __init__(self, number_classes, features):
        self.number_classes = number_classes
        self.features = features
        self.prior_probabilities = []
        self.likelihoods = {}
        self.feature_set = set()

        # Set the Laplace smoothing parameter to optimal for chi-square feature selection
        if number_classes == 3:
            self.lconst = 1
        else:
            self.lconst = 0.87
    
    def train(self,training_data):
        """
        Train the classifier using provided training data.

        Parameters:
        training_data (DataFrame): The training data.
        """

        # Calculate the prior probabilities for each class
        self.prior_probabilities = (
            training_data['Sentiment']
            .value_counts(normalize=True)
            .sort_index()
            .tolist()
        )

        # Calculate the likelihoods of each feature for each class
        self.calculate_likelihoods(training_data)

    def predict(self, data):
        """
        Generate predictions for provided given data.

        Parameters:
        data (DataFrame): The data to generate predictions for.

        Returns:
        predictions (dict): A dictionary of the predictions for each sentence {sentence_id: prediction}.
        """
        predictions = {}

        for _, row in data.iterrows():
            words = row['Phrase']

            # Calculate the probabilities of each class for the phrase
            posterior_probs = self.calculate_probabilities(words)

            # Store the result in the predictions dictionary
            predictions[row['SentenceId']] = posterior_probs.index(max(posterior_probs))

        return predictions

    def calculate_probabilities(self, words):
        """
        Calculate the posterior probabilities of each class for the given phrase.

        Parameters:
        words (list): The words in the phrase.

        Returns:
        posterior_probs (list): A list of the posterior probabilities for each class.
        """
        posterior_probs = []

        if self.number_classes == 3:
            default_values = [len(self.feature_set)] * 3
        else:
            default_values = [len(self.feature_set)] * 5

        for c in range(self.number_classes):
            # Add the prior probability for the class
            posterior_prob = self.prior_probabilities[c]
    
            for word in words:
                # Multiply the prior probabilitiy with the likelihoods
                posterior_prob *= self.likelihoods.get(word, default_values)[c]

            posterior_probs.append(posterior_prob)

        return posterior_probs

    def calculate_likelihoods(self, training_data):
        """
        Calculate the likelihoods of each feature for each class.

        Parameters:
        training_data (DataFrame): The training data.

        Returns:
        likelihoods (dict): A dictionary of the likelihoods for each feature {feature: [likelihoods]}.
        """
        # Get the all features for each class
        total_features = [
            Counter(training_data[training_data['Sentiment'] == c]['Phrase'].explode())
            for c in range(self.number_classes)
        ]

        # Get a set of all the unique features in the training data (for laplase smoothing)
        self.feature_set = set(training_data['Phrase'].explode().unique())

        # Calculate the likelihoods of each feature for each class
        for feature in self.feature_set:
            f_likelihoods = []

            for c in range(self.number_classes):
                # Calculate the number of times the feature appears in the training data for the class
                feature_count = total_features[c].get(feature, 0)

                # Calculate the likelihood of the feature for the class
                f_likelihoods.append((feature_count + 1*self.lconst) / (sum(total_features[c].values()) + len(self.feature_set)*self.lconst))
            
            self.likelihoods[feature] = f_likelihoods