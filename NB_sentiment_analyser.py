# -*- coding: utf-8 -*-
"""
NB sentiment analyser. 

Start code.
"""
import argparse

from utils import Utils
from classifier import Classifier

# List of methods to be used for feature extraction and selection: all experimental methods have
# been left in, however the best performing method has been currently selected.
# EXTRACT_METHODS = [None, 'TF', 'TF-IDF'] # Will not work for all for final version.
SELECT_METHODS = ['TF-IDF', 'NVAR', 'chi-square']   

EXTRACT_METHOD = None
SELECT_METHOD = SELECT_METHODS[2]

def parse_args():
    parser=argparse.ArgumentParser(description="A Naive Bayes Sentiment Analyser for the Rotten Tomatoes Movie Reviews dataset")
    parser.add_argument("training")
    parser.add_argument("dev")
    parser.add_argument("test")
    parser.add_argument("-classes", type=int)
    parser.add_argument('-features', type=str, default="all_words", choices=["all_words", "features"])
    parser.add_argument('-output_files', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('-confusion_matrix', action=argparse.BooleanOptionalAction, default=False)
    args=parser.parse_args()
    return args

def main():
    
    inputs=parse_args()
    
    #input files
    training = inputs.training
    dev = inputs.dev
    test = inputs.test
    
    #number of classes
    number_classes = inputs.classes
    
    #accepted values "features" to use your features or "all_words" to use all words (default = all_words)
    features = inputs.features
    
    #whether to save the predictions for dev and test on files (default = no files)
    output_files = inputs.output_files
     
    #whether to print confusion matrix (default = no confusion matrix)
    confusion_matrix = inputs.confusion_matrix
    
    utils = Utils(EXTRACT_METHOD, SELECT_METHOD, number_classes, features, training, dev, test)

    # Load the data from the tsv files, and apply preprocessing steps
    training_data, dev_data, test_data = utils.process_data()

    # Train the classifier
    model = Classifier(number_classes, features)
    model.train(training_data)

    # Get the predictions for the dev set
    dev_predictions = model.predict(dev_data)

    # Get the predictions for the test set
    test_predictions = model.predict(test_data)
    
    # Get the macro-F1 score for the dev set
    f1_score = utils.calculate_macro_f1(dev_data, dev_predictions)

    # Save the predictions to files
    if output_files:
        utils.save_predictions(dev_predictions, test_predictions)

    # Generate the confusion matrix
    if confusion_matrix:
        utils.generate_heatmap(dev_data, dev_predictions)

    print("Number of classes\tFeatures\tmacro-F1(dev)")
    print("%d\t%s\t%f" % (number_classes, features, f1_score))

if __name__ == "__main__":
    main()