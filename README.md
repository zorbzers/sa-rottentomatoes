# Sentiment Analysis of Movie Reviews

This project was completed as an assignment for the module titled 'Text Processing'. The assignment description was as follows:

>The aim of this project is to implement a multinomial Naive Bayes model for a sentiment analysis task using the Rotten Tomatoes movie review dataset. This dataset is derived from the "Sentiment Analysis on Movie Reviews" [Kaggle competition](https://www.kaggle.com/competitions/sentiment-analysis-on-movie-reviews/overview), that uses data from the works of [Pang and Lee](https://aclanthology.org/P05-1015) and [Socher at al.](https://aclanthology.org/D13-1170). Obstacles like sentence negation, sarcasm, terseness, language ambiguity, and many others make this task very challenging.

The success of the project was based on the improvements made over a majority class baseline. Below, the confusion matrices for a majority class baseline can be seen. One for a 3-class classification (positive, neutral, negative), one for a 5-class (very positive, positive, neutral, negative, very negative).

![image](images\cf_baseline.png)

Here, the confusion matrices can be seen for the final model; split into using all words, and using selected features.
![image](images\cf_final.png)

##### Majority Class Baseline Macro-f1 Scores
| Configuration | Macro-f1 Score |
| - | - |
| 3-class | 0.485913 | 
| 5-class | 0.284448 |

##### Final Implementation Macro-f1 Scores and % Improvement
| | Macro-f1 Score | Improvement from baseline(%) |
| - | - | - |
| 3-class (features) | 0.536483 | 10.41% |
| 5-class (features) | 0.369824 | 30.01% |
| 3-class (all-words) | 0.527507 | 8.56% |
| 5-class (all-words) | 0.345890 | 21.60% |



## Implementation Details
The full submitted project report can be viewed [here](submission_report.pdf).

#### Classifier
The project implements a multinomial Naive Bayes classifier with Laplace smoothing from scratch, based on the decision below:

\[ s^* = \underset{s_i}{argmax} \ p(s_i) \prod_{j=1}^{N} p(t_j|s_i) \]
\[ p(t_j|s_i) = \frac{count(t_j,s_i) + \alpha}{(\sum_{f}count(t_f,s_i)) + \alpha|V|} \]

Here, \(\alpha > 0\) is the Laplace smoothing parameter (typically 1).

#### Preprocessing
A variety of preprocessing methods are fully implemented, including:
- Lowercasing
- Stoplisting
- Expanding abbreviations
- Negation
- Binarisation
- Stemming
- Lemmatization

The best models, however, do not use all methods in unison. Through empirical testing optimal combinations of preprocessing methods where identified (see full report for more).

#### Feature Selection
Three methods are fully implemented:
- Chi<sup>2</sup>
- TF-IDF
- Selecting nouns, verbs, adjectives and adverbs

Here, Chi<sup>2</sup> was selected as the most optimal method.

## Usage
#### Requirements
- Python v3.11.7
- Pandas v2.1.4
    - Used for loading the .tsv file data.
- Seaborn v0.13.0
    - Used for rendering the confusion matrices.
- Matplotlib v3.8.2
    - Used for displaying the confusion matrices.
- NLTK v3.8.1
    - Used to implement NVAR using wordnet.
    - Used to implement stemming and lemmatisation preprocessing methods.
- Scikit-learn v1.3.2
    - Used to implement Chi-squared feature selection, with CountVectorizer, SelectKBest and chi2.
    - Was strictly not used for any implementation regarding the classifier.

#### Cloning the Repository
To clone this repository to run the code locally, run the following command in the directory you wish the code to be in:
```sh
git clone <repo url>
```

#### Creating a Virtual Environment
While not necessary, it is recommended to create a virtual environment:
1. Create
    ```sh
    python -m venv venv
    ```

2. Activate
    - On Windows systems:
        ```sh
        .\venv\Scripts\activate
        ```
    - On Unix systems:
        ```sh
        source venv/bin/activate
        ```

#### Install Requirements
```sh
pip install -r requirements.txt
```

#### Run instructions
- To run the application:
`python NB_sentiment_analyser.py <TRAINING_FILE> <DEV_FILE> <TEST_FILE> -classes <NUMBER_CLASSES> -features <all_words,features> -confusion_matrix -output_files`
- Where:
    - <TRAINING_FILE> <DEV_FILE> <TEST_FILE> are the paths to the training, dev and test files, respectively;
    - -classes <NUMBER_CLASSES> should be either 3 or 5, i.e. the number of classes being predicted;
    - -features is a parameter to define whether to use features or all words as features;
    - -output_files is an optional value defining whether or not the prediction files should be saved;
    - -confusion_matrix is an optional value defining whether confusion matrices should be shown.

- Alternatively, a script `auto_results.py` is included to run all 4 configurations.