# Sentiment Analysis of Movie Reviews

This project was completed as an assignment for the module titled 'Text Processing'. The assignment description was as follows:

>The aim of this project is to implement a multinomial Naive Bayes model for a sentiment analysis task using the Rotten Tomatoes movie review dataset. This dataset is derived from the "Sentiment Analysis on Movie Reviews" [Kaggle competition](https://www.kaggle.com/competitions/sentiment-analysis-on-movie-reviews/overview), that uses data from the works of [Pang and Lee](https://aclanthology.org/P05-1015) and [Socher at al.](https://aclanthology.org/D13-1170). Obstacles like sentence negation, sarcasm, terseness, language ambiguity, and many others make this task very challenging.

The success of the project was based on the improvements made over a majority class baseline. Below, the confusion matrices for a majority class baseline can be seen. One for a 3-class classification (positive, neutral, negative), one for a 5-class (very positive, positive, neutral, negative, very negative).
<p align="center">
  <img src="https://github.com/zorbzers/sa-rottentomatoes/blob/master/images/cf_baseline.png"/>
</p>

Here, the confusion matrices can be seen for the final model; split into using all words, and using selected features.
<p align="center">
  <img src="https://github.com/zorbzers/sa-rottentomatoes/blob/master/images/cf_final.png"/>
</p>


##### Majority Class Baseline Macro-f1 Scores
<table align="center">
    <thead style="background-color: #f0f0f0;">
        <tr>
            <th>Configuration</th>
            <th>Macro-f1 Score</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td style="background-color: #f0f0f0;"><strong>3-class</strong></td>
            <td>0.485913</td>
        </tr>
        <tr>
            <td style="background-color: #f0f0f0;"><strong>5-class</strong></td>
            <td>0.284448</td>
        </tr>
    </tbody>
</table>

##### Final Implementation Macro-f1 Scores and % Improvement
<table align="center">
    <thead style="background-color: #f0f0f0;">
        <tr>
            <th></th>
            <th>Macro-f1 Score</th>
            <th>Improvement From Baseline(%)</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td style="background-color: #f0f0f0;"><strong>3-class (features)</strong></td>
            <td>0.536483</td>
            <td>10.41%</td>
        </tr>
        <tr>
            <td style="background-color: #f0f0f0;"><strong>5-class (features)</strong></td>
            <td>0.369824</td>
            <td>30.01%</td>
        </tr>
        <tr>
            <td style="background-color: #f0f0f0;"><strong>3-class (all words)</strong></td>
            <td>0.527507</td>
            <td>8.56%</td>
        </tr>
        <tr>
            <td style="background-color: #f0f0f0;"><strong>5-class (all words)</strong></td>
            <td>0.345890</td>
            <td>21.60%</td>
        </tr>
    </tbody>
</table>

<hr>

## Implementation Details
The full submitted project report can be viewed [here](submission_report.pdf).

#### Classifier
The project implements a multinomial Naive Bayes classifier with Laplace smoothing from scratch, based on the decision below:

$$ s^* = \underset{s_i}{argmax} \ p(s_i) \prod_{j=1}^{N} p(t_j|s_i) $$

$$ p(t_j|s_i) = \frac{count(t_j,s_i) + \alpha}{(\sum_{f}count(t_f,s_i)) + \alpha|V|} $$

Here, $`\alpha > 0`$ is the Laplace smoothing parameter (typically 1).

#### Preprocessing
A variety of preprocessing methods are fully implemented, including:
- Lowercasing
- Stoplisting
- Expanding abbreviations
- Negation
- Binarisation
- Stemming
- Lemmatization

The best models, however, do not use all methods in unison. Through empirical testing optimal combinations of preprocessing methods where identified (see the [full report](submission_report.pdf) and [this section](#analysis-and-optimisations) for more).

#### Feature Selection
Three methods are fully implemented:
- Chi<sup>2</sup>
- TF-IDF
- Selecting nouns, verbs, adjectives and adverbs

Here, Chi<sup>2</sup> was selected as the most optimal method (again, see the [full report](submission_report.pdf) and [this section](#analysis-and-optimisations) for more).

<hr>

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

<hr>

## Analysis and Optimisations
The [full report](submission_report.pdf) contains the critical points to be made about the analysis and optimisations conducted for this project, however, the report was restricted to a page limit. In this section, full details of the empirical testing conducted will be included.


#### Feature selection: analysis of the three methods
As mentioned, three methods were chosen to compare:

1. NVAR (Nouns, verbs, adjectives and adverbs)
    - As in the name, this involves selecting all nouns, verbs, adjectives and adverbs as features to be used.
    - Initially, this was chosen through the comparison of all combinations of nouns, verbs, adjectives and/or adverbs as a method. Despite this comparison, the use of all 4 in conjunction yielded the best results.
2. TF-IDF
    - TF-IDF was selected for it's ability to capture the relevance of a term within a phrase.
    - The TF-IDF of a given term in a given document is calculated as:
        $$ \textit{TF-IDF}_{(t,d)} = TF_{(t,d)} \times \log{\frac{|D|}{DF_{t}}} $$ 
    Where TF(t,d) is the frequency of term t in document d. D is the document set, and DF<sub>t</sub> is the number of documents containing term t.
3. Chi<sup>2</sup>
    - Chi<sup>2</sup> was selected due to its effectiveness in assessing the independence between features and the target class.
    - The Chi<sup>2</sup> statistic for a feature is given by:

    $$ \chi^2 = \sum \frac{(O_{ij} - E_{ij})^2}{E_{ij}} $$

    Here O<sub>ij</sub> is the observed frequency for a feature and class, and E<sub>ij</sub> is the expected frequency.

All three methods were compared when using just lowercasing for preprocessing, and $`\alpha = 1`$. The table below shows the performance of each under these conditions:

<!-- <img src="images\initial-f1-results.png" width="400"/> -->

<table align="center">
    <thead style="background-color: #f0f0f0;">
        <tr>
            <th>Feature Selection</th>
            <th>TF-IDF</th>
            <th>NVAR</th>
            <th>Chi<sup>2</sup></th>
        </tr>
        <tr>
            <th>Preprocessing</th>
            <th>LC</th>
            <th>LC</th>
            <th>LC</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td style="background-color: #f0f0f0;"><strong>3-class (features)</strong></td>
            <td>0.509871</td>
            <td>0.510401</td>
            <td>0.507740</td>
        </tr>
        <tr>
            <td style="background-color: #f0f0f0;"><strong>5-class (features)</strong></td>
            <td>0.336089</td>
            <td>0.313913</td>
            <td>0.337628</td>
        </tr>
        <tr>
            <td style="background-color: #f0f0f0;"><strong>3-class (all words)</strong></td>
            <td>0.487747</td>
            <td>0.487747</td>
            <td>0.487747</td>
        </tr>
        <tr>
            <td style="background-color: #f0f0f0;"><strong>5-class (all words)</strong></td>
            <td>0.297355</td>
            <td>0.297355</td>
            <td>0.297355</td>
        </tr>
    </tbody>
</table>

Overall, at this point, all 3 methods performed similarly. So, all 3 were used in the next section, in hopes of creating a larger disparity between the methods.

<hr>

#### Preprocessing: finding the optimal combination for each feature selection method
Through empirical testing, all 3 methods were tested with all possible combinations of preprocessing method(s). Typically, the Laplace constant is 1, so during this testing, this was kept constant due to time constraints on the assignment. The 20 best-performing combinations are displayed for each method below, for both 3-class and 5-class.

<h5 align="center">Top Combinations for NVAR</h5>
<p align="center"><img src="images\optimal-pre-nvar.png" width="600"/></p>
<hr>
<h5 align="center">Top Combinations for TF-IDF</h5>
<p align="center"><img src="images\optimal-pre-tfidf.png" width="600"/></p>
<hr>
<h5 align="center">Top Combinations for Chi<sup>2</sup></h5>
<p align="center"><img src="images\optimal-pre-chi.png" width="600"/></p>
<hr>

##### Overall Comparison
The table below shows a comparison between each method using their most optimal preprocessing configuration.
<table align="center">
    <thead>
        <tr style="background-color: #f0f0f0;">
            <th>Feature Selection</th>
            <th>TF-IDF</th>
            <th>NVAR</th>
            <th>Chi<sup>2</sup></th>
        </tr>
    </thead>
    <tbody>
        <tr style="background-color: #f0f0f0; font-weight: bold;">
            <td>Preprocessing</td>
            <td>Optimal</td>
            <td>Optimal</td>
            <td>Optimal</td>
        </tr>
        <tr style="background-color: #f0f0f0; font-weight: bold;">
            <td>Laplace Constant</td>
            <td>1</td>
            <td>1</td>
            <td>1</td>
        </tr>
        <tr>
            <td style="background-color: #f0f0f0;"><strong>3-class (features)</strong></td>
            <td>0.527507</td>
            <td>0.515927</td>
            <td>0.536483</td>
        </tr>
        <tr>
            <td style="background-color: #f0f0f0;"><strong>5-class (features)</strong></td>
            <td>0.345227</td>
            <td>0.336637</td>
            <td>0.359535<br></td>
        </tr>
        <tr>
            <td style="background-color: #f0f0f0;"><strong>3-class (all words)</strong></td>
            <td>0.524420</td>
            <td>0.504836</td>
            <td>0.503334</td>
        </tr>
        <tr>
            <td style="background-color: #f0f0f0;"><strong>5-class (all words)</strong></td>
            <td>0.333311</td>
            <td>0.324886</td>
            <td>0.307602</td>
        </tr>
    </tbody>
</table>

It's much clearer now that TF-IDF and Chi<sup>2</sup> perform better than NVAR. At this point, Chi<sup>2</sup> was chosen as the focus. Although TF-IDF outperforms it when feature selection is not applied, the better competitor when using feature selection was chosen as it was deemed more valuable.
<hr>

#### Laplace Parameter: finding the optimal smoothing parameter
Finally, testing was conducted to find the optimal smoothing parameter for Laplace when using Chi<sup>2</sup>. The plots below show this for 3-class and 5-class, both with and without feature selection.

<p align="center"><img src="images\laplace-opti.png" width="600"/></p>

An interesting observation is that the optimal value was close to 1 as expected when feature selection is applied, however, when all words are used, it was much closer to 0.4. This could be because feature selection reduces the number of features, so there is more need for smoothing to handle the reduced feature space effectively. In contrast, using all words increases the feature space, which might require less smoothing to achieve optimal performance.