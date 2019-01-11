import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)

class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores
        score_bics = []
        """
        Bayesian information criteria: BIC = -2 * logL + p * logN
        where L is the likelihood of the fitted model, p is the number of parameters, and N is the number of data points.
        The term -2 * log L decreases with increasing model complexity (more parameters), 
        whereas the penalties 2*p or p*logN increase with increasing complexity. The BIC applies a larger penalty when N > e ** 2 = 7.4
        
        loop from min_n_components to max_n_components to find the lowest BIC score as the best model
        """
        for num_states in range(self.min_n_components, self.max_n_components + 1):
            try:
                hmm_model = self.base_model(num_states)
                log_likelihood = hmm_model.score(self.X, self.lengths)
                num_data_points = sum(self.lengths)
                num_free_params = ( num_states ** 2 ) + ( 2 * num_states * num_data_points ) - 1
                score_bic = (-2 * log_likelihood) + (num_free_params * np.log(num_data_points))
                score_bics.append((score_bic, hmm_model))
            except:
                continue
        return min(score_bics, key = lambda x: x[0])[1] if score_bics else None


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        """
        X is input training data fiven in the form of a word dictionary
        X(i) is the current word being evaluated
        log(P(X(i)) = log(P(original word))
        1/(M-1)SUM(log(P(X(all but i)) = average of log(P(all except original word))
        
        so the max score the best model 
        
        """

        other_words = []
        models = []
        score_dics = []
        #get 'all except original word' list
        for word in self.words:
            if word != self.this_word:
                other_words.append(self.hwords[word])
        for num_states in range(self.min_n_components, self.max_n_components + 1):
            try:
                hmm_model = self.base_model(num_states)
                log_likelihood_original_word = hmm_model.score(self.X, self.lengths)
                models.append((log_likelihood_original_word, hmm_model))
            except:
                continue

        for model in models:
            log_likelihood_original_word, hmm_model = model
            score_dic = log_likelihood_original_word - np.mean([model[1].score(word[0], word[1]) for word in other_words])
            score_dics.append((score_dic, model[1]))
        return max(score_dics, key = lambda x: x[0])[1] if score_dics else None


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using cross-validation
        """
        
        
        
        find average Log Likelihood of  cross-validation fold
        the max score the best model
        
        """
        kf = KFold()
        log_likelihoods = []
        score_cvs = []

        for num_states in range(self.min_n_components, self.max_n_components + 1):
            try:
           # Check sufficient data to split using KFold. Number of folds. Must be at least 2.
                if len(self.sequences) > 2:
                    for train, test in kf.split(self.sequences):
                        self.X, self.lengths = combine_sequences(train, self.sequences)
                        X_test, lengths_test = combine_sequences(test, self.sequences)

                        hmm_model = self.base_model(num_states)
                        log_likelihood = hmm_model.score(X_test, lengths_test)
                else:
                    hmm_model = self.base_model(num_states)
                    log_likelihood = hmm_model.score(self.X, self.lengths)
    
                log_likelihoods.append(log_likelihood)

                score_cvs_avg = np.mean(log_likelihoods)
                score_cvs.append((score_cvs_avg, hmm_model))
            except:
                continue

        return max(score_cvs, key = lambda x: x[0])[1] if score_cvs else None
