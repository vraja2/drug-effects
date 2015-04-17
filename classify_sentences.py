import xlrd
import nltk
from nltk.corpus import stopwords
from sklearn import svm
from collections import defaultdict
from itertools import compress
from itertools import chain
import random
import gensim
import math
import re

class Classifier:
    
    def __init__(self):
        excel_fpath = 'data.xls'
        #get all side effect and comment sentences
        self.side_effects,self.comments = self.initialize_dataset(excel_fpath)
        #9000 sentences
        self.num_se_sentences = len(self.side_effects)
        #6198 sentences
        self.num_comment_sentences = len(self.comments)
        self.total_num_sentences = self.num_se_sentences+self.num_comment_sentences
        print self.total_num_sentences
        #interleave comments with side effects to create a well balanced set
        self.interleaved_sentences = self.interleave_sentences()
        self.folds = self.create_k_folds(6,self.interleaved_sentences)
        self.stops = set(stopwords.words('english'))
        #set of scraped side effects from SIDER
        self.se_unigrams = set(self.initialize_se_list())
        self.scraped_se_length = len(self.se_unigrams)
        #percent of data used for training
        #training_percent = 0.8
        #self.training_se, self.training_comments, self.all_training, self.test_sentences = self.initialize_traintest_data(training_percent)
        #maps from word to the frequency is occurs in the side effect and comment corpuses
        #self.side_effect_freq = defaultdict(float) #10957 words
        #self.comment_freq = defaultdict(float) #15174 words
        #self.all_words = defaultdict(float)
        #self.total_word_count = 0.
        #self.compute_unigram_freq(self.training_se, self.training_comments)
        #maps from word to index in feature set 
        #self.se_feature_ind = self.build_unigram_feature_set(self.side_effect_freq.keys())
        #self.comment_feature_ind = self.build_unigram_feature_set(self.comment_freq.keys())
        #se_samples,se_labels = self.create_samples_labels(self.training_se,self.se_feature_ind,len(self.side_effect_freq.keys()))
        self.MI = defaultdict(float)
        #to calculate mutual info pass in all the words (features) 
        #self.feature_word_set = self.compute_MI()
        #print self.feature_word_set
        #self.tfidf_se = defaultdict(float)
        #self.tfidf_comments = defaultdict(float)
        #se_features, comment_features = self.build_tfidf(self.training_se, self.training_comments)
        #put together all words that will be used ashttps://docs.google.com/spreadsheets/d/1LbZnzWfwkQcXrwBqBKtIP9dd_0iUt7uSQvN3l_462HU/edit#gid=0 features
        #self.feature_word_set = set(se_features).union(set(comment_features))
        #self.feature_word_set = self.feature_word_set.union(self.se_unigrams)
        #self.feature_ind = defaultdict(int)
        #print "building unigram feature set"
        #assign each feature to a certain index, create a dictionary that maps words to feature positions
        #self.feature_ind = self.build_unigram_feature_set(self.feature_word_set)
        #create feature vectors for each training example and labels as well
        #self.samples,self.labels = self.create_samples_labels(self.training_se+self.training_comments,self.feature_ind,len(self.feature_word_set))
        self.perform_cross_validation(self.folds)
    
    def interleave_sentences(self):
        interleaved = []
        i = 0
        while i < self.num_comment_sentences:
            interleaved.extend([self.comments[i],self.side_effects[i]])
            i += 1
        while i < self.num_se_sentences:
            interleaved.extend([self.side_effects[i]])
            i += 1
        return interleaved

    def initialize_dataset(self,fpath):
        workbook = xlrd.open_workbook(fpath)
        sheet_name = workbook.sheet_names()[0]
        worksheet = workbook.sheet_by_name(sheet_name)
        side_effects = []
        comments = []
        num_rows = worksheet.nrows
        for row in range(1,num_rows):
            se_split = [word.lower() for word in re.findall(r"\w+", worksheet.cell_value(row,0))]
            comment_split = [word.lower() for word in re.findall(r"\w+", worksheet.cell_value(row,1))]
            side_effects.append([se_split,1])
            #fewer comments than side effects
            if comment_split:
                comments.append([comment_split,0])
        return side_effects,comments
    
    def initialize_se_list(self):
        f = open('./scripts/side_effect_scraper/side_effect_scraper/spiders/side_effects.txt','r')
        seffects = f.readlines()
        seffect_set = set()
        for se in seffects:
            if se.lower().rstrip() not in self.stops:
                seffect_set.add(se.lower().rstrip())
        return seffect_set
    
    """def build_unigram_feature_set(self,num_features=1000):
        #maps from feature (specific word) to the index of its binry feature
        self.feature_ind = defaultdict(int)
        curr_ind = 0
        for word in self.feature_word_set:
            self.feature_ind[word] = curr_ind
            curr_ind += 1"""
   
    def perform_cross_validation(self,folds):
        num_folds = len(folds)
        indices = range(0,num_folds)
        for hold_out in indices:
            mask = [1]*num_folds
            mask[hold_out] = 0
            self.side_effect_freq = defaultdict(float) #10957 words
            self.comment_freq = defaultdict(float) #15174 words
            self.all_words = defaultdict(float)
            self.total_word_count = 0.
            training = list(compress(folds,mask))
            training = list(chain.from_iterable(training))
            self.training_se = [train_data for train_data in training if train_data[1] == 1]
            self.training_comments = [train_data for train_data in training if train_data[1] == 0]
            self.compute_unigram_freq(self.training_se, self.training_comments)
            self.test_sentences = folds[hold_out]
            self.feature_word_set = self.compute_MI()
            print self.feature_word_set
            self.feature_ind = defaultdict(int)
            self.feature_ind = self.build_unigram_feature_set(self.feature_word_set)
            self.samples,self.labels = self.create_samples_labels(self.training_se+self.training_comments,self.feature_ind,len(self.feature_word_set))
            self.run()

    def compute_MI(self): 
        se_conditional = self.compute_conditionals(self.training_se)
        comment_conditional = self.compute_conditionals(self.training_comments)
        num_se_sentences = len(self.training_se)
        num_comments = len(self.training_comments)
        #p(C=1) or p(C=0), equally likely to contain side effects or not
        class_prob = 0.5
        for word in self.all_words.keys():
            p_u1_c1 = se_conditional[word]/num_se_sentences
            p_u1_c0 = comment_conditional[word]/num_comments
            p_u0_c1 = 1-p_u1_c1
            p_u0_c0 = 1-p_u1_c0
            p_u1 = self.all_words[word]/self.total_word_count
            p_u0 = 1-p_u1
            mi_u1_c1 = p_u1_c1*class_prob*self.log_wrapper(p_u1_c1/(p_u1*class_prob))
            mi_u1_c0 = p_u1_c0*class_prob*self.log_wrapper(p_u1_c0/(p_u1*class_prob))
            mi_u0_c1 = p_u0_c1*class_prob*self.log_wrapper(p_u0_c1/(p_u0*class_prob))
            mi_u0_c0 = p_u0_c0*class_prob*self.log_wrapper(p_u0_c0/(p_u0*class_prob))
            self.MI[word] = mi_u1_c1+mi_u1_c0+mi_u0_c1+mi_u0_c0 
        sorted_feature_words = sorted(self.MI,key=self.MI.get,reverse=True)
        stop_words_removed = []
        for word in sorted_feature_words:
            if word not in self.stops:
                stop_words_removed.append(word)
        return set(stop_words_removed[:200])

    def log_wrapper(self,num):
        if num <= 0.:
            return 0
        return math.log(num)

    def compute_conditionals(self,sentence_label_pairs):
        conditional = defaultdict(float)
        for pair in sentence_label_pairs:
            sentence = pair[0]
            for word in sentence:
                conditional[word] += 1
        return conditional

    def build_unigram_feature_set(self, word_set):
        feature_ind = defaultdict(int)
        curr_ind = 0
        for word in word_set:
            feature_ind[word] = curr_ind
            curr_ind += 1
        return feature_ind

    def create_samples_labels(self,training_data,feature_ind,num_features=1000):
        samples = []
        labels = []
        for sen_pair in training_data:
            sen = sen_pair[0]
            label = sen_pair[1]
            curr_sample = [0]*num_features
            for word in sen:
                if word in feature_ind:
                    feature_index = feature_ind[word]
                    curr_sample[feature_index] = 1
            samples.append(curr_sample)
            labels.append(label)
        return samples,labels



    """
    def build_unigram_feature_set(self,num_features=1000):
        #maps from feature (specific word) to the index of its binry feature
        self.feature_ind = defaultdict(int)
        curr_ind = 0
        for se in self.se_unigrams:
            self.feature_ind[se] = curr_ind
            curr_ind += 1
        #TODO: fix this later, limiting number of sentences to go through due to too many features from unigram
        for sen_pair in random.sample(self.all_training,2000):
            sen = sen_pair[0]
            label = sen_pair[1]
            for word in sen:
                if curr_ind == num_features:
                    break
                if word not in self.stops and word not in self.feature_word_set:
                    self.feature_word_set.add(word)
                    self.feature_ind[word] = curr_ind
                    curr_ind += 1"""
    """ 
    def create_samples_labels(self,num_features=1000):
        samples = []
        labels = []
        for sen_pair in self.all_training:
            sen = sen_pair[0]
            label = sen_pair[1]
            curr_sample = [0]*num_features
            for word in sen:
                if word in self.feature_word_set:
                    feature_index = self.feature_ind[word]
                    curr_sample[feature_index] = 1
            samples.append(curr_sample)
            labels.append(label)
        return samples,labels
    """
    
    def create_k_folds(self, k, data):
        folds = []
        fold_length = self.total_num_sentences/k
        for i in range(k):
            start = i*fold_length
            end = start+fold_length
            folds.append(data[start:end])
        return folds

    def initialize_traintest_data(self, training_percent):
        #get an equal number of positive and negative (comments) sentences
        num_training_se_sentences = int(training_percent*self.num_se_sentences)
        num_training_comments_sentences = int(training_percent*self.num_comment_sentences)
        training_se = self.side_effects[:num_training_se_sentences]
        training_comments = self.comments[:num_training_comments_sentences]
        testing_se = self.side_effects[num_training_se_sentences:self.num_se_sentences]
        testing_comments = self.comments[num_training_comments_sentences:self.num_comment_sentences]
        self.num_test_comments = len(testing_comments)
        self.num_test_se = len(testing_se)
        training_all = training_se+training_comments
        return training_se,training_comments,training_all,testing_se+testing_comments 

    def initialize_word2vec(self):
        sentences = [sentence_pair[0] for sentence_pair in self.all_training]
        print "Initializing Model"
        model = gensim.models.Word2Vec(sentences, size=100, window=5, min_count=5, workers=4)
        print model.most_similar(positive=['drug'])

    def compute_unigram_freq(self, side_effects, comments):
        print "computing unigram freq"
        for se_pair in side_effects:
            se_sen = se_pair[0]
            for word in se_sen:
                self.all_words[word] += 1
                self.side_effect_freq[word] += 1
                self.total_word_count += 1
        for comment_pair in comments:
            comment_sen = comment_pair[0]
            for word in comment_sen:
                self.all_words[word] += 1
                self.comment_freq[word] += 1
                self.total_word_count += 1
        print len(sorted(self.side_effect_freq,key=self.side_effect_freq.get))
        print len(sorted(self.comment_freq,key=self.comment_freq.get))

    def build_tfidf(self, side_effects, comments):
        print "intiializing tfidf"
        side_effect_freq = defaultdict(float)
        comment_freq = defaultdict(float)
        for se_pair in side_effects:
            se_sen = se_pair[0]
            for word in se_sen:
                side_effect_freq[word] += 1
        for comment_pair in comments:
            comment_sen = comment_pair[0]
            for word in comment_sen:
                comment_freq[word] += 1
        for key,val in side_effect_freq.iteritems():
            tf = val
            idf = math.log(2/(1+(comment_freq[key]>0)))
            self.tfidf_se[key] = tf*idf
        for key,val in comment_freq.iteritems():
            tf = val 
            idf = math.log(2/(1+(side_effect_freq[key]>0)))
            self.tfidf_comments[key] = tf*idf
        return sorted(self.tfidf_comments,key=self.tfidf_comments.get)[:500], sorted(self.tfidf_se,key=self.tfidf_se.get)[:500]

    def evaluate(self):
        print "Running evaluation"
        pos_correct = 0.
        pos_total = 0.
        neg_correct = 0.
        neg_total = 0.
        overall_correct = 0.
        for sen_pair in self.test_sentences:
            sen = sen_pair[0]
            label = sen_pair[1]
            feat_vector = [0]*len(self.feature_word_set)
            for word in sen:
                if word in self.feature_word_set:
                    feature_index = self.feature_ind[word]
                    feat_vector[feature_index] = 1
            pred = self.classifier.predict(feat_vector)[0]
            if label == 1:
                pos_total += 1
            if label == 0:
                neg_total += 1
            if pred == label:
                overall_correct += 1
                if label == 1:
                    pos_correct += 1
                else:
                    neg_correct += 1
        print "Accuracy is {} overall".format(overall_correct/(pos_total+neg_total))
        print "Accuracy is {} for positives".format(pos_correct/pos_total)
        if neg_total > 0:
            print "Accuracy is {} for negatives".format(neg_correct/neg_total)
    
    def run(self):
        print "creating svm"
        self.classifier = svm.SVC()
        self.classifier.fit(self.samples, self.labels)
        print "evaluating svm"
        self.evaluate()

       

if __name__ == "__main__":
    c = Classifier()
    
