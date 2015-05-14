import xlrd
import nltk
import string
import random
import gensim
import math
import re
import random
from simplejson import loads
from nltk.corpus import stopwords
from sklearn import svm
from sklearn import linear_model
from sklearn.metrics import precision_recall_fscore_support
from collections import defaultdict
from itertools import compress
from itertools import chain
from stanford_corenlp_pywrapper import sockwrap
import cPickle as pickle

class Classifier:
    
    def __init__(self):
        random.seed(3)
        excel_fpath = 'data.xls'
        self.se_raw_sentences = []
        self.comment_raw_sentences = []
        #get all side effect and comment sentences
        self.nonlemma_side_effects,self.nonlemma_comments = self.initialize_dataset(excel_fpath)
        #self.swrap = sockwrap.SockWrap("pos",corenlp_jars=["../stanford-corenlp-python/stanford-corenlp-full-2015-04-20/stanford-corenlp-3.5.2.jar",
        #  "../stanford-corenlp-python/stanford-corenlp-full-2015-04-20/stanford-corenlp-3.5.2-models.jar"])
        #self.se_lemmatized = self.lemmatize_sentences(self.se_raw_sentences,self.side_effects,swrap,1)
        #self.comment_lemmatized = self.lemmatize_sentences(self.comment_raw_sentences,self.comments,swrap,0)
        #pickle.dump(self.se_lemmatized,open('se_lemmatized.pkl','wb'))
        #pickle.dump(self.comment_lemmatized,open('comment_lemmatized.pkl','wb'))
        #print "Finished lemmatizing"
        self.se_lemmatized = pickle.load(open('./se_lemmatized.pkl','rb'))
        self.comment_lemmatized = pickle.load(open('./comment_lemmatized.pkl','rb'))
        self.side_effects = self.se_lemmatized
        self.comments = self.comment_lemmatized
        #downsample positive samples to create a balanced dataset
        self.side_effects = random.sample(self.side_effects,len(self.comments))
        #9000 sentences
        self.num_se_sentences = len(self.side_effects)
        #6198 sentences
        self.num_comment_sentences = len(self.comments)
        self.total_num_sentences = self.num_se_sentences+self.num_comment_sentences
        #interleave comments with side effects to create a well balanced set
        self.interleaved_sentences,self.raw_interleaved,self.nonlemma_interleaved = self.interleave_sentences()
        #self.write_sentences_to_file(self.raw_interleaved)
        #load dependency parses for each sentence
        #self.dep_parses = pickle.load(open('./dep_parse_final.pkl','rb'))
        #self.lemmatized_deps = self.lemmatize_dependencies(self.swrap,self.dep_parses)
        #pickle.dump(self.lemmatized_deps,open('lemmatized_deps.pkl','wb'))
        self.dep_parses = pickle.load(open('./lemmatized_deps.pkl','rb'))
        self.folds = self.create_k_folds(12,self.interleaved_sentences)
        self.dep_parse_folds = self.create_k_folds(12,self.dep_parses)
        self.non_lemma_folds = self.create_k_folds(12,self.nonlemma_interleaved)
        self.stops = set(stopwords.words('english'))
        #set of scraped side effects from SIDER
        self.sider_feature_set = set(self.initialize_se_list())
        self.scraped_se_length = len(self.sider_feature_set)
        self.MI = defaultdict(float)
        self.bigram_MI = defaultdict(float)
        self.perform_cross_validation(self.folds)   
    
    def write_sentences_to_file(self,sentences):
        f = open('raw_sen_out.txt','w')
        for sentence in sentences:
            s = filter(lambda x: x in string.printable, sentence)
            f.write("%s\n" % s)
        print "finished writing"

    def lemmatize_sentences(self,sentences,tokenized,swrap,label):
        lemmatized_sentences = []
        for k,sen in enumerate(sentences):
            sentence = sen.lower()
            print k
            try:
                sen_parses = swrap.parse_doc(sentence)['sentences']
                sen_lemmas = []
                for parse in sen_parses:
                    lemmas = parse['lemmas'] 
                    sen_lemmas.extend(lemmas) 
                lemmatized_sentences.append([sen_lemmas,label])
            except:
                lemmatized_sentences.append(tokenized[k])
        return lemmatized_sentences
    
    def lemmatize_dependencies(self,swrap,dep_parses):
        lemmatized_deps = []
        print len(dep_parses)
        for k,sen_deps in enumerate(dep_parses):
            print k
            new_deps = []
            if sen_deps:
                for dep in sen_deps:
                    w1_parse = swrap.parse_doc(dep[1])['sentences']
                    w2_parse = swrap.parse_doc(dep[2])['sentences']
                    if w1_parse:
                        w1_lemma = w1_parse[0]['lemmas'][0]
                    else:
                        w1_lemma = dep[1]
                    if w2_parse:
                        w2_lemma = w2_parse[0]['lemmas'][0]
                    else:
                        w2_lemma = dep[2]
                    new_dep = (dep[0],w1_lemma,w2_lemma)
                    new_deps.append(new_dep)
            lemmatized_deps.append(new_deps)
        return lemmatized_deps 

    def compute_POS_freq(self):
        se_tag_freq = defaultdict(float)
        comment_tag_freq = defaultdict(float)
        print "Working on positive examples"
        for sen_pair in self.side_effects[:10]:
            se_sen = sen_pair[0]
            tagged_sen = nltk.pos_tag(se_sen)
            for tag_tup in tagged_sen:
                se_tag_freq[tag_tup[1]] += 1
        print "Working on negative examples"
        for sen_pair in self.comments[:10]:
            comment = sen_pair[0]
            tagged_sen = nltk.pos_tag(se_sen) 
            for tag_tup in tagged_sen:
                comment_tag_freq[tag_tup[1]] += 1
        se_output = open('se.pkl','wb')
        comment_output = open('comment.pkl','wb')
        pickle.dump(sorted(se_tag_freq,reverse=True),se_output)
        pickle.dump(sorted(comment_tag_freq,reverse=True),comment_output)

    def interleave_sentences(self):
        interleaved = []
        raw_interleaved = []
        nonlemma_interleaved = []
        i = 0
        while i < self.num_comment_sentences:
            interleaved.extend([self.comments[i],self.side_effects[i]])
            raw_interleaved.extend([self.comment_raw_sentences[i],self.se_raw_sentences[i]])
            nonlemma_interleaved.extend([self.nonlemma_comments[i],self.nonlemma_side_effects[i]])
            i += 1
        while i < self.num_se_sentences:
            interleaved.extend([self.side_effects[i]])
            raw_interleaved.extend([self.se_raw_sentences[i]])
            nonlemma_interleaved.extend([self.nonlemma_side_effects[i]])
            i += 1
        return interleaved,raw_interleaved,nonlemma_interleaved

    def initialize_dataset(self,fpath):
        workbook = xlrd.open_workbook(fpath)
        sheet_name = workbook.sheet_names()[0]
        worksheet = workbook.sheet_by_name(sheet_name)
        side_effects = []
        comments = []
        num_rows = worksheet.nrows
        for row in range(1,num_rows):
            self.se_raw_sentences.append(worksheet.cell_value(row,0))
            se_split = [word.lower() for word in re.findall(r"\w+\'*\w*", worksheet.cell_value(row,0))]
            comment_split = [word.lower() for word in re.findall(r"\w+\'*\w*", worksheet.cell_value(row,1))]
            side_effects.append([se_split,1])
            #fewer comments than side effects
            if comment_split:
                self.comment_raw_sentences.append(worksheet.cell_value(row,1))
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
    
    def perform_cross_validation(self,folds):
        num_folds = len(folds)
        indices = range(0,num_folds)
        avg_fscore = 0.
        for hold_out in indices:
            #if hold_out == num_folds-1:
            #    continue
            mask = [1]*num_folds
            mask[hold_out] = 0
            self.side_effect_freq = defaultdict(float) #10957 words
            self.comment_freq = defaultdict(float) #15174 words
            self.all_words = defaultdict(float)
            self.total_word_count = 0.
            training = list(compress(folds,mask))
            training = list(chain.from_iterable(training))
            training_deps = list(compress(self.dep_parse_folds,mask))
            training_deps = list(chain.from_iterable(training_deps))
            #training_deps = self.lemmatize_dependencies(self.swrap,training_deps)
            self.training_se = [train_data for train_data in training if train_data[1] == 1]
            self.training_comments = [train_data for train_data in training if train_data[1] == 0]
            self.training_se_deps = [deps for k,deps in enumerate(training_deps) if training[k][1] == 1]
            self.training_comments_deps = [deps for k,deps in enumerate(training_deps) if training[k][1] == 0]
            self.training_all = training 
            self.compute_unigram_freq(self.training_se, self.training_comments)
            self.compute_bigram_freq(self.training_se,self.training_comments)
            self.compute_trigram_freq(self.training_se,self.training_comments)
            self.compute_dependency_freq(training_deps)
            self.test_sentences = folds[hold_out]
            self.test_deps = self.dep_parse_folds[hold_out]
            self.feature_word_set = self.compute_MI(self.training_se,self.training_comments,
                self.all_words,self.total_word_count,self.compute_conditionals,200)
            self.bigram_feature_set = self.compute_MI(self.training_se,self.training_comments,
                self.bigrams,self.total_bigram_count,self.compute_bigram_conditionals,300)
            self.dep_feature_set = self.compute_MI(self.training_se_deps,self.training_comments_deps,
                self.dep_freq,self.total_dep_count,self.compute_dep_conditional,200)
            self.trigram_feature_set = self.compute_MI(self.training_se,self.training_comments,
                self.trigrams,self.total_trigram_count,self.compute_trigram_conditionals,100)
            #print self.feature_word_set
            #print self.bigram_feature_set
            #print self.dep_feature_set
            #print self.feature_word_set
            #self.feature_word_set = self.feature_word_set.union(self.sider_feature_set)
            self.feature_ind = defaultdict(int)
            self.feature_ind  = self.build_feature_set(self.feature_word_set,self.bigram_feature_set,self.trigram_feature_set,self.dep_feature_set)
            self.total_features = len(self.feature_word_set)+len(self.bigram_feature_set)+len(self.trigram_feature_set)+len(self.dep_feature_set)
            #self.feature_ind = self.build_unigram_feature_set(self.feature_word_set)
            self.samples, self.labels = self.create_empty_samples_labels(self.total_features,len(self.training_all))
            self.update_unigram_features(self.training_all,self.feature_ind)
            self.update_bigram_features(self.training_all,self.feature_ind)
            self.update_trigram_features(self.training_all,self.feature_ind)
            self.update_dep_features(training_deps,self.feature_ind)
            #self.samples,self.labels = self.create_samples_labels(self.training_se+self.training_comments,self.feature_ind,self.total_features)
            fscore = self.run()
            avg_fscore += fscore
        #downsampling so don't include last fold in the average
        print "Average F-score: {}".format(avg_fscore/num_folds)
    
    def create_empty_samples_labels(self,num_features,num_training_examples):
        labels = [0]*num_training_examples
        samples = []
        for i in range(num_training_examples):
            samples.append([0]*num_features)
        return samples,labels
    
    def update_unigram_features(self,training_data,feature_ind):
        for k,sen_pair in enumerate(training_data):
            sen = sen_pair[0]
            label = sen_pair[1]
            curr_sample = self.samples[k] 
            for word in sen:
                if word in feature_ind:
                    feature_index = feature_ind[word]
                    curr_sample[feature_index] = 1
            self.labels[k] = label
    
    def update_bigram_features(self,training_data,feature_ind):
        for k,sen_pair in enumerate(training_data):
            sen = sen_pair[0]
            label = sen_pair[1]
            curr_sample = self.samples[k] 
            for j,word in enumerate(sen):
                if j==0:
                    continue
                else:
                    if (sen[j-1],word) in self.bigrams:  
                        feature_index = feature_ind[(sen[j-1],word)]
                        curr_sample[feature_index] = 1
    
    def update_trigram_features(self,training_data,feature_ind):
        for k,sen_pair in enumerate(training_data):
            sen = sen_pair[0]
            label = sen_pair[1]
            curr_sample = self.samples[k] 
            for j,word in enumerate(sen):
                if j==0 or j==1:
                    continue
                else:
                    if (sen[j-2],sen[j-1],word) in self.trigrams:  
                        feature_index = feature_ind[(sen[j-2],sen[j-1],word)]
                        curr_sample[feature_index] = 1
    
    def update_dep_features(self,training_data,feature_ind):
        for k,dep_sen in enumerate(training_data):
            curr_sample = self.samples[k]
            if dep_sen:
                for dep in dep_sen:
                  if dep in feature_ind:
                      feature_index = feature_ind[dep] 
                      curr_sample[feature_index] = 1

    def get_dependency_parse(self):
        #dep_freq = defaultdict(float)
        #we assume that stanford corenlp is running at 127.0.0.1:8080  
        swrap = sockwrap.SockWrap("parse",corenlp_jars=["../stanford-corenlp-python/stanford-corenlp-full-2014-08-27/stanford-corenlp-3.4.1.jar",
          "../stanford-corenlp-python/stanford-corenlp-full-2014-08-27/stanford-corenlp-3.4.1-models.jar"])
        sentence_deps = []
        for k,sen in enumerate(self.raw_interleaved):
            se_sen = sen.lower()
            try:
                sen_parses = swrap.parse_doc(se_sen)['sentences']
            except:
                continue
            if sen_parses == None:
                sentence_deps.append(None)
                continue
            deps = []
            for parse in sen_parses:
                parse_deps = parse['deps_basic']
                #deps.extend(parse['deps_basic'])
                offsets = parse['char_offsets']
                words = []
                for offset in offsets:
                    words.append(se_sen[offset[0]:offset[1]])
                for dep in parse_deps:
                    if dep[0] == 'root':
                        deps.append((dep[0],'',words[dep[2]]))
                    else:
                        deps.append((dep[0],words[dep[1]],words[dep[2]]))
            sentence_deps.append(deps)
        pickle.dump(sentence_deps,open('dep_parses.pkl','wb'))

    def compute_bigram_freq(self,side_effects,comments):
        all_sentences = side_effects+comments
        self.bigrams = defaultdict(float)
        self.total_bigram_count = 0.
        for sen_pair in all_sentences:
            sen = sen_pair[0]
            for k,word in enumerate(sen):
                if k==0: 
                    continue
                else:
                    self.bigrams[(sen[k-1],word)] += 1
                    self.total_bigram_count += 1 
        #return bigrams,total_bigram_count
    
    def compute_trigram_freq(self,side_effects,comments):
        all_sentences = side_effects+comments
        self.trigrams = defaultdict(float)
        self.total_trigram_count = 0.
        for sen_pair in all_sentences:
            sen = sen_pair[0]
            for k,word in enumerate(sen):
                if k==0 or k==1:
                    continue
                else:
                    self.trigrams[(sen[k-2],sen[k-1],word)] += 1
                    self.total_trigram_count += 1

    def compute_dependency_freq(self,dep_parses):
        self.dep_freq = defaultdict(float)
        self.total_dep_count = 0.
        for sentence_deps in dep_parses:
            if sentence_deps:
                for dep in sentence_deps:
                    self.dep_freq[dep] += 1
                    self.total_dep_count += 1
    
    def compute_MI(self,training_pos,training_neg,feature_freq,total_feature_count,conditional_fn,num_features):
        se_conditional = conditional_fn(training_pos)
        comment_conditional = conditional_fn(training_neg)
        num_se_sentences = len(self.training_se)
        num_comments = len(self.training_comments)
        class_prob = 0.5
        MI = defaultdict(float)
        for feature in feature_freq.keys():
            p_u1_c1 = se_conditional[feature]/num_se_sentences
            p_u1_c0 = comment_conditional[feature]/num_comments
            p_u0_c1 = 1-p_u1_c1
            p_u0_c0 = 1-p_u1_c0
            p_u1 = feature_freq[feature]/total_feature_count
            p_u0 = 1-p_u1
            mi_u1_c1 = p_u1_c1*class_prob*self.log_wrapper(p_u1_c1/(p_u1*class_prob))
            mi_u1_c0 = p_u1_c0*class_prob*self.log_wrapper(p_u1_c0/(p_u1*class_prob))
            mi_u0_c1 = p_u0_c1*class_prob*self.log_wrapper(p_u0_c1/(p_u0*class_prob))
            mi_u0_c0 = p_u0_c0*class_prob*self.log_wrapper(p_u0_c0/(p_u0*class_prob))
            MI[feature] = mi_u1_c1+mi_u1_c0+mi_u0_c1+mi_u0_c0 
        sorted_features = sorted(MI,key=MI.get,reverse=True)
        return set(sorted_features[:num_features])

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
    
    def compute_bigram_conditionals(self,sentence_label_pairs):
        conditional = defaultdict(float)
        for sen_pair in sentence_label_pairs:
            sen = sen_pair[0]
            for k,word in enumerate(sen):
                if k==0: 
                    continue
                else:
                    conditional[(sen[k-1],word)] += 1
        return conditional
    
    def compute_trigram_conditionals(self,sentence_label_pairs):
        conditional = defaultdict(float)
        for sen_pair in sentence_label_pairs:
            sen = sen_pair[0]
            for k,word in enumerate(sen):
                if k==0 or k==1: 
                    continue
                else:
                    conditional[(sen[k-2],sen[k-1],word)] += 1
        return conditional
    
    def compute_dep_conditional(self,dependencies):
        conditional = defaultdict(float)
        for dep_sen in dependencies:
            if dep_sen:
                for dep in dep_sen:
                    conditional[dep] += 1
        return conditional 

    def build_unigram_feature_set(self, word_set):
        feature_ind = defaultdict(int)
        curr_ind = 0
        for word in word_set:
            feature_ind[word] = curr_ind
            curr_ind += 1
        return feature_ind
    
    def build_feature_set(self,unigram_set,bigram_set,trigram_set,dep_feature_set):
        feature_ind = defaultdict(int)
        curr_ind = 0
        #create unigram features
        for word in unigram_set:
            feature_ind[word] = curr_ind
            curr_ind += 1
        for bigram in bigram_set:
            feature_ind[bigram] = curr_ind
            curr_ind += 1
        for trigram in trigram_set:
            feature_ind[trigram] = curr_ind
            curr_ind += 1
        for dep in dep_feature_set:
            feature_ind[dep] = curr_ind
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
        incorrectly_labeled = open('incorrectly_labeled.txt','a')
        y_true = []
        y_pred = []
        for k,sen_pair in enumerate(self.test_sentences):
            sen = sen_pair[0]
            label = sen_pair[1]
            y_true.append(label)
            feat_vector = [0]*self.total_features
            for word in sen:
                if word in self.feature_word_set:
                    feature_index = self.feature_ind[word]
                    feat_vector[feature_index] = 1
            for j,word in enumerate(sen):
                if j==0:
                    continue
                else:
                    if (sen[j-1],word) in self.bigrams:  
                        feature_index = self.feature_ind[(sen[j-1],word)]
                        feat_vector[feature_index] = 1
            for j,word in enumerate(sen):
                if j==0 or j==1:
                    continue
                else:
                    if (sen[j-2],sen[j-1],word) in self.trigrams:  
                        feature_index = self.feature_ind[(sen[j-2],sen[j-1],word)]
                        feat_vector[feature_index] = 1
            dep_sen = self.test_deps[k]
            if dep_sen:
                for dep in dep_sen:
                    if dep in self.dep_feature_set:
                        feature_index = self.feature_ind[dep] 
                        feat_vector[feature_index] = 1
            pred = self.classifier.predict(feat_vector)[0]
            y_pred.append(pred)
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
            else:
                incorrectly_labeled.write("Classified sentence '{}' with true label {} incorrectly as {}\n".format(sen,label,pred))
        prfs = precision_recall_fscore_support(y_true,y_pred,average='weighted')
        print prfs
        print "Num Wrong: {}".format(len(self.test_sentences)-overall_correct)
        print "Accuracy is {} overall".format(overall_correct/(pos_total+neg_total))
        print "Accuracy is {} for positives".format(pos_correct/pos_total)
        if neg_total > 0:
            print "Accuracy is {} for negatives".format(neg_correct/neg_total)
        return prfs[2]
    
    def run(self):
        print "creating svm"
        self.classifier = svm.LinearSVC()
        #self.classifier = linear_model.LogisticRegression()
        self.classifier.fit(self.samples, self.labels)
        print "evaluating svm"
        fscore = self.evaluate()
        return fscore

if __name__ == "__main__":
    c = Classifier()
    
