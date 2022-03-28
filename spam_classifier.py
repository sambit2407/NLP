from Application_logging import App_Logger
import pandas as pd
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


class SpamClassifier:
    def __init__(self):
        self.log_writer = App_Logger()
        self.file = open('spam_classifier_log', '+a')
        self.data = pd.read_excel('spamdata.xlsx', names=['target', 'messages'])
        self.log_writer.log(self.file, 'Data Loaded successfully')
        # self.log_writer.log(self.file, f'Data Loaded successfully{self.data.info()}')
        # self.log_writer.log(self.file, f'Data columns : {len(self.data)}')

    def cleaning_text(self):
        lammet = WordNetLemmatizer()
        self.corpus = []
        try:
            for i in range(0, len(self.data)):
                cln_data = re.sub('[^a-zA-Z]', ' ', str(self.data['messages'][i]))
                cln_data = cln_data.lower()
                cln_data = cln_data.split()
                cln_data = [lammet.lemmatize(word) for word in cln_data if word not in stopwords.words('english')]
                cln_data = ' '.join(cln_data)
                self.corpus.append(cln_data)
            self.log_writer.log(self.file, 'Data cleaned  successfully!!')

        except Exception as e:
            self.log_writer.log(self.file, 'Error occured in cleaning method : %s' % e)
            raise e

    def tfid(self):
        tf = TfidfVectorizer()
        self.x = tf.fit_transform(self.corpus).toarray()
        self.log_writer.log(self.file, 'TFID Vectorization Done !!')
        # self.log_writer.log(self.file, f'Error occured in cleaning method : {x.shape}')

    def train_test_split(self):
        y = pd.get_dummies(self.data['target'])
        self.y = y.iloc[:, 1].values
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=0.20,
                                                                                random_state=0)
        self.log_writer.log(self.file, 'train test split done !!')

    def naive_bayes(self):
        spam_model = MultinomialNB().fit(self.x_train, self.y_train)
        self.y_pred = spam_model.predict(self.x_test)
        # self.log_writer.log(self.file, 'testing ......')
        # self.log_writer.log(self.file, f' y_test  : {self.y_test}')
        # self.log_writer.log(self.file, f'y_pred : {y_pred}')

    def testing_model(self):
        self.log_writer.log(self.file, 'Testing Started .....')
        confusion_mat = confusion_matrix(self.y_test, self.y_pred)
        acc_scr = accuracy_score(self.y_test, self.y_pred)
        self.log_writer.log(self.file, f'Confusion matrix : {confusion_mat}')
        self.log_writer.log(self.file, f'Accuracy Score  : {acc_scr}')
        self.log_writer.log(self.file, 'Testing Ended .....')


sc = SpamClassifier()
sc.cleaning_text()
sc.tfid()
sc.train_test_split()
sc.naive_bayes()
sc.testing_model()
