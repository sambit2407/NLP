"""Process of reducing infected words to their word stem ex- gone ,goes ,going --->>go"""
"""lemmatization is better than stemming as it can give you proper words than stemming and lemmatization takes more time"""
"""Historical,historic,histoer --->> histori(stemming)
Gone,goes,going -->> go(lemmatization-proper word)"""

"""Application- stemming can be used for sentiment analysis,spam classifier as it does not need proper meaning of base word
Lemmatization can be used for caht bots"""
import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
class NLP:
    def __init__(self, para):
        self.para = para

    def tokens(self):
        self.sentences = nltk.sent_tokenize(self.para)
        print(self.sentences)
        self.words = nltk.word_tokenize(self.para)
        print(self.words)

    def length(self):
        print('sentence length : ',len(self.sentences))
        print('words length : ', len(self.words))

    def stemming(self):
        print('Before stemming : ',self.sentences)
        stem=PorterStemmer()
        for i in range(len(self.sentences)):
            word=nltk.word_tokenize(self.sentences[i])
            stem_words=[stem.stem(w) for w in word if w not in stopwords.words('english')]
            self.sentences[i]=' '.join(stem_words)

        print('After stemming : ',self.sentences)

    def lammetization(self):

        print('Before lammeting : ',self.sentences)
        lammet=WordNetLemmatizer()
        for i in range(len(self.sentences)):
            word = nltk.word_tokenize(self.sentences[i])
            lammet_words = [lammet.lemmatize(w) for w in word if w not in stopwords.words('english')]
            self.sentences[i] = ' '.join(lammet_words)
        print('After lammeting : ',self.sentences)







para='This study was a preliminary study of high school student value changes because of the terrorist attack on the ' \
     'U.S. The major limitations of this study were that the student population was from California and might not ' \
     'truly represent all high school students in the U.S. Further, this study could not be considered a truly ' \
     'longitudinal study because of privacy issues that prevented the researchers from identifying all the students ' \
     'who returned surveys before the attack. In addition, the senior class had graduated the previous year, ' \
     'and a much larger freshman class entered the school. These issues not only made the samples similar, ' \
     'but also different in their composition. The researchers will conduct periodic studies to explore whether these ' \
     'value changes are permanent and continue into adulthood. We do not know what if any changes will take place in ' \
     'their values as they grow older, and we will continue to explore their values in our longitudinal studies of ' \
     'the impact of the 9/11 terrorist attacks. '



nlp=NLP(para)
nlp.tokens()
#nlp.stemming()
nlp.lammetization()