"""Tokenization - creating lists of sentences and words from a paragraphs"""
import nltk


# nltk.download()

class Tokenization:
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

para='In Harry’s world fate works not only through powers and objects such as prophecies, the Sorting Hat, wands, ' \
     'and the Goblet of Fire, but also through people. Repeatedly, other characters decide Harry’s future for him, ' \
     'depriving him of freedom and choice. For example, before his eleventh birthday, the Dursleys control Harry’s ' \
     'life, keeping from him knowledge of his past and understanding of his identity (Sorcerer’s 49). In Harry Potter ' \
     'and the Chamber of Secrets, Dobby repeatedly assumes control over events by intercepting Ron’s and Hermione’s ' \
     'letters during the summer; by sealing the barrier to Platform 93⁄4, causing Harry to miss the Hogwarts Express; ' \
     'and by sending a Bludger after Harry in a Quidditch match, breaking his wrist. Yet again, in Harry Potter and ' \
     'the Prisoner of Azkaban, many adults intercede while attempting to protect Harry from perceived danger, ' \
     'as Snape observes: “Everyone from the Minister of Magic downward has been trying to keep famous Harry Potter ' \
     'safe from Sirius Black” (284). All these characters, as enactors of fate, unknowingly drive Harry toward his ' \
     'destiny by attempting to control or to direct his life, while themselves controlled and directed by fate. '
token=Tokenization(para)
token.tokens()
token.length()