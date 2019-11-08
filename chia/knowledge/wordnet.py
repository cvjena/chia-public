from nltk.corpus import wordnet

from . import RelationSource


class WordNetAccess(RelationSource):
    def get_hypernyms(self, synset):
        return {
            f"WordNet3.0::{hsynset.name()}"
            for hsynset in wordnet.synset(synset).hypernyms()
        }

    def get_hyponyms(self, synset):
        return {
            f"WordNet3.0::{hsynset.name()}"
            for hsynset in wordnet.synset(synset).hyponyms()
        }

    def get_right_for(self, uid_left):
        if uid_left.startswith("WordNet3.0::"):
            return self.get_hypernyms(uid_left[12:])
        else:
            return set()

    def get_left_for(self, uid_right):
        if uid_right.startswith("WordNet3.0::"):
            return self.get_hyponyms(uid_right[12:])
        else:
            return set()
