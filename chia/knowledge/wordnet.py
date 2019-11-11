from nltk.corpus import wordnet

from . import RelationSource
from chia.framework import caching


class WordNetAccess(RelationSource):
    def _get_hypernyms(self, synset):
        return {
            f"WordNet3.0::{hsynset.name()}"
            for hsynset in wordnet.synset(synset).hypernyms()
        }

    def _get_hyponyms(self, synset):
        return {
            f"WordNet3.0::{hsynset.name()}"
            for hsynset in wordnet.synset(synset).hyponyms()
        }

    @caching.read_only_for_positional_args
    def get_right_for(self, uid_left):
        if uid_left.startswith("WordNet3.0::"):
            return self._get_hypernyms(uid_left[12:])
        else:
            return set()

    @caching.read_only_for_positional_args
    def get_left_for(self, uid_right):
        if uid_right.startswith("WordNet3.0::"):
            return self._get_hyponyms(uid_right[12:])
        else:
            return set()
