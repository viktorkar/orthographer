from symspellpy import SymSpell, Verbosity

class SymSpellClass:
    def __init__(self, language):
        self._sym_spell = SymSpell(max_dictionary_edit_distance=3, count_threshold=1)
        self._path = '../data/dictionaries/'
        self.language = language
        if self.language == 'de':
            self._sym_spell.load_dictionary(self._path+'de-100k.txt', 0, 1)
        if self.language == 'en':
            self._sym_spell.load_dictionary(self._path+'en-80k.txt', 0, 1)
        if self.language == 'es':
            self._sym_spell.load_dictionary(self._path+'es-100k.txt', 0, 1)
        if self.language == 'fr':
            self._sym_spell.load_dictionary(self._path+'fr-100k.txt', 0, 1)

    ###############################################################################################
    def get_suggestion(self, word):
        """Returns a suggestions for the given word."""
        result = self._sym_spell.lookup(word, Verbosity.TOP, max_edit_distance=3)
        if result:
            return result[0].term
        else:
            return ""

    ###############################################################################################
    def __call__(self, pred_strings):
        """Perform spell correction on the given strings. Returns a list of tuples 
        (IsIncorrectWord, Suggestion)."""
        spellchecks = []

        for word in pred_strings:
            # If word is an empty string or contains numbers, we don't perform lookup.
            if word == "" or any(char.isdigit() for char in word):
                spellchecks.append((False, ""))
                continue
    
            # We perform lookup and look at the top result.
            result = self._sym_spell.lookup(word, Verbosity.TOP, max_edit_distance=3)
            
            # If we have a result, look at the top result.
            if result:
                word = result[0].term
                dist = result[0].distance

                if dist > 0:
                    spellchecks.append((True, word))  # Incorrect word, append True and suggestion.
                else:
                    spellchecks.append((False, word)) # dist == 0 means word is correct.

            # Else, we did not find any suggestion to match the word.
            else:
                spellchecks.append((True, ""))
        
        return spellchecks
       
