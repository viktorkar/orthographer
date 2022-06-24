import neuspell
import math

class NeuSpellClass:
    def __init__(self):
        self.checker =  neuspell.BertChecker()
        self.checker.from_pretrained()

    def correct_and_split(self, pred_string):
        """Performs spell correction on a string of length less than 512. Returns a list of tuples
        (IsIncorrectWord, Suggestion)."""
        spellchecks = []

        prediction = self.checker.correct(pred_string).lower()
        prediction = list(filter(None, prediction.split(' ')))
        pred_string = list(filter(None, pred_string.split(' ')))

        for (correct, pred) in list(zip(prediction, pred_string)):
            spellchecks.append((correct != pred, correct)) # True, means incorrectly spelled
        return spellchecks

    def __call__(self, pred_strings):
        """Perform spell correction on the given strings. Returns a list of tuples
        (IsIncorrectWord, Suggestion)."""
        pred_string = ' '.join(pred_strings)
        max_len = 512
        cut_index = 400
        current_add = 0

        spellchecks = []
        while pred_string:
            if len(pred_string) > cut_index:
                c = pred_string[cut_index+current_add]
                while c != ' ' and cut_index + current_add != 512 and cut_index + current_add != len(pred_string):
                    current_add += 1
                    c = pred_string[cut_index + current_add - 1]
                temp = pred_string[0:cut_index + current_add]
                pred_string = pred_string[cut_index+current_add:]
                spellchecks.extend(self.correct_and_split(temp))
            else:
                spellchecks.extend(self.correct_and_split(pred_string))
                return spellchecks
            current_add = 0
        return spellchecks