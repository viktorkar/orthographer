import cv2 as cv

from symspell import SymSpellClass

class SpellCheckModule:
    """The module used for spellchecking. The module can be changed to use different spellcheckers as long
    as run() returns the same format."""
    def __init__(self, sc_type, language):
        self.tickmeter = cv.TickMeter()
        self.sc_type = sc_type
        self.language = language

        self.__setup()

    ####################################################################################################
    def __setup(self):
        """Creates a new spellchecker. Done when language is changed to make sure the used spellchecker 
        supports the language."""
        if self.sc_type == 'symspell':
            self.spellchecker = SymSpellClass(self.language)
    
    ####################################################################################################
    def update_language(self, language):
        """Update the used language and create a new spellchecker."""
        if self.language != language:
            self.language = language
            self.__setup()

    ####################################################################################################
    def run(self, pred_strings):
        """Perform spell correction on a given list of strings.

        Args:
            pred_strings: A list of strings to be spellchecked.

        Returns:
            sc_infTime: The inference time for the spellchecker.
            spellchecks: A list of tuples (IsIncorrectWord, Suggestion).
        """
        self.tickmeter.start()

        spellchecks = self.spellchecker(pred_strings)
        # Save inference time.
        self.tickmeter.stop()
        sc_infTime = self.tickmeter.getTimeMilli()
        self.tickmeter.reset()

        return sc_infTime, spellchecks
