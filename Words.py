# Funcs used to split the raw str into each unique word
def SplitSentence(Sentence : str, SepLetterList : list[str]) -> list[str]:
    SentenceList = []

    PrevSepLetterIdx = 0
    
    for LetterIdx in range(len(Sentence)):
        if Sentence[LetterIdx] in SepLetterList:
            SentenceList.append(Sentence[PrevSepLetterIdx:LetterIdx])
            SentenceList.append(Sentence[LetterIdx])
            PrevSepLetterIdx = LetterIdx
        elif LetterIdx == len(Sentence)-1:
            SentenceList.append(Sentence[PrevSepLetterIdx:])
    
    return SentenceList

def Split(Text, SepLetterList) -> list[list[str]]:
    RetText = []
    for Sentence in Text:
        RetText.append(SplitSentence(Sentence, SepLetterList))
    return RetText

# The Deafult and required Tokens
START_TOKEN = "<sos>"
END_TOKEN = "<eos>"
EMPTY_TOKEN = ''
DEAFULT_TOKETS = [EMPTY_TOKEN, START_TOKEN, END_TOKEN]

class WordStorage:
    def __init__(self, UseDeafultTokens = True) -> None:
        """
            - UseDeafultTokens: Specifies if the Tokens from DEAFULT_TOKETS should be added or not
        """

        self.N_Words = 0
        self.Words2Idx = {}  
        self.Words = []

        if UseDeafultTokens:
            for x in DEAFULT_TOKETS:
                self.AddWord(x)

    def AddSentence(self, Sentence : str):
        SentenceList = SplitSentence(Sentence, SepLetterList)
        
        for Word in SentenceList:
            self.AddWord(Word)

    def AddSentenceList(self, SentenceList : list[str]):        
        for Word in SentenceList:
            self.AddWord(Word)

    def AddWord(self, Word):
        if Word not in self.Words:
            self.Words.append(Word)
            self.Words2Idx[Word] = len(self.Words2Idx)
            self.N_Words += 1

def CreateWordStorage(Text, SepLetterList, UseDeafultTokens = True):
    # print(Text)
    TextLists = Split(Text, SepLetterList)
    
    wordstorage = WordStorage(UseDeafultTokens=UseDeafultTokens)

    print(TextLists[0])
    for x in TextLists:
        wordstorage.AddSentenceList(x)
    print(wordstorage.Words)

    return wordstorage
