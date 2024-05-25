from typing import Any

Text = ""

FilePaths2NLines = {"Data/yelp_labelled.txt" : 1000, "Data/amazon_cells_labelled.txt" : 1000}

import time
Since = time.time()

for File in FilePaths2NLines:
    try:
        f = open(File)

        for x in range(FilePaths2NLines[File]):   
            Text = Text + f.readline()

        f.close()
    except IOError:
        raise IOError(f"{File} Not Found")
Now = time.time() - Since
print(f"Time Taken: {(Now*100):.4f}ms")

Text = Text.split('\n')
# print(Text)

InputTexts = []
TargetTexts = []

for x in Text:
    SentenceList = x.split('\t')
    # print(SentenceList)

    input_text, target_text = [], []
    if len(SentenceList) == 2:
        input_text = SentenceList[0]
        target_text = SentenceList[1]

        InputTexts.append(input_text)
        TargetTexts.append(target_text)

print(InputTexts[0])
print(TargetTexts[0])

SepLetterList = [' ', '.', '!', '?', ';', ',', '-', ')', '(', '"', "'"]

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

InputTexts = Split(InputTexts, SepLetterList)

# print(InputTexts[0])
# print(TargetTexts[0])

START_TOKEN = "<sos>"
END_TOKEN = "<eos>"
EMPTY_TOKEN = ''
DEAFULT_TOKETS = [EMPTY_TOKEN, START_TOKEN, END_TOKEN]

class WordStorage:
    def __init__(self, UseDeafultTokens = True) -> None:
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

InputWords = WordStorage()
TargetWords = WordStorage(False)
Since = time.time()

for x in InputTexts:
    InputWords.AddSentenceList(x)

TargetWords.AddWord("negative")
TargetWords.AddWord("positive")

Now = time.time() - Since
print(f"Time Taken: {(Now*100):.4f}ms")

# print(InputWords.Words)
# print(InputWords.Words2Idx)

def GetIndicieSentence(wordstorage : WordStorage, SentenceList : list[str]):    
    Indicies = [wordstorage.Words2Idx[START_TOKEN]]
    
    for y in SentenceList:
        Indicies.append(wordstorage.Words2Idx[y])
    Indicies.append(wordstorage.Words2Idx[END_TOKEN])
    
    return Indicies

def GetIndicies(wordstorage : WordStorage, Text) -> (list[list[int]], int):
    RetIndicies = []
    MaxLen = 0

    for x in Text:
        i = len(RetIndicies)
        RetIndicies.append(GetIndicieSentence(wordstorage, x))
        
        if len(RetIndicies[i]) > MaxLen:
            MaxLen = len(RetIndicies[i])

    return RetIndicies, MaxLen

InputIndicies, MaxInputLen = GetIndicies(InputWords, InputTexts)

def Indicies2Str(Indicies, wordstorage : WordStorage):
    Sentence = ""
    for x in Indicies:
        Sentence = Sentence + wordstorage.Words[x]
    return Sentence

# Shape of (Input_Len, NLines)

def PadSequenceSentence(Len, Indicies):
    # Make all the seq of same len = MaxLen
    if len(Indicies) < Len:
        MaxRange : int = Len - len(Indicies)
        if MaxRange > 0:
            for x in range(MaxRange):
                Indicies.append(InputWords.Words2Idx[EMPTY_TOKEN])

def PadSequence(Len, Input):
    # Make all the seq of same len = MaxLen
    for Indicies in Input:
        PadSequenceSentence(Len, Indicies)

PadSequence(MaxInputLen, InputIndicies)

print(InputIndicies[0])
print(Indicies2Str(InputIndicies[0], InputWords))
print(MaxInputLen)

import numpy as np

TargetIndicies = []
for x in TargetTexts:
    TargetIndicies.append(int(x))

InputIndicies = np.array(InputIndicies)
TargetIndicies = np.array(TargetIndicies)

print(InputIndicies.shape)
print(TargetIndicies.shape)

import torch

InputTensor = torch.from_numpy(InputIndicies).long()

def GetTensor(Indicies):
    Indicies = np.array(Indicies)
    return torch.from_numpy(Indicies).long()

TargetTensor = torch.from_numpy(TargetIndicies).float().unsqueeze(dim=1)

print(TargetTensor.shape)
print(InputTensor.shape)

from torch import nn

# Model Creation
class PosOrNeg(nn.Module):
    def __init__(self, VocabSize, EmbedSize, InputLen,HiddenSize, OutputSize, NLayers) -> None:
        """
            - VocabSize  : The Num of unique words in the text
            - EmbedSize  : The embedding vector size, preffered range: (50, 300)
            - InputLen   : The Len of the Input 
            - HiddenSize : The Hidden Size for the LSTM, preffered range: (50, 300)
            - OutputSize : The Output Dim of Model
            - NLayers    : The Num of Layers in the Lstm 
        """
        
        super().__init__()

        self.HiddenSize = HiddenSize
        self.NLayers = NLayers

        self.Embedding = nn.Embedding(VocabSize, EmbedSize)
        self.Lstm = nn.LSTM(EmbedSize * InputLen, HiddenSize, num_layers=NLayers, batch_first=True)
        self.Classifier = nn.Linear(HiddenSize, OutputSize)
        self.Sigmoid = nn.Sigmoid()

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        # print(x.shape)
        Embed : torch.Tensor = self.Embedding(x)
        Embed = Embed.view((Embed.shape[0], Embed.shape[1] * Embed.shape[2])).unsqueeze(dim=1)
        # print(Embed.shape)

        hc = torch.zeros(self.NLayers, x.size(0), self.HiddenSize)
        cc = torch.zeros(self.NLayers, x.size(0), self.HiddenSize)
        
        Output, _ = self.Lstm(Embed, (hc, cc))
        # print(Output.shape)

        Output = self.Sigmoid(self.Classifier(Output))
        # print(Output.shape)

        return Output

N_LAYERS = 1
EMBED_SIZE = 128
HIDDEN_SIZE = 128
N_EPOCHS = 100
LEARNING_RATE = 0.001

Model = PosOrNeg(
    VocabSize=InputWords.N_Words,
    EmbedSize=EMBED_SIZE,
    InputLen=MaxInputLen,
    HiddenSize=HIDDEN_SIZE,
    OutputSize=1,
    NLayers=N_LAYERS
)

print(Model)

from pathlib import Path

Model_Pth = Path("Model.pth")

if Model_Pth.is_file():
    Model.load_state_dict(torch.load(f="Model.pth"))

def Predict(Sentence):    
    Ind = GetIndicieSentence(InputWords, SplitSentence(Sentence, SepLetterList))
    PadSequenceSentence(MaxInputLen, Ind)

    return int(torch.round(Model(GetTensor(Ind).unsqueeze(dim=0))).item())

Sentence = "I am impressed."

from enum import Enum

class States(Enum):
    TRAIN = 1
    TEST = 2

CURRENT_STATE = States.TRAIN

if CURRENT_STATE == States.TEST:
    while(Sentence != "Quit"):
        Sentence = input("Enter Sentence To Predict(Quit): ")
        
        if Sentence != "Quit":
            print("The Sentence is: ", TargetWords.Words[Predict(Sentence)])

def GetAccuracy(TargetTensor, PredTensor, IndexRange):
    Acc = 0

    for x in range(IndexRange):
        if PredTensor[x] == TargetTensor[x]:
            Acc += 1
    return (Acc/IndexRange)*100

if CURRENT_STATE == States.TRAIN:
    Loss_Fn = nn.BCELoss()
    Optim = torch.optim.Adam(params=Model.parameters(), lr=LEARNING_RATE)

    for epoch in range(N_EPOCHS):
        Since = time.time()

        Model.train()

        y_pred : torch.Tensor = Model(InputTensor).squeeze(dim=2)
        Loss = Loss_Fn(y_pred, TargetTensor)
        
        Acc = GetAccuracy(TargetTensor, torch.round(y_pred), 2000)

        Optim.zero_grad()
        Loss.backward()
        Optim.step()

        Now = time.time() - Since

        print(f"Epoch: {epoch}/{N_EPOCHS} Loss: {Loss:.4f} Acc: {Acc}%")
        print(f"Took: {Now}s Estimated Remaining Time: {(Now*(N_EPOCHS-epoch))}s")

    torch.save(f="Model.pth", obj=Model.state_dict())
