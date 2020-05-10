
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
from youtube_transcript_api import YouTubeTranscriptApi
from PyQt5 import QtCore, QtGui, QtWidgets


# -------------------------------------Latent Semantic Analysis----------------------------------------------------------------------------------------------------------------------------------------#
def createSummaryByLSA(d, val):
    global Kth
    Kth = int(val)
    Lexicon = []
    processedDocument = ''
    document = d
    sentences = nltk.sent_tokenize(document)
    raw_sentences = sentences
    for j in range(len(sentences)):
        processedSentence = re.sub('[^a-zA-Z]', ' ', sentences[j])
        processedSentence = processedSentence.lower()
        wordList = processedSentence.split()
        processedSentence = ''
        newSentence = ''
        for word in wordList:
            if word not in set(stopwords.words('english')):
                newSentence = newSentence + ' ' + word
                if word not in Lexicon:
                    if len(word) > 1:
                        Lexicon.append(word)
            sentences[j] = newSentence
            processedDocument = '.'.join(sentences)
    # Creating matrix A in which rows would represent our terms and columns represent sentences

    # Rows for terms,cols for documents/sentences
    A = []
    for i in range(len(Lexicon)):
        A.append([])
    i = 0
    for i in range(len(Lexicon)):
        for j in range(len(sentences)):
            if Lexicon[i] in sentences[j]:
                A[i].append(1)
            else:
                A[i].append(0)

                ##

    from numpy import array
    import numpy as np
    from scipy.linalg import svd

    A = array(A)

    U, s, VT = svd(A)
    # Dimensiontality reduction
    taken = set()
    # Taking col. values
    print("--------------------------Summary By LSA--------------------")
    sLsa = ""
    st = nltk.sent_tokenize(document)
    for i in range(Kth):
        m = float('-inf')
        for k in range(len(VT[0])):
            expressive_value = abs(VT[0][k])
            if expressive_value > m and k not in taken:
                m = expressive_value
                sIndex = k
                # VT[0][sIndex] = float('-inf')

        if sIndex not in taken:
            print('Index:' + str(sIndex))
            print(raw_sentences[sIndex])
            sLsa = sLsa + st[sIndex]
            if sLsa[-1] != ".":
                sLsa = sLsa + ". "
        taken.add(sIndex)
    return sLsa



# --------------------------------Relevance Measure----------------------------------------------------------------------------------------#
Kth = 0


def Reforming_Of_DocVector_After_Deletion(document, l):
    newDoc = ""
    passage = ""
    sentences = nltk.sent_tokenize(document)
    for i in range(l, len(sentences)):
        newDoc = newDoc + sentences[i]
        newDoc = newDoc + ". "
    passage = newDoc
    passageVector = {}
    termInPassage = passage.replace(",", " ")
    termInPassage = termInPassage.lower()
    termInPassage = termInPassage.replace(".", " ")
    termInPassage = termInPassage.replace(":", " ")
    termInPassage = termInPassage.replace(";", " ")
    termInPassage = termInPassage.replace("?", " ")
    termInPassage = termInPassage.replace("[", " ")
    termInPassage = termInPassage.replace("]", " ")
    termInPassage = termInPassage.split(" ")
    termInPassage = termInPassage
    uniquePassage = list(dict.fromkeys(termInPassage))
    tf = 0
    for k in range(len(uniquePassage)):
        count = 0
        for j in range(len(termInPassage)):
            if uniquePassage[k] == termInPassage[j]:
                count = count + 1
        passageVector[uniquePassage[k]] = count
    return passageVector


def createVector(passage):
    passageVector = {}
    termInPassage = passage.replace(",", " ")
    termInPassage = termInPassage.lower()
    termInPassage = termInPassage.replace(".", " ")
    termInPassage = termInPassage.replace(":", " ")
    termInPassage = termInPassage.replace(";", " ")
    termInPassage = termInPassage.replace("?", " ")
    termInPassage = termInPassage.replace("[", " ")
    termInPassage = termInPassage.replace("]", " ")
    termInPassage = termInPassage.split(" ")
    termInPassage = termInPassage
    uniquePassage = list(dict.fromkeys(termInPassage))
    tf = 0
    for k in range(len(uniquePassage)):
        count = 0
        for j in range(len(termInPassage)):
            if uniquePassage[k] == termInPassage[j]:
                count = count + 1
        passageVector[uniquePassage[k]] = count
    return passageVector


def takeDotProduct(d1, d2):
    dot_product = sum(d1[key] * d2.get(key, 0) for key in d1)
    return dot_product


def CreateRelevanceIndex(doc, sentences):
    stopword = stopwords.words("english")
    docVector = createVector(doc)
    for o in range(len(stopword)):
        if stopword[o] in docVector:
            del docVector[stopword[o]]
    # print(docVector)
    sentencesRelevence = {}
    for i in range(len(sentences)):
        sentenceVector = createVector(sentences[i])
        sentencesRelevence[i] = takeDotProduct(docVector, sentenceVector) / len(sentenceVector)
        # print(wc)
        docVector = Reforming_Of_DocVector_After_Deletion(doc, i)

    # print("#-------RELEVANCE MEASURE SCORE OF SENTENCES------#")
    # print(sentencesRelevence)
    return sentencesRelevence


def CreateSummaryByRelevanceMeasure(doc, val):
    global Kth

    sentByDot = doc.split('.')
    sentences = nltk.sent_tokenize(doc)
    """
    """
    for i in range(len(sentences)):
        sentences[i] = re.sub('[^a-zA-Z]', ' ', sentences[i])

    if int(val) > len(sentences):
        Kth = int(len(sentences))
        val = Kth
    dict1 = CreateRelevanceIndex(doc, sentences)
    dict1 = {k: v for k, v in sorted(dict1.items(), key=lambda item: item[1])}
    # print(dict1)
    SentenceOrder = []
    for key, value in dict1.items():
        SentenceOrder.append(key)

    print(SentenceOrder)

    c = len(sentences)
    summary = []
    for i in range(int(val)):
        summary.append(sentences[SentenceOrder[i]])

        c = c - 1

    return summary


def printSummary(summary):
    s = ""
    for i in range(len(summary)):
        print(summary[i])
        s = s + summary[i]

    return s


# -------------------------------------------------UI-------------------------------------------------------------------------------------------------------#

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(666, 576)
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(178, 190, 181))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(178, 190, 181))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(178, 190, 181))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(178, 190, 181))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Window, brush)
        MainWindow.setPalette(palette)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.youtube_link = QtWidgets.QTextEdit(self.centralwidget)
        self.youtube_link.setGeometry(QtCore.QRect(130, 20, 361, 31))
        font = QtGui.QFont()
        font.setFamily("MS Sans Serif")
        font.setPointSize(10)
        self.youtube_link.setFont(font)
        self.youtube_link.setObjectName("youtube_link")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(20, 20, 91, 20))
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(85, 0, 127))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.ButtonText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0, 128))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.PlaceholderText, brush)
        brush = QtGui.QBrush(QtGui.QColor(85, 0, 127))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.ButtonText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0, 128))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.PlaceholderText, brush)
        brush = QtGui.QBrush(QtGui.QColor(120, 120, 120))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(120, 120, 120))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(120, 120, 120))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.ButtonText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0, 128))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.PlaceholderText, brush)
        self.label.setPalette(palette)
        font = QtGui.QFont()
        font.setFamily("MS Sans Serif")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.summary_lsa = QtWidgets.QTextEdit(self.centralwidget)
        self.summary_lsa.setGeometry(QtCore.QRect(90, 240, 471, 91))
        font = QtGui.QFont()
        font.setFamily("MS Sans Serif")
        font.setPointSize(11)
        self.summary_lsa.setFont(font)
        self.summary_lsa.setObjectName("summary_lsa")
        self.summary_rm = QtWidgets.QTextEdit(self.centralwidget)
        self.summary_rm.setGeometry(QtCore.QRect(90, 370, 471, 91))
        font = QtGui.QFont()
        font.setFamily("MS Sans Serif")
        font.setPointSize(11)
        self.summary_rm.setFont(font)
        self.summary_rm.setObjectName("summary_rm")
        self.button_generate = QtWidgets.QPushButton(self.centralwidget)
        self.button_generate.setGeometry(QtCore.QRect(90, 490, 471, 41))
        self.button_generate.clicked.connect(self.pressed)
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(85, 0, 127))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(85, 0, 127))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.ButtonText, brush)
        brush = QtGui.QBrush(QtGui.QColor(85, 0, 127))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(85, 0, 127))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.ButtonText, brush)
        brush = QtGui.QBrush(QtGui.QColor(120, 120, 120))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(120, 120, 120))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.ButtonText, brush)
        self.button_generate.setPalette(palette)
        font = QtGui.QFont()
        font.setFamily("MS Sans Serif")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.button_generate.setFont(font)
        self.button_generate.setObjectName("button_generate")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(200, 210, 211, 20))
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(85, 0, 127))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(85, 0, 127))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(120, 120, 120))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, brush)
        self.label_2.setPalette(palette)
        font = QtGui.QFont()
        font.setFamily("MS Sans Serif")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(230, 340, 161, 20))
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(85, 0, 127))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(85, 0, 127))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(120, 120, 120))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, brush)
        self.label_3.setPalette(palette)
        font = QtGui.QFont()
        font.setFamily("MS Sans Serif")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.document = QtWidgets.QTextEdit(self.centralwidget)
        self.document.setGeometry(QtCore.QRect(100, 100, 441, 100))
        font = QtGui.QFont()
        font.setFamily("MS Sans Serif")
        font.setPointSize(12)
        self.document.setFont(font)
        self.document.setObjectName("document")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(240, 60, 131, 20))
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(85, 0, 127))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(85, 0, 127))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(120, 120, 120))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, brush)
        self.label_4.setPalette(palette)
        font = QtGui.QFont()
        font.setFamily("MS Sans Serif")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(10, 200, 81, 31))
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(85, 0, 127))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(85, 0, 127))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(120, 120, 120))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, brush)
        self.label_5.setPalette(palette)
        font = QtGui.QFont()
        font.setFamily("MS Sans Serif")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.K = QtWidgets.QTextEdit(self.centralwidget)
        self.K.setGeometry(QtCore.QRect(20, 100, 51, 41))
        font = QtGui.QFont()
        font.setFamily("MS Sans Serif")
        font.setPointSize(15)
        font.setBold(True)
        font.setWeight(75)
        self.K.setFont(font)
        self.K.setObjectName("K")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(20, 60, 41, 31))
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(85, 0, 127))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.ButtonText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0, 128))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.PlaceholderText, brush)
        brush = QtGui.QBrush(QtGui.QColor(85, 0, 127))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.ButtonText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0, 128))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.PlaceholderText, brush)
        brush = QtGui.QBrush(QtGui.QColor(120, 120, 120))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(120, 120, 120))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(120, 120, 120))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.ButtonText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0, 128))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.PlaceholderText, brush)
        self.label_6.setPalette(palette)
        font = QtGui.QFont()
        font.setFamily("MS Sans Serif")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 666, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Summary Generator"))
        self.youtube_link.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Sans Serif\'; font-size:10pt; font-weight:400; font-style:normal;\">\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p></body></html>"))
        self.label.setText(_translate("MainWindow", "Youtube Link"))
        self.summary_lsa.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Sans Serif\'; font-size:11pt; font-weight:400; font-style:normal;\">\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-family:\'MS Shell Dlg 2\'; font-size:8.25pt;\"><br /></p></body></html>"))
        self.summary_rm.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Sans Serif\'; font-size:11pt; font-weight:400; font-style:normal;\">\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-family:\'MS Shell Dlg 2\';\"><br /></p></body></html>"))
        self.button_generate.setText(_translate("MainWindow", "Generate Summary"))
        self.label_2.setText(_translate("MainWindow", " Latent Semantic Analysis"))
        self.label_3.setText(_translate("MainWindow", "Relevance Measure"))
        self.document.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Sans Serif\'; font-size:12pt; font-weight:400; font-style:normal;\">\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p></body></html>"))
        self.label_4.setText(_translate("MainWindow", "Raw Document"))
        self.label_5.setText(_translate("MainWindow", "Summary"))
        self.K.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Sans Serif\'; font-size:15pt; font-weight:600; font-style:normal;\">\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p></body></html>"))
        self.label_6.setText(_translate("MainWindow", "Set K"))


    def pressed(self):
        K = self.K.toPlainText()
        if K.isdigit()==False:
            K=0
        else:
            K=int(K)
        Document = ""
        success=True
        try:
            youtube_link=self.youtube_link.toPlainText()
            video_id = self.youtube_link.toPlainText()[32:]
            trans = YouTubeTranscriptApi.get_transcript(video_id)
            for i in range(len(trans)):
                Document = Document + " " + trans[i]['text'].rstrip('\r\n')
        except:
            print('Url is not accessible')
            success=False
        if success:
            self.document.setText(Document)
            self.summary_lsa.setText(createSummaryByLSA(Document, K))
            self.summary_rm.setText(printSummary(CreateSummaryByRelevanceMeasure(Document, K)))
            success=False
        try:
            Document=self.document.toPlainText()
            self.document.setText(Document)
            self.summary_lsa.setText(createSummaryByLSA(Document,K))
            self.summary_rm.setText(printSummary(CreateSummaryByRelevanceMeasure(Document,K)))
        except:
            print('Document is not defined')


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
