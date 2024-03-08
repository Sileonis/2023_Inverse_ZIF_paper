from scipy import stats

class Statistical_Tests():
    
    def __init__(self,testName):
        self.name = testName

    def getTest(self, arrayA, arrayB):
        
        if self.name == "pairedT":
            return self.getPairedT(arrayA, arrayB)
        else:
            raise NotImplementedError("No other Statistical Test Implemented Yet.")

    def getPairedT(self,arrayA, arrayB):
        return stats.ttest_rel(arrayA,arrayB)