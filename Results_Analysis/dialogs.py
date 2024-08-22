import utils

def getMainAction():
    return ["Plot All Results", 
                "Plot For Threshold Criterion",
                "Plot For Low Progression Criterion",
                "Analyse By Data Size",
                "Plot Cumulative Results For Mae Threshold",
                "Print Success Rate For All Zifs",
                "Print Datasets Used Against a Zif",
                "Get best datasets by probability",
                "Get best datasets by data size",
                "Test selected model against all Zifs",
                "Plot Error To Dataset Size",
                "Test top100 mofs prediction (methane absorption)",
                "Exit"]

def getNumOfActions():
    return len(getMainAction())

def selectMainAction():
    print("----------   MAIN MENU   ----------")

    mainMenu = getMainAction()

    utils.printOptions(mainMenu)
    action = utils.validateInput(mainMenu)

    return action

def yesNoInput():
    acceptablePos = ["yes","y"]
    acceptableNeg = ["no","n"]

    answer = input()
    answer = answer.lower()
    while answer not in acceptablePos + acceptableNeg:
        print("Please respond with a yes or a no.")
        answer = input().lower()

    if answer in acceptablePos:
        return True
    else:
        return False