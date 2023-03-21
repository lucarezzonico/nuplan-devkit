import multiprocessing as mp
from gui import runGUI, guiStateInit

SILENCE = False

def startThreads(manager, silence):
    guiState = manager.dict()
    guiStateInit(guiState)
    
    pGUI = mp.Process(target=runGUI, args=(guiState, silence))
    
    pGUI.start()
    
    pGUI.join()


if __name__ == "__main__":
    mp.freeze_support()
    mp.set_start_method('spawn')
    with mp.Manager() as manager:
        startThreads(manager=manager, silence=SILENCE)