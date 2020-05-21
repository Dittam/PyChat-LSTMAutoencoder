from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from keras.models import load_model
from preprocessing import clean
from inference import runInference
import numpy as np
import platform
import time
import pickle


def login(email, password):
    print("loggining in...")
    driver.get('https://www.messenger.com/')
    driver.find_element_by_xpath('//*[@id="email"]').send_keys(email)
    driver.find_element_by_xpath('//*[@id="pass"]').send_keys(password)
    driver.find_element_by_xpath('//*[@id="loginbutton"]').click()
    time.sleep(4)


def initChatHistory():
    """
    iterates through all chats saving the number of visible messages in each
    chat in a dictionary
    """
    print("initializing chat history...")
    chatLengths = dict()
    allChats = driver.find_elements_by_xpath("//li[contains(@class,'_5l-3')]")
    for chat in allChats:
        chat.click()
        time.sleep(2)
        chatName = driver.find_element_by_xpath("//span[@class='_3oh-']").text
        messages = driver.find_elements_by_xpath(
            "//span[@class='_3oh- _58nk']")
        chatLengths[chatName] = len(messages)
    return chatLengths


def mainLoop(pollFreq, chatName="Test Env"):
    print("listening...")
    while True:
        # check for new messages in all chats that are not currently selected
        try:
            newAlert = driver.find_element_by_xpath(
                "//li[contains(@class,'_1ht3')]")
            newAlert.click()
            chatName = driver.find_element_by_xpath(
                "//span[@class='_3oh-']").text
        except:
            pass
        # compare number of new messages to number of old messages for current
        # chat, and only respond to new messages
        time.sleep(pollFreq)
        newMessages = driver.find_elements_by_xpath(
            "//span[@class='_3oh- _58nk']")
        if chatLengths[chatName] < len(newMessages):
            for j in range(chatLengths[chatName], len(newMessages)):
                onNewMessage(newMessages[j].text)
                if "<EXIT>" in newMessages[j].text:
                    print("exiting...")
                    return 0
        chatLengths[chatName] = len(newMessages)


def onNewMessage(text):
    """
    Generate and send response from pybot's ML model
    """
    if "@PyBot Alpha" in text:
        print(text)
        response = runInference(clean(text.split(' ', 2)[2]), eModel, dModel,
                                word2idx, MAX_SENTENCE_LENGTH)
        print(response)
        actions = ActionChains(driver)
        actions.send_keys(response, Keys.ENTER).perform()


if __name__ == '__main__':
    # Load Model
    MAX_SENTENCE_LENGTH = 20
    paths = "C:/Users/ditta/Desktop/LSTMPyBotData/"
    word2idx = pickle.load(open(paths + "word2idx.pkl", "rb"))
    trainModel = load_model(
        paths + "trainedModels/trainModel_20EPOCHS_ 2.63LOSS.h5")
    eModel = load_model(
      paths + "trainedModels/eModel_20EPOCHS_ 2.63LOSS.h5", compile=False)
    dModel = load_model(
      paths + "trainedModels/dModel_20EPOCHS_ 2.63LOSS.h5", compile=False)
    # print(runInference("hello how are you", eModel,
    #                   dModel, word2idx, MAX_SENTENCE_LENGTH))

    # Run selenium bot
    driver = webdriver.Chrome(
        'C:/Program Files (x86)/Google/Chrome/Application/chromedriver.exe')

    login("username", "password")
    chatLengths = initChatHistory()
    driver.find_element_by_xpath(
        "//*[contains(text(), '" + "Test Env" + "')]").click()
    time.sleep(2)
    mainLoop(0.5)
    driver.close()
