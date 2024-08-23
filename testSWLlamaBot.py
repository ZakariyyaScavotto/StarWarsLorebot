from swLlamaBot import swLlamaBot
import time
import matplotlib.pyplot as plt
import pandas as pd


def plotChatTimes(chatTimes, simNum):
    plt.plot(chatTimes)
    plt.ylabel("Time taken to answer (s)")
    plt.xlabel("Question number")
    plt.title(f"Chat times for simulation number {simNum+1}")
    # plt.show()
    plt.savefig(f"chatSimPlots/chatTimes_{simNum+1}.png")
    plt.close()

def chatAndTime(bot, question, chatTimes):
    start = time.time()
    bot.chat(question)
    print(f"Time taken to answer: {time.time() - start} seconds")
    chatTimes.append(time.time() - start)

def main():
    allChatTimesDF = pd.DataFrame()
    # bot = swLlamaBot(vecStorePath="FAISSvectorstore", loadVecStore=True)
    bot = swLlamaBot(vecStorePath="FAISSvectorStoreIVFPQ", loadVecStore=True) # DO NOT USE IVFFLAT WE DONT HAVE THE RAM FOR IT
    for i in range(10):
        chatTimes = []
        chatAndTime(bot, "Who is Darth Vader?", chatTimes)
        chatAndTime(bot, "How did he turn to the dark side?", chatTimes)
        chatAndTime(bot, "Who is Luke Skywalker?", chatTimes)
        chatAndTime(bot, "How did he come to be a Jedi?", chatTimes)
        chatAndTime(bot, "I thought Luke was a Sith Lord?", chatTimes)
        chatAndTime(bot, "So how did he defeat Darth Vader?", chatTimes)
        chatAndTime(bot, "What is the Force?", chatTimes)
        chatAndTime(bot, "What is the dark side?", chatTimes)
        chatAndTime(bot, "What is the light side?", chatTimes)
        chatAndTime(bot, "Which side is stronger?", chatTimes)
        chatAndTime(bot, "What is the Galactic Empire?", chatTimes)
        chatAndTime(bot, "What is the Rebel Alliance?", chatTimes)
        chatAndTime(bot, "Whose ideology is better between the two?", chatTimes)
        plotChatTimes(chatTimes, i)
        allChatTimesDF[f"Simulation {i}"] = chatTimes
        bot.reset_chat()
        print(f"Simulation {i} completed")
    # Get the average and standard deviation of the chat times
    allChatTimesDF["Average"] = allChatTimesDF.mean(axis=1)
    allChatTimesDF["Standard Deviation"] = allChatTimesDF.std(axis=1)
    allChatTimesDF.to_csv("chatSimPlots/allChatTimes.csv")
    print("All simulations completed")

if __name__ == "__main__":
    main()