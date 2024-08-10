from swLlamaBot import swLlamaBot

def main():
    bot = swLlamaBot(vecStorePath="FAISSvectorstore", loadVecStore=True)
    bot.chat("Who is Darth Vader?")
    bot.chat("How did he turn to the dark side?")
    bot.chat("Who is Luke Skywalker?")
    bot.chat("How did he come to be a Jedi?")
    bot.chat("I thought Luke was a Sith Lord?")

if __name__ == "__main__":
    main()