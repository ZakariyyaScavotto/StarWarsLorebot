from swLlamaBot import swLlamaBot

def main():
    bot = swLlamaBot(vecStorePath="FAISSvectorstore", loadVecStore=True)
    bot.chat("Who is Anakin Skywalker?")
    bot.chat("Why did he become Darth Vader?")

if __name__ == "__main__":
    main()