from swLlamaBot import swLlamaBot

def main():
    bot = swLlamaBot(vecStorePath="FAISSvectorstore", loadVecStore=True)
    bot.chat("Who was Darth Revan?")

if __name__ == "__main__":
    main()