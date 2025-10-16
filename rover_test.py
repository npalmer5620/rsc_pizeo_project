from megapi import *

if __name__ == '__main__':
    bot = MegaPi()
    print("start")
    bot.start('/dev/cu.usbserial-120')
    print("connected")
    time.sleep(2)
    bot.motorRun(M1, 100)
    time.sleep(2)
    bot.motorStop(M1)
    print("stop")

