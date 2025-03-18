import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
#Interactions_Final_v10_Dataset (1)
def main():
   idf = pd.read_csv(r'C:\Users\Grand\source\repos\GameDevStats\GameDevStats\Interactions_Final_v10_Dataset (1).csv')
   ondf = pd.read_csv(r'C:\Users\Grand\source\repos\GameDevStats\GameDevStats\Updated_Old_NPCs.csv')
   sndf = pd.read_csv(r'C:\Users\Grand\source\repos\GameDevStats\GameDevStats\Updated_Seasonal_NPCs.csv')

   idf["Test"] = idf["NPC friendliness"]

   x = np.arange(0, radians(1800), radians(12))
   plt.scatter(idf["Test"], idf["Interaction length"], .5, 'b')
   plt.show()
   

main()