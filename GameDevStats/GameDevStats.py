from asyncio.windows_events import NULL
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math

CSVFiles = ["User ID", "NPC ID", "User level","NPC friendliness", "Interest", "Interaction length", "Interaction quests acquired"]

def MaxAndMin(x):
  newFrame = x.copy()
  max = x.max()
  min = x.min()
  for i in range(len(x)):
      newFrame[i] = (newFrame[i]-min)/(max-min)

  return newFrame

def predict(x1, w1, b):
  return (w1 * ((x1)**.5)) + b

# Function to compute cost (Mean Squared Error)
def compute_cost(x_train, y_true, w, b):
    m = len(y_true)
    y_pred = predict(x_train, w, b)
    return (1 / (2 * m)) * np.sum((y_pred - y_true) ** 2)

# Function to perform one iteration of gradient descent
def gradient_descent_step(x_train, y_true, w, b, alpha):
    m = len(y_true)
    y_pred = predict(x_train, w, b)
    dw = (1 / m) * np.sum((y_pred - y_true) * x_train)
    db = (1 / m) * np.sum(y_pred - y_true)
    w -= alpha * dw
    b -= alpha * db
    return w, b

# Training function
def train_linear_regression(x_train, y_true, w, b, alpha, iterations):
    cost_history = []
    w_history = []
    b_history = []
    for i in range(iterations):
        w, b = gradient_descent_step(x_train, y_true, w, b, alpha)
        cost_history.append(compute_cost(x_train, y_true, w, b))
        w_history.append(w)
        b_history.append(b)
    return w, b, cost_history, w_history, b_history

def sorter(ID, Sortie, value, df):
    Data = []
    for index, row in df.iterrows():
        if row[Sortie] == value:
            Data.append(row[ID])

    return Data

def main():
   idf = pd.read_csv(r'C:\Users\Grand\source\repos\Grandcircuit424\GameDevStat\GameDevStats\Interactions_Final_v10_Dataset (1).csv')
   ondf = pd.read_csv(r'C:\Users\Grand\source\repos\Grandcircuit424\GameDevStat\GameDevStats\Updated_Old_NPCs.csv')
   sndf = pd.read_csv(r'C:\Users\Grand\source\repos\Grandcircuit424\GameDevStat\GameDevStats\Updated_Seasonal_NPCs.csv')

   idf_4 = idf[idf['User level'] >= 40].reset_index(drop=True)
   idf_3 = idf[(idf['User level'] < 40) & (idf['User level'] >= 30)].reset_index(drop=True)
   idf_2 = idf[(idf['User level'] < 30) & (idf['User level'] >= 21)].reset_index(drop=True)
   idf_1 = idf[idf['User level'] < 20].reset_index(drop=True)


   pd.set_option('display.max_rows', None)
   pd.set_option('display.max_column', None)
   best_w, best_b, cost_history, w_history, b_history = train_linear_regression(idf_1['NPC friendliness'], idf_1['Interest'], 0,0,.0001,100)
   print(f"{best_w} : {best_b} : {cost_history[len(cost_history)-1]}")
 


   # MSE vs Iteration graph
   plt.plot(range(100), cost_history)
   plt.title("MSE vs iteration (LVL 10)")
   plt.xlabel("Iteration")
   plt.ylabel("MSE")
   plt.show()
  
   
 
   ## The 4 populations devided
   fig, axes = plt.subplots(2, 2, figsize=(30, 24))

   #Scattering data for each population
   axes[0,0].scatter(idf_1['NPC friendliness'], idf_1['Interest'], 5, 'red')
   axes[0,1].scatter(idf_2['NPC friendliness'], idf_2['Interest'], 5, 'blue')
   axes[1,0].scatter(idf_3['NPC friendliness'], idf_3['Interest'], 5, 'green')
   axes[1,1].scatter(idf_4['NPC friendliness'], idf_4['Interest'], 5, 'black')

   ## Title and axis
   axes[0,0].set_title(f"NPC friendliness V Interest <LVL20")
   axes[0,1].set_title(f"NPC friendliness V Interest LVL21-30")
   axes[1,0].set_title(f"NPC friendliness V Interest LVL31-40")
   axes[1,1].set_title(f"NPC friendliness V Interest >LVL40")
   for i in range(2):
        for j in range(2):
            axes[i-1,j-1].set_xlabel("NPC friendliness")
            axes[i-1,j-1].set_ylabel("Interest")
   plt.show()

    #Gradient Descent graph for lvl20>
   plt.title("NPC friendliness v Interest LVL21>")
   idf_1 = idf_1.sort_values('NPC friendliness')

   plt.scatter(idf_1["NPC friendliness"], idf_1["Interest"], 5, 'b')
   plt.xlabel("NPC friendliness")
   plt.ylabel("Interest")

 
   plt.plot(idf_1["NPC friendliness"], predict(idf_1["NPC friendliness"], best_w, best_b), 'r')
   plt.title("NPC friendliness v Interest (LVL10)")
   plt.xlabel("NPC friendliness")
   plt.ylabel("Interest")
 
   plt.show()
   
   # 7x7 Grid on interaction data
   x = np.arange(0, math.radians(1800), math.radians(12))
   fig, axes = plt.subplots(7, 7, figsize=(30, 24))
   for i in range(7):
       for j in range(7):
            axes[i,j].scatter(idf[CSVFiles[i]], idf[CSVFiles[j]], .5, 'g')
   

   plt.show()
   


    
main()