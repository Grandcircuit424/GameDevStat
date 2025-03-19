import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math

CSVFiles = ["User ID", "NPC ID", "User level","NPC friendliness", "Interest", "Interaction length", "Interaction quests acquired"]

def predict(x1, w1, b):
  return (w1 * (x1)**2)+x1-100 + b

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

   lvl10 = sorter("NPC ID", "NPC area level", 10, sndf)
   lvl20 = sorter("NPC ID", "NPC area level", 20, sndf)
   lvl30 = sorter("NPC ID", "NPC area level", 30, sndf)
   lvl40 = sorter("NPC ID", "NPC area level", 40, sndf)

   FillerDF1 = idf[idf['NPC ID'].isin(lvl10)]
   FillerDF2 = idf[idf['NPC ID'].isin(lvl20)]
   FillerDF3 = idf[idf['NPC ID'].isin(lvl30)]
   FillerDF4 = idf[idf['NPC ID'].isin(lvl40)]
   FillerArray = [FillerDF1,FillerDF2,FillerDF3,FillerDF4]
   pd.set_option('display.max_rows', None)


   best_w, best_b, cost_history, w_history, b_history = train_linear_regression(FillerDF3['NPC friendliness'], FillerDF3['Interest'], 0,0,.0001,20)
   print(f"{best_w} : {best_b} : {cost_history[len(cost_history)-1]}")

   plt.plot(range(20), cost_history)
   plt.title("MSE vs iteration (LVL 10)")
   plt.xlabel("Iteration")
   plt.ylabel("MSE")

  
   fig, axes = plt.subplots(2, 2, figsize=(30, 24))

   axes[0,0].scatter(FillerDF1['NPC friendliness'], FillerDF1['Interest'], 5, 'red')
   axes[0,1].scatter(FillerDF2['NPC friendliness'], FillerDF2['Interest'], 5, 'blue')
   axes[1,0].scatter(FillerDF3['NPC friendliness'], FillerDF3['Interest'], 5, 'green')
   axes[1,1].scatter(FillerDF4['NPC friendliness'], FillerDF4['Interest'], 5, 'black')

   z=1
   for i in range(2):
       for j in range(2):
            axes[i,j].set_title(f"NPC friendliness V Interest LVL{z}0")
            axes[i,j].set_xlabel("NPC friendliness")
            axes[i,j].set_ylabel("Interest")
            z=z+1

   '''
   idf["X"] = idf["NPC friendliness"]
   idf["Y"] = idf["Interest"]

   plt.scatter(idf["X"], idf["Y"])
   plt.title("NPC friendliness v Interest")
   plt.xlabel("NPC friendliness")
   plt.ylabel("Interest")
   '''

   '''
   x = np.arange(0, math.radians(1800), math.radians(12))
   fig, axes = plt.subplots(7, 7, figsize=(30, 24))
   for i in range(7):
       for j in range(7):
            axes[i,j].scatter(FillerDF2[CSVFiles[i]], FillerDF2[CSVFiles[j]], .5, 'g')
   '''

   plt.show()
   


    
main()