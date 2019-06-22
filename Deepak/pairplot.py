import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('creditcard.csv')
sns.set(style="ticks")
sns.pairplot(df, hue="Class")
plt.savefig('pairplots.png')
