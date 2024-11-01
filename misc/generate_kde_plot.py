import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

geyser = sns.load_dataset("geyser")
sns.kdeplot(data=geyser, x="waiting", y="duration")

# Save the plot pdf
plt.savefig("kde_plot.pdf")