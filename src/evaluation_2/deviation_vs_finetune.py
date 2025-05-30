import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from evaluation2 import get_results

x_coord = "model_type"
y_coord = "deviation"

plt.rcParams.update({
        "text.usetex": True,
        "font.size": 16
    })

df = get_results(onlyv2=True)
plt.figure(figsize=(6, 4))
ax = sns.boxplot(y=x_coord, x=y_coord, data=df, whis=(0, 100), orient='y',width=.5 )
plt.title('Finetunning vs. Deviation')
plt.xlabel("Finetunning")
plt.ylabel("Deviation")
plt.xticks(rotation=45)

ax.axvline(0, ls='--', color='red', lw=0.7)

plt.tight_layout()
plt.show()