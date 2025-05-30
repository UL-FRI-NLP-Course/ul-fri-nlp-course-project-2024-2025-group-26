import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from evaluation2 import get_results

x_coord = "sloberta_treshold"
y_coord = "input_length"

df = get_results(onlyv2=False)

# Filter out rows where excel_rows is 10
df = df[df["excel_rows"] == 10]

plt.rcParams.update({
        "text.usetex": True,
        "font.size": 16
    })

plt.figure(figsize=(6, 4))
ax = sns.boxplot(y=x_coord, x=y_coord, data=df, whis=(0, 100), orient='y',width=.5 )
plt.title('Sloberta treshold vs. Input Length')
plt.ylabel("Sloberta Treshold")
plt.xlabel("Input Length")
plt.xticks(rotation=45)




plt.tight_layout()
plt.show()