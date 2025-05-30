import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from evaluation2 import get_results

x_coord = "excel_rows"

df = get_results(onlyv2=True)
plt.figure(figsize=(6, 6))
ax = sns.boxplot(x=x_coord, y='bleu_score', data=df)
plt.title(f'BLEU Score vs. {x_coord}')
plt.xlabel(x_coord)
plt.ylabel('BLEU Score')
plt.xticks(rotation=45)

# Calculate means for each group
means = df.groupby(x_coord)['bleu_score'].mean()

# Add mean labels to the plot
for i, count in enumerate(sorted(df[x_coord].unique())):
    mean_val = means[count]
    ax.text(i, mean_val, f'{mean_val:.2f}', 
            horizontalalignment='center', size='small', color='black', weight='semibold')



plt.tight_layout()
plt.show()