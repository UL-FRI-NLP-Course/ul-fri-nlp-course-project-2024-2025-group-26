import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from evaluation2 import get_results

df = get_results(onlyv2=True)
plt.figure(figsize=(6, 6))
ax = sns.boxplot(x='model_type', y='bleu_score', data=df)
plt.title('BLEU Score vs. Model Type')
plt.xlabel('Model Type')
plt.ylabel('BLEU Score')
plt.xticks(rotation=45)

# Calculate means for each group
means = df.groupby('model_type')['bleu_score'].mean()

# Add mean labels to the plot
for i, count in enumerate(sorted(df['model_type'].unique())):
    mean_val = means[count]
    ax.text(i, mean_val, f'{mean_val:.2f}', 
            horizontalalignment='center', size='small', color='black', weight='semibold')



plt.tight_layout()
plt.show()