import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from evaluation2 import get_results

plt.figure(figsize=(6, 6))
df = get_results(onlyv2=True)


ax = sns.boxplot(x='few_shot_count', y='bleu_score', data=df)
plt.title('BLEU Score vs. Few-Shot Count')
plt.xlabel('Few-Shot Count')
plt.ylabel('BLEU Score')

# Calculate means for each group
means = df.groupby('few_shot_count')['bleu_score'].mean()

# Add mean labels to the plot
for i, count in enumerate(sorted(df['few_shot_count'].unique())):
    mean_val = means[count]
    ax.text(i, mean_val, f'{mean_val:.2f}', 
            horizontalalignment='center', size='small', color='black', weight='semibold')

plt.tight_layout()
plt.show()