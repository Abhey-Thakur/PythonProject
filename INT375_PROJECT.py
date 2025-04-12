import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the data
df = pd.read_csv('innings_deliveries.csv')

# Data cleaning
# Convert empty strings to NaN in player_out and wicket_kind columns
df['player_out'] = df['player_out'].replace('', np.nan)
df['wicket_kind'] = df['wicket_kind'].replace('', np.nan)

# Create a flag for wicket deliveries
df['is_wicket'] = df['player_out'].notna()

# Create a batsman-team mapping since team column represents batting team
df['batter_team'] = df['team']

# Set style for visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = [12, 6]

# Visualization 1: Runs Distribution by Team (Histogram)
plt.figure(figsize=(12, 6))
sns.histplot(data=df, x='runs_total', hue='team', bins=range(0, 8), 
             multiple='dodge', shrink=0.8, palette=['red', 'blue'])
plt.title('Distribution of Runs per Delivery by Team')
plt.xlabel('Runs per Delivery')
plt.ylabel('Count of Deliveries')
plt.xticks(range(0, 7))
plt.legend(title='Team')
plt.show()

# Visualization 2: Top Scorers (Bar Chart)
batter_runs = df.groupby(['batter', 'batter_team'])['runs_batter'].sum().reset_index()
top_scorers = batter_runs.sort_values('runs_batter', ascending=False).head(8)

plt.figure(figsize=(12, 6))
sns.barplot(data=top_scorers, x='batter', y='runs_batter', hue='batter_team',
            palette=['red', 'blue'], dodge=False)
plt.title('Top Run Scorers in the Match')
plt.xlabel('Batsman')
plt.ylabel('Total Runs')
plt.legend(title='Team')
plt.show()

# Visualization 3: Wickets by Bowler (Pie Chart)
wickets = df[df['is_wicket']].groupby('bowler').size().reset_index(name='wickets')
top_bowlers = wickets.sort_values('wickets', ascending=False).head(5)

plt.figure(figsize=(8, 8))
plt.pie(top_bowlers['wickets'], labels=top_bowlers['bowler'], autopct='%1.1f%%',
        startangle=90, colors=sns.color_palette('pastel'))
plt.title('Top Wicket-Takers in the Match')
plt.show()

# Visualization 4: Run Rate Progression (Line Plot)
# Calculate cumulative runs over time
df['cumulative_runs'] = df.groupby('team')['runs_total'].cumsum()
df['cumulative_balls'] = df.groupby('team').cumcount() + 1
df['run_rate'] = (df['cumulative_runs'] / df['cumulative_balls']) * 6

plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x='cumulative_balls', y='run_rate', hue='team', 
             palette=['red', 'blue'], linewidth=2.5)
plt.title('Run Rate Progression Throughout the Innings')
plt.xlabel('Balls Bowled')
plt.ylabel('Run Rate (per over)')
plt.legend(title='Team')
plt.grid(True, alpha=0.3)
plt.show()

# Visualization 5: Dismissal Types (Bar Chart)
wicket_types = df[df['is_wicket']]['wicket_kind'].value_counts().reset_index()
wicket_types.columns = ['wicket_kind', 'count']

plt.figure(figsize=(10, 6))
sns.barplot(data=wicket_types, x='wicket_kind', y='count', 
            palette=sns.color_palette('husl'))
plt.title('Types of Dismissals in the Match')
plt.xlabel('Dismissal Type')
plt.ylabel('Count')
plt.show()

# Visualization 6: Heatmap of Runs per Over
# Create a pivot table of runs per over per team
runs_per_over = df.pivot_table(index='over', columns='team', 
                              values='runs_total', aggfunc='sum')

plt.figure(figsize=(12, 6))
sns.heatmap(runs_per_over, cmap='YlOrRd', annot=True, fmt='g', 
            linewidths=0.5, cbar_kws={'label': 'Runs Scored'})
plt.title('Runs Scored per Over by Each Team')
plt.xlabel('Team')
plt.ylabel('Over Number')
plt.show()
