import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("C:/Users/aboda/Downloads/daily_weather_mock.csv", parse_dates=["Date"])
df.set_index("Date", inplace=True)


df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)


print("\nSummary Statistics:\n", df.describe())
print("\nWeather Condition Counts:\n", df["Weather_Condition"].value_counts())


plt.figure(figsize=(8,5))
sns.histplot(df["Temperature_Avg"], kde=True, bins=30)
plt.title("Distribution of Average Temperature")
plt.show()

plt.figure(figsize=(6,4))
df["Weather_Condition"].value_counts().plot(kind='bar', color='skyblue')
plt.title("Weather Condition Frequency")
plt.ylabel("Days")
plt.show()


df['Temperature_Avg'].resample('M').mean().plot(figsize=(10,4), title="Monthly Average Temperature")
plt.ylabel("Avg Temp (\u00b0C)")
plt.show()

df[['Humidity', 'Precipitation']].resample('M').mean().plot(figsize=(10,4), title="Humidity and Precipitation Trends")
plt.ylabel("Average")
plt.show()


plt.figure(figsize=(8,5))
sns.scatterplot(data=df, x="Temperature_Avg", y="Humidity", alpha=0.5)
plt.title("Temperature vs Humidity")
plt.show()

plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

plt.figure(figsize=(8,5))
sns.boxplot(data=df, x="Weather_Condition", y="Temperature_Avg")
plt.title("Temperature Distribution by Weather Condition")
plt.show()


hottest_day = df["Temperature_Max"].idxmax()
coldest_day = df["Temperature_Min"].idxmin()
print("\nHottest day:", hottest_day)
print("Coldest day:", coldest_day)

correlation = df["Temperature_Avg"].corr(df["Humidity"])
print("\nCorrelation between Temperature and Humidity:", correlation)

rainy_days_temp = df[df["Weather_Condition"] == "Rainy"]["Temperature_Avg"]
non_rainy_days_temp = df[df["Weather_Condition"] != "Rainy"]["Temperature_Avg"]
print("\nAverage Temp on Rainy Days:", rainy_days_temp.mean())
print("Average Temp on Non-Rainy Days:", non_rainy_days_temp.mean())

print("\nAverage Wind Speed by Weather Condition:\n", df.groupby("Weather_Condition")["Wind_Speed"].mean())
