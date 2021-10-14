import pandas as pd
import numpy as np

df=pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_1.csv")

# Percentage of the missing value in each attribute
missing_value = df.isnull().sum()/df.count()*100
print(missing_value)
print(df.dtypes)

# Calculating number of launches on each site
launch_site = df["LaunchSite"].value_counts()
print(launch_site)

# Calculating number and occurrence each orbit
orbit = df["Orbit"].value_counts("Orbit")
print(orbit)

# Calculating number and outcomes of mission outcome per orbit type
landing_outcomes = df["Outcome"].value_counts()
print(landing_outcomes)



for i,outcome in enumerate(landing_outcomes.keys()):
    print(i,outcome)


# Instance where second stage not landed successfully
bad_outcomes=set(landing_outcomes.keys()[[1,3,5,6,7]])
print(bad_outcomes)


# Landing outcome labels
landing_class = []
for key,value in df["Outcome"].items():
     if value in bad_outcomes:
        landing_class.append(0)
     else:
        landing_class.append(1)
df['Class'] = landing_class
print(df['Class'])

mean = df['Class'].mean()
print(mean)

df.to_csv("dataset_part_2.csv", index=False)