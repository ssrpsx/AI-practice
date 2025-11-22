import pandas as pd
# import os
# print(os.getcwd())

df = pd.read_csv('pandas/pokemon_data.csv')
# print(df.head(3))
# print(df.tail(3))

# df_xlsx = pd.read_excel('pokemon_data.xlsx')
# print(df_xlsx)

# df = pd.read_csv('pokemon_data.txt', delimiter='\t')

# Read Headers
# print(df.columns)

# Read Each Column
# print(df[['Name', 'Type 1', 'HP']][0:5])

# Read Each Row
# print(df.head(4))
# print(df.iloc[1:4])

# for index, row in df.iterrows():
#     print(index, row['Name'], '\n')

# print(df.loc[df['Type 1'] == "Grass"])

# Read a specific location such as Search Name is Venusaur
# print(df.iloc[2, 1])


## Sorting/Describing Data
# print(df.describe())
# print(df.sort_values(['Type 1', 'HP'], ascending=[True, False]))

## Making changes to the data
# df['Total'] = df['HP'] + df['Attack'] + df['Defense'] + df['Sp. Atk'] + df['Sp. Def'] + df['Speed']
# print(df, '\n')

# df = df.drop(columns=['Total'])
# print(df, '\n')

# df['Total'] = df.iloc[:, 4:10].sum(axis=1)
# print(df, '\n')

## Save into CSV file
# df.to_csv('pandas/modified.csv', index=False)
# df.to_csv('pandas/modified.txt', index=False, sep='\t')

## Filtering Data
# print(df.loc[(df['Type 1'] == 'Grass') & (df['Type 2'] == 'Poison')])

# u can push ~ in the front of condition, it mean opposite sign
# print(df.loc[~df['Name'].str.contains('Mega')])

# re.I ignore Uppercase and lowercase letters
# import re
# print(df.loc[df['Type 1'].str.contains('fire|grass', flags=re.I ,regex=True)])

# (* or star) means str according to pi with 0 number or more && ^ The operation is The first string condition.
# print(df.loc[df['Name'].str.contains('^pi[a-z]*', flags=re.I ,regex=True)])

## Conditional Changes
# ส่วนแรกคือเงืื่อนไข และตรง Legendary คือ column ที่จะเปลี่ยน
# df.loc[df['Type 1'] == 'Fire', 'Legendary'] = True

# df.loc[df['Total'] > 500, ['Generation', 'Legendary']] = ['TEST VALUE', 'TEST TEST']
# print(df)

## Aggregate Statistics (GROUP BY)
# print(df.groupby(['Type 1']).mean(numeric_only=True).sort_values('Attack', ascending=False))

# df['count'] = 1
# print(df.groupby(['Type 1', 'Type 2']).count()['count'])

## working with large amounts of data
# for df in pd.read_csv('pandas/modified.csv', chunksize=5):
    # print("Chunksize : ")
    # print(df)