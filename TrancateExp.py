import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# Example input
df = pd.DataFrame({
    'Position': ['Cloud Developer', 'Data Engineer', 'SDN Developer'],
    'Skill1': ['devOps', 'couchbase', 'maven'],
    'Skill2': ['aws', 'kafka', 'opendaylight'],
    'Skill3': ['', '', 'onos']
})

# Combine skills into one comma-separated string
df['all_skills'] = df[['Skill1', 'Skill2', 'Skill3']].fillna('').agg(','.join, axis=1)
df['all_skills'] = df['all_skills'].str.replace(',,', ',').str.strip(',')

print(df)

vectorizer = TfidfVectorizer(tokenizer=lambda x: x.split(','), lowercase=False)
X_tfidf = vectorizer.fit_transform(df['all_skills'])  # This is a sparse matrix
print("----------------------------")
print(vectorizer.get_feature_names_out())
print(X_tfidf)

svd = TruncatedSVD(n_components=3, random_state=42)  # compress to 30 features
X_skills_reduced = svd.fit_transform(X_tfidf)

print(X_skills_reduced)

# Create feature names
skill_feature_names = [f'skill_svd_{i}' for i in range(X_skills_reduced.shape[1])]
df_svd = pd.DataFrame(X_skills_reduced, columns=skill_feature_names)

# Merge into original dataframe
df_final = pd.concat([df, df_svd], axis=1)

print(df_final)
df_final.to_csv('output.csv', index=False)