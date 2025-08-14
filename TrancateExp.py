import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA

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

# for testing set
df_test = pd.DataFrame({
    'Position': ['AI Developer', 'Software Engineer', 'System Engineer'],
    'Skill1': ['pytorch', 'maven', 'ubuntu'],
    'Skill2': ['LLM', 'kafka', ''],
    'Skill3': ['', 'couchbase', '']
})

# Combine skills into one comma-separated string
df_test['all_skills'] = df_test[['Skill1', 'Skill2', 'Skill3']].fillna('').agg(','.join, axis=1)
df_test['all_skills'] = df_test['all_skills'].str.replace(',,', ',').str.strip(',')


print(df)
print('************* Testing SVD *****************')
print(df_test)

pca = PCA(n_components=2)

vectorizer = TfidfVectorizer(tokenizer=lambda x: x.split(','), lowercase=False)
X_tfidf = vectorizer.fit_transform(df['all_skills'])  # This is a sparse matrix

X_tfidf_test = vectorizer.transform(df_test['all_skills'])

print("----------------------------")
print(vectorizer.get_feature_names_out())
print(X_tfidf)
print("-----------Testing Matrix -----------------")
df_test_matrix = pd.DataFrame(X_tfidf_test.toarray(), columns=vectorizer.get_feature_names_out())
print(df_test_matrix.head())
print("-----------xx -----------------")
print(X_tfidf_test)


svd = TruncatedSVD(n_components=3, random_state=42)  # compress to 30 features
X_skills_reduced = svd.fit_transform(X_tfidf)

X_skills_reduced_test = svd.transform(X_tfidf_test)

print("---- X Skills Reduced (Training) ----")
print(X_skills_reduced)
print("---- X Skills Reduced (Test) ----")
print(X_skills_reduced_test)


# Create feature names
skill_feature_names = [f'skill_svd_{i}' for i in range(X_skills_reduced.shape[1])]
df_svd = pd.DataFrame(X_skills_reduced, columns=skill_feature_names)

df_svd_test = pd.DataFrame(X_skills_reduced_test, columns=skill_feature_names)

# Merge into original dataframe
df_final = pd.concat([df, df_svd], axis=1)
df_final_test = pd.concat([df_test, df_svd_test], axis=1)

print('---------- DF Final (training) ----------')
print(df_final)
print('---------- DF Final (test) ----------')
print(df_final_test)
df_final.to_csv('output.csv', index=False)