import pandas as pd
from collections import Counter

# Load data
df = pd.read_csv('teams.csv')

print("=" * 80)
print("DATASET OVERVIEW")
print("=" * 80)
print(f"Total rows: {len(df):,}")
print(f"Unique companies: {df['company'].nunique()}")
print(f"Unique teams (raw): {df['team'].nunique():,}")
print(f"\nCompany distribution:")
print(df['company'].value_counts())

print("\n" + "=" * 80)
print("SAMPLE DATA BY COMPANY")
print("=" * 80)
for company in df['company'].unique():
    if company != 'company':  # Skip header if present
        company_df = df[df['company'] == company]
        print(f"\n{company} - {len(company_df):,} rows, {company_df['team'].nunique()} unique teams")
        print("Sample teams:")
        print(company_df['team'].value_counts().head(15))

print("\n" + "=" * 80)
print("PATTERN ANALYSIS")
print("=" * 80)

# Common suffixes
all_teams = df['team'].dropna().unique()
suffixes = []
for team in all_teams:
    words = team.lower().split()
    if words:
        suffixes.append(words[-1])

print("\nMost common last words (potential suffixes):")
suffix_counts = Counter(suffixes)
for suffix, count in suffix_counts.most_common(20):
    print(f"  {suffix}: {count}")
