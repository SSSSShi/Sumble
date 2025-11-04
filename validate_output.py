"""
Validation and Analysis Script

This script analyzes the normalization results to provide quality metrics
and insights into the team name normalization pipeline.
"""

import pandas as pd
from collections import Counter
from team_normalizer import TeamNormalizer


def analyze_normalization():
    """Analyze the normalization results and print quality metrics."""

    # Load input and output
    input_df = pd.read_csv('teams.csv')
    output_df = pd.read_csv('output.csv')

    print("=" * 80)
    print("NORMALIZATION QUALITY ANALYSIS")
    print("=" * 80)

    # Overall statistics
    print("\n1. OVERALL STATISTICS")
    print("-" * 80)
    print(f"Total records: {len(output_df):,}")
    print(f"Original unique teams: {input_df['team'].nunique():,}")
    print(f"Normalized unique teams: {output_df['processed_team'].nunique():,}")
    reduction = (1 - output_df['processed_team'].nunique() / input_df['team'].nunique()) * 100
    print(f"Reduction: {reduction:.1f}%")

    # Per-company analysis
    print("\n2. PER-COMPANY BREAKDOWN")
    print("-" * 80)
    for company in output_df['company'].unique():
        company_input = input_df[input_df['company'] == company]
        company_output = output_df[output_df['company'] == company]

        orig_unique = company_input['team'].nunique()
        norm_unique = company_output['processed_team'].nunique()
        comp_reduction = (1 - norm_unique / orig_unique) * 100 if orig_unique > 0 else 0

        print(f"\n{company}:")
        print(f"  Records: {len(company_output):,}")
        print(f"  Original unique teams: {orig_unique:,}")
        print(f"  Normalized unique teams: {norm_unique:,}")
        print(f"  Reduction: {comp_reduction:.1f}%")

    # Consolidation examples
    print("\n3. TOP CONSOLIDATIONS (Multiple teams → One normalized name)")
    print("-" * 80)
    for company in ['Apple', 'Nvidia', 'OpenAI', 'Tesla']:
        company_df = output_df[output_df['company'] == company]

        # Group by processed_team and count unique original teams
        consolidations = company_df.groupby('processed_team')['team'].apply(
            lambda x: list(x.unique())
        ).reset_index()

        consolidations['count'] = consolidations['team'].apply(len)
        consolidations = consolidations[consolidations['count'] > 1].sort_values(
            'count', ascending=False
        )

        if len(consolidations) > 0:
            print(f"\n{company} - Top 5 consolidations:")
            for idx, (_, row) in enumerate(consolidations.head(5).iterrows(), 1):
                print(f"  {idx}. '{row['processed_team']}' ← {row['count']} variations")
                for orig in row['team'][:3]:
                    print(f"     • {orig}")
                if len(row['team']) > 3:
                    print(f"     • ... and {len(row['team']) - 3} more")

    # Most common normalized teams
    print("\n4. MOST COMMON NORMALIZED TEAMS (per company)")
    print("-" * 80)
    for company in ['Apple', 'Nvidia', 'OpenAI', 'Tesla']:
        company_df = output_df[output_df['company'] == company]
        top_teams = company_df['processed_team'].value_counts().head(5)

        print(f"\n{company}:")
        for team, count in top_teams.items():
            print(f"  {count:>4}x  {team}")

    # Bidirectional validation report
    print("\n" + "=" * 80)
    print("5. BIDIRECTIONAL VALIDATION REPORT")
    print("=" * 80)
    print("\nRe-running normalizer to collect rejection statistics...")

    normalizer = TeamNormalizer(fuzzy_threshold=85)
    input_df = pd.read_csv('teams.csv')
    _ = normalizer.normalize_dataframe(input_df)

    rejected = normalizer.rejected_matches

    print(f"\nTotal matches rejected: {len(rejected):,}")

    if rejected:
        # Count by reason
        reasons = Counter([r['reason'] for r in rejected])
        print(f"\nRejection reasons:")
        for reason, count in reasons.items():
            reason_label = {
                'opposing_keywords': 'Opposing keywords (active/passive, RF/CPU, etc.)',
                'bidirectional_validation': 'Bidirectional validation (missing important keywords)'
            }.get(reason, reason)
            print(f"  {reason_label}: {count:,}")

        # Show examples of bidirectional validation rejections
        bidir_rejections = [r for r in rejected if r['reason'] == 'bidirectional_validation']
        if bidir_rejections:
            print(f"\nTop 15 examples of bidirectional validation rejections:")
            print("(These would have been FALSE MATCHES without validation)\n")

            # Sort by score (highest first - these were closest to being matched)
            bidir_rejections.sort(key=lambda x: x['score'], reverse=True)

            for i, rejection in enumerate(bidir_rejections[:15], 1):
                print(f"{i:2}. Score: {rejection['score']}")
                print(f"    ✗ '{rejection['team1']}'")
                print(f"      would have matched → '{rejection['team2']}'")
                print()

        # Show examples of opposing keyword rejections
        opposing_rejections = [r for r in rejected if r['reason'] == 'opposing_keywords']
        if opposing_rejections:
            print(f"\nTop 10 examples of opposing keyword rejections:")
            print("(Teams with conflicting domains like RF vs CPU, Active vs Passive)\n")

            opposing_rejections.sort(key=lambda x: x['score'], reverse=True)

            for i, rejection in enumerate(opposing_rejections[:10], 1):
                print(f"{i:2}. Score: {rejection['score']}")
                print(f"    ✗ '{rejection['team1']}'")
                print(f"      vs '{rejection['team2']}'")
                print()

    print("=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    analyze_normalization()
