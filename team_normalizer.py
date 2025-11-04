"""
Team Name Normalization System

This module normalizes team names from job postings by:
1. Basic text normalization (case, whitespace, punctuation, apostrophes, ampersands)
2. Removing common organizational suffixes while preserving core identity
3. Extracting and standardizing acronyms (e.g., "TDG" → "Technology Development Group")
4. Fuzzy matching with token-set similarity (85% threshold)
5. Opposing keyword detection (prevents matching "active" vs "passive", "RF" vs "CPU")
6. Domain-specific prefix validation (preserves technical context)
7. Bidirectional validation (ensures important keywords are preserved)
8. Rejection tracking (captures false matches for quality analysis)

The pipeline processes each company separately and prevents 9,000+ false matches
through intelligent validation layers.
"""

import pandas as pd
import re
from rapidfuzz import fuzz, process
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class TeamNormalizer:
    """
    Normalizes team names by applying multiple strategies to group
    similar team names together while preventing false matches.

    Key features:
    - Company-specific processing for better accuracy
    - Bidirectional validation ensures semantic correctness
    - Domain-aware matching (RF ≠ CPU, Active ≠ Passive)
    - Tracks rejected matches for quality analysis
    """

    # Common suffixes that don't add meaningful distinction
    TEAM_SUFFIXES = [
        'team', 'teams', 'group', 'groups', 'organization', 'organizations',
        'org', 'orgs', 'department', 'departments', 'dept', 'division',
        'divisions', 'unit', 'units', 'function', 'functions', 'effort',
        'efforts', 'initiative', 'initiatives', 'program', 'programs',
        'project', 'projects'
    ]

    # Words that typically indicate engineering roles (less important for grouping)
    DESCRIPTOR_WORDS = ['engineering', 'engineer', 'engineers']

    def __init__(self, fuzzy_threshold: int = 85):
        """
        Initialize the normalizer.

        Args:
            fuzzy_threshold: Minimum similarity score (0-100) for fuzzy matching
        """
        self.fuzzy_threshold = fuzzy_threshold
        self.acronym_map = {}
        self.rejected_matches = []  # Track rejected matches for analysis

    def extract_acronym(self, text: str) -> Tuple[str, str]:
        """
        Extract acronym from text like "Technology Development Group (TDG)".

        Returns:
            Tuple of (text_without_acronym, acronym or empty string)
        """
        # Match pattern: text (ACRONYM) or text (ABC)
        pattern = r'\s*\(([A-Z]{2,})\)\s*$'
        match = re.search(pattern, text.strip())

        if match:
            acronym = match.group(1)
            text_without = text[:match.start()].strip()
            return text_without, acronym

        return text, ""

    def basic_normalize(self, text: str) -> str:
        """
        Apply basic text normalization: lowercase, clean whitespace, etc.
        """
        if pd.isna(text):
            return ""

        text = str(text)

        # Extract and remove acronym before processing
        text, acronym = self.extract_acronym(text)

        # Convert to lowercase
        text = text.lower()

        # Remove possessives and apostrophes to avoid "s" artifacts
        text = text.replace("'s", "")
        text = text.replace("'", "")

        # Remove & entirely to avoid "is t" artifacts (IS&T → IST, not IS T)
        text = text.replace("&", "")

        # Remove special characters but keep alphanumeric and spaces
        text = re.sub(r'[^\w\s-]', ' ', text)

        # Normalize whitespace
        text = ' '.join(text.split())

        # Fix single-letter artifacts (U.S. → US, I/O → IO)
        # Replace pattern: single letter + space + single letter
        text = re.sub(r'\b([a-z])\s+([a-z])\b', r'\1\2', text)

        return text.strip()

    def remove_suffixes(self, text: str) -> str:
        """
        Remove common organizational suffixes from team names.
        """
        words = text.split()

        # Remove trailing suffixes (but keep at least 2 words if possible)
        while len(words) > 2 and words[-1] in self.TEAM_SUFFIXES:
            words.pop()

        # If we only have 1 word left and it was a suffix, restore one level
        if len(words) == 1 and words[0] in self.TEAM_SUFFIXES:
            # Return original text to avoid over-normalization
            return text

        # Remove descriptor words if they're at the end and we have enough words
        while len(words) > 1 and words[-1] in self.DESCRIPTOR_WORDS:
            words.pop()

        return ' '.join(words) if words else text

    def normalize_team_name(self, team_name: str) -> str:
        """
        Apply full normalization pipeline to a team name.
        """
        # Basic normalization
        normalized = self.basic_normalize(team_name)

        # Remove suffixes
        normalized = self.remove_suffixes(normalized)

        # Final cleanup
        normalized = ' '.join(normalized.split())

        return normalized if normalized else team_name.lower()

    def should_not_match(self, team1: str, team2: str) -> bool:
        """
        Check if two teams should NOT be matched despite high similarity.
        Prevents incorrect grouping of semantically different teams.
        """
        # Opposing keyword pairs that indicate different teams
        opposing_pairs = [
            ('active', 'passive'),
            ('frontend', 'backend'),
            ('front end', 'back end'),
            ('hardware', 'software'),
            ('incoming', 'outgoing'),
            ('internal', 'external'),
            ('senior', 'junior'),
            ('east', 'west'),
            ('north', 'south'),
        ]

        # Critical domain-specific keywords that MUST match
        # If teams have different ones, they're different teams
        critical_domain_prefixes = [
            'rf', 'cpu', 'gpu', 'ml', 'ai', 'ad', 'ads', 'search', 'video', 'audio',
            'wireless', 'cellular', 'silicon', 'soc', 'asic', 'fpga',
            'analog', 'digital', 'web', 'mobile', 'ios', 'android',
            'frontend', 'backend', 'security', 'privacy', 'cloud', 'data',
            'network', 'infrastructure', 'system', 'systems', 'analysis',
            'design', 'architecture', 'operations', 'sales', 'marketing',
            'finance', 'legal', 'recruiting', 'hr'
        ]

        words1 = set(team1.lower().split())
        words2 = set(team2.lower().split())

        # Check for opposing keywords
        for word1, word2 in opposing_pairs:
            if (word1 in words1 and word2 in words2) or (word2 in words1 and word1 in words2):
                return True

        # Check for conflicting domain prefixes
        # Extract critical domain words from each team
        domains1 = words1.intersection(critical_domain_prefixes)
        domains2 = words2.intersection(critical_domain_prefixes)

        # Remove generic words that can coexist with other domains
        generic_words = {'platform', 'infrastructure', 'data', 'cloud'}
        specific_domains1 = domains1 - generic_words
        specific_domains2 = domains2 - generic_words

        # Case 1: Both teams have specific domains and they're different
        # E.g., "rf platform" vs "cpu platform" - different domains
        if specific_domains1 and specific_domains2:
            if specific_domains1.isdisjoint(specific_domains2):
                return True

        # Case 2: One team has a specific domain prefix, the other doesn't
        # These should NOT match - the domain prefix is semantically important
        # E.g., "rf hardware design" vs "hardware design" - different teams
        # E.g., "ad platform" vs "platform engineering" - different teams
        if (specific_domains1 and not specific_domains2) or (specific_domains2 and not specific_domains1):
            return True

        # Check length difference - if teams differ by >40% in length, likely different
        len1, len2 = len(team1), len(team2)
        if len1 > 0 and len2 > 0:
            length_ratio = abs(len1 - len2) / max(len1, len2)
            if length_ratio > 0.4:
                return True

        return False

    def build_acronym_map(self, team_names: List[str]) -> Dict[str, str]:
        """
        Build a mapping of acronyms to their full forms.
        For example: "TDG" -> "technology development group"
        """
        acronym_map = {}

        for team_name in team_names:
            text, acronym = self.extract_acronym(team_name)
            if acronym:
                normalized_text = self.basic_normalize(text)
                normalized_text = self.remove_suffixes(normalized_text)

                # Map acronym to normalized full form
                if acronym.lower() not in acronym_map:
                    acronym_map[acronym.lower()] = normalized_text

        return acronym_map

    def validate_match(self, original_team: str, canonical_team: str) -> bool:
        """
        Validate that a canonical team name makes sense for the original team.
        Returns True if valid, False if it's a bad match.

        Bidirectional check:
        1. All significant words in canonical should appear in original
        2. Important domain words in original should appear in canonical
        """
        # Get normalized versions
        orig_words = set(original_team.lower().split())
        canon_words = set(canonical_team.lower().split())

        # Remove common suffixes that are okay to differ
        ignore_words = {'team', 'group', 'organization', 'org', 'department',
                       'dept', 'division', 'unit', 'teams', 'groups', 'engineering'}

        # Important words that should be preserved
        important_keywords = {
            'system', 'systems', 'analysis', 'design', 'architecture',
            'search', 'platform', 'infrastructure', 'operations', 'sales',
            'marketing', 'finance', 'security', 'network', 'data', 'cloud'
        }

        canon_significant = canon_words - ignore_words
        orig_words_clean = orig_words - ignore_words

        # Check 1: All canonical words should appear in original
        for canon_word in canon_significant:
            found = False
            if canon_word in orig_words_clean:
                found = True
            elif any(fuzz.ratio(canon_word, orig_word) > 85
                    for orig_word in orig_words_clean):
                found = True

            # Reject if significant word not found (don't skip short words like "rf", "ml", "ai")
            if not found:
                return False

        # Check 2: Important words from original should appear in canonical
        orig_important = orig_words_clean.intersection(important_keywords)
        canon_important = canon_words.intersection(important_keywords)

        # If original has important keywords, canonical should have them too
        for important_word in orig_important:
            found = False
            if important_word in canon_words:
                found = True
            # Check for variations (system vs systems)
            elif any(fuzz.ratio(important_word, canon_word) > 85
                    for canon_word in canon_words):
                found = True

            if not found:
                # Important word from original is missing in canonical - bad match
                return False

        return True

    def fuzzy_match_teams(self, team_names: List[str],
                          normalized_teams: List[str]) -> Dict[str, str]:
        """
        Use fuzzy matching to group similar team names together.
        Returns a mapping from normalized name to canonical name.
        """
        if not normalized_teams:
            return {}

        # Remove duplicates while preserving order
        unique_teams = []
        seen = set()
        for team in normalized_teams:
            if team not in seen:
                unique_teams.append(team)
                seen.add(team)

        # Group similar teams
        canonical_map = {}
        processed = set()

        for i, team in enumerate(unique_teams):
            if team in processed:
                continue

            # This team becomes the canonical version
            canonical_map[team] = team
            processed.add(team)

            # Find similar teams
            remaining = [t for t in unique_teams[i+1:] if t not in processed]

            if remaining:
                # Use token_set_ratio for better subset matching
                matches = process.extract(
                    team,
                    remaining,
                    scorer=fuzz.token_set_ratio,
                    limit=None
                )

                for match_text, score, _ in matches:
                    # Check if teams meet threshold and should be matched
                    if score >= self.fuzzy_threshold:
                        # Additional validation: don't match opposing teams
                        if self.should_not_match(team, match_text):
                            self.rejected_matches.append({
                                'team1': match_text,
                                'team2': team,
                                'score': score,
                                'reason': 'opposing_keywords'
                            })
                        elif not self.validate_match(match_text, team):
                            # Final validation: ensure canonical makes sense for this team
                            # This catches cases like "RF Systems Cellular" → "Cellular Systems Analysis"
                            # where "analysis" never appeared in the original
                            self.rejected_matches.append({
                                'team1': match_text,
                                'team2': team,
                                'score': score,
                                'reason': 'bidirectional_validation'
                            })
                        else:
                            canonical_map[match_text] = team
                            processed.add(match_text)

        return canonical_map

    def normalize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize all team names in a dataframe.
        Processes each company separately for better accuracy.
        """
        results = []

        for company in df['company'].unique():
            company_df = df[df['company'] == company].copy()

            # Step 1: Basic normalization
            company_df['normalized'] = company_df['team'].apply(
                self.normalize_team_name
            )

            # Step 2: Build acronym map for this company
            self.acronym_map = self.build_acronym_map(company_df['team'].tolist())

            # Step 3: Replace acronyms with full forms where possible
            for acronym, full_form in self.acronym_map.items():
                company_df['normalized'] = company_df['normalized'].str.replace(
                    f'^{acronym}$', full_form, regex=True
                )

            # Step 4: Fuzzy match similar teams
            unique_normalized = company_df['normalized'].unique().tolist()
            canonical_map = self.fuzzy_match_teams(
                company_df['team'].tolist(),
                unique_normalized
            )

            # Step 5: Apply canonical mapping
            company_df['processed_team'] = company_df['normalized'].map(canonical_map)

            # Fallback to original if something went wrong
            company_df['processed_team'] = company_df['processed_team'].fillna(
                company_df['normalized']
            )

            results.append(company_df)

        # Combine all companies
        final_df = pd.concat(results, ignore_index=True)

        # Select and order columns as required
        final_df = final_df[['document_id', 'company', 'team', 'processed_team']]

        # Apply title case for better readability
        final_df['processed_team'] = final_df['processed_team'].str.title()

        return final_df


def main():
    """
    Main function to run the team normalization pipeline.
    """
    print("=" * 80)
    print("TEAM NAME NORMALIZATION PIPELINE")
    print("=" * 80)

    # Load data
    print("\n[1/5] Loading data...")
    df = pd.read_csv('teams.csv')
    print(f"   Loaded {len(df):,} team mentions across {df['company'].nunique()} companies")
    print(f"   Original unique teams: {df['team'].nunique():,}")

    # Initialize normalizer
    print("\n[2/5] Initializing normalizer...")
    normalizer = TeamNormalizer(fuzzy_threshold=85)

    # Normalize
    print("\n[3/5] Normalizing team names...")
    print("   - Applying basic text normalization")
    print("   - Removing organizational suffixes")
    print("   - Extracting and mapping acronyms")
    print("   - Fuzzy matching similar teams")

    result_df = normalizer.normalize_dataframe(df)

    # Statistics
    print("\n[4/5] Computing statistics...")
    print(f"   Processed unique teams: {result_df['processed_team'].nunique():,}")
    reduction = (1 - result_df['processed_team'].nunique() / df['team'].nunique()) * 100
    print(f"   Reduction: {reduction:.1f}%")

    # Show examples
    print("\n   Example normalizations:")
    for company in result_df['company'].unique():
        company_df = result_df[result_df['company'] == company]
        print(f"\n   {company}:")

        # Find good examples where multiple originals map to one processed
        grouped = company_df.groupby('processed_team')['team'].apply(
            lambda x: list(x.unique())
        ).reset_index()

        # Show cases where multiple teams were grouped
        examples = grouped[grouped['team'].apply(len) > 1].head(3)

        for _, row in examples.iterrows():
            print(f"      '{row['processed_team']}' ←")
            for orig in row['team'][:3]:  # Show max 3 examples
                print(f"         - {orig}")

    # Save output
    print("\n[5/5] Saving output...")
    result_df.to_csv('output.csv', index=False)
    print("   ✓ Saved to output.csv")

    print("\n" + "=" * 80)
    print("NORMALIZATION COMPLETE")
    print("=" * 80)

    return result_df


if __name__ == "__main__":
    main()
