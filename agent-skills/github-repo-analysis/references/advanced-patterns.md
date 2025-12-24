# Advanced GitHub Repository Analysis Patterns

## Visualization Examples

### Commit Activity Heatmap

Generate a heatmap showing commit activity by day of week and hour:

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def create_commit_heatmap(commits):
    # Extract timestamps
    timestamps = [datetime.fromisoformat(c['commit']['author']['date'].replace('Z', '+00:00')) 
                  for c in commits]
    
    # Create DataFrame
    df = pd.DataFrame({
        'day_of_week': [t.weekday() for t in timestamps],
        'hour': [t.hour for t in timestamps]
    })
    
    # Create heatmap data
    heatmap_data = df.groupby(['day_of_week', 'hour']).size().unstack(fill_value=0)
    
    # Plot
    plt.figure(figsize=(12, 6))
    sns.heatmap(heatmap_data, cmap='YlOrRd', cbar_kws={'label': 'Commits'})
    plt.xlabel('Hour of Day')
    plt.ylabel('Day of Week')
    plt.title('Commit Activity Heatmap')
    plt.yticks([0, 1, 2, 3, 4, 5, 6], ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    plt.tight_layout()
    plt.savefig('commit_heatmap.png')
```

### Contributor Growth Over Time

Track how the contributor base has grown:

```python
def analyze_contributor_growth(commits):
    contributors_over_time = []
    seen_contributors = set()
    
    # Sort commits chronologically
    sorted_commits = sorted(commits, 
                           key=lambda c: c['commit']['author']['date'])
    
    for commit in sorted_commits:
        author = commit['commit']['author']['name']
        seen_contributors.add(author)
        date = datetime.fromisoformat(commit['commit']['author']['date'].replace('Z', '+00:00'))
        
        contributors_over_time.append({
            'date': date,
            'total_contributors': len(seen_contributors),
            'is_new_contributor': author not in seen_contributors
        })
    
    return contributors_over_time
```

### Release Velocity Chart

Visualize release frequency trends:

```python
def plot_release_velocity(releases):
    if not releases:
        return
    
    release_dates = [datetime.fromisoformat(r['published_at'].replace('Z', '+00:00')) 
                     for r in releases]
    
    plt.figure(figsize=(12, 6))
    plt.plot(release_dates, range(len(release_dates), 0, -1), marker='o')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Releases')
    plt.title('Release Timeline')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('release_velocity.png')
```

## Advanced Metrics

### Commit Size Analysis

Analyze the size of commits (additions/deletions):

```python
def analyze_commit_sizes(owner, repo, token):
    headers = {'Authorization': f'token {token}'} if token else {}
    
    commits_detail = []
    # Fetch detailed commit data (limited to recent commits to avoid rate limits)
    for page in range(1, 4):  # First 300 commits
        url = f'https://api.github.com/repos/{owner}/{repo}/commits'
        response = requests.get(url, headers=headers, 
                               params={'page': page, 'per_page': 100})
        
        for commit in response.json():
            sha = commit['sha']
            detail_url = f'https://api.github.com/repos/{owner}/{repo}/commits/{sha}'
            detail = requests.get(detail_url, headers=headers).json()
            
            commits_detail.append({
                'sha': sha,
                'additions': detail['stats']['additions'],
                'deletions': detail['stats']['deletions'],
                'total_changes': detail['stats']['total']
            })
    
    return commits_detail
```

### Bus Factor Analysis

Estimate project risk based on contributor concentration:

```python
def calculate_bus_factor(contributor_stats):
    """
    Calculate bus factor: minimum contributors responsible for 50% of commits
    """
    total_commits = sum(c['commits'] for c in contributor_stats)
    threshold = total_commits * 0.5
    
    cumulative = 0
    bus_factor = 0
    
    for contributor in contributor_stats:
        cumulative += contributor['commits']
        bus_factor += 1
        if cumulative >= threshold:
            break
    
    return {
        'bus_factor': bus_factor,
        'interpretation': 'Low risk' if bus_factor >= 3 else 'High risk',
        'top_contributor_percentage': contributor_stats[0]['percentage'] if contributor_stats else 0
    }
```

### Activity Trends

Detect increasing or decreasing activity:

```python
def analyze_activity_trends(commits_by_month):
    """
    Calculate 3-month moving average to detect trends
    """
    months = sorted(commits_by_month.keys())
    values = [commits_by_month[m] for m in months]
    
    moving_avg = []
    for i in range(len(values) - 2):
        avg = sum(values[i:i+3]) / 3
        moving_avg.append({
            'month': months[i+1],
            'commits': values[i+1],
            'moving_average': round(avg, 1)
        })
    
    # Detect trend
    if len(moving_avg) >= 2:
        recent_avg = sum(ma['moving_average'] for ma in moving_avg[-3:]) / 3
        older_avg = sum(ma['moving_average'] for ma in moving_avg[:3]) / 3
        trend = 'increasing' if recent_avg > older_avg * 1.2 else \
                'decreasing' if recent_avg < older_avg * 0.8 else 'stable'
    else:
        trend = 'insufficient_data'
    
    return {
        'trend': trend,
        'monthly_data': moving_avg
    }
```

## GitHub GraphQL API Examples

For more efficient queries, use GraphQL:

```python
import requests

def graphql_query(query, token):
    headers = {'Authorization': f'bearer {token}'}
    response = requests.post('https://api.github.com/graphql', 
                            json={'query': query}, 
                            headers=headers)
    return response.json()

# Example: Get commit count and contributor count efficiently
query = """
{
  repository(owner: "OWNER", name: "REPO") {
    defaultBranchRef {
      target {
        ... on Commit {
          history(first: 100) {
            totalCount
            edges {
              node {
                author {
                  name
                  email
                }
                committedDate
              }
            }
          }
        }
      }
    }
    releases(first: 100) {
      totalCount
      edges {
        node {
          name
          publishedAt
          tagName
        }
      }
    }
  }
}
"""
```

## Git-based Analysis (Local Repository)

### Detailed Contributor Analysis

```bash
# Get detailed contributor stats
git log --all --format='%aN <%aE>' | sort -u > contributors.txt

# Commits per contributor with date range
git log --all --format='%aN|%aI' | awk -F'|' '
{
    author=$1
    commits[author]++
    if (!first[author]) first[author]=$2
    last[author]=$2
}
END {
    for (a in commits) {
        print a "|" commits[a] "|" first[a] "|" last[a]
    }
}'

# Lines of code per contributor (current state)
git ls-files | while read file; do git blame -w -M -C "$file"; done | \
    awk '{print $2}' | sort | uniq -c | sort -rn
```

### File Change Frequency

```bash
# Most frequently changed files
git log --all --name-only --format="" | sort | uniq -c | sort -rn | head -20

# Files changed per commit (churn analysis)
git log --all --oneline --numstat | awk '
    NF==3 {files++}
    NF==1 {print files; files=0}
' | awk '{sum+=$1; count++} END {print "Average files per commit:", sum/count}'
```

### Commit Message Analysis

```python
def analyze_commit_messages(commits):
    """
    Analyze commit message patterns
    """
    patterns = {
        'fix': 0,
        'feat': 0,
        'refactor': 0,
        'docs': 0,
        'test': 0,
        'chore': 0,
        'other': 0
    }
    
    for commit in commits:
        message = commit['commit']['message'].lower()
        
        categorized = False
        for pattern in patterns:
            if message.startswith(pattern):
                patterns[pattern] += 1
                categorized = True
                break
        
        if not categorized:
            patterns['other'] += 1
    
    return patterns
```

## Performance Optimization Tips

### Caching Results

```python
import json
from pathlib import Path

def cache_commits(owner, repo, commits):
    cache_dir = Path('.cache')
    cache_dir.mkdir(exist_ok=True)
    
    cache_file = cache_dir / f'{owner}_{repo}_commits.json'
    with open(cache_file, 'w') as f:
        json.dump(commits, f)

def load_cached_commits(owner, repo):
    cache_file = Path(f'.cache/{owner}_{repo}_commits.json')
    if cache_file.exists():
        with open(cache_file) as f:
            return json.load(f)
    return None
```

### Parallel Processing

```python
from concurrent.futures import ThreadPoolExecutor
import requests

def fetch_commit_details_parallel(owner, repo, shas, token, max_workers=5):
    headers = {'Authorization': f'token {token}'} if token else {}
    
    def fetch_one(sha):
        url = f'https://api.github.com/repos/{owner}/{repo}/commits/{sha}'
        return requests.get(url, headers=headers).json()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(fetch_one, shas))
    
    return results
```

## Export Formats

### CSV Export

```python
import csv

def export_to_csv(contributors, filename='contributors.csv'):
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['name', 'email', 'commits', 'percentage'])
        writer.writeheader()
        writer.writerows(contributors)
```

### Markdown Report

```python
def generate_markdown_report(report):
    md = f"""# {report['repository']} Analysis

## Overview
- **Total Commits**: {report['commits']['total']}
- **Contributors**: {report['contributors']['total_contributors']}
- **Releases**: {report['releases']['total']}

## Top Contributors
| Rank | Name | Commits | Percentage |
|------|------|---------|------------|
"""
    
    for i, c in enumerate(report['contributors']['top_10'], 1):
        md += f"| {i} | {c['name']} | {c['commits']} | {c['percentage']}% |\n"
    
    return md
```
