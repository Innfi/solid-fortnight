#!/usr/bin/env python3
"""
GitHub Repository Analyzer
Analyzes commit frequency, contributors, and release timeline for a GitHub repository.
"""

import requests
import argparse
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import json


class GitHubAnalyzer:
    def __init__(self, owner, repo, token=None):
        self.owner = owner
        self.repo = repo
        self.token = token
        self.headers = {'Authorization': f'token {token}'} if token else {}
        self.base_url = f'https://api.github.com/repos/{owner}/{repo}'
    
    def _make_request(self, url, params=None):
        """Make API request with error handling"""
        response = requests.get(url, headers=self.headers, params=params)
        
        if response.status_code == 404:
            raise ValueError(f"Repository {self.owner}/{self.repo} not found")
        elif response.status_code == 403:
            raise ValueError("API rate limit exceeded. Use authentication token.")
        elif response.status_code != 200:
            raise ValueError(f"API error: {response.status_code}")
        
        return response
    
    def fetch_all_commits(self, max_pages=None):
        """Fetch all commits with pagination"""
        commits = []
        page = 1
        
        while True:
            if max_pages and page > max_pages:
                break
            
            url = f'{self.base_url}/commits'
            response = self._make_request(url, params={'page': page, 'per_page': 100})
            page_commits = response.json()
            
            if not page_commits:
                break
            
            commits.extend(page_commits)
            page += 1
            
            # Check rate limit
            remaining = int(response.headers.get('X-RateLimit-Remaining', 0))
            if remaining < 10:
                print(f"Warning: Only {remaining} API requests remaining")
        
        return commits
    
    def fetch_releases(self):
        """Fetch all releases"""
        url = f'{self.base_url}/releases'
        response = self._make_request(url)
        return response.json()
    
    def analyze_commit_frequency(self, commits):
        """Analyze commit frequency by time period"""
        commits_by_day = defaultdict(int)
        commits_by_week = defaultdict(int)
        commits_by_month = defaultdict(int)
        
        for commit in commits:
            date_str = commit['commit']['author']['date']
            date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            
            day = date.strftime('%Y-%m-%d')
            week = date.strftime('%Y-W%W')
            month = date.strftime('%Y-%m')
            
            commits_by_day[day] += 1
            commits_by_week[week] += 1
            commits_by_month[month] += 1
        
        return {
            'by_day': dict(sorted(commits_by_day.items())),
            'by_week': dict(sorted(commits_by_week.items())),
            'by_month': dict(sorted(commits_by_month.items())),
            'total': len(commits)
        }
    
    def analyze_contributors(self, commits):
        """Analyze contributor statistics"""
        contributor_commits = Counter()
        contributor_emails = {}
        
        for commit in commits:
            author = commit['commit']['author']['name']
            email = commit['commit']['author']['email']
            
            contributor_commits[author] += 1
            contributor_emails[author] = email
        
        total_commits = len(commits)
        contributors = []
        
        for author, count in contributor_commits.most_common():
            contributors.append({
                'name': author,
                'email': contributor_emails[author],
                'commits': count,
                'percentage': round((count / total_commits) * 100, 2)
            })
        
        return {
            'total_contributors': len(contributors),
            'contributors': contributors,
            'top_10': contributors[:10]
        }
    
    def analyze_releases(self, releases):
        """Analyze release timeline"""
        if not releases:
            return {'total': 0, 'releases': []}
        
        release_data = []
        
        for i, release in enumerate(releases):
            published = datetime.fromisoformat(release['published_at'].replace('Z', '+00:00'))
            
            days_since_previous = None
            if i < len(releases) - 1:
                prev_published = datetime.fromisoformat(
                    releases[i + 1]['published_at'].replace('Z', '+00:00')
                )
                days_since_previous = (published - prev_published).days
            
            release_data.append({
                'name': release['name'] or release['tag_name'],
                'tag': release['tag_name'],
                'published_at': release['published_at'],
                'days_since_previous': days_since_previous,
                'prerelease': release['prerelease']
            })
        
        # Calculate average time between releases
        intervals = [r['days_since_previous'] for r in release_data if r['days_since_previous']]
        avg_interval = sum(intervals) / len(intervals) if intervals else None
        
        return {
            'total': len(releases),
            'releases': release_data,
            'average_days_between_releases': round(avg_interval, 1) if avg_interval else None
        }
    
    def generate_report(self, commits, releases):
        """Generate comprehensive analysis report"""
        commit_freq = self.analyze_commit_frequency(commits)
        contributors = self.analyze_contributors(commits)
        release_analysis = self.analyze_releases(releases)
        
        # Calculate time range
        if commits:
            first_commit_date = datetime.fromisoformat(
                commits[-1]['commit']['author']['date'].replace('Z', '+00:00')
            )
            last_commit_date = datetime.fromisoformat(
                commits[0]['commit']['author']['date'].replace('Z', '+00:00')
            )
        else:
            first_commit_date = last_commit_date = None
        
        return {
            'repository': f'{self.owner}/{self.repo}',
            'analysis_date': datetime.now().isoformat(),
            'time_range': {
                'first_commit': first_commit_date.isoformat() if first_commit_date else None,
                'last_commit': last_commit_date.isoformat() if last_commit_date else None,
                'days_active': (last_commit_date - first_commit_date).days if first_commit_date else None
            },
            'commits': commit_freq,
            'contributors': contributors,
            'releases': release_analysis
        }


def print_report(report):
    """Print formatted text report"""
    print(f"\n{'='*80}")
    print(f"GitHub Repository Analysis: {report['repository']}")
    print(f"{'='*80}\n")
    
    # Time range
    time_range = report['time_range']
    if time_range['first_commit']:
        print(f"Analysis Period: {time_range['first_commit'][:10]} to {time_range['last_commit'][:10]}")
        print(f"Days Active: {time_range['days_active']}\n")
    
    # Commit summary
    commits = report['commits']
    print(f"Commit Activity:")
    print(f"  Total Commits: {commits['total']}")
    if commits['by_month']:
        avg_per_month = commits['total'] / len(commits['by_month'])
        print(f"  Average Commits/Month: {avg_per_month:.1f}")
    print()
    
    # Top contributors
    contributors = report['contributors']
    print(f"Contributors:")
    print(f"  Total Contributors: {contributors['total_contributors']}")
    print(f"\nTop 10 Contributors:")
    for i, contrib in enumerate(contributors['top_10'], 1):
        print(f"  {i}. {contrib['name']} - {contrib['commits']} commits ({contrib['percentage']}%)")
    print()
    
    # Releases
    releases = report['releases']
    print(f"Releases:")
    print(f"  Total Releases: {releases['total']}")
    if releases['average_days_between_releases']:
        print(f"  Average Days Between Releases: {releases['average_days_between_releases']}")
    
    if releases['releases']:
        print(f"\nRecent Releases:")
        for release in releases['releases'][:5]:
            days_info = f" ({release['days_since_previous']} days since previous)" if release['days_since_previous'] else ""
            prerelease = " [PRERELEASE]" if release['prerelease'] else ""
            print(f"  - {release['name']} ({release['published_at'][:10]}){days_info}{prerelease}")
    
    print(f"\n{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description='Analyze GitHub repository')
    parser.add_argument('repo', help='Repository in format owner/repo')
    parser.add_argument('--token', help='GitHub personal access token')
    parser.add_argument('--max-commits', type=int, help='Maximum number of commits to analyze')
    parser.add_argument('--json', action='store_true', help='Output JSON format')
    
    args = parser.parse_args()
    
    # Parse repository
    try:
        owner, repo = args.repo.split('/')
    except ValueError:
        print("Error: Repository must be in format owner/repo")
        return
    
    # Analyze
    analyzer = GitHubAnalyzer(owner, repo, args.token)
    
    print(f"Fetching commits for {owner}/{repo}...")
    max_pages = args.max_commits // 100 if args.max_commits else None
    commits = analyzer.fetch_all_commits(max_pages=max_pages)
    
    print(f"Fetching releases...")
    releases = analyzer.fetch_releases()
    
    print(f"Generating report...")
    report = analyzer.generate_report(commits, releases)
    
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_report(report)


if __name__ == '__main__':
    main()
