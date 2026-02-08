# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "argparse",
#     "matplotlib",
#     "psycopg2-binary",
#     "numpy",
#     "python-dotenv",
# ]
# ///

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime, timezone
import matplotlib.dates as mdates
import psycopg2
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from urllib.parse import urlparse
from collections import defaultdict

def connect_to_database():
    """Connect to PostgreSQL database using environment variables"""
    # Get connection parameters
    db_url = os.getenv('DATABASE_URL')
    if not db_url:
        print("Error: DATABASE_URL environment variable is required")
        sys.exit(1)

    result = urlparse(db_url)
    user = result.username
    password = result.password
    database = result.path[1:]
    host = result.hostname
    port = result.port

    print(f"Connecting to database: {user}@{host}:{port}/{database}")

    try:
        conn = psycopg2.connect(
            host=host,
            port=port,
            database=database,
            user=user,
            password=password
        )
        return conn
    except psycopg2.Error as e:
        print(f"Error connecting to database: {e}")
        print("Make sure PostgreSQL is running and environment variables are set correctly")
        sys.exit(1)

def fetch_submission_data(conn):
    """Fetch submissions from database"""
    query = """
    SELECT
        s.id as submission_id,
        s.field_id,
        f.base_id as base,
        f.range_size,
        s.search_mode,
        s.submit_time,
        s.elapsed_secs,
        s.username,
        s.client_version,
        s.disqualified
    FROM submissions s
    LEFT JOIN fields f ON s.field_id = f.id
    WHERE
        s.disqualified = false;
    """

    try:
        with conn.cursor() as cur:
            cur.execute(query)
            results = cur.fetchall()
    except psycopg2.Error as e:
        print(f"Error executing query: {e}")
        sys.exit(1)

    submission_dicts = []
    for row in results:
        submission_dicts.append({
            'submission_id': row[0],
            'field_id': row[1],
            'base': row[2],
            'range_size': row[3],
            'search_mode': row[4],
            'submit_time': row[5],
            'elapsed_secs': row[6],
            'username': row[7],
            'client_version': row[8],
            'disqualified': row[9]
        })

    return submission_dicts

def total_progress_bar_chart(submissions):
    # Create a bar chart of range_size completed per submission time period
    # Group by date (day) and separate anvil from others
    daily_ranges_anvil = defaultdict(int)
    daily_ranges_others = defaultdict(int)

    # Calculate index: daily average range_size per second for specified user
    daily_index_range = defaultdict(float)
    daily_index_time = defaultdict(float)

    for submission in submissions:
        if submission['submit_time'] and submission['range_size']:
            # Extract date (without time)
            date = submission['submit_time'].date()
            if submission['username'] == 'anvil':
                daily_ranges_anvil[date] += submission['range_size']
            else:
                daily_ranges_others[date] += submission['range_size']

            # Track index user's range/time for calculating average
            if submission['username'] == 'pailiah' and submission['elapsed_secs']:
                daily_index_range[date] += float(submission['range_size'])
                daily_index_time[date] += float(submission['elapsed_secs'])

    # Sort by date (union of all dates)
    all_dates = set(daily_ranges_anvil.keys()) | set(daily_ranges_others.keys())
    sorted_dates = sorted(all_dates)

    # Convert dates to datetime objects for matplotlib
    date_objects = [datetime.combine(date, datetime.min.time()) for date in sorted_dates]
    anvil_totals = [daily_ranges_anvil[date] for date in sorted_dates]
    others_totals = [daily_ranges_others[date] for date in sorted_dates]

    # Calculate index values (average range_size per second)
    index_values = []
    index_dates = []
    for date in sorted_dates:
        if daily_index_time[date] > 0:
            avg_rate = daily_index_range[date] / daily_index_time[date]
            baseline_rate = 1.4e6
            index_val = avg_rate / baseline_rate
            index_values.append(index_val)
            index_dates.append(datetime.combine(date, datetime.min.time()))

    # Create the stacked bar chart with secondary y-axis for index
    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax1.bar(date_objects, others_totals, color='steelblue', alpha=0.7, label='Others')
    ax1.bar(date_objects, anvil_totals, bottom=others_totals, color='orange', alpha=0.7, label='Anvil')
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Total Range Size Completed', fontsize=12)
    ax1.set_title('Range Size Completed per Day', fontsize=14, fontweight='bold')

    # Add secondary y-axis for the index
    ax2 = ax1.twinx()
    if index_values:
        ax2.plot(index_dates, index_values, color='black', linewidth=1, marker='',
                label=f'Pailiah Index', alpha=0.8)
        ax2.set_ylabel(f'Pailiah Index (Nice-Only Multiplier)', fontsize=12, color='black')
        ax2.tick_params(axis='y', labelcolor='black')

    # Format x-axis with dates
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=0, ha='center')

    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels() if index_values else ([], [])
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    ax1.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    # Save the chart
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f'progress_chart_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nChart saved to: {output_file}")

def main():
    # Load environment variables
    load_dotenv()

    # Connect to database
    print("Connecting to database...")
    conn = connect_to_database()
    print("Successfully connected to database")

    # Fetch submissions
    print("Fetching data...")
    submissions = fetch_submission_data(conn)
    print(f"Found {len(submissions)} submissions")

    # Filter the points of interest
    cutoff_date = datetime(2025, 10, 1, tzinfo=timezone.utc)
    submissions = [s for s in submissions if s['search_mode'] == 'niceonly' and s['submit_time'] > cutoff_date]
    print(f"Filtered to {len(submissions)} submissions")

    # Create a total progress bar chart
    print("Generating charts...")
    # Configure the index username here
    total_progress_bar_chart(submissions)

    # Close database connection
    conn.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
