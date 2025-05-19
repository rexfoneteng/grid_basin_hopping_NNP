#!/usr/bin/env python3
# _*_ coding: utf-8 _*_
# ==============================================================
# Copyright(c) 2025-, Huu Trong Phan (phanhuutrong93@gmail.com)
# See the README file in the top-level directory for license.
# ==============================================================
# Creation Time : 20250511 10:24:52
# ==============================================================
#!/usr/bin/env python3
# monitor_progress.py
import os
import glob
import json
import time
from datetime import datetime

def monitor_progress(output_dir="output", interval=300):
    """Monitor progress of all basin hopping jobs"""
    print(f"Starting progress monitor - checking every {interval} seconds")
    print(f"Press Ctrl+C to stop")

    try:
        while True:
            job_dirs = sorted(glob.glob(os.path.join(output_dir, "job_*")))
            total_jobs = len(job_dirs)
            completed_jobs = 0
            total_accepted = 0
            total_rejected = 0
            best_energy = float('inf')
            best_job = None

            print("\n" + "="*80)
            print(f"Progress report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("="*80)

            for job_dir in job_dirs:
                job_id = os.path.basename(job_dir)
                stats_file = os.path.join(job_dir, "statistics.json")

                if os.path.exists(stats_file):
                    try:
                        with open(stats_file, "r") as f:
                            stats = json.load(f)

                        if stats.get("stopping_reason"):
                            completed_jobs += 1

                        total_accepted += stats.get("accepted_steps", 0)
                        total_rejected += stats.get("rejected_steps", 0)

                        if "best_energy" in stats and stats["best_energy"] < best_energy:
                            best_energy = stats["best_energy"]
                            best_job = job_id
                    except:
                        pass

            if total_jobs > 0:
                print(f"Jobs: {completed_jobs}/{total_jobs} completed ({completed_jobs/total_jobs*100:.1f}%)")
                print(f"Structures: {total_accepted} accepted, {total_rejected} rejected")
                if best_job:
                    print(f"Best energy so far: {best_energy:.6f} (in {best_job})")
            else:
                print("No jobs found")

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\nMonitoring stopped")

if __name__ == "__main__":
    monitor_progress()
