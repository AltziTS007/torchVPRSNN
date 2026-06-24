#!/usr/bin/env python3
import os
import shutil
import argparse
import csv

def prepare_nordland(raw_dir, out_dir):
    seasons = ['spring', 'summer', 'fall', 'winter']
    tools_dir = os.path.dirname(os.path.abspath(__file__))
    
    for season in seasons:
        list_file = os.path.join(tools_dir, "dataset_imagenames", f"nordland_imageNames_{season}.txt")
        if not os.path.exists(list_file):
            print(f"Warning: {list_file} not found. Skipping {season}.")
            continue
            
        raw_season_dir = os.path.join(raw_dir, season)
        out_season_dir = os.path.join(out_dir, season)
        os.makedirs(out_season_dir, exist_ok=True)
        
        with open(list_file, 'r') as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
            
        print(f"Copying {len(lines)} files for Nordland {season}...")
        copied = 0
        missing = 0
        for line in lines:
            filename = os.path.basename(line)
            src = os.path.join(raw_season_dir, filename)
            dst = os.path.join(out_season_dir, filename)
            
            if os.path.exists(src):
                shutil.copy2(src, dst)
                copied += 1
            else:
                missing += 1
                
        print(f"Nordland {season}: Copied {copied}/{len(lines)} frames (Missing: {missing})")

def prepare_oxford(raw_dir, out_dir, condition):
    if condition is None:
        print("Error: --condition must be specified for Oxford dataset (Sun, Rain, Dusk).")
        return

    cond_map = {
        'Sun': 'ORC_sun/2015-08-12-15-04-18',
        'Rain': 'ORC_rain/2015-10-29-12-18-17',
        'Dusk': 'ORC_dusk/2015-11-12-13-27-51'
    }
    
    if condition not in cond_map:
        print(f"Error: Invalid condition {condition}. Choose from Sun, Rain, Dusk.")
        return

    tools_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file = os.path.join(tools_dir, "dataset_imagenames", f"ORC_{condition}_timestamps.csv")
    
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found.")
        return
        
    out_stereo_dir = os.path.join(out_dir, cond_map[condition], 'stereo', 'left')
    os.makedirs(out_stereo_dir, exist_ok=True)
    
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        next(reader) # skip header
        mappings = list(reader)
        
    print(f"Copying and renaming {len(mappings)} files for Oxford {condition}...")
    copied = 0
    missing = 0
    for row in mappings:
        repo_name = row[0]
        raw_name = row[1]
        
        src = os.path.join(raw_dir, raw_name)
        dst = os.path.join(out_stereo_dir, repo_name)
        
        if os.path.exists(src):
            shutil.copy2(src, dst)
            copied += 1
        else:
            missing += 1
            
    print(f"Oxford {condition}: Copied {copied}/{len(mappings)} frames (Missing: {missing})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset Preparation Utility for VPR SNN")
    parser.add_argument("--dataset", type=str, choices=["nordland", "oxford"], required=True, help="Which dataset to prepare")
    parser.add_argument("--raw-dir", type=str, required=True, help="Path to raw images (for Nordland: parent of season folders. For Oxford: parent of stereo images)")
    parser.add_argument("--out-dir", type=str, required=True, help="Path where the clean dataset will be saved (e.g., nordland_clean/data or ORC)")
    parser.add_argument("--condition", type=str, choices=["Sun", "Rain", "Dusk"], help="Oxford dataset condition to prepare")
    
    args = parser.parse_args()
    
    print(f"Preparing {args.dataset} dataset...")
    if args.dataset == "nordland":
        prepare_nordland(args.raw_dir, args.out_dir)
    elif args.dataset == "oxford":
        prepare_oxford(args.raw_dir, args.out_dir, args.condition)
