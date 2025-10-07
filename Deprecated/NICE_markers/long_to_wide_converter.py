import os
import pandas as pd
import argparse
import glob


def convert_to_wide_format(input_file, output_file, analysis_type):
    """
    Convert data from long format to wide format where each row represents
    a subject-round combination, and columns represent different markers.
    
    Parameters:
    -----------
    input_file : str
        Path to the input CSV file (all_subjects_*.csv)
    output_file : str
        Path to save the wide format CSV
    analysis_type : str
        Type of analysis: 'whole_brain', 'per_electrode', or 'per_roi'
    """
    print(f"Converting {input_file} to wide format...")
    
    # Read the long format data
    df = pd.read_csv(input_file)
    
    # Make sure 'subject' and 'round' columns exist
    required_cols = ['subject_id', 'round']
    if not all(col in df.columns for col in required_cols):
        msg = f"Input file must contain columns: {required_cols}"
        raise ValueError(msg)
    
    # Convert round to string if it's not already
    df['round'] = df['round'].astype(str)
    
    # Process based on analysis type
    if analysis_type == 'whole_brain':
        # For whole brain analysis, markers as columns
        
        # Data is already in the format we want (each row is a subject-round combo)
        # We just need to make sure each marker is a column
        wide_df = df
        
    elif analysis_type == 'per_electrode':
        # For per-electrode analysis, one row per subject/round, cols for markers
        
        # Check if 'channel' and 'marker' columns exist
        req_cols = ['channel', 'marker', 'value']
        if not all(col in df.columns for col in req_cols):
            err_msg = "Per-electrode data must have required columns"
            raise ValueError(err_msg)
        
        # Pivot to get one row per subject-round with marker_channel as columns
        wide_df = df.pivot_table(
            index=['subject_id', 'round'],
            columns=['marker', 'channel'],
            values='value',
            aggfunc='mean'  # Use mean if there are duplicates
        )
        
        # Flatten the multi-level columns
        wide_df.columns = [
            f"{marker}_{channel}" 
            for marker, channel in wide_df.columns
        ]
        
        # Reset index to make 'subject' and 'round' regular columns
        wide_df.reset_index(inplace=True)
        
    elif analysis_type == 'per_roi':
        # For per-ROI analysis, one row per subject/round, columns for markers
        
        # Check if 'roi' and 'marker' columns exist
        roi_req_cols = ['roi', 'marker', 'value']
        if not all(col in df.columns for col in roi_req_cols):
            err_msg = "Per-ROI data must have required columns"
            raise ValueError(err_msg)
        
        # Pivot to get one row per subject-round with marker_roi as columns
        wide_df = df.pivot_table(
            index=['subject_id', 'round'],
            columns=['marker', 'roi'],
            values='value',
            aggfunc='mean'  # Use mean if there are duplicates
        )
        
        # Flatten the multi-level columns
        wide_df.columns = [
            f"{marker}_{roi}" 
            for marker, roi in wide_df.columns
        ]
        
        # Reset index to make 'subject' and 'round' regular columns
        wide_df.reset_index(inplace=True)
        
    else:
        raise ValueError(f"Unknown analysis type: {analysis_type}")
    
    # Save the wide format data
    wide_df.to_csv(output_file, index=False)
    print(f"Saved wide format data to {output_file}")
    
    n_rows = wide_df.shape[0]
    n_cols = wide_df.shape[1] - 2
    print(f"Converted data to wide format with {n_rows} rows")
    print(f"(subject-round combinations)")
    print(f"Each row has {n_cols} marker columns")
    
    return wide_df


def process_directory(input_dir, output_dir=None):
    """
    Process all marker files in the input directory and convert them to wide format.
    
    Parameters:
    -----------
    input_dir : str
        Directory containing the marker CSV files
    output_dir : str, optional
        Directory to save the wide format files, defaults to input_dir
    """
    if output_dir is None:
        output_dir = input_dir
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all marker files
    all_files = glob.glob(os.path.join(input_dir, 'all_subjects_*.csv'))
    
    if not all_files:
        print(f"No marker files found in {input_dir}")
        return
    
    print(f"Found {len(all_files)} marker files to convert")
    
    for file in all_files:
        # Determine analysis type based on filename
        basename = os.path.basename(file)
        
        if 'whole_brain' in basename:
            analysis_type = 'whole_brain'
        elif 'per_electrode' in basename:
            analysis_type = 'per_electrode'
        elif 'per_roi' in basename:
            analysis_type = 'per_roi'
        else:
            msg = f"Could not determine analysis type for {basename}, skipping"
            print(msg)
            continue
        
        # Create output filename
        output_file = os.path.join(
            output_dir,
            basename.replace('all_subjects_', 'all_subjects_wide_')
        )
        
        # Convert file
        try:
            convert_to_wide_format(file, output_file, analysis_type)
        except Exception as e:
            print(f"Error converting {basename}: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Convert EEG marker data from long to wide format'
    )
    
    default_input = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '..', 'Data', 'EEG', 'markers_output'
    )
    
    parser.add_argument(
        '--input_dir',
        type=str,
        default=default_input,
        help='Directory containing the marker CSV files'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '..', 'Data', 'EEG'),
        help='Directory to save the wide format files (defaults to input_dir)'
    )
    
    parser.add_argument(
        '--file',
        type=str,
        default=None,
        help='Specific CSV file to convert (overrides input_dir)'
    )
    
    parser.add_argument(
        '--type',
        type=str,
        choices=['whole_brain', 'per_electrode', 'per_roi'],
        default=None,
        help='Analysis type for specific file conversion'
    )
    
    args = parser.parse_args()
    
    if args.file:
        # Convert a specific file
        if not args.type:
            msg = "When converting a specific file, specify the analysis type"
            print(msg)
            parser.print_help()
            exit(1)
            
        # Determine output file
        output_file = args.file.replace('.csv', '_wide.csv')
        if args.output_dir:
            output_file = os.path.join(
                args.output_dir,
                os.path.basename(output_file)
            )
            
        convert_to_wide_format(args.file, output_file, args.type)
    else:
        # Process all files in directory
        process_directory(args.input_dir, args.output_dir) 