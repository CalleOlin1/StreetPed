from paralane_to_drivestudio import convert_clip
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a ParLane clip to DriveStudio format.")
    parser.add_argument("input_clip", type=str, help="Path to the input ParLane clip.")
    parser.add_argument("output_clip", type=str, help="Path to the output DriveStudio clip.")
    args = parser.parse_args()
    # Example cli usage
    

    convert_clip(args.input_clip, args.output_clip)