#!/usr/bin/env python3
"""
Process video files with various filters.

Usage:
    python process_input.py INPUT_VIDEO [OUTPUT_VIDEO] --filter FILTER [filter-options]

Examples:
    python process_input.py input/video.mp4 --filter sobel --threshold 0.05
    python process_input.py input/video.mp4 --filter cartoon --color-levels 6
    python process_input.py input/video.mp4 --filter thermal --colormap hot
    python process_input.py input/video.mp4 --filter pixel --pixel-size 32
"""

import sys
import argparse
from pathlib import Path
from sources import VideoFileSource
from pipeline import VideoPipeline
from filters import list_filters, get_filter_info, create_filter


def build_parser():
    """Build argument parser with dynamically generated filter arguments."""
    parser = argparse.ArgumentParser(
        description='Process video files with various filters',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        'input_video',
        type=str,
        help='Path to input video file'
    )

    parser.add_argument(
        'output_video',
        type=str,
        nargs='?',
        help='Path to output video file (default: output/<filter>_<input_name>)'
    )

    parser.add_argument(
        '--filter',
        type=str,
        choices=list_filters(),
        default='sobel',
        help='Filter to apply (default: sobel)'
    )

    # General options
    parser.add_argument(
        '--preview',
        action='store_true',
        help='Show preview window during processing'
    )

    parser.add_argument(
        '--no-audio',
        action='store_true',
        help='Do not preserve audio from source video'
    )

    # Add arguments for all filters
    for filter_name in list_filters():
        filter_info = get_filter_info(filter_name)
        params = filter_info['parameters']

        for param_name, param_info in params.items():
            arg_name = f'--{param_name}'
            arg_kwargs = {
                'help': param_info.get('help', ''),
                'default': param_info.get('default')
            }

            param_type = param_info.get('type')

            if param_type == bool:
                arg_kwargs['action'] = 'store_true'
                arg_kwargs.pop('default', None)
            else:
                arg_kwargs['type'] = param_type

            if 'choices' in param_info:
                arg_kwargs['choices'] = param_info['choices']

            parser.add_argument(arg_name, **arg_kwargs)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Setup paths
    input_path = Path(args.input_video)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    if args.output_video:
        output_path = Path(args.output_video)
    else:
        output_path = Path("output") / f"{args.filter}_{input_path.name}"

    # Get filter info
    filter_info = get_filter_info(args.filter)
    params = filter_info['parameters']

    # Build filter kwargs from parsed args
    filter_kwargs = {}
    for param_name in params.keys():
        arg_value = getattr(args, param_name.replace('-', '_'), None)
        if arg_value is not None:
            filter_kwargs[param_name.replace('-', '_')] = arg_value

    # Create filter
    filter_obj = create_filter(args.filter, **filter_kwargs)

    # Print configuration
    print(f"Filter: {filter_info['description']}")
    for param_name, value in filter_kwargs.items():
        print(f"  {param_name}: {value}")

    print(f"\nInput: {input_path}")
    print(f"Output: {output_path}")
    print(f"Preview: {args.preview}")
    print(f"Preserve audio: {not args.no_audio}\n")

    # Initialize source and run pipeline
    source = VideoFileSource(input_path)
    pipeline = VideoPipeline(source, filter_obj, source_path=input_path)
    pipeline.process_to_file(
        output_path,
        show_preview=args.preview,
        preserve_audio=not args.no_audio
    )


if __name__ == "__main__":
    main()
