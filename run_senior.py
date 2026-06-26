"""Entry point script for the Senior Citizen Identification system.

Usage:
    python run_senior.py [--source SOURCE] [--output OUTPUT] [--format FORMAT]
                         [--gui] [--model-dir MODEL_DIR]

Examples:
    python run_senior.py --source video.mp4
    python run_senior.py --source 0 --gui
    python run_senior.py --source video.avi --output results.csv --format csv
"""

from src.senior.app import SeniorCitizenApp

if __name__ == "__main__":
    SeniorCitizenApp().run()
