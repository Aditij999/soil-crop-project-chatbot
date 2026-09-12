import os
import json


BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")


class ConversationContext:
    """Tracks the active crop within a chat session for follow-up questions."""

    def __init__(self, default_crop=None):
        self.default_crop = default_crop
        self.active_crop = default_crop

    def set_active_crop(self, crop):
        if crop:
            self.active_crop = crop

    def reset(self, default_crop):
        self.default_crop = default_crop
        self.active_crop = default_crop


def load_latest_report():
    """
    Load the most recently generated Smart Agri report.
    """

    report_files = [
        file for file in os.listdir(OUTPUT_DIR)
        if file.startswith("report_") and file.endswith(".json")
    ]

    if not report_files:
        raise FileNotFoundError(
            "No Smart Agri report found. Run pipeline.py first."
        )

    latest_file = max(
        report_files,
        key=lambda file: os.path.getmtime(
            os.path.join(OUTPUT_DIR, file)
        )
    )

    report_path = os.path.join(OUTPUT_DIR, latest_file)

    with open(report_path, "r", encoding="utf-8") as file:
        report = json.load(file)

    return report


if __name__ == "__main__":
    report = load_latest_report()

    print("Latest Smart Agri report loaded.")
    print(f"Crop: {report['chosen_crop']}")
