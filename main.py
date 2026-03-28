# main.py
# A simple command line version of the Student Performance Predictor.
# If you don't want to run the web app, you can just use this from the terminal.
# Run it with: python main.py

import os
from model import predict_performance, load_artifacts, main as train_main

# terminal color codes to make output readable
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

BANNER = f"""
{CYAN}{BOLD}╔══════════════════════════════════════════════╗
║     🎓  Student Performance Predictor       ║
║        Powered by Decision Tree ML          ║
╚══════════════════════════════════════════════╝{RESET}
"""


def get_float_input(prompt, min_val, max_val):
    # keeps asking until the user gives a valid number in the correct range
    while True:
        try:
            val = float(input(f"  {prompt}: "))
            if min_val <= val <= max_val:
                return val
            else:
                print(f"  {YELLOW}Please enter a value between {min_val} and {max_val}.{RESET}")
        except ValueError:
            print(f"  {YELLOW}That doesn't look like a number. Try again.{RESET}")


def run_prediction():
    print(f"\n{BOLD}Enter the student's details below:{RESET}\n")

    study_hours    = get_float_input("Study hours per day      (0 to 12)",    0, 12)
    sleep_hours    = get_float_input("Sleep hours per night    (3 to 10)",    3, 10)
    attendance     = get_float_input("Attendance percentage    (0 to 100)",   0, 100)
    previous_marks = get_float_input("Previous exam marks      (0 to 100)",   0, 100)

    # train model if it hasn't been trained yet
    if not os.path.exists("model.pkl"):
        print(f"\n  {YELLOW}Model not found, training it now...{RESET}")
        train_main()

    model, le = load_artifacts()
    result = predict_performance(study_hours, sleep_hours, attendance, previous_marks, model, le)

    # print result with color
    print("\n" + "─" * 48)
    if result == "Good":
        print(f"  {GREEN}{BOLD}Result: {result} Performance ✅{RESET}")
        print(f"  {GREEN}This student is likely to do well!{RESET}")
    elif result == "Average":
        print(f"  {YELLOW}{BOLD}Result: {result} Performance ⚠️{RESET}")
        print(f"  {YELLOW}There's room for improvement in a few areas.{RESET}")
    else:
        print(f"  {RED}{BOLD}Result: {result} Performance ❌{RESET}")
        print(f"  {RED}This student needs help and focused attention.{RESET}")
    print("─" * 48)

    # give suggestions based on what was entered
    print(f"\n  {CYAN}Some suggestions:{RESET}")
    count = 0
    if study_hours < 4:
        print("   → Study at least 4-6 hours a day.")
        count += 1
    if sleep_hours < 6:
        print("   → Try to get 7-8 hours of sleep each night.")
        count += 1
    if attendance < 75:
        print("   → Attendance is low, aim for at least 75%.")
        count += 1
    if previous_marks < 50:
        print("   → Work on strengthening fundamentals and practice more.")
        count += 1
    if count == 0:
        print(f"   {GREEN}No issues found - keep it up! 🌟{RESET}")


def main():
    print(BANNER)

    # check if we already have a trained model
    if not os.path.exists("model.pkl"):
        print(f"  {YELLOW}Running for the first time - training the model...{RESET}\n")
        train_main()
    else:
        accuracy = 86.05
        if os.path.exists("accuracy.txt"):
            with open("accuracy.txt") as f:
                accuracy = float(f.read().strip())
        print(f"  {GREEN}Model ready | Accuracy: {BOLD}{accuracy}%{RESET}\n")

    # keep predicting until user says no
    while True:
        run_prediction()
        print()
        again = input("  Predict for another student? (y/n): ").strip().lower()
        if again != "y":
            print(f"\n  {CYAN}Thanks for using the predictor. Goodbye! 👋{RESET}\n")
            break


if __name__ == "__main__":
    main()
