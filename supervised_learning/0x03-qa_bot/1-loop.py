#!/usr/bin/env python3
"""Question answer loop script prototype."""


def question_loop():
    """Qustion answer loop."""

    leave = ["exit", "quit", "goodbye", "bye"]
    A = ""

    while True:
        print("Q: ", end="")
        q = input()
        if q in leave:
            A = "Goodbye"
        print("A: ", A)

        if A == "Goodbye":
            break

if __name__ == "__main__":
    question_loop()
