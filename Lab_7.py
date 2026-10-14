#####################################################
# Bryan Hernandez Rodriguez
# Lab 7
# OLS Regression
# This program reads a CSV file and makes a
# scatterplot and regression line.
#####################################################

import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm


# clean column names
def to_snake_case(name):
    name = str(name).strip()
    name = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name)
    name = re.sub(r"[^0-9a-zA-Z]+", "_", name)
    return name.strip("_").lower()


# yes or no question
def ask_yes_no(prompt):
    while True:
        answer = input(prompt + " (y/n): ").strip().lower()

        if answer == "y" or answer == "yes":
            return True
        elif answer == "n" or answer == "no":
            return False
        else:
            print("Please enter y or n.")


# choose a column
def ask_column(prompt, columns):
    while True:
        choice = to_snake_case(input(prompt).strip())

        if choice in columns:
            return choice
        else:
            print("That column was not found. Try again.")


# save the graph
def save_figure(fig, default_name):
    name = input("Enter a file name or press Enter to use the default: ").strip()

    if name == "":
        name = default_name

    if not name.endswith(".png"):
        name = name + ".png"

    fig.savefig(name, dpi=300)
    print("Saved:", name)


def main():

    # get the csv file
    while True:
        path = input("Enter the dataset file path (CSV): ").strip()
        path = path.strip('"').strip("'")

        try:
            df = pd.read_csv(path)
            break
        except FileNotFoundError:
            print("File not found. Try again.")
        except Exception as e:
            print("Could not read file:", e)

    print("Rows:", len(df))
    print("Columns:", len(df.columns))


    # clean column names
    df.columns = [to_snake_case(col) for col in df.columns]


    # remove names column if there is one
    if "names" in df.columns:
        df = df.drop(columns=["names"])


    # show columns
    print("\nAvailable columns:")

    for col in df.columns:
        print("-", col)


    # choose X
    x_col = ask_column(
        "Enter the column name for the X variable: ",
        df.columns
    )


    # choose Y
    y_col = ask_column(
        "Enter the column name for the Y variable: ",
        df.columns
    )


    # make sure X and Y are not the same
    while y_col == x_col:
        print("X and Y cannot be the same.")
        y_col = ask_column(
            "Choose a different Y variable: ",
            df.columns
        )


    # remove missing values
    data = df[[x_col, y_col]].apply(
        pd.to_numeric,
        errors="coerce"
    ).dropna()


    if len(data) < 3:
        print("Not enough data for regression.")
        return


    print("Rows after cleaning:", len(data))


    # scatterplot
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(
        data[x_col],
        data[y_col],
        alpha=0.6
    )

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"{y_col} vs {x_col}")
    ax.grid(alpha=0.3)

    fig.tight_layout()

    plt.show(block=False)
    plt.pause(0.5)


    # save scatterplot
    if ask_yes_no("Save the scatterplot?"):
        save_figure(
            fig,
            f"{x_col}_vs_{y_col}_scatter.png"
        )


    # OLS regression
    X = sm.add_constant(data[x_col])
    model = sm.OLS(data[y_col], X).fit()

    print("\nOLS Regression Summary:")
    print(model.summary())


    # regression line
    x_line = np.linspace(
        data[x_col].min(),
        data[x_col].max(),
        100
    )

    y_line = model.params["const"] + model.params[x_col] * x_line


    # add line to graph
    ax.plot(
        x_line,
        y_line,
        linewidth=2,
        label=f"OLS line R2 = {model.rsquared:.3f}"
    )

    ax.set_title(f"{y_col} vs {x_col} with regression line")
    ax.legend()

    fig.canvas.draw_idle()
    plt.pause(0.5)


    # save final graph
    if ask_yes_no("Save the graph with regression line?"):
        save_figure(
            fig,
            f"{x_col}_vs_{y_col}_ols.png"
        )

    plt.show()


if __name__ == "__main__":
    main()






