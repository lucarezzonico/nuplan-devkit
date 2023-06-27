import pandas as pd
import numpy as np


# Create the DataFrame
data = {
    '& all': [0.39, 0.44, 0.48],
    '& changing lane': [0.34, 0.41, 0.45],
    '& stopping with lead': [0.79, 0.83, 0.71],
    '& starting left turn': [0.19, 0.29, 0.39],
    '& low magnitude speed': [0.49, 0.54, 0.43],
    '& starting right turn': [0.25, 0.35, 0.42],
    '& behind long vehicle': [0.64, 0.51, 0.59],
    '& high magnitude speed': [0.18, 0.43, 0.44],
    '& stationary in traffic': [0.84, 0.88, 0.81],
    '& near multiple vehicles': [0.67, 0.48, 0.65],
    '& following lane with lead': [0.25, 0.25, 0.29],
    '& traversing pickup drop off': [0.22, 0.29, 0.44],
    '& high lateral acceleration': [0.20, 0.08, 0.18],
    '& waiting for pedestrian to cross': [0.30, 0.43, 0.48],
    '& starting straight traffic light intersection traversal': [0.12, 0.35, 0.42],
}

df = pd.DataFrame(data, index=['AutoBotEgo', 'Urban Driver Closed-Loop', 'Urban Driver Closed-Loop with velocity']).T

# Create a function to highlight the max in a series.
def bold_max(s):
    is_max = s == s.max()
    return [' & '.join(['\\textbf{' + str(v) + '}' if vmax else str(v) for v, vmax in zip(s, is_max)])]

# Apply the function to each row
df = df.apply(bold_max, axis=1)

# Convert to LaTeX
latex_str = df.to_latex(header=False, escape=False)

# Removing square brackets from the string
latex_str = latex_str.replace('[', '').replace(']', '')

# Formatting the string
latex_table = """
\\begin{{table}}[h]
\\begin{{center}}
    \\begin{{tabular}}{{clccc}}
        \\Xhline{{2\\arrayrulewidth}}
        & & \\multicolumn{{3}}{{c}}{{Model}} \\\\
        & & \\rotatebox{{90}}{{AutoBotEgo}} & \\rotatebox{{90}}{{\\parbox{{2.1cm}}{{Urban Driver Closed-Loop}}}} & \\rotatebox{{90}}{{\\parbox{{2.1cm}}{{Urban Driver Closed-Loop with velocity}}}} \\\\
        \\Xhline{{0.5\\arrayrulewidth}}
        \\multirow{{14}}{{*}}{{\\rotatebox{{90}}{{Scenario Scores}}}}
        {} % Here goes the tabular data
        \\Xhline{{0.5\\arrayrulewidth}}
        \\multirow{{1}}{{*}}{{\\rotatebox{{90}}{{Scenario Scores}}}}
        & all & 0.39 & 0.44 & 0.48 \\\\
        \\Xhline{{2\\arrayrulewidth}}
    \\end{{tabular}}
\\end{{center}}
\\caption{{Open-Loop Simulation Results}}
\\label{{tab:open_loop_simulation}}
\\end{{table}}
""".format(latex_str[latex_str.find("\\midrule")+8:latex_str.find("\\bottomrule")].strip()) # extracting only the tabular data

# Print the LaTeX table
latex_str = latex_str.replace('\toprule', '').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ')

print(latex_str)