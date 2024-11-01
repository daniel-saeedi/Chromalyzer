import csv

def generate_latex_from_csv(csv_file):
    with open(csv_file, mode='r') as file:
        reader = csv.DictReader(file)
        latex_lines = []
        rownum = 0
        for row in reader:
            rownum += 1
            coefficient = float(row['coefficient'])
            mz = float(row['m/z'])
            RT1 = row['RT1']
            RT2 = row['RT2']
            samples = row['samples']
            feature_index = row['feature_index']
            feature_class = row['class']
            # Format RT1 and RT2 to have the lower value first
            RT1_range = eval(RT1)
            RT2_range = eval(RT2)
            RT1_sorted = f"[{min(RT1_range):.3f}, {max(RT1_range):.3f}]"
            RT2_sorted = f"[{min(RT2_range):.3f}, {max(RT2_range):.3f}]"

            latex_line = f"\\stepcounter{{rownum}}\\arabic{{rownum}}  & {coefficient:.4f} & {mz:.1f} & {RT1_sorted} & {RT2_sorted} & {samples} & - \\\\ \\hline"
            latex_lines.append(latex_line)
    
    return latex_lines

# Example usage
csv_file = '/usr/scratch/chromalyzer/lr_l2_results/fragments/fragments_info.csv'  # Replace with your actual CSV file path
latex_lines = generate_latex_from_csv(csv_file)
for line in latex_lines:
    print(line)
