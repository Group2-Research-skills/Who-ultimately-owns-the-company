import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ================== PART 1: MATRIX CREATION & NORMALIZATION ==================

def normalize_ownership_matrix_column(A):
    """Column-normalize matrix."""
    column_sums = A.sum(axis=0)
    zero_columns = column_sums == 0
    column_sums[zero_columns] = 1
    return A / column_sums


# Load raw data
df = pd.read_csv('/Users/fazila/Desktop/synthetic_company_structures_500_links.csv')

# Get unique IDs
unique_shareholder_ids = df['shareholder_id'].unique()
unique_company_ids = df['company_id'].unique()

# Add 6 new dummy shareholders
new_shareholder_ids = [1000, 1001, 1002, 1003, 1004, 1005]
all_shareholder_ids = np.concatenate([unique_shareholder_ids, new_shareholder_ids])

# Create mappings
shareholder_id_to_index = {id: idx for idx, id in enumerate(all_shareholder_ids)}
company_id_to_index = {id: idx for idx, id in enumerate(unique_company_ids)}

# Initialize square matrix
matrix = np.zeros((len(all_shareholder_ids), len(unique_company_ids)))

# Populate matrix
for _, row in df.iterrows():
    sh_idx = shareholder_id_to_index[int(row['shareholder_id'])]
    co_idx = company_id_to_index[int(row['company_id'])]
    matrix[sh_idx, co_idx] = row['shareholder_percent']

# Normalize matrix (now column-stochastic)
normalized_matrix = normalize_ownership_matrix_column(matrix)


# ================== PART 2: POWER ITERATION ==================

def power_iteration(A, max_iter=1000, tol=1e-6):
    """Compute dominant eigenvector for column-stochastic matrix."""
    n = A.shape[0]
    # Random initial vector P0, sum = 1
    P = np.random.rand(n, 1)
    P = P / P.sum()

    for i in range(max_iter):
        P_new = A @ P  # Matrix-vector multiplication
        P_new /= P_new.sum()  # Maintain L1 norm

        if np.linalg.norm(P_new - P) < tol:
            print(f'Converged at iteration {i}')
            break
        P = P_new

    return P_new

dominant_vector = power_iteration(normalized_matrix)

# ================== PART 3: SAVE RESULTS ==================

# Create DataFrame with shareholder IDs (including dummies)
result_df = pd.DataFrame(
    dominant_vector,
    index=all_shareholder_ids,
    columns=['Influence']
)

# Save final result
result_df.to_csv('/Users/fazila/Desktop/dominant_vector.csv')
print("Dominant eigenvector saved with dummy shareholders included.")


#===========================Part 4: Sorting Eigenvector ===========================
# Extract shareholder IDs from the normalized matrix (same as all_shareholder_ids)
shareholder_ids = all_shareholder_ids.tolist()

# Flatten dominant vector and sort it in descending order
dominant_flat = dominant_vector.flatten()
sorted_indices = np.argsort(-dominant_flat)  # Sort by influence descending

# Sort shareholder IDs and corresponding influence values
sorted_probs = dominant_flat[sorted_indices]
sorted_shareholders = [shareholder_ids[i] for i in sorted_indices]

# Convert to percentages
sorted_probs_percent = sorted_probs * 100

# ============================== Part 5 Visualization =============================



# Plot Top 10 Shareholders Graph
top_n = 10
plt.figure(figsize=(8, 5))
plt.bar(range(top_n), sorted_probs_percent[:top_n], tick_label=sorted_shareholders[:top_n])
plt.xlabel('Top 10 Shareholder IDs')
plt.ylabel('Influence probability (%)')
plt.title('Top 10 Most Influential Shareholders (Power Iteration)')
plt.tight_layout()
plt.savefig('/Users/fazila/Desktop/power_iteration_top10_shareholders.png', dpi=300)
plt.show()

print("Top 10 Shareholder graph saved as 'power_iteration_top10_shareholders.png'")

# Plot Influence Distribution among all shareholders
plt.figure(figsize=(10, 6))
plt.plot(range(len(sorted_probs_percent)), sorted_probs_percent, label="Influence", color="b", lw=2)
plt.xlabel('Shareholder Rank')
plt.ylabel('Influence Probability (%)')
plt.title('Influence Distribution for All Shareholders')
plt.grid(True, linestyle='--', linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.show()


# Plot Probability Distribution of Shareholders

# Smooth sorted_probs_percent using a rolling average
window_size = 5  # You can adjust this (e.g., 5, 10, etc.)

smoothed_probs = pd.Series(sorted_probs_percent).rolling(window=window_size, center=True, min_periods=1).mean()

# Plotting the smoothed curve
plt.figure(figsize=(10, 6))
x_vals = range(1, len(smoothed_probs) + 1)

plt.plot(x_vals, smoothed_probs, label='Smoothed Probability', color='blue')
plt.xlabel('Shareholder Rank (1 = Most Influential)', fontsize=12)
plt.ylabel('Probability (%)', fontsize=12)
plt.title('Smoothed Probability Distribution of Shareholders', fontsize=14)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()



#==================== Part 6: Calculate Weight Influence======================

# Calculate the weighted influence by combining eigenvector values and ownership percentages
weighted_influence = dominant_vector.flatten() * np.sum(normalized_matrix, axis=1)
# Sort by descending influence (highest first)
weighted_influence_sorted = np.argsort(-weighted_influence)


# Set top_n to 6 to display the top 6 influential shareholders
top_n = 6

# Plot the top owners' influence



# Visualize Influence Across All Shareholders
# Plot influence across all shareholders
sorted_shareholders = [all_shareholder_ids[i] for i in weighted_influence_sorted]
sorted_influence = weighted_influence[weighted_influence_sorted]

plt.figure(figsize=(10, 6))
plt.plot(sorted_shareholders, sorted_influence, label="Weighted Influence", color="blue", lw=2)
plt.xlabel('Shareholder ID')
plt.ylabel('Weighted Influence (%)')
plt.title('Influence Distribution for All Shareholders')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()

# Save the plot to a file
plt.savefig('/Users/fazila/Desktop/influence_distribution_all_shareholders.png', dpi=300)
plt.show()
print("Influence distribution graph saved as 'influence_distribution_all_shareholders.png'")

# Extract the top 6 indices based on sorted order
top_indices = sorted_indices[:6]

# Get the corresponding top owners (shareholder IDs) and their influences
top_owners = [all_shareholder_ids[idx] for idx in top_indices]
top_influences = dominant_flat[top_indices]

# Normalize influences to percentages (if needed)
top_influences_percent = top_influences * 100

# Print for verification
print("Top 6 Owners (Real Shareholder IDs):", top_owners)
print("Top 6 Weighted Influences:", top_influences_percent)

# Visualize the Top Owners' Influence with the correct weights
plt.figure(figsize=(20, 12))  # Increase figure size for wider bars
bars = plt.bar(top_owners, top_influences_percent, color='blue', width=4)  # Adjust width to make bars thicker

# Set labels and title
plt.xlabel('Shareholder IDs', fontsize=14)
plt.ylabel('Weighted Influence (%)', fontsize=14)
plt.title('Top 6 Most Influential Shareholders (Weighted Influence)', fontsize=16)

# Ensure the layout is tight and clear
plt.tight_layout()

# Save the plot to a file
plt.savefig('/Users/fazila/Desktop/top_6_influential_shareholders.png', dpi=300)
plt.show()
print("Top 6 influential shareholders graph saved as 'top_6_influential_shareholders.png'")





def power_iteration_with_tracking(A, max_iter=1000, tol=1e-6):
    n = A.shape[0]
    P = np.random.rand(n, 1)
    P = P / P.sum()

    diffs = []

    for i in range(max_iter):
        P_new = A @ P
        P_new /= P_new.sum()

        diff = np.linalg.norm(P_new - P)
        diffs.append(diff)

        if diff < tol:
            print(f'Converged at iteration {i}')
            break

        P = P_new

    return P_new, diffs

# Run the tracking version of power iteration
dominant_vector, diffs = power_iteration_with_tracking(normalized_matrix)


# Plot convergence
plt.plot(diffs)
plt.xlabel("Iteration")
plt.ylabel("Difference (L2 norm)")
plt.title("Convergence of Power Iteration")
plt.grid(True)
plt.tight_layout()
plt.show()



direct_ownership = matrix.sum(axis=1)
influence = dominant_vector.flatten()

plt.figure(figsize=(8, 6))
plt.scatter(direct_ownership, influence, alpha=0.6)
plt.xlabel("Cumulative Direct Ownership (%)")
plt.ylabel("Influence Score")
plt.title("Cumulative Direct Ownership vs. Influence Score")
plt.grid(True)
plt.tight_layout()
plt.show()




cumulative_influence = np.cumsum(sorted_probs_percent)
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(cumulative_influence) + 1), cumulative_influence)
plt.axhline(80, color='r', linestyle='--', label='80% Influence')
plt.xlabel("Number of Top Shareholders")
plt.ylabel("Cumulative Influence (%)")
plt.title("Pareto Analysis of Shareholder Influence")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



import seaborn as sns

plt.figure(figsize=(10, 6))
sns.heatmap(normalized_matrix, cmap='Blues', cbar=True)
plt.title("Heatmap of Column-Normalized Ownership Matrix")
plt.xlabel("Companies")
plt.ylabel("Shareholders")
plt.tight_layout()
plt.show()




# === Step 1: Compute direct ownership for each shareholder ===
direct_ownership = matrix.sum(axis=1)  # sum across all companies
direct_ownership_percent = direct_ownership / direct_ownership.sum() * 100

# === Step 2: Influence score ===
influence = dominant_vector.flatten()
influence_percent = influence / influence.sum() * 100

# === Step 3: Combine into one DataFrame ===
summary_df = pd.DataFrame({
    'Shareholder_ID': all_shareholder_ids,
    'Direct_Ownership (%)': direct_ownership_percent,
    'Influence Score (%)': influence_percent
})

# === Step 4: Sort by influence descending ===
summary_df = summary_df.sort_values(by='Influence Score (%)', ascending=False).reset_index(drop=True)

# === Step 5: Add Rank Column ===
summary_df.insert(0, 'Rank', summary_df.index + 1)

# === Step 6: Get Top N ===
top_n = 10
top_shareholders_df = summary_df.head(top_n)

# === Step 7: Print / Export ===
print(top_shareholders_df.to_string(index=False))
top_shareholders_df.to_csv('/Users/fazila/Desktop/top_shareholders_summary.csv', index=False)



