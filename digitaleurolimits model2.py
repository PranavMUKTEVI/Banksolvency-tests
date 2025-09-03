import numpy as np
import matplotlib.pyplot as plt

# Time horizon
T = 10
time = np.arange(T)

# Parameters
loan_to_deposit_ratio = 0.94

# Proportions for deposit sectors
household_deposits_ratio = 0.60
corporate_deposits_ratio = 0.30
other_deposits_ratio = 0.10

# Deposit and loan variables
total_deposits = np.zeros(T)
total_loans = np.zeros(T)
household_deposits = np.zeros(T)
corporate_deposits = np.zeros(T)
other_deposits = np.zeros(T)
household_loans = np.zeros(T)
corporate_loans = np.zeros(T)
other_loans = np.zeros(T)

# Assuming an initial value for deposits (e.g., €1 trillion)
initial_deposits = 1e12

# Define dynamics over time (example: simple step function or linear growth)
for t in range(T):
    if t == 0:
        total_deposits[t] = initial_deposits
    else:
        total_deposits[t] = total_deposits[t-1] * 1.01  # Growth of 1% per period (adjust as needed)

    # Break down deposits into sectors
    household_deposits[t] = total_deposits[t] * household_deposits_ratio
    corporate_deposits[t] = total_deposits[t] * corporate_deposits_ratio
    other_deposits[t] = total_deposits[t] * other_deposits_ratio

    # Loans based on loan-to-deposit ratio
    household_loans[t] = household_deposits[t] * loan_to_deposit_ratio
    corporate_loans[t] = corporate_deposits[t] * loan_to_deposit_ratio
    other_loans[t] = other_deposits[t] * loan_to_deposit_ratio

    # Sum of loans
    total_loans[t] = household_loans[t] + corporate_loans[t] + other_loans[t]

# Additional parameters (examples)
interest_rate = np.zeros(T)
reserves = np.zeros(T)
consumption = np.zeros(T)
labor_supply = np.zeros(T)
wage = np.zeros(T)
digital_euro_deposits = np.zeros(T)
inflation = np.zeros(T)

# Assuming digital euro has limits of €3000 per user, this is factored in
digital_euro_limit = 3000
population = 100e6  # Assume 100 million people
max_digital_euro_deposits = digital_euro_limit * population

# Define dynamics for other variables
for t in range(T):
    # Sample dynamics for other variables (adjust based on your model)
    interest_rate[t] = 0.01 - 0.001 * t  # Decrease over time
    reserves[t] = total_loans[t] * 0.1    # Reserves proportional to loans
    consumption[t] = 1.01 + 0.001 * t     # Small growth in consumption
    labor_supply[t] = 1.005 + 0.001 * t   # Labor supply increases slightly
    wage[t] = 1.005 + 0.001 * t           # Wages also grow over time
    digital_euro_deposits[t] = min(max_digital_euro_deposits, total_deposits[t] * 0.15)  # 15% cap on digital euro deposits
    inflation[t] = 0.0015 + 0.0001 * t    # Small increase in inflation

# Plotting
fig, axs = plt.subplots(3, 3, figsize=(15, 7))
axs[0, 0].plot(time, consumption)
axs[0, 0].set_title("Consumption")
axs[0, 1].plot(time, labor_supply)
axs[0, 1].set_title("Labor Supply")
axs[0, 2].plot(time, wage)
axs[0, 2].set_title("Wage")

axs[1, 0].plot(time, interest_rate)
axs[1, 0].set_title("Interest Rate (Traditional Deposits)")
axs[1, 1].plot(time, total_deposits)
axs[1, 1].set_title("Traditional Deposits")
axs[1, 2].plot(time, digital_euro_deposits)
axs[1, 2].set_title("Digital Euro Deposits (Non-Interest Bearing)")

axs[2, 0].plot(time, reserves)
axs[2, 0].set_title("Reserves")
axs[2, 1].plot(time, total_loans)
axs[2, 1].set_title("Loans")
axs[2, 2].plot(time, inflation)
axs[2, 2].set_title("Inflation")

plt.tight_layout()
plt.show()
