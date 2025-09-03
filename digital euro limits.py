import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

# Parameters (adjusted based on thesis)
beta = 0.99  # Discount factor
gamma = 0.5  # Loan conversion ratio
phi = 1.0    # Risk aversion in labor supply
lambda_param = 0.01  # Reserve requirement ratio (adjusted based on thesis)
phi_pi = 1.5  # Response to inflation
phi_y = 0.5   # Response to output gap
phi_R = 0.2   # Response to liquidity ratio
rho = 0.8     # Policy inertia
kappa = 0.5   # Adjustment cost
reserve_ratio_threshold = 0.15  # Threshold to trigger monetary policy change

# Digital Euro Parameters
digital_euro_limit = 3000  # Based on thesis conclusion
GDP_ratio_min = 0.15  # Minimum digital euro circulation relative to GDP
GDP_ratio_max = 0.30  # Maximum digital euro circulation relative to GDP

# Loan-to-deposit ratio
loan_to_deposit_ratio = 0.94

# Sectoral Deposit Ratios
household_deposits_ratio = 0.60
corporate_deposits_ratio = 0.30
other_deposits_ratio = 0.10

# Initial Guess for Steady State
C_ss = 1.0  # Consumption steady state
N_ss = 1.0  # Labor supply steady state
W_ss = 1.0  # Wage rate steady state
r_ss = 0.03 # Interest rate steady state (applies only to traditional deposits)
D_ss = 1.0  # Traditional deposits steady state
R_ss = lambda_param * D_ss  # Reserves steady state

# User tiers and holding limits in euros
TIER_LIMITS = {
    'Retail': 3000,  # Max limit for retail users
    'SMEs': 100000,  # Max limit for small businesses
    'Institutions': 1000000  # Max limit for large institutions
}

# Initialize user accounts by type
user_accounts = {
    'Retail': 0,
    'SMEs': 0,
    'Institutions': 0
}

# Function to deposit digital euros, respecting tiered limits
def deposit_digital_euro(user_type, amount):
    max_limit = TIER_LIMITS[user_type]
    if user_accounts[user_type] + amount > max_limit:
        print(f"Cannot deposit {amount} euros. Exceeds {user_type} limit of {max_limit} euros.")
        amount = max_limit - user_accounts[user_type]
    user_accounts[user_type] += amount
    print(f"{amount} euros deposited to {user_type} account. Current balance: {user_accounts[user_type]} euros.")

# Define the system of equations without interest on digital euro
def equations_with_non_interest_digital_euro(vars):
    C, N, W, r, D_T, D_E, R, L, pi = vars
    
    # Apply the €3,000 limit on digital euro holdings
    if D_E > digital_euro_limit:
        excess_amount = D_E - digital_euro_limit
        D_E = digital_euro_limit
        D_T += excess_amount
    
    # Monitor reserve ratio and adjust monetary policy if reserves are too low
    reserve_ratio = R / (D_T + D_E)
    if reserve_ratio < reserve_ratio_threshold:
        lambda_param_dynamic = 0.02  # Increase reserve ratio if reserves are too low
    else:
        lambda_param_dynamic = lambda_param
    
    # Household utility
    eq1 = beta * (1 + r) * (C / C_ss) - 1
    # Labor supply
    eq2 = W - N ** phi
    # Wage rate
    eq3 = W - C / N
    # Bank liquidity
    eq4 = L + R - (D_T + D_E)
    # Loan provision
    eq5 = L - gamma * (D_T + D_E - R)
    # Reserve requirement
    eq6 = R - lambda_param_dynamic * (D_T + D_E)
    # Central bank policy (only applies to traditional deposits)
    eq7 = r - (rho * r + (1 - rho) * (phi_pi * pi + phi_y * (C - C_ss) / C_ss + phi_R * R / D_T))  # D_T only
    # Market clearing
    eq8 = C + D_T + D_E - (1 + r_ss) * D_T - D_E - W * N  # No interest on D_E
    # Inflation dynamics (Phillips Curve)
    eq9 = pi - kappa * (N - N_ss)
    
    return [eq1, eq2, eq3, eq4, eq5, eq6, eq7, eq8, eq9]

# Initial guess for steady-state values
initial_guess_policy = [C_ss, N_ss, W_ss, r_ss, D_ss, min(D_ss, digital_euro_limit), R_ss, gamma * (2 * D_ss - R_ss), 0]

# Solve the steady state with policy changes and non-interest-bearing digital euro
steady_state_policy = fsolve(equations_with_non_interest_digital_euro, initial_guess_policy)

# Extract results from steady state
C_ss, N_ss, W_ss, r_ss, D_T_ss, D_E_ss, R_ss, L_ss, pi_ss = steady_state_policy
print(f"Steady State Values with Non-Interest-Bearing Digital Euro:\nC: {C_ss}, N: {N_ss}, W: {W_ss}, r: {r_ss}, D_T: {D_T_ss}, D_E: {D_E_ss}, R: {R_ss}, L: {L_ss}, pi: {pi_ss}")

# Simulate a shock and policy response over time
shock_size = 0.01
shock_response_policy = []

for t in range(10):
    if t == 0:
        shock_response_policy.append([C_ss, N_ss, W_ss, r_ss + shock_size, D_T_ss, D_E_ss, R_ss, L_ss, pi_ss])
    else:
        # Update with the shock and new policy
        shock_response_policy.append(fsolve(equations_with_non_interest_digital_euro, shock_response_policy[-1]))

# Convert to numpy array for easy plotting
shock_response_policy = np.array(shock_response_policy)

# Plot the results
plt.figure(figsize=(12, 8))
plt.subplot(3, 3, 1)
plt.plot(shock_response_policy[:, 0], label='Consumption')
plt.title('Consumption')
plt.subplot(3, 3, 2)
plt.plot(shock_response_policy[:, 1], label='Labor Supply')
plt.title('Labor Supply')
plt.subplot(3, 3, 3)
plt.plot(shock_response_policy[:, 2], label='Wage')
plt.title('Wage')
plt.subplot(3, 3, 4)
plt.plot(shock_response_policy[:, 3], label='Interest Rate')
plt.title('Interest Rate (Traditional Deposits)')
plt.subplot(3, 3, 5)
plt.plot(shock_response_policy[:, 4], label='Traditional Deposits')
plt.title('Traditional Deposits')
plt.subplot(3, 3, 6)
plt.plot(shock_response_policy[:, 5], label='Digital Euro Deposits')
plt.title('Digital Euro Deposits (Non-Interest Bearing)')
plt.subplot(3, 3, 7)
plt.plot(shock_response_policy[:, 6], label='Reserves')
plt.title('Reserves')
plt.subplot(3, 3, 8)
plt.plot(shock_response_policy[:, 7], label='Loans')
plt.title('Loans')
plt.subplot(3, 3, 9)
plt.plot(shock_response_policy[:, 8], label='Inflation')
plt.title('Inflation')
plt.tight_layout()
plt.show()
def simulate_bank_profitability(traditional_deposits, digital_euro_deposits):
    loan_creation_rate = 0.9  # 90% of traditional deposits can be used for loans
    bank_profit_margin = 0.02  # Banks earn 2% on loans
    
    # Loans generated from traditional deposits
    loans_from_traditional_deposits = traditional_deposits * loan_creation_rate
    bank_profit = loans_from_traditional_deposits * bank_profit_margin

    # Substitution effect: more digital euros means fewer traditional deposits
    if digital_euro_deposits > 0:
        traditional_deposits -= digital_euro_deposits * 0.5  # Assume 50% substitution
        loans_from_traditional_deposits = traditional_deposits * loan_creation_rate
        bank_profit = loans_from_traditional_deposits * bank_profit_margin
        print(f"Bank profit reduced due to shift to digital euro deposits.")

    return bank_profit

# Example usage
total_deposits = 1e6  # Total deposits in the system (example value)
household_deposits = total_deposits * household_deposits_ratio
corporate_deposits = total_deposits * corporate_deposits_ratio
other_deposits = total_deposits * other_deposits_ratio
total_traditional_deposits = household_deposits + corporate_deposits + other_deposits

digital_euro_deposits = 3000  # Assume initial retail digital euro deposit
bank_profit = simulate_bank_profitability(total_traditional_deposits, digital_euro_deposits)
print(f"Bank profitability after shift to digital euro: {bank_profit} euros")
# Function to simulate bank profitability before and after shift to digital euro deposits
def simulate_bank_profitability(traditional_deposits, digital_euro_deposits, loan_to_deposit_ratio, bank_profit_margin=0.02):
    """
    Simulates the impact of digital euro deposits on bank profitability by reducing traditional deposits
    and therefore reducing loan creation capacity.
    
    traditional_deposits: total traditional bank deposits (initial)
    digital_euro_deposits: total digital euro deposits held by households or firms
    loan_to_deposit_ratio: loan-to-deposit ratio (94% for the user)
    bank_profit_margin: percentage profit banks make on loans (default 2%)
    """
    # Calculate total loans from traditional deposits
    loans_from_traditional_deposits = traditional_deposits * loan_to_deposit_ratio
    bank_profit_initial = loans_from_traditional_deposits * bank_profit_margin
    
    # Now, adjust for the substitution effect: digital euro reduces traditional deposits
    if digital_euro_deposits > 0:
        # Assume a 50% substitution effect (i.e., 50% of digital euro deposits substitute traditional deposits)
        traditional_deposits_after_shift = traditional_deposits - digital_euro_deposits * 0.5
        loans_from_traditional_deposits_after_shift = traditional_deposits_after_shift * loan_to_deposit_ratio
        bank_profit_after_shift = loans_from_traditional_deposits_after_shift * bank_profit_margin
        
        print(f"Traditional Deposits Before: {traditional_deposits}")
        print(f"Traditional Deposits After Digital Euro Shift: {traditional_deposits_after_shift}")
        print(f"Loans Before Digital Euro Shift: {loans_from_traditional_deposits}")
        print(f"Loans After Digital Euro Shift: {loans_from_traditional_deposits_after_shift}")
        print(f"Bank Profit Before Shift: {bank_profit_initial}")
        print(f"Bank Profit After Shift: {bank_profit_after_shift}")
        
        # Calculate the impact on profitability
        profit_impact = bank_profit_initial - bank_profit_after_shift
        print(f"Profitability Reduction Due to Digital Euro: {profit_impact} euros")
        
        return bank_profit_initial, bank_profit_after_shift, profit_impact
    else:
        # No shift if digital_euro_deposits is 0
        print(f"No digital euro deposits. Bank profit unchanged: {bank_profit_initial} euros")
        return bank_profit_initial, bank_profit_initial, 0

# Example usage
total_deposits = 1e6  # Example total traditional deposits (1,000,000 euros)
digital_euro_deposits = 50000  # Assume 50,000 euros shifted to digital euro holdings

# Simulate the impact on bank profitability
bank_profit_before, bank_profit_after, profit_impact = simulate_bank_profitability(
    traditional_deposits=total_deposits,
    digital_euro_deposits=digital_euro_deposits,
    loan_to_deposit_ratio=loan_to_deposit_ratio
)

print(f"\nFinal Results:")
print(f"Bank Profit Before Digital Euro Shift: {bank_profit_before} euros")
print(f"Bank Profit After Digital Euro Shift: {bank_profit_after} euros")
print(f"Total Profit Impact: {profit_impact} euros")
# Function to simulate bank profitability and assess liquidity risk before and after shift to digital euro deposits
def simulate_bank_profitability_and_liquidity_risk(traditional_deposits, digital_euro_deposits, loan_to_deposit_ratio, reserve_requirement_ratio, liquidity_threshold=0.05, bank_profit_margin=0.02):
    """
    Simulates the impact of digital euro deposits on bank profitability and liquidity risk.
    
    traditional_deposits: total traditional bank deposits (initial)
    digital_euro_deposits: total digital euro deposits held by households or firms
    loan_to_deposit_ratio: loan-to-deposit ratio (94% for the user)
    reserve_requirement_ratio: required reserves as a fraction of total deposits (e.g., 1%)
    liquidity_threshold: threshold below which liquidity risk occurs (default 5%)
    bank_profit_margin: percentage profit banks make on loans (default 2%)
    """
    # Calculate total loans and reserves from traditional deposits
    loans_from_traditional_deposits = traditional_deposits * loan_to_deposit_ratio
    bank_profit_initial = loans_from_traditional_deposits * bank_profit_margin
    
    # Calculate initial required reserves before the shift
    initial_required_reserves = traditional_deposits * reserve_requirement_ratio
    print(f"Initial Required Reserves: {initial_required_reserves} euros")

    # Substitution effect: digital euro reduces traditional deposits
    if digital_euro_deposits > 0:
        # Assume a 50% substitution effect (i.e., 50% of digital euro deposits substitute traditional deposits)
        traditional_deposits_after_shift = traditional_deposits - digital_euro_deposits * 0.5
        loans_from_traditional_deposits_after_shift = traditional_deposits_after_shift * loan_to_deposit_ratio
        bank_profit_after_shift = loans_from_traditional_deposits_after_shift * bank_profit_margin
        
        # Calculate required reserves after the shift
        required_reserves_after_shift = traditional_deposits_after_shift * reserve_requirement_ratio
        print(f"Required Reserves After Digital Euro Shift: {required_reserves_after_shift} euros")
        
        # Calculate the liquidity risk (if reserves fall below a certain liquidity threshold)
        liquidity_risk = required_reserves_after_shift < liquidity_threshold * traditional_deposits_after_shift
        if liquidity_risk:
            print("Warning: Liquidity Risk Detected! Reserves are below the liquidity threshold.")
        else:
            print("No Liquidity Risk: Reserves are above the liquidity threshold.")
        
        print(f"Traditional Deposits Before: {traditional_deposits}")
        print(f"Traditional Deposits After Digital Euro Shift: {traditional_deposits_after_shift}")
        print(f"Loans Before Digital Euro Shift: {loans_from_traditional_deposits}")
        print(f"Loans After Digital Euro Shift: {loans_from_traditional_deposits_after_shift}")
        print(f"Bank Profit Before Shift: {bank_profit_initial}")
        print(f"Bank Profit After Shift: {bank_profit_after_shift}")
        
        # Calculate the impact on profitability
        profit_impact = bank_profit_initial - bank_profit_after_shift
        print(f"Profitability Reduction Due to Digital Euro: {profit_impact} euros")
        
        return bank_profit_initial, bank_profit_after_shift, profit_impact, liquidity_risk
    else:
        # No shift if digital_euro_deposits is 0
        print(f"No digital euro deposits. Bank profit and liquidity unchanged: {bank_profit_initial} euros")
        return bank_profit_initial, bank_profit_initial, 0, False

# Example usage
total_deposits = 1e6  # Example total traditional deposits (1,000,000 euros)
digital_euro_deposits = 50000  # Assume 50,000 euros shifted to digital euro holdings
loan_to_deposit_ratio = 0.94  # Loan to deposit ratio (user's data)
reserve_requirement_ratio = 0.01  # Reserve requirement ratio (1%)
liquidity_threshold = 0.05  # Liquidity threshold (5%)

# Simulate the impact on bank profitability and liquidity risk
bank_profit_before, bank_profit_after, profit_impact, liquidity_risk = simulate_bank_profitability_and_liquidity_risk(
    traditional_deposits=total_deposits,
    digital_euro_deposits=digital_euro_deposits,
    loan_to_deposit_ratio=loan_to_deposit_ratio,
    reserve_requirement_ratio=reserve_requirement_ratio,
    liquidity_threshold=liquidity_threshold
)

print(f"\nFinal Results:")
print(f"Bank Profit Before Digital Euro Shift: {bank_profit_before} euros")
print(f"Bank Profit After Digital Euro Shift: {bank_profit_after} euros")
print(f"Total Profit Impact: {profit_impact} euros")
print(f"Liquidity Risk Detected: {liquidity_risk}")
# Function to simulate bank profitability and assess liquidity risk, with monetary policy change suggestion
def simulate_bank_profitability_and_policy_change(traditional_deposits, digital_euro_deposits, loan_to_deposit_ratio, reserve_requirement_ratio, liquidity_threshold=0.05, bank_profit_margin=0.02):
    """
    Simulates the impact of digital euro deposits on bank profitability, liquidity risk, and suggests changes to the monetary policy system.
    
    traditional_deposits: total traditional bank deposits (initial)
    digital_euro_deposits: total digital euro deposits held by households or firms
    loan_to_deposit_ratio: loan-to-deposit ratio (94% for the user)
    reserve_requirement_ratio: required reserves as a fraction of total deposits (e.g., 1%)
    liquidity_threshold: threshold below which liquidity risk occurs (default 5%)
    bank_profit_margin: percentage profit banks make on loans (default 2%)
    """
    # Calculate total loans and reserves from traditional deposits
    loans_from_traditional_deposits = traditional_deposits * loan_to_deposit_ratio
    bank_profit_initial = loans_from_traditional_deposits * bank_profit_margin
    
    # Calculate initial required reserves before the shift
    initial_required_reserves = traditional_deposits * reserve_requirement_ratio
    print(f"Initial Required Reserves: {initial_required_reserves} euros")

    # Substitution effect: digital euro reduces traditional deposits
    if digital_euro_deposits > 0:
        # Assume a 50% substitution effect (i.e., 50% of digital euro deposits substitute traditional deposits)
        traditional_deposits_after_shift = traditional_deposits - digital_euro_deposits * 0.5
        loans_from_traditional_deposits_after_shift = traditional_deposits_after_shift * loan_to_deposit_ratio
        bank_profit_after_shift = loans_from_traditional_deposits_after_shift * bank_profit_margin
        
        # Calculate required reserves after the shift
        required_reserves_after_shift = traditional_deposits_after_shift * reserve_requirement_ratio
        print(f"Required Reserves After Digital Euro Shift: {required_reserves_after_shift} euros")
        
        # Calculate the liquidity risk (if reserves fall below a certain liquidity threshold)
        liquidity_risk = required_reserves_after_shift < liquidity_threshold * traditional_deposits_after_shift
        
        # Policy change recommendation based on liquidity risk
        if liquidity_risk:
            print("Warning: Liquidity Risk Detected! Reserves are below the liquidity threshold.")
            
            # Recommendation based on the reserves level
            if required_reserves_after_shift < 0.03 * traditional_deposits_after_shift:
                print("Policy Recommendation: Shift to a Ceiling System - Banks need more access to central bank lending.")
            elif required_reserves_after_shift < liquidity_threshold * traditional_deposits_after_shift:
                print("Policy Recommendation: Shift to a Corridor System - Manage liquidity with tighter reserves.")
            else:
                print("Policy Recommendation: Continue with Floor System - Sufficient reserves.")
        else:
            print("No Liquidity Risk: Reserves are above the liquidity threshold. No immediate change to the monetary system is required.")
        
        print(f"Traditional Deposits Before: {traditional_deposits}")
        print(f"Traditional Deposits After Digital Euro Shift: {traditional_deposits_after_shift}")
        print(f"Loans Before Digital Euro Shift: {loans_from_traditional_deposits}")
        print(f"Loans After Digital Euro Shift: {loans_from_traditional_deposits_after_shift}")
        print(f"Bank Profit Before Shift: {bank_profit_initial}")
        print(f"Bank Profit After Shift: {bank_profit_after_shift}")
        
        # Calculate the impact on profitability
        profit_impact = bank_profit_initial - bank_profit_after_shift
        print(f"Profitability Reduction Due to Digital Euro: {profit_impact} euros")
        
        return bank_profit_initial, bank_profit_after_shift, profit_impact, liquidity_risk
    else:
        # No shift if digital_euro_deposits is 0
        print(f"No digital euro deposits. Bank profit and liquidity unchanged: {bank_profit_initial} euros")
        return bank_profit_initial, bank_profit_initial, 0, False

# Example usage
total_deposits = 1e6  # Example total traditional deposits (1,000,000 euros)
digital_euro_deposits = 50000  # Assume 50,000 euros shifted to digital euro holdings
loan_to_deposit_ratio = 0.94  # Loan to deposit ratio (user's data)
reserve_requirement_ratio = 0.01  # Reserve requirement ratio (1%)
liquidity_threshold = 0.05  # Liquidity threshold (5%)

# Simulate the impact on bank profitability, liquidity risk, and policy change recommendations
bank_profit_before, bank_profit_after, profit_impact, liquidity_risk = simulate_bank_profitability_and_policy_change(
    traditional_deposits=total_deposits,
    digital_euro_deposits=digital_euro_deposits,
    loan_to_deposit_ratio=loan_to_deposit_ratio,
    reserve_requirement_ratio=reserve_requirement_ratio,
    liquidity_threshold=liquidity_threshold
)

print(f"\nFinal Results:")
print(f"Bank Profit Before Digital Euro Shift: {bank_profit_before} euros")
print(f"Bank Profit After Digital Euro Shift: {bank_profit_after} euros")
print(f"Total Profit Impact: {profit_impact} euros")
print(f"Liquidity Risk Detected: {liquidity_risk}")
# Assuming you have defined the total traditional deposits and the effect of digital euro substitution
traditional_deposits_before_shift = traditional_deposits

# Assuming digital_euro_deposits is calculated as the sum of digital euro accounts
# Shift from traditional deposits to digital euro deposits (50% substitution)
traditional_deposits_after_shift = traditional_deposits_before_shift - (0.5 * digital_euro_deposits)

# Required reserves after the shift based on the updated reserve requirement ratio
required_reserves_after_shift = reserve_requirement_ratio * traditional_deposits_after_shift

# ECB-specific policy change recommendation based on liquidity risk
if liquidity_risk:
    print("Warning: Liquidity Risk Detected! Reserves are below the liquidity threshold.")
    
    # Recommendation based on the reserves level (for ECB's de facto floor system)
    if required_reserves_after_shift < 0.03 * traditional_deposits_after_shift:
        print("ECB Policy Recommendation: Shift from a Floor to a Ceiling System - Banks will need greater access to central bank lending.")
    elif required_reserves_after_shift < liquidity_threshold * traditional_deposits_after_shift:
        print("ECB Policy Recommendation: Shift to a Corridor System - Manage liquidity more actively as reserves fall.")
    else:
        print("ECB Policy Recommendation: Maintain Floor System - Reserves are sufficient under current conditions.")





