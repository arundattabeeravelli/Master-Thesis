
from gurobipy import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Define the different emission intensity levels and associated costs for hydrogen scenarios
hydrogen_scenarios = [
    {"name": "PEM Electrolysis", "cost": 8.4, "emission": 0.0},
    {"name": "Alkaline Electrolysis", "cost": 7.19, "emission": 0.0},
    {"name": "SMR", "cost": 2, "emission": 12.34},
    {"name": "SMR + CCS", "cost": 3.82, "emission": 2.5},
    {"name": "ATR + CCS", "cost": 3.77, "emission": 3}
]

# Initialize a list to store results for all scenarios
scenario_results = []
infeasible_scenarios=[]

for scenario in hydrogen_scenarios:
    
    HESOPT=Model('Hybrid_energy_system_optimization')

    # Read the data from the text file
    df = pd.read_csv(r"D:\TU Delft\2nd year\Master Thesis\Code\Code with pareto fronts\Load profiles\interpolated_data2.txt", sep="\t", header=None)
    df.columns=('Timestep', 'Load')

    Time_Step=df['Timestep'].tolist()
    Load_Profile=df['Load'].tolist()

    #Input Parameters
    #EASy Marine 80Ah(3.1kWH) Battery
    Costbattery=150         #$/kWh
    Battery_efficiency=0.95
    V_batt=0.01515
    W_batt=20
    SOC_min=0.1
    SOC_max=0.9
    C_ratemax=1
    Module_rating=3.1
    
    #ZEPP 150kW fuel cell parameters
    CostFC=635     #(Baldi etal, ritari etal, Fuel cell system production cost modeling and analysis-Achim Kampker)            #$/kW
    DeltaP_fc=15
    F_start=0.1
    k_1fc=0.0074
    k_2fc=0.3236
    k_1p=0.1245
    k_2p=8.8333
    P_FCmax=150
    I_fcmin=0
    I_fcmax=1500
    k_FCmin=0.1
    k_FCmax=0.9
    V_fc=0.00396
    W_fc=2.36
    Number_of_Stacks=8
    F_FCmax=8.82
    
    #Hydrogen Storage onboard Parameters
    H2_storagecost=200
    liquefaction_cost=1
    
    #Engine room restrictions
    Weight_Eng=11600
    Volume_Eng=470
    C_ov=0.2
    
    Electricity_price=0.095
    CO2tax=0.096 #$/kg
    M=100000

    #Sets
    T=Time_Step
    I=range(1,Number_of_Stacks+1)

    #Decision Variables

                 
    #Fuel Cell variables
    n_FC=HESOPT.addVar(lb=0, vtype=GRB.INTEGER)

    #Binary variable to calculate the number of stacks needed
    x_FC = {}
    for i in I:
        x_FC[i] = HESOPT.addVar(vtype=GRB.BINARY)

    #on/off status of ith fuel cell at time t
    deltaFC={}
    for i in I:
        for t in T:
            deltaFC[i,t]=HESOPT.addVar(vtype=GRB.BINARY)

    #Power output of ith fuel cell stack at time t
    P_FC={}
    for i in I:
        for t in T:
            P_FC[i,t]=HESOPT.addVar(lb=0, ub=150, vtype=GRB.CONTINUOUS)
            
    #Current density of individual fuel cell
    I_FC={}
    for i in I:
        for t in T:
            I_FC[i,t]=HESOPT.addVar(lb=0, ub= 1500, vtype=GRB.CONTINUOUS)

    #SFC of hydrogen for ith fuel cell at time t
    F_fc={}
    for i in I:
        for t in T:
            F_fc[i,t]=HESOPT.addVar(lb=0, vtype=GRB.CONTINUOUS)

    #startup phase of ith fuel cell at time t
    delta_stup={}
    for i in I:
        for t in T:
            delta_stup[i,t]=HESOPT.addVar(vtype=GRB.BINARY)
            
            
    #Battery Variables
    #Battery capacity
    E_Battmax=HESOPT.addVar(lb=0, vtype=GRB.CONTINUOUS)

    #Battery SOC at time t
    E_batt={}
    for t in T:
        E_batt[t]=HESOPT.addVar(vtype=GRB.CONTINUOUS)
        
    
    # Number of battery modules        
    n_Batt=HESOPT.addVar(lb=0, ub=97, vtype=GRB.INTEGER)
        
    # Battery discharge power
    P_Battplus={}
    for t in T:
        P_Battplus[t]=HESOPT.addVar(lb=0, vtype=GRB.CONTINUOUS)
        
    # Battery charge power
    P_Battminus={}
    for t in T:
        P_Battminus[t]=HESOPT.addVar(lb=0, vtype=GRB.CONTINUOUS)
        
    #Binary variables to prevent charge and discharge of the battery at the same time step
    y_c={}
    for t in T:
        y_c[t]=HESOPT.addVar(vtype=GRB.BINARY)
        
    y_dc={}
    for t in T:
        y_dc[t]=HESOPT.addVar(vtype=GRB.BINARY)

    HESOPT.update()
   
    #Constraints
    
    #fuel consumption of ith stack at time t
    con1={}
    for i in I:
        for t in T:
                con1[i,t]=HESOPT.addConstr(F_fc[i,t] == (k_1fc*I_FC[i,t] + k_2fc*deltaFC[i,t] + delta_stup[i,t]*F_start*F_FCmax)*(1/60))
                
    con2={}
    for i in I:
        for t in T:
            con2[i,t]=HESOPT.addConstr(P_FC[i,t]== k_1p* I_FC[i,t] + k_2p*deltaFC[i,t])

    # Lower current density limit of the stack
    con3={}
    for i in I: 
        for t in T:
            con3[i,t]=HESOPT.addConstr((I_fcmin*x_FC[i]) <= I_FC[i,t])

    #Upper current density limit of the stack
    con4={}
    for i in I:
        for t in T:
            con4[i,t]=HESOPT.addConstr(I_FC[i,t] <= (I_fcmax*x_FC[i]))

    #Allowable load variation of fuel cell stacks between time steps
    con5={}
    for i in I:
        for t in range(1,len(T)):
            con5[i,t]=HESOPT.addConstr(DeltaP_fc >= P_FC[i,t]-P_FC[i,t-1])
            
    con6={}
    for i in I:
        for t in range(1,len(T)):
            con6[i,t]=HESOPT.addConstr(DeltaP_fc >= -(P_FC[i,t]-P_FC[i,t-1]))
            
    # Constraints to ensure that all the selected stacks deliver the same amount of power
    con7={}
    for i in I:
        for t in T:
            con7[i,t]=HESOPT.addConstr(P_FC[i,t] <= M*x_FC[i])
            
    con8={}
    for i in I:
        for t in T:
            con8[i,t]=HESOPT.addConstr(P_FC[i,t] <= P_FC[1,t])
            
    con9={}
    for i in I:
        for t in T:
            con9[i,t]=HESOPT.addConstr(P_FC[i,t] >= P_FC[1,t] - (1-x_FC[i])*M)
            
    con10={}
    for i in I:
        for t in T:
            con10[i,t]=HESOPT.addConstr(P_FC[i,t]>=0)

    #Calculating the number of fuel cell stacks required
    con11=HESOPT.addConstr(quicksum(x_FC[i] for i in I)  == n_FC)
    
    
    #Equation to find the total number of start/stops that the FC stack undergoes in a single trip
    con12={}
    for i in I:
        for t in range(len(T)-1):
            con12[i,t]=HESOPT.addConstr(0 <= deltaFC[i,t] - deltaFC[i,t+1] + delta_stup[i,t])
            
        
    #Lower power limit of the FC stack
    con13={}
    for i in I:
        for t in T:
            con13[i,t]=HESOPT.addConstr(P_FC[i,t] >= k_FCmin*P_FCmax*deltaFC[i,t])
            
    #Upper power limit of the FC stack
    con14={}
    for i in I:
        for t in T:
            con14[i,t]=HESOPT.addConstr(P_FC[i,t] <= k_FCmax*P_FCmax*deltaFC[i,t])
            
    #Li-ion Battery Model
    
    #energy available in the battery at each time step
    con15={}
    for t in range(1,len(T)):
        con15[t]=HESOPT.addConstr((E_batt[t]) == (E_batt[t-1] + (Battery_efficiency*P_Battminus[t] - (1/Battery_efficiency)*P_Battplus[t])*(1/60)))

                                  
    #Lower SOC limit of the battery
    con16={}
    for t in T:
        con16[t]=HESOPT.addConstr( SOC_min*E_Battmax <= E_batt[t])

    #Upper SOC limit of the battery    
    con17={}
    for t in T:
        con17[t]=HESOPT.addConstr(SOC_max*E_Battmax >= E_batt[t])


    #Discharge power limit of the battery
    con18={}
    for t in T:
        con18[t]=HESOPT.addConstr(P_Battplus[t] <= C_ratemax*E_Battmax)
        
        
    #Charge power limit of the battery
    con19={}
    for t in T:
        con19[t]=HESOPT.addConstr(P_Battminus[t] <= C_ratemax*E_Battmax)
        

    #initial state of charge of the battery at time step 0
    con20=HESOPT.addConstr(E_batt[0]== 0.8*E_Battmax)
    
    #Number of battery modules
    con21=HESOPT.addConstr(E_Battmax==n_Batt*Module_rating)
            
    con22={}
    for t in T:
        con22[t]=HESOPT.addConstr(P_Battplus[t] <= M*y_dc[t])
        
    con23={}
    for t in T:
        con23[t]=HESOPT.addConstr(P_Battminus[t] <= M*y_c[t])
        
    con24={}
    for t in T:
        con24[t]=HESOPT.addConstr(y_dc[t] + y_c[t] <= 1)
        
    #Energy Balance constraint
    con25={}
    for t in T:
        con25[t]=HESOPT.addConstr( quicksum(P_FC[i,t] for i in I) + P_Battplus[t] == Load_Profile[t] + P_Battminus[t])
            

    #Volume Limit Constraint
    con26=HESOPT.addConstr(E_Battmax*V_batt + n_FC*P_FCmax*V_fc <= Volume_Eng*(1+C_ov))

    HESOPT.update()
    
    # Update the objectives with the new weights
    HESOPT.setObjective(E_Battmax * Costbattery + n_FC * P_FCmax * CostFC 
                         + quicksum(F_fc[i, t] * H2_storagecost for i in I for t in T)
                         + quicksum(F_fc[i, t] * scenario["cost"] for i in I for t in T)
                         +  quicksum(F_fc[i, t] * scenario["emission"]*CO2tax for i in I for t in T) + 
                            quicksum(F_fc[i, t] * liquefaction_cost for i in I for t in T) +
                            Electricity_price*0.8*E_Battmax*(1/Battery_efficiency))
    
    
    HESOPT.modelSense = GRB.MINIMIZE
    HESOPT.update()
    # Solve the optimization problem
    HESOPT.setParam( 'OutputFlag', True)        # silencing gurobi output or not
    #HESOPT.setParam('NonConvex', 2)
    #HESOPT.setParam('Timelimit', 3600)
    #HESOPT.setParam ('MIPGap', 0.005);         # Optimality gap
    #HESOPT.setParam('MIPFocus', 2)
    HESOPT.write("PEMFCBattdes.lp")             # print the model in .lp format file
    HESOPT.optimize ()
    
    if HESOPT.status == GRB.Status.OPTIMAL:
        # Extract and store the results for the current scenario
        result = {
            "Scenario": scenario["name"],
            "Total_Cost": HESOPT.ObjVal,
            "CAPEX": n_FC.x * CostFC * P_FCmax + E_Battmax.x * Costbattery + float(sum(F_fc[i, t].x for i in I for t in T)* H2_storagecost),
            "Num_FCStacks": int(n_FC.x),
            "Cost_FCs": n_FC.x * CostFC * P_FCmax,
            "H2_storagecost": float(sum(F_fc[i, t].x for i in I for t in T)* H2_storagecost),
            "Batt_Cap(kWH)": E_Battmax.x,
            "Num_Battmods":n_Batt.x,
            "Cost_Batt": E_Battmax.x * Costbattery,
            "OPEX": Electricity_price*0.8*E_Battmax.x*(1/Battery_efficiency) + float(sum(F_fc[i, t].x for i in I for t in T) * scenario["cost"]) + float(sum(F_fc[i, t].x for i in I for t in T) * liquefaction_cost)
                    + float(sum(F_fc[i,t].x for i in I for t in T)*scenario["emission"]*CO2tax),
            "Batt_charge_cost": Electricity_price*0.8*E_Battmax.x*(1/Battery_efficiency),
            "H2_Wt": float(sum(F_fc[i, t].x for i in I for t in T)),
            "Cost_H2": float(sum(F_fc[i, t].x for i in I for t in T) * scenario["cost"]),
            "H2_liquefactioncost": float(sum(F_fc[i, t].x for i in I for t in T) * liquefaction_cost),
            "Emissions": float(sum(F_fc[i,t].x for i in I for t in T)*scenario["emission"]),
            "Cost_CO2(H2)":float(sum(F_fc[i,t].x for i in I for t in T)*scenario["emission"]*CO2tax),
        }
        scenario_results.append(result)
        

        total_power_fc = []                 # List to store the total power output of all stacks
        power_battery_discharge = []        # List to store battery discharge power
        power_battery_charge=[]             # list to store battery charge power
        load_profile = Load_Profile         # Load profile data
        Difference=[]                       # List to store the difference between the supply and demand power
        StateOfCharge=[]                    # List to store the battery SOC at each time step
        power_output_fc = {}                # Dictionary to store individual FC stack power output
        
        for t in T:
            total_power_fc.append(sum(P_FC[i, t].x for i in I))                                                             # Calculate total FC power output
            power_battery_discharge.append(P_Battplus[t].x)                                                                 # Collect battery discharge power
            power_battery_charge.append(P_Battminus[t].x)                                                                   # Collect battery charge power
            Difference.append(sum(P_FC[i, t].x for i in I) + P_Battplus[t].x - P_Battminus[t].x - Load_Profile[t])          # Collect the power difference
            StateOfCharge.append(E_batt[t].x/E_Battmax.x)                                                                   # Collect the SOC values of the battery at each time step
            for i in I:
                power_output_fc[(i, t)] = P_FC[i, t].x                                                                      # Collect individual stack power output values for each stack at each time step
            
        hours = [t / 60 for t in T]
            
        # Create a separate image for each individual FC stack
        for i in I:
            plt.figure(figsize=(25, 15))
            plt.plot(hours, [power_output_fc[(i, t)] for t in T], label='Power output FC {i}', color='red')
            plt.xlabel('Time(hours)', fontsize=30)
            plt.xticks(fontsize=30)
            plt.ylabel(f'Power Output FC Stack {i} (kW)', fontsize=30)
            plt.yticks(fontsize=30)
            plt.title(f'Power Output of FC Stack {i} vs Time', fontsize=30)
            plt.show()
             
        # Create a plot for the battery charge/discharge power and SOC
        fig, ax1 = plt.subplots(figsize=(25, 15))
        ax1.plot(hours, power_battery_discharge, label='Battery discharge power', color='red')
        ax1.plot(hours, [-p for p in power_battery_charge], label='Battery charge power', color='purple')
        ax1.set_xlabel('Time(hours)', fontsize=30)
        ax1.tick_params(axis='x', labelcolor='black', labelsize=30)
        ax1.set_ylabel('Power in KW', color='black', fontsize=30)
        ax1.tick_params(axis='y', labelcolor='black', labelsize=30)

        # Create a second y-axis (right) for energy content
        ax2 = ax1.twinx()
        ax2.plot(hours, StateOfCharge, label='SOC', color='blue')
        ax2.set_ylabel('State of Charge', color='blue', fontsize=30)
        ax2.tick_params(axis='y', labelcolor='blue', labelsize=30)

        # Set titles for each axis
        ax1.set_title(f'Battery Power vs SOC Scenario:' + scenario["name"], fontsize=30)
        # Combine legends for both axes
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines = lines1 + lines2
        labels = labels1 + labels2
        ax1.legend(lines, labels, loc='upper left', fontsize=30)
        plt.show()
    
  
    
        # Create a plot for all the component power output and load profile
        plt.figure(figsize=(25, 15))
        # Plot the total power output of all the fuel cell stacks
        plt.plot(hours, total_power_fc, label='Total FC Power Output', color='green')
        # Plot the power of the battery
        plt.plot(hours, power_battery_discharge, label='Battery discharge Power', color='red')
        plt.plot(hours, [-p for p in power_battery_charge], label='Battery charge power', color='purple')
        # Plot the load profile
        plt.plot(hours, load_profile, label='Load Profile', linestyle='dotted', color='black')
        plt.xlabel('Time(hours)', fontsize=35)
        plt.xticks(fontsize=30)
        plt.ylabel('Power (kW)', fontsize=35)
        plt.yticks(fontsize=30)
        plt.title(f'Power Output of FC Stacks and Battery Over Time Scenario:' + scenario["name"], fontsize=26)
        plt.legend(fontsize=22, loc='best')
        plt.show()
 
        # Plot the difference in supply and demand power
        plt.figure(figsize=(10, 6))
        plt.plot(hours, Difference, label=f'Difference of power', color='darkcyan')
        plt.xlabel('Time(hours)')
        plt.ylabel('Load - (Battery + FC) (kW)')
        plt.title(f'Load and delivered power difference Scenario:' + scenario["name"])
        plt.legend(loc='upper right')
        plt.show()
        
    else:
        infeasible_scenarios.append(scenario["name"])


if scenario_results:
    # Create a Pandas DataFrame from the list of solution dictionaries    
    solutions_df = pd.DataFrame(scenario_results)   
    
    # Normalize the values within each scenario
    solutions_df_normalized = solutions_df.copy()
    solutions_df_normalized["CAPEX"] /= solutions_df_normalized["CAPEX"].max()
    solutions_df_normalized["OPEX"] /= solutions_df_normalized["OPEX"].max()
    solutions_df_normalized["Emissions"] /= solutions_df_normalized["Emissions"].max()

    # Plotting grouped bar graph
    plt.figure(figsize=(15, 8))

    bar_width = 0.2  # Adjust as needed
    index = range(len(solutions_df_normalized))

    # Bar graph for CAPEX
    plt.bar(index, solutions_df_normalized["CAPEX"], width=bar_width, color='tab:blue', label='CAPEX')

    # Bar graph for OPEX
    plt.bar([i + bar_width for i in index], solutions_df_normalized["OPEX"], width=bar_width, color='tab:orange', label='OPEX')

    # Bar graph for Emissions
    plt.bar([i + 2 * bar_width for i in index], solutions_df_normalized["Emissions"], width=bar_width, color='tab:grey', label='Emissions')

    # Display actual values on top of the bars
    for i, value in enumerate(solutions_df["CAPEX"]):
        plt.text(i, solutions_df_normalized["CAPEX"][i], f'{value:.2f}', ha='center', va='top', rotation=90, fontsize=12)
    for i, value in enumerate(solutions_df["OPEX"]):
        plt.text(i + bar_width, solutions_df_normalized["OPEX"][i], f'{value:.2f}', ha='center', va='top', rotation=90,fontsize=12)
    for i, value in enumerate(solutions_df["Emissions"]):
        if value != 0:
            plt.text(i + 2 * bar_width, solutions_df_normalized["Emissions"][i], f'{value:.2f}', ha='center', va='top', rotation=90, fontsize=12)
        elif value == 0:
            plt.text(i + 2 * bar_width, solutions_df_normalized["Emissions"][i]+0.01, f'{value:.2f}', ha='center', va='bottom', rotation=90, fontsize=12)

    # Customize the plot
    plt.title('CAPEX, OPEX, and Emissions for Different Hydrogen Grades in PEMFC/LIB solution')
    plt.xlabel('Scenario')
    plt.xticks([i + bar_width for i in index], solutions_df["Scenario"])
    plt.ylabel('Values')
    plt.legend()

    plt.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)

    plt.tight_layout()
    plt.show()
    
    pd.options.display.float_format = '{:.2f}'.format  # Set the format for floating-point numbers
    pd.set_option('display.max_columns', None)  # Display all columns

    # Print the DataFrame as a string
    solution_string = solutions_df.to_string(index=False) 
    
    # Loop through the hydrogen scenarios
    for scenario in hydrogen_scenarios:
    # Find the index of the 'Scenario' column
        scenario_col_idx = solutions_df.columns.get_loc("Scenario")
    
        # Filter the results DataFrame to exclude the "Scenario" column and maintain the original order
        selected_columns = [col for col in solutions_df.columns if col != "Scenario"]
        scenario_df = solutions_df.loc[solutions_df.iloc[:, scenario_col_idx] == scenario["name"], selected_columns]
    
        # Print the scenario name before printing the scenario's results
        print("Scenario: " + scenario["name"])
        
        # Print the filtered DataFrame for this scenario
        print(scenario_df)
        print("\n")
        
for scenario in infeasible_scenarios:
    print(f"No feasible solution found for scenario: {scenario}")

