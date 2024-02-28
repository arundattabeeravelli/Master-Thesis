
from gurobipy import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define the different emission intensity levels and associated costs for hydrogen scenarios
hydrogen_scenarios = [
    {"name": "PEM Electrolysis", "cost": 8.4, "emission": 0.0},
    {"name": "Alkaline Electrolysis", "cost": 7.19, "emission": 0.0},
    {"name": "SMR", "cost": 2, "emission": 12},
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
    Number_of_Stacks=7
    F_FCmax=8.82
    
    #Hydrogen Storage onboard Parameters
    H2_storagecost=200
    liquefaction_cost=1
    
    #Engine room restrictions
    Volume_Eng=470
    C_ov=0.2        #Oversizing parameter
    
    #Global parameters
    Electricity_price=0.095    #cost of charging the battery at the port
    CO2tax=0.096 #$/kg
    
    #Diesel generator(Wartsila 800kW) parameters
    CostDG=300  #$/kW
    k_ENGmin=0.11   #lower limit of 22kW/cyl=88kW 
    k_ENGmax=1
    Number_of_DGs=2
    GenEff=0.95
    P_ENGmax=800
    DeltaP_DG=0.33*P_ENGmax
    V_DG=0.0296     #m3/kW(ENGmax)
    
    CostMGO=0.844 #€/kg 1 year average(Oct2022-2023)
    CO2eqMGO=4.211 #kgCO2eq/kgLFO or DMA fuel which is MGO(Product guide of Warsila L20)
    Storagecost_MGO=1.3     #1.161Euros/kg  #storage costs of MGO 27Euros/GJ (Power-2-Fuel  Cost Analysis-TNO)
    M=100000
    
    #Sets
    T=Time_Step
    I=range(1,Number_of_Stacks+1)
    J=range(1,Number_of_DGs+1)

    #Decision Variables
    
    #Diesel Generator variables
    
    #Power output of ENGINE
    P_ENG={}
    for j in J:
        for t in T:
            P_ENG[j,t]=HESOPT.addVar(lb=0, ub=800, vtype=GRB.CONTINUOUS)

    P_DG={}
    for j in J:
        for t in T:
            P_DG[j,t]=HESOPT.addVar(lb=0, vtype=GRB.CONTINUOUS)

    #Number of Diesel Generators
    n_DG=HESOPT.addVar(lb=0, vtype=GRB.INTEGER)

    #Binary Variable to calculate the number of DG's
    x_DG={}
    for j in J:
        x_DG[j]=HESOPT.addVar(vtype=GRB.BINARY)

    #Fuel Consumption of DG
    FC_ENG={}
    for j in J:
        for t in T:
            FC_ENG[j,t]=HESOPT.addVar(lb=0, vtype=GRB.CONTINUOUS)

                 
    #Fuel Cell variables
    
    #Number of Fuel cell stacks
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

    #Hydrogen consumption for ith fuel cell at time t
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
        
        
    #Number of battery modules required    
    n_Batt=HESOPT.addVar(lb=0, ub=97, vtype=GRB.INTEGER)
    
    #Battery discharge power    
    P_Battplus={}
    for t in T:
        P_Battplus[t]=HESOPT.addVar(lb=0, vtype=GRB.CONTINUOUS)
    
    #Battery charge power
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
               
    #Equation relating Output power vs Fuel consumption for 1 minute PWL
    con1={}
    for j in J:
        for t in T:
            con1[j,t]=HESOPT.addGenConstrPWL(P_ENG[j,t], FC_ENG[j,t], [400, 600, 680, 800], [1.406, 1.99, 2.245, 2.65])        
                
    
    #constraints to ensure that all the selected diesel generators deliver the same power at each time step t        
    con2={}
    for j in J:
        for t in T:
            con2[j,t]=HESOPT.addConstr(P_ENG[j,t] <= M*x_DG[j])
            
    con3={}
    for j in J:
        for t in T:
            con3[j,t]=HESOPT.addConstr(P_ENG[j,t] <= P_ENG[1,t])
            
    con4={}
    for j in J:
        for t in T:
            con4[j,t]=HESOPT.addConstr(P_ENG[j,t] >= P_ENG[1,t] - (1-x_DG[j])*M)
            
    con5={}
    for j in J:
        for t in T:
            con5[j,t]=HESOPT.addConstr(P_ENG[j,t]>=0)
    
    
    #Engine Output to Generator Output
    con6={}
    for j in J:
        for t in T:
            con6[j,t]=HESOPT.addConstr(P_DG[j,t]==P_ENG[j,t]*GenEff)
            
    #Lower Power limit of DG
    con7={}
    for j in J:
        for t in T:
            con7[j,t]=HESOPT.addConstr(P_ENG[j,t] >= k_ENGmin*P_ENGmax*x_DG[j])
            
    #Upper Power limit of DG
    con8={}
    for j in J:
        for t in T:
            con8[j,t]=HESOPT.addConstr(P_ENG[j,t] <= k_ENGmax*P_ENGmax*x_DG[j])
       
            
    #Number of DG's required
    con9=HESOPT.addConstr(quicksum(x_DG[j] for j in J) == n_DG)
    
    #Equation relating the H2 consumption to the current density of the stack
    con10={}
    for i in I:
        for t in T:
                con10[i,t]=HESOPT.addConstr(F_fc[i,t] == (k_1fc*I_FC[i,t] + k_2fc*deltaFC[i,t] + delta_stup[i,t]*F_start*F_FCmax)*(1/60))
    
    #Equation relating the Stack Power output to the current density
    con11={}
    for i in I:
        for t in T:
            con11[i,t]=HESOPT.addConstr(P_FC[i,t]== k_1p* I_FC[i,t] + k_2p*deltaFC[i,t])

    #lower power limit of all the fuel cell stacks at each time step
    con12={}
    for i in I: 
        for t in T:
            con12[i,t]=HESOPT.addConstr((I_fcmin*x_FC[i]) <= I_FC[i,t])

    #upper power limit of all the fuel cell stacks
    con13={}
    for i in I:
        for t in T:
            con13[i,t]=HESOPT.addConstr(I_FC[i,t] <= (I_fcmax*x_FC[i]))

    #Allowable load variation of fuel cell stacks between time steps
    con14={}
    for i in I:
        for t in range(1,len(T)):
            con14[i,t]=HESOPT.addConstr(DeltaP_fc >= P_FC[i,t]-P_FC[i,t-1])
            
    con15={}
    for i in I:
        for t in range(1,len(T)):
            con15[i,t]=HESOPT.addConstr(DeltaP_fc >= -(P_FC[i,t]-P_FC[i,t-1]))

    #Calculating the number of fuel cell stacks required
    con16=HESOPT.addConstr(quicksum(x_FC[i] for i in I)  == n_FC)
    

    #constraints to ensure that all the selected stacks deliver the same power
    con17={}
    for i in I:
        for t in T:
            con17[i,t]=HESOPT.addConstr(P_FC[i,t] <= M*x_FC[i])
            
    con18={}
    for i in I:
        for t in T:
            con18[i,t]=HESOPT.addConstr(P_FC[i,t] <= P_FC[1,t])
            
    con19={}
    for i in I:
        for t in T:
            con19[i,t]=HESOPT.addConstr(P_FC[i,t] >= P_FC[1,t] - (1-x_FC[i])*M)
            
    con20={}
    for i in I:
        for t in T:
            con20[i,t]=HESOPT.addConstr(P_FC[i,t]>=0)
    
 
    #constraint to find the number of start/stops of the fuel cell stack
    con21={}
    for i in I:
        for t in range(len(T)-1):
            con21[i,t]=HESOPT.addConstr(0 <= deltaFC[i,t] - deltaFC[i,t+1] + delta_stup[i,t])
            
    #Lower power limit of FC stack
    con22={}
    for i in I:
        for t in T:
            con22[i,t]=HESOPT.addConstr(P_FC[i,t] >= k_FCmin*P_FCmax*deltaFC[i,t])
            
    #Upper power limit of FC stack
    con23={}
    for i in I:
        for t in T:
            con23[i,t]=HESOPT.addConstr(P_FC[i,t] <= k_FCmax*P_FCmax*deltaFC[i,t])

            
    #Li-ion Battery Model
    
    #energy available in the battery at each time step
    con24={}
    for t in range(1,len(T)):
        con24[t]=HESOPT.addConstr((E_batt[t]) == (E_batt[t-1] + (Battery_efficiency*P_Battminus[t] - (1/Battery_efficiency)*P_Battplus[t])*(1/60)))

                                  
    #Lower SOC limit of the battery
    con25={}
    for t in T:
        con25[t]=HESOPT.addConstr( SOC_min*E_Battmax <= E_batt[t])

    #Upper SOC limit of the battery    
    con26={}
    for t in T:
        con26[t]=HESOPT.addConstr(E_batt[t] <= SOC_max*E_Battmax)


    #lower Power limit of battery
    con27={}
    for t in T:
        con27[t]=HESOPT.addConstr(P_Battplus[t] <= C_ratemax*E_Battmax)

        
    #Upper Power limit of Battery
    con28={}
    for t in T:
        con28[t]=HESOPT.addConstr(P_Battminus[t] <= C_ratemax*E_Battmax)
        
    #initial state of charge of the battery at time step 0
    con29=HESOPT.addConstr(E_batt[0]== 0.8*E_Battmax)
    
    #Number of battery modules required
    con30=HESOPT.addConstr(E_Battmax==n_Batt*Module_rating)

    con32={}
    for t in T:
        con32[t]=HESOPT.addConstr(P_Battplus[t] <= M*y_dc[t])
        
    con33={}
    for t in T:
        con33[t]=HESOPT.addConstr(P_Battminus[t] <= M*y_c[t])
        
    con34={}
    for t in T:
        con34[t]=HESOPT.addConstr(y_dc[t] + y_c[t] <= 1)
                
    #Energy Balance constraint
    con35={}
    for t in T:
        con35[t]=HESOPT.addConstr( quicksum(P_FC[i,t] for i in I) + P_Battplus[t] + quicksum(P_DG[j,t] for j in J) == Load_Profile[t] + P_Battminus[t])
            
    #Engine Volume Limit Constraint
    con35=HESOPT.addConstr(E_Battmax*V_batt + n_FC*P_FCmax*V_fc + n_DG*P_ENGmax*V_DG <= Volume_Eng*(1+C_ov))

    HESOPT.update()
       
    HESOPT.setObjective(E_Battmax * Costbattery + n_FC * P_FCmax * CostFC + n_DG * P_ENGmax * CostDG 
                         + quicksum(FC_ENG[j, t]*Storagecost_MGO for j in J for t in T) +
                         quicksum(F_fc[i, t]*H2_storagecost for i in I for t in T) +
                         quicksum(F_fc[i, t] * scenario["cost"] for i in I for t in T) +
                         quicksum(F_fc[i, t] * scenario["emission"]*CO2tax for i in I for t in T) +
                         quicksum(F_fc[i, t] * liquefaction_cost for i in I for t in T) +
                         quicksum(FC_ENG[j, t] * CostMGO for j in J for t in T) +
                         quicksum(FC_ENG[j, t] * CO2eqMGO * CO2tax for j in J for t in T) + 
                         Electricity_price*0.8*E_Battmax*(1/Battery_efficiency))
    
    
    HESOPT.modelSense = GRB.MINIMIZE
    HESOPT.update()
    # Solve the optimization problem
    HESOPT.setParam( 'OutputFlag', True)    # silencing gurobi output or not
    #HESOPT.setParam('Method', -1)          # -1=automatic, 0=primal simplex,1=dual simplex,2=barrier,3=concurrent,4=deterministic concurrent, and 5=deterministic concurrent simplex.    
    #HESOPT.setParam('NonConvex', 2)
    #HESOPT.setParam('Timelimit', 3600)     # Optimization time limit
    #HESOPT.setParam ('MIPGap', 0.005);     # find the optimal solution
    HESOPT.write("DGPEMFCbattdes.lp")           # print the model in .lp format file
    HESOPT.optimize ()
    
    if HESOPT.status == GRB.Status.OPTIMAL:
        
        total_power_fc = []                 # List to store the total power output of all stacks
        power_battery_discharge = []        # List to store battery discharge power
        power_battery_charge=[]             # List to store battery charge power
        DG_power=[]                         # list to store DG power
        load_profile = Load_Profile         # Load profile data
        Difference=[]                       # List for the difference in supply and demand data
        StateOfCharge=[]                    # List to store the SOC values of the battery
        power_output_fc = {}                # Dictionary to store the individual FC stack power output
        power_output_DG={}                  # Dictionary to store the individual DG power output
        
        for t in T:
            total_power_fc.append(sum(P_FC[i, t].x for i in I))                                                                                         # Collect the total stack power output at each time step
            power_battery_discharge.append(P_Battplus[t].x)                                                                                             # Collect the battery discharge power values at each time step
            power_battery_charge.append(P_Battminus[t].x)                                                                                               # Collect the battery charge power at each time step
            DG_power.append(sum(P_DG[j,t].x for j in J))                                                                                                # Collect the total DG power output at each time step
            Difference.append(sum(P_FC[i, t].x for i in I) + P_Battplus[t].x + sum(P_DG[j,t].x for j in J) - P_Battminus[t] - Load_Profile[t])          # Collect the difference between supply and demand at each time step        
            StateOfCharge.append(E_batt[t].x/E_Battmax.x)                                                                                               # Collect SOC values of battery at each time step
            for i in I:
                power_output_fc[(i, t)] = P_FC[i, t].x                                                                                                  # Collect P_FC values for each stack at each time step
            for j in J:
                power_output_DG[(j,t)]=P_DG[j,t].x                                                                                                      # Collect P_DG values for each DG at each time step
        

        # Extract and store the results for the current scenario
        result = {
            "Scenario": scenario["name"],
            "Total_Cost": HESOPT.ObjVal,
            "CAPEX": n_FC.x * CostFC * P_FCmax + E_Battmax.x * Costbattery + n_DG.x * P_ENGmax * CostDG
                    + float(sum(F_fc[i, t].x for i in I for t in T)* H2_storagecost) 
                    + float(sum(FC_ENG[j, t].x for j in J for t in T)*Storagecost_MGO),
            "Num_FCStacks": int(n_FC.x),
            "Cost_FCs": n_FC.x * CostFC * P_FCmax,
            "H2_storagecost": float(sum(F_fc[i, t].x for i in I for t in T) * H2_storagecost),
            "Num_DGs": int(n_DG.x),
            "Cost_DGs": n_DG.x * CostDG * P_ENGmax,
            "MGO_storagecost": float(sum(FC_ENG[j, t].x for j in J for t in T) * Storagecost_MGO),
            "Batt_Cap(kWH)": E_Battmax.x,
            "Num_Battmods":n_Batt.x,
            "Cost_Batt": E_Battmax.x * Costbattery,
            "OPEX": Electricity_price*0.8*E_Battmax.x*(1/Battery_efficiency) 
                    + float(sum(F_fc[i, t].x for i in I for t in T) * scenario["cost"]) 
                    + float(sum(F_fc[i, t].x for i in I for t in T) * liquefaction_cost)
                    + float(sum(F_fc[i,t].x for i in I for t in T)*scenario["emission"]*CO2tax) 
                    + float(sum(FC_ENG[j, t].x for j in J for t in T) * CostMGO) 
                    + float(sum(FC_ENG[j,t].x for j in J for t in T)*CO2eqMGO*CO2tax),
            "Batt_charge_cost": Electricity_price*0.8*E_Battmax.x*(1/Battery_efficiency),
            "H2_Wt": float(sum(F_fc[i, t].x for i in I for t in T)),
            "Cost_H2": float(sum(F_fc[i, t].x for i in I for t in T) * scenario["cost"]),
            "Cost_CO2(H2)":float(sum(F_fc[i,t].x for i in I for t in T)*scenario["emission"]*CO2tax),
            "H2_liquefactioncost": float(sum(F_fc[i, t].x for i in I for t in T) * liquefaction_cost),
            "MGO_Wt": float(sum(FC_ENG[j, t].x for j in J for t in T)),
            "Cost_MGO": float(sum(FC_ENG[j, t].x for j in J for t in T) * CostMGO),
            'CO2_Cost(MGO)': float(sum(FC_ENG[j,t].x for j in J for t in T)*CO2eqMGO*CO2tax),
            "Emissions": float(sum(F_fc[i,t].x for i in I for t in T)*scenario["emission"]) 
                        + float(sum(FC_ENG[j,t].x for j in J for t in T)*CO2eqMGO),
        }
        scenario_results.append(result)
        

        hours = [t / 60 for t in T]        
       
        # Create a separate plot for each individual FC stack
        for i in I:
             plt.figure(figsize=(25, 15))
             plt.plot(hours, [power_output_fc[(i, t)] for t in T], label='Power output FC {i}', color='red')
             plt.xlabel('Time(hours)', fontsize=30)
             plt.xticks(fontsize=30)
             plt.ylabel(f'Power Output FC Stack {i} (kW)', fontsize=30)
             plt.yticks(fontsize=30)
             plt.title(f'Power Output of FC Stack {i} vs Time', fontsize=30)
             plt.show()
              
        # Create a seperate plot for each individual Diesel Generator
        for j in J:
            plt.figure(figsize=(25,15))
            plt.plot(hours, [power_output_DG[(j,t)] for t in T], label= 'Power output DG {j}', color= 'purple')
            plt.xlabel('Time(hours)', fontsize=30)
            plt.xticks(fontsize=30)
            plt.ylabel(f'Power Output of DG {j} (kW)', fontsize=30)
            plt.yticks(fontsize=30)
            plt.title(f'Power output of DG {j} vs Time', fontsize=30)
            plt.show()
            
            
        # Battery charge/discharge power vs SOC
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
        ax1.set_title(f'Battery Power vs SOC Scenario: ' + scenario["name"], fontsize=30)
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines = lines1 + lines2
        labels = labels1 + labels2
        ax1.legend(lines, labels, loc='upper left', fontsize=30)
        plt.show()
        
  
        
        # Combined plot of all the component power output
        plt.figure(figsize=(25, 15))
        # Plot the total power output of all the fuel cell stacks
        plt.plot(hours, total_power_fc, label='Total FC Power Output', color='green')
        #plot the DG power
        plt.plot(hours, DG_power, label='Total DG power', color='Purple')
        # Plot the power of the battery
        plt.plot(hours, power_battery_discharge, label='Battery discharge Power', color='red')
        plt.plot(hours, [-p for p in power_battery_charge], label='Battery charge power', color='blue')
        # Plot the load profile
        plt.plot(hours, load_profile, label='Load Profile', linestyle='dotted', color='black')
        plt.xlabel('Time(hours)', fontsize=35)
        plt.xticks(fontsize=30)
        plt.ylabel('Power (kW)', fontsize=35)
        plt.yticks(fontsize=30)
        plt.title(f'Power Output of FC Stacks, DGs and Battery Over Time Scenario:' + scenario["name"], fontsize=26)
        plt.legend(fontsize=22, loc='best')
        plt.show()
     
        # Plot of the difference between supply and demand
        plt.figure(figsize=(10, 6))
        plt.plot(T, Difference, label=f'Difference of power', color='darkcyan')
        plt.xlabel('Time Step')
        plt.ylabel('Load - (DG + Battery + FC) (kW)')
        plt.title(f'Load and delivered power difference Scenario:' + scenario["name"])
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

    plt.title('CAPEX, OPEX, and Emissions for Different Hydrogen Grades in DG/PEMFC/LIB Solution')
    plt.xlabel('Scenario')
    plt.xticks([i + bar_width for i in index], solutions_df["Scenario"])
    plt.ylabel('Values')
    plt.legend()

    plt.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)

    plt.tight_layout()
    plt.show()
    
    pd.options.display.float_format = '{:.2f}'.format  
    pd.set_option('display.max_columns', None) 

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

