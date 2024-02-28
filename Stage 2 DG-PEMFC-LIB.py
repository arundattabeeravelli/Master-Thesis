
from gurobipy import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

# Define the different emission intensity levels and associated costs for hydrogen scenarios
hydrogen_scenarios = [
    {"name": "PEM Electrolysis", "cost": 8.4, "emission": 0.0},
]

# Initialize a list to store results for all scenarios
scenario_results = []
infeasible_scenarios=[]

for scenario in hydrogen_scenarios:
    Vcumulative = 0                                 # degradation of the fuel cell at start of life
    PEMFC_SOH = 1                                   # State of Health of PEMFC at start of life
    Month=0                                         # Number of months initialization
    k_1p=0.1245                                     # Coefficient of current density at the start of life
    Battdegcumulative = 0                           # Battery degradation at start of life
    Batt_SOH = 1                                    # State of Health of battery at the start of life
    Monthly_hydrogen_consumption=[]                 # List to store the increase in monthly hydrogen consumption values
    SOH_PEMFC=[]                                    # List to store the SOH of the PEMFC at the end of each month
    SOH_Battery=[]                                  # List to store the SOH of battery at the end of each month
    OPEX=[]                                         # List to store the increase in OPEX due to the degradations and reduced efficiency of the plant
    while PEMFC_SOH >0.8 and Batt_SOH > 0.8 :
    
        HESOPT=Model('Hybrid_energy_system_optimization')
    
        # Read the data from the text file with space as the separator
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
        E_Battmax= Module_rating*59*Batt_SOH
       
        #ZEPP 150kW fuel cell parameters
        CostFC=635
        DeltaP_fc=15
        F_start=0.1
        
        #coefficients relaing h2 consumption to current density
        k_1fc=0.0074
        k_2fc=0.3236
        
        #coefficients relating output power to current density
        #k_1p=0.1245
        k_2p=8.8333
        
        #coefficients relating the current density to degradation
        k_1dv=0.0018
        k_2dv=9.4166
        deltavload=0.0441   #microvolts/deltakW
        deltavstup=23.91    #microvolts/cycle
        
        P_FCmax=150
        I_fcmin=0
        I_fcmax=1500
        V_fc=0.00396
        W_fc=2.36
        Number_of_Stacks=2
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
        
        #Global parameters
        Electricity_price=0.095    #cost of charging the battery at the port
        CO2tax=0.096 #$/kg
        
        #Diesel generator(Wartsila 800kW) parameters
        CostDG=300  #$/kW
        k_ENGmin=0.11   #lower limit of 22kW/cyl=88kW 
        k_ENGmax=1
        Number_of_DGs=1
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
                P_FC[i,t]=HESOPT.addVar(lb=0, ub=135, vtype=GRB.CONTINUOUS)
                
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
    
        #Battery SOC at time t
        E_batt={}
        for t in T:
            E_batt[t]=HESOPT.addVar(vtype=GRB.CONTINUOUS)
            
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
                    
        #Variables representing the degradation associated with operating the fuel cell
        
        #degradation due to transient loading
        Vdelta_loadchange={}
        for i in I:
            for t in T:
                Vdelta_loadchange[i,t]=HESOPT.addVar(vtype=GRB.CONTINUOUS)
                
        #degradation due to start-stop cycles
        Vdelta_stup={}
        for i in I:
            for t in T:
                Vdelta_stup[i,t]=HESOPT.addVar(vtype=GRB.CONTINUOUS)
                
        #degradation due to operating the fuel cell
        Vdelta_PFC={}
        for i in I:
            for t in T:
                Vdelta_PFC[i,t]=HESOPT.addVar(vtype=GRB.CONTINUOUS)
                
        #Total degradation at each time step
        Vdelta_total={}
        for i in I:
            for t in T:
                Vdelta_total[i,t]=HESOPT.addVar(vtype=GRB.CONTINUOUS)
                
        #auxiliary variable
        auxvar={}
        for i in I:
            for t in T:
                auxvar[i,t]=HESOPT.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS)
                
        #absolute power difference
        PFCdeltaabs={}
        for i in I:
            for t in T:
                PFCdeltaabs[i,t]=HESOPT.addVar(vtype=GRB.CONTINUOUS)  
    
    
        HESOPT.update()
    
        #Constraints
        
        #Equation relating the current density to the h2 consumption at each time step
        con1={}
        for i in I:
            for t in T:
                    con1[i,t]=HESOPT.addConstr(F_fc[i,t] == ((k_1fc*I_FC[i,t] + k_2fc*deltaFC[i,t] + delta_stup[i,t]*F_start*F_FCmax)*(1/60)))

        # Equation relating the current density to the power output of the stack                    
        con2={}
        for i in I:
            for t in T:
                con2[i,t]=HESOPT.addConstr(P_FC[i,t]== k_1p* I_FC[i,t] + k_2p*deltaFC[i,t])
    
        #lower limit of current density
        con3={}
        for i in I: 
            for t in T:
                con3[i,t]=HESOPT.addConstr((I_fcmin*x_FC[i]) <= I_FC[i,t])
    
        #upper limit of current density
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
    
        #Calculating the number of fuel cell stacks required
        con7=HESOPT.addConstr(quicksum(x_FC[i] for i in I)  == n_FC)
        
        #constraints to ensure that all the selected stacks deliver the same power
        con8={}
        for i in I:
            for t in T:
                con8[i,t]=HESOPT.addConstr(P_FC[i,t] <= M*x_FC[i])
                
        con9={}
        for i in I:
            for t in T:
                con9[i,t]=HESOPT.addConstr(P_FC[i,t] <= P_FC[1,t])
                
        con10={}
        for i in I:
            for t in T:
                con10[i,t]=HESOPT.addConstr(P_FC[i,t] >= P_FC[1,t] - (1-x_FC[i])*M)
                
        con11={}
        for i in I:
            for t in T:
                con11[i,t]=HESOPT.addConstr(P_FC[i,t]>=0)
        
        
        #constraint to find the startup phase for fuel consumption calculation
        con12={}
        for i in I:
            for t in range(len(T)-1):
                con12[i,t]=HESOPT.addConstr(0 <= deltaFC[i,t] - deltaFC[i,t+1] + delta_stup[i,t])
    
                
        #Li-ion Battery Model
        
        #initial state of charge of the battery at time step 0
        con13=HESOPT.addConstr(E_batt[0]== 0.8*E_Battmax)
        
        #energy available in the battery at each time step
        con14={}
        for t in range(1,len(T)):
            con14[t]=HESOPT.addConstr((E_batt[t]) == (E_batt[t-1] + (Battery_efficiency*P_Battminus[t] - (1/Battery_efficiency)*P_Battplus[t])*(1/60)))
    
                                      
        #Lower SOC limit of the battery
        con15={}
        for t in T:
            con15[t]=HESOPT.addConstr( SOC_min*E_Battmax <= E_batt[t])
    
        #Upper SOC limit of the battery    
        con16={}
        for t in T:
            con16[t]=HESOPT.addConstr(E_batt[t] <= SOC_max*E_Battmax)
               
    
        #lower Power limit of battery
        con17={}
        for t in T:
            con17[t]=HESOPT.addConstr(P_Battplus[t] <= C_ratemax*E_Battmax)
            
            
        #Upper Power limit of Battery
        con18={}
        for t in T:
            con18[t]=HESOPT.addConstr(P_Battminus[t] <= C_ratemax*E_Battmax)
            
        con19={}
        for t in T:
            con19[t]=HESOPT.addConstr(P_Battplus[t] <= M*y_dc[t])
            
        con20={}
        for t in T:
            con20[t]=HESOPT.addConstr(P_Battminus[t] <= M*y_c[t])
            
        con21={}
        for t in T:
            con21[t]=HESOPT.addConstr(y_dc[t] + y_c[t] <= 1)

            
        #Energy Balance constraint
        con22={}
        for t in T:
            con22[t]=HESOPT.addConstr( quicksum(P_FC[i,t] for i in I) + P_Battplus[t] + quicksum(P_DG[j,t] for j in J) == Load_Profile[t] + P_Battminus[t])
                
              
        #auxilary constraint to calculate the voltage loss due to load changes
        con23={}
        for i in I:
            for t in range(1,len(T)):
                con23[i,t]=HESOPT.addConstr(auxvar[i,t]==P_FC[i,t]-P_FC[i,t-1])
                     
        con24={}
        for i in I:
            for t in T:
                con24[i,t]=HESOPT.addGenConstrAbs(PFCdeltaabs[i,t], auxvar[i,t])
                
        con25={}
        for i in I:
            for t in T:
                con25[i,t]=HESOPT.addConstr(Vdelta_loadchange[i,t]==PFCdeltaabs[i,t]*deltavload)
                
        con26={}
        for i in I:
            for t in T:
                con26[i,t]=HESOPT.addConstr(Vdelta_stup[i,t]== delta_stup[i,t]*deltavstup)
                
        con27={}
        for i in I:
            for t in T:
                con27[i,t]=HESOPT.addConstr(Vdelta_PFC[i,t]== (k_1dv*I_FC[i,t] + k_2dv*deltaFC[1,t])*(1/60))
                
        #Total degradation
        con28={}
        for t in T:
            con28[t]=HESOPT.addConstr(Vdelta_total[1,t] == Vdelta_PFC[1,t] + Vdelta_loadchange[1,t] + Vdelta_stup[1,t])
            
        #constraint relating Output power vs Fuel consumption for 1 minute PWL
        con29={}
        for j in J:
            for t in T:
                con29[j,t]=HESOPT.addGenConstrPWL(P_ENG[j,t], FC_ENG[j,t], [400, 600, 680, 800], [1.406, 1.99, 2.245, 2.65])        
                    
         
        #Engine Output to Generator Output
        con30={}
        for j in J:
            for t in T:
                con30[j,t]=HESOPT.addConstr(P_DG[j,t]==P_ENG[j,t]*GenEff)
                
        #Lower limit of DG
        con31={}
        for j in J:
            for t in T:
                con31[j,t]=HESOPT.addConstr(P_ENG[j,t] >= k_ENGmin*P_ENGmax*x_DG[j])
                
        #Upper limit of DG
        con32={}
        for j in J:
            for t in T:
                con32[j,t]=HESOPT.addConstr(P_ENG[j,t] <= k_ENGmax*P_ENGmax*x_DG[j])
               
                    
        #Constraint to calculate the number of DG's
        con33=HESOPT.addConstr(quicksum(x_DG[j] for j in J) == n_DG)
        
        HESOPT.update()
        
        # Update the objectives with the new weights
        HESOPT.setObjectiveN((quicksum(F_fc[i, t] * scenario["cost"] for i in I for t in T) +
                              quicksum(F_fc[i, t] * scenario["emission"]*CO2tax for i in I for t in T) +
                              quicksum(F_fc[i, t] * liquefaction_cost for i in I for t in T) +
                              quicksum(FC_ENG[j, t] * CostMGO for j in J for t in T) +
                              quicksum(FC_ENG[j, t] * CO2eqMGO * CO2tax for j in J for t in T) + 
                              Electricity_price*0.8*E_Battmax*(1/Battery_efficiency)), index=0, priority=2, reltol=0)    #1% tolerance=0.01
        HESOPT.setObjectiveN(quicksum(Vdelta_total[1,t] for t in T), index=1, priority=1, reltol=0)
        
        
        HESOPT.modelSense = GRB.MINIMIZE
        HESOPT.update()
        # Solve the optimization problem
        HESOPT.setParam( 'OutputFlag', True) # silencing gurobi output or not
        #HESOPT.setParam('NonConvex', 2)
        #HESOPT.setParam('Timelimit', 3600)
        #HESOPT.setParam ('MIPGap', 0.05);       # find the optimal solution
        HESOPT.write("stage2DPB.lp")            # print the model in .lp format file
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
                StateOfCharge.append(E_batt[t].x/E_Battmax)                                                                                               # Collect SOC values of battery at each time step
                for i in I:
                    power_output_fc[(i, t)] = P_FC[i, t].x                                                                                                  # Collect P_FC values for each stack at each time step
                for j in J:
                    power_output_DG[(j,t)]=P_DG[j,t].x                                                                                                      # Collect P_DG values for each DG at each time step
            

           
            # Create a separate image for each fuel cell stack
            for i in I:
                 # Create a new figure for each stack
                 plt.figure(figsize=(25, 15))
                 plt.plot(T, [power_output_fc[(i, t)] for t in T], label='Power output FC {i}', color='red')
                 plt.xlabel('Time Step', fontsize=20)
                 plt.xticks(fontsize=18)
                 plt.ylabel(f'Power Output FC Stack {i} (kW)', fontsize=20)
                 plt.yticks(fontsize=18)
                 plt.title(f'Power Output of FC Stack {i} vs Time', fontsize=22)
                 plt.show()
                     
            for j in J:
                plt.figure(figsize=(25,15))
                plt.plot(T, [power_output_DG[(j,t)] for t in T], label= 'Power output DG {j}', color= 'purple')
                plt.xlabel('Time step', fontsize=20)
                plt.xticks(fontsize=18)
                plt.ylabel(f'Power Output of DG {j} (kW)', fontsize=20)
                plt.yticks(fontsize=18)
                plt.title(f'Power output of DG {j} vs Time', fontsize=22)
                plt.show()
                
            SOC_chg=0.8
            SOC_dchg=0
            Cycle_chg=0
            Cycle_dchg=0
            
            for t in range(1, len(StateOfCharge)):
                dSOC = StateOfCharge[t]- StateOfCharge[t-1]
                if dSOC > 0:
                    Up = dSOC
                    SOC_chg = SOC_chg + Up
                    if SOC_chg >= 1:
                        Cycle_chg +=1
                        SOC_chg = 0
                elif dSOC < 0:
                    Down = -dSOC
                    SOC_dchg = SOC_dchg + Down
                    if SOC_dchg >= 1:
                        Cycle_dchg +=1
                        SOC_dchg = 0
              
            Total_cycles = (Cycle_chg + Cycle_dchg + SOC_chg + SOC_dchg)/2           
            
            #degradation of battery per month due to cycling aging 
            Cycle_deg=(Total_cycles*5)/6250
            
            
            #degradation due to calendar aging per month-fitting parameters
            a1 = 0.00157
            a2 = 1.317  
            b1 = 142300
            b2 = -3492
            c1 = 0.48
            
            #calendar aging per month of 6 days at 25 deg celcius  
            calendar_aging = a1 * (math.e)**(a2*StateOfCharge[len(T)-1]) * b1 * (math.e)**(b2/298.15) * (6)**c1
            
            Battdegcumulative += calendar_aging + Cycle_deg
            
            SOH_Battery.append(Batt_SOH)
            
            # If battery is to be replaced at it's end of life
            if Battdegcumulative > 0.2:
                Batt_SOH = 1
                Battdegcumulative = 0
            elif Battdegcumulative < 0.2:
                Batt_SOH = 1- Battdegcumulative
            
            #Batt_SOH = 1- Battdegcumulative      
            
            Vref=0.979
            #monthly 5 trips and are added to the cumulative value
            Vcumulative += float(sum(Vdelta_total[1,t].x for t in T))*10**-6*5
            
            SOH_PEMFC.append(PEMFC_SOH)
            PEMFC_SOH= (Vref-Vcumulative)/Vref

            Month += 1
            
            k_1deg = 0.09866
            k_2deg = 0.1245
            
            k_1p = k_1deg * (-Vcumulative) + k_2deg
            
            #store the increase in hydrogen conumption in a list
            total_hydrogen_consumption = float(sum(F_fc[i,t].x for i in I for t in T))
            Monthly_hydrogen_consumption.append(total_hydrogen_consumption)

            #list to store the OPEX of each month
            opex = HESOPT.getObjective(0).getValue(),
            OPEX.append(opex)
            
            # Extract and store the results for the current scenario
            result = {
                "Scenario": scenario["name"],
                "OPEX": HESOPT.getObjective(0).getValue(),
                "Month": Month,
                "PEMFC_SOH": PEMFC_SOH,
                "Batt_SOH": Batt_SOH,
                "Battcapacity": E_Battmax,
                "Batt_charge_cost": Electricity_price*0.6*E_Battmax*(1/Battery_efficiency),
                "H2_Wt": float(sum(F_fc[i, t].x for i in I for t in T)),
                "Cost_H2": float(sum(F_fc[i, t].x for i in I for t in T) * scenario["cost"]),
                "H2_liquefactioncost": float(sum(F_fc[i, t].x for i in I for t in T) * liquefaction_cost),
                "Cost_CO2(H2)":float(sum(F_fc[i,t].x for i in I for t in T)*scenario["emission"]*CO2tax),
                "MGO_Wt": float(sum(FC_ENG[j, t].x for j in J for t in T)),
                "Cost_MGO": float(sum(FC_ENG[j, t].x for j in J for t in T) * CostMGO),
                'CO2_Cost(MGO)': float(sum(FC_ENG[j,t].x for j in J for t in T)*CO2eqMGO*CO2tax),
                "Emissions": float(sum(F_fc[i,t].x for i in I for t in T)*scenario["emission"]) 
                            + float(sum(FC_ENG[j,t].x for j in J for t in T)*CO2eqMGO),
                "Total_degradation": HESOPT.getObjective(1).getValue(),
                'Voltageloss_stup':float(sum(Vdelta_stup[1,t].x for t in T)),
                'Voltageloss_loadchange':float(sum(Vdelta_loadchange[1,t].x for t in T)),
                'Voltage_losspower':float(sum(Vdelta_PFC[1,t].x for t in T)),
                'Total_batterycycles': Total_cycles,
            }
            scenario_results.append(result)
            
            hours = [t / 60 for t in T]
            
            
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
            ax1.set_title(f'Battery Power vs SOC for Month:' + str(Month), fontsize=30)
            # Combine legends for both axes
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            lines = lines1 + lines2
            labels = labels1 + labels2
            ax1.legend(lines, labels, loc='upper left', fontsize=30)
    
            #plt.savefig(f'Power of battery vs SOC_{weight_CAPEX:.2f}.png', dpi=600)
            plt.show()
        
      
        
        # Create a Matplotlib figure
            plt.figure(figsize=(25, 15))
        # Plot the total power output of all the fuel cell stacks
            plt.plot(hours, total_power_fc, label='Total FC Power Output', color='green')
            #plot the DG power
            plt.plot(hours, DG_power, label='Total DG power', color='purple')
            # Plot the power of the battery
            plt.plot(hours, power_battery_discharge, label='Battery discharge Power', color='red')
            plt.plot(hours, [-p for p in power_battery_charge], label='Battery charge power', color='blue')
            # Plot the load profile
            plt.plot(hours, load_profile, label='Load Profile', linestyle='dotted', color='black')
            # Customize the plot with labels and a title
            plt.xlabel('Time(hours)', fontsize=35)
            plt.xticks(fontsize=30)
            plt.ylabel('Power (kW)', fontsize=35)
            plt.yticks(fontsize=30)
            plt.title(f'Power Output of DG, FC Stacks and Battery Over Time for Month:' + str(Month), fontsize=26)
        
            plt.legend(fontsize=22, loc='best')
            #plt.savefig(f'Power output of all components_{weight_CAPEX:.2f}.png', dpi=600)
            # Show the plot
            plt.show()
     
            #plt.figure(figsize=(10, 6))
            #plt.plot(T, Difference, label=f'Difference of power', color='darkcyan')
            # Customize the plot with labels and a title
            #plt.xlabel('Time Step')
            #plt.ylabel('Load - (Battery + FC) (kW)')
            #plt.title(f'Load and delivered power difference (Weight_CAPEX = {weight_combo["weight_CAPEX"]})  Scenario:' + scenario["name"])
    
            # Add a legend to distinguish different weight combinations
            #plt.legend(loc='upper right')
            #plt.savefig(f'Power_difference_{weight_CAPEX:.2f}.png', dpi=600)
            # Show the plot
            #plt.show()
            
        else:
            infeasible_scenarios.append(scenario["name"])
            break
        
    plt.figure(figsize=(15, 8))
    plt.plot(range(1, Month + 1), Monthly_hydrogen_consumption, marker='o', linestyle='-', color='blue')
    plt.xlabel('Month', fontsize=14)
    plt.ylabel('Total Hydrogen Consumption (kg)', fontsize=14)
    plt.title('Increase in Hydrogen Consumption Over Months', fontsize=18)
    plt.grid(True)
    plt.xticks(range(1, Month + 1, 3), fontsize=14)
    plt.yticks(fontsize=14)
    plt.show()
    

    SOH_Battery.append(Batt_SOH)
    SOH_PEMFC.append(PEMFC_SOH)
    
    # Create a plot to display Battery SOH and PEMFC SOH values over months
    plt.figure(figsize=(15, 8))
    plt.plot(range(1, Month + 2), SOH_Battery, marker='o', linestyle='-', color='blue', label='Battery SOH')
    plt.plot(range(1, Month + 2), SOH_PEMFC, marker='o', linestyle='-', color='green', label='PEMFC SOH')
     
    # Add labels and title to the plot
    plt.xlabel('Month', fontsize=14)
    plt.ylabel('State of Health (SOH)', fontsize=14)
    plt.title('Battery and PEMFC State of Health Over Months', fontsize=18)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.xticks(range(1, Month + 2, 3), fontsize=14)
    plt.yticks(fontsize=14)
    plt.show()


if scenario_results:
    # Create a Pandas DataFrame from the list of solution dictionaries    
    solutions_df = pd.DataFrame(scenario_results)   
   
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

