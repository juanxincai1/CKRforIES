import numpy as np
from gym import spaces
import xlrd

class IesEnv():
    def __init__(self, period, day ,time_step):
        self.period = period
        self.day = day
        self.time_step = time_step
        #Data preparation
        excel = xlrd.open_workbook('data.xls',encoding_override="utf-8")
        sheet_load = excel.sheet_by_index(0)
        list_elec = []
        list_heat = []
        list_ld = []
        for x in range(300):
            value = sheet_load.row_values(x,0,72)
            list_elec.append(value[0:24])
            list_heat.append(value[24:48])
            list_ld.append(value[48:72])
        #Parameter definition
        self.load_elec = np.array(list_elec)
        self.load_heat = np.array(list_heat)
        self.load_ld = np.array(list_ld)
        self.Yita_CHP_rec = 0.85
        self.Yita_CHP_e = 0.42
        self.Yita_CHP_h = 0.35
        self.Cop_HP_h = 4
        self.Yita_GB = 0.76
        self.Yita_RG = 1
        self.Yita_ch = -0.95
        self.Yita_dis = -0.95
        #running cost
        self.c_mr = np.array(([0.05, 0.04, 0.02, 0.02, 0.02])) .reshape(1,-1)
        self.Gas_caloritic_value = 33.812
        self.Yita_energy_init_capacity = 0.5
        self.energy_storage_capacity_upper = 0.9
        self.energy_storage_capacity_lower = 0.1
        self.RG_to_elec=self.load_ld
        # Price definition
        self.Gas_price = 0.0321
        self.peak = 0.1999
        self.flat = 0.1199
        self.cereal = 0.0399
        self.Elec_price = np.concatenate(
            (self.cereal * (np.ones(7)), self.flat * np.ones(5), self.peak * np.ones(7),
            self.flat * np.ones(4), self.cereal * np.ones(1))).reshape(1,-1)
        #Upper and lower limits
        self.max_output_CHP = 2560 * np.ones(self.period).reshape(1,-1)
        self.min_output_CHP = np.zeros(self.period).reshape(1,-1)
        self.max_output_HP = 2400 * np.ones(self.period).reshape(1,-1)
        self.min_output_HP = np.zeros(self.period).reshape(1,-1)
        self.max_output_GB = 2560 * np.ones(self.period).reshape(1,-1)
        self.min_output_GB = np.zeros(self.period).reshape(1,-1)
        self.max_buyElec = 4000 * np.ones(self.period).reshape(1,-1)
        self.min_buyElec = -4000 * np.ones(self.period).reshape(1,-1)
        self.max_ES = 320 * np.ones(self.period).reshape(1,-1)
        self.min_ES = -320 * np.ones(self.period).reshape(1,-1)
        self.ES_capacity = 1600 * np.ones(self.period + 1).reshape(1,-1)
        self.ES_capacity_upper = self.energy_storage_capacity_upper * self.ES_capacity
        self.ES_capacity_lower = self.energy_storage_capacity_lower * self.ES_capacity
        self.energy_storage_init_capacity = np.zeros(self.period + 1).reshape(1,-1)
        self.energy_storage_init_capacity[0,0] = self.Yita_energy_init_capacity * self.ES_capacity[0,0]
        self.energy_storage_init_capacity[0,self.time_step] = np.clip(self.energy_storage_init_capacity[0,self.time_step],
                                                           self.ES_capacity_lower[0,self.time_step],
                                                           self.ES_capacity_upper[0,self.time_step])
        max_output_action = np.hstack((self.max_output_CHP[0,self.time_step],
                                      2*self.max_ES[0,self.time_step],
                                      self.max_output_HP[0,self.time_step])).reshape(1,-1)
        min_output_action = np.hstack((self.min_output_CHP[0,self.time_step],
                                      self.min_ES[0,self.time_step],
                                      self.min_output_HP[0,self.time_step])).reshape(1,-1)
        max_output_action1=max_output_action.reshape(-1,)
        min_output_action1=min_output_action.reshape(-1,)
        self.action_space = spaces.Box(low=min_output_action1, high=max_output_action1, dtype=np.float32)
        high = np.ones((5, 1))
        high = high.reshape((-1,))
        low = np.vstack(np.zeros((5, 1)))
        low = low.reshape((-1,))
        self.observation_space = spaces.Box(low=low, high=high,
                                            dtype=np.float32)
        self.state=[]
    #Initialization status
    def reset(self,day):
        self.time_step=0
        self.state = np.hstack((self.load_elec[day,self.time_step], self.load_heat[day,self.time_step], self.RG_to_elec[day,self.time_step], self.energy_storage_init_capacity[0,self.time_step], self.time_step )).reshape(1,-1)
        return self.state
    #Status Replacement and Reward Value Return
    def forward(self, action, state ,day):
        CHP_heat_out=action[0,0]*state[0,1]
        out_HP_heat=action[0,1]*state[0,1]
        out_GB_heat=action[0,2]*state[0,1]
        CHP_elc_out=CHP_heat_out*self.Yita_CHP_e/self.Yita_CHP_h
        out_HP_elec=out_HP_heat/self.Cop_HP_h
        elec_demand=state[0,0]+out_HP_elec-CHP_elc_out-state[0,2]-action[0,3]
        if action[0,3] < 0:
            self.energy_storage_init_capacity[0,self.time_step + 1] = self.energy_storage_init_capacity[0,self.time_step] + \
                                                              action[0,3] * self.Yita_ch
        else:
            self.energy_storage_init_capacity[0,self.time_step + 1] = self.energy_storage_init_capacity[0,self.time_step] + \
                                                              action[0,3] / self.Yita_dis
        self.energy_storage_init_capacity[0,self.time_step + 1] = np.clip(self.energy_storage_init_capacity[0,self.time_step + 1], self.ES_capacity_lower[0,self.time_step + 1],
                                                    self.ES_capacity_upper[0,self.time_step + 1])
        if self.time_step==23:
            es_loss = self.compute_esloss(self.energy_storage_init_capacity)
        else:
            es_loss=0
        current_cost, mr_cost ,elc_cost, gas_cost = self.compute_cost(state,elec_demand,CHP_elc_out,out_GB_heat,out_HP_elec)
        #Transfinite punishment
        dhr_loss = np.zeros((4))
        dhr_loss[0] = self.compute_over_cost(CHP_elc_out, self.min_output_CHP[0,0], self.max_output_CHP[0,0])  # CHP
        dhr_loss[1] = self.compute_over_cost(out_HP_heat, self.min_output_HP[0,0], self.max_output_HP[0,0])  #HP
        dhr_loss[2] = self.compute_over_cost(out_GB_heat, self.min_output_GB[0,0], self.max_output_GB[0,0])  #GB
        dhr_loss[3] = self.compute_over_cost(elec_demand, self.min_buyElec[0,0], self.max_buyElec[0,0])  #elec
        uneq_loss = np.sum(dhr_loss)
        loss = 0.01*current_cost + uneq_loss + es_loss
        self.time_step += 1
        if self.time_step < 24:
            state_ = np.hstack((self.load_elec[day,self.time_step], self.load_heat[day,self.time_step], self.RG_to_elec[day,self.time_step], self.energy_storage_init_capacity[0,self.time_step], self.time_step )).reshape(1,-1)
        else:
            state_=np.hstack((self.load_elec[day,self.time_step-24], self.load_heat[day,self.time_step-24], self.RG_to_elec[day,self.time_step-24], self.energy_storage_init_capacity[0,self.time_step-24], self.time_step-24 )).reshape(1,-1)
        return state_, -loss,current_cost
    #Calculation of overrun cost
    def compute_over_cost(self, value, down, high):
        if value>high:
            loss=np.linalg.norm(value-high)
        elif value<down:
            loss=np.linalg.norm(value-down)
        else:
            loss=0
        return loss
    #Calculating costs
    def compute_cost(self,state,elec_demand,CHP_out_elc,out_GB_heat,out_HP_elec):
        if elec_demand>0:
            cost_elec =np.sum(elec_demand * self.Elec_price[0,self.time_step])
        else:
            cost_elec=-np.sum(elec_demand * self.Elec_price[0,self.time_step]*0.5)
        chp_gas = CHP_out_elc / self.Yita_CHP_e
        gb_gas = out_GB_heat/self.Yita_GB
        cost_gas = np.sum(chp_gas + gb_gas) * self.Gas_price
        cost_mr = np.sum(self.c_mr[0,0] * CHP_out_elc + self.c_mr[0,1] * (out_HP_elec) + self.c_mr[0,2] * state[0,3]+ self.c_mr[0,3] * out_GB_heat \
                 + self.c_mr[0,4] * state[0,2] )
        cost = cost_elec + cost_gas + cost_mr
        return cost, cost_mr,cost_elec, cost_gas
    #Calculate the cost of battery energy storage
    def compute_esloss(self, cur_state_pre):
        es_loss = np.linalg.norm(cur_state_pre[0,0] - cur_state_pre[0,self.period-1])
        return es_loss
