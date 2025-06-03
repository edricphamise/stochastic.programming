#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  3 13:21:09 2025

@author: edricpham
"""

import numpy as np
import pyomo.environ as pyo
import sys
from pyomo.environ import *
from pyomo.opt import SolverFactory

class Benders:
    def __init__(self):
        self.stage = [0, 1, 2]
        self.node = [[0], [0, 1], [0, 1, 2, 3]]
        self.c = [[1], [7, 5], [0, 0, 0, 0]]
        self.f = [[1], [3, 2], [3, 7, 2, 3]]
        self.b = [[0], [700, 50], [210, 300, 350, 500]]
        self.prob_t1 = 0.3
        self.prob_t21 = 0.4
        self.prob_t22 = 0.5
        self.probability = [[1], 
                            [self.prob_t1, (1 - self.prob_t1)], 
                            [self.prob_t1*self.prob_t21, self.prob_t1*(1-self.prob_t21), 
                             (1 - self.prob_t1)*self.prob_t22, (1 - self.prob_t1)*(1 - self.prob_t22)]]
        self.d = [[50], [30, 40], [0, 0, 0, 0]]
        self.A = [[0], [2, 1], [1, 2, 3, 1]]
        self.B = [[0], [2, 3], [1, 2, 2, 2]]
        self.C = [[0], [0, 0], [2, 1, 0, 0]]
        self.D = [[0], [0, 0], [0, 0, 3, 5]]
        self.eps = 1e-3
        self.max_iterations = 50
        self.LB = sys.float_info.min
        self.UB = sys.float_info.max
        self.x = [[0] * len(i) for i in self.node]
        self.y = [[0] * len(i) for i in self.node]
        self.lower_bounds = []
        self.upper_bounds = []
        self.optimality_cuts = []
    
    def subproblem(self, x, t, s):
        m = ConcreteModel()
        m.T = Set(initialize=self.stage)
        m.S = Set(initialize=self.node[t])
        m.y = Var(m.T, m.S, domain=NonNegativeReals)
        m.obj = Objective(expr = self.f[t][s]*m.y[t,s], sense = minimize)
        m.constraint_demand = Constraint(expr = self.A[t][s] * x[0][0] + self.C[t][s] * x[1][0] 
                                  + self.D[t][s] * x[1][1] + self.B[t][s] * m.y[t,s] == self.b[t][s])
        m.dual = Suffix(direction=Suffix.IMPORT)
        SolverFactory("glpk").solve(m)
        return m.dual[m.constraint_demand], value(m.obj), value(m.y[t,s])

    def masterproblem(self, x, t, s, investment_cost):
        m = ConcreteModel()
        m.T = Set(initialize=self.stage)
        m.S = Set(initialize=self.node[t])
        m.x = Var(m.T, m.S, domain = NonNegativeReals)
        m.n = Var(m.T, m.S, domain = NonNegativeReals)
        m.constraint_investcap = Constraint(expr = m.x[t,s] <= self.d[t][s])
        m.constraint_optcut = ConstraintList()
        for pi in self.optimality_cuts:
            if [t,s] == [0,0]:
                optcut = m.n[t,s] >= (sum(self.probability[t0][s0] * pi[t0][s0] * (self.b[t0][s0] - self.A[t0][s0]*m.x[t,s] 
                                        - self.C[t0][s0]*x[1][0] - self.D[t0][s0]*x[1][1])
                                          for t0 in m.T if t0 > t
                                          for s0 in self.node[t0])
                                        + investment_cost)
                
            elif [t,s] == [1,0]:
                optcut = m.n[t,s] >= sum(self.probability[t0][s0] * pi[t0][s0] * (self.b[t0][s0] - self.A[t0][s0]*x[0][0] 
                                        - self.C[t0][s0]*m.x[t,s] - self.D[t0][s0]*x[1][1])
                                         for t0 in m.T if t0 > t
                                         for s0 in self.node[t0] if s0 in [0,1])
                                       
            elif [t,s] == [1,1]:
                optcut = m.n[t,s] >= sum(self.probability[t0][s0] * pi[t0][s0] * (self.b[t0][s0] - self.A[t0][s0]*x[0][0] 
                                        - self.C[t0][s0]*x[1][0] - self.D[t0][s0]*m.x[t,s]) 
                                         for t0 in m.T if t0 > t
                                         for s0 in self.node[t0] if s0 in [2,3])
            
            m.constraint_optcut.add(optcut)
                
        m.obj = Objective(expr = self.probability[t][s]*self.c[t][s]*m.x[t,s] + m.n[t,s], sense = minimize)
        SolverFactory('glpk').solve(m)
        return value(m.x[t,s]), value(m.obj), value(m.n[t,s]), value(m.obj) - value(m.n[t,s])

    def solve_problem(self):
        i = 0
        self.x[0][0] = 1
        self.x[1][0] = 3
        self.x[1][1] = 1

        while abs((self.UB - self.LB)/self.UB) >= self.eps and i <= self.max_iterations:
            objective_2nd_stage = []
            pi_list = [[0] * len(i) for i in self.node]
            
            # Solve sub problem
            for t in range(1, len(self.stage)):
                for s in self.node[t]:
                    pi, objective, _ = self.subproblem(self.x, t, s)
                    objective_2nd_stage.append(objective * self.probability[t][s])
                    pi_list[t][s] = pi
            total_objective_2nd_stage = np.sum(objective_2nd_stage, axis=0)
            
            self.optimality_cuts.append(pi_list)
          
            # Update upper bound
            self.UB = min(self.UB,
                            self.probability[0][0]*self.c[0][0]*self.x[0][0]
                          + self.probability[1][0]*self.c[1][0]*self.x[1][0]
                          + self.probability[1][1]*self.c[1][1]*self.x[1][1] 
                          + total_objective_2nd_stage)

            # Solve master problem
            total_investment = 0
            for t in reversed(self.stage[:-1]):
                for s in self.node[t]:
                    x, objective_master, n, investment_cost = self.masterproblem(self.x, t, s, total_investment)
                    total_investment += investment_cost
                    self.x[t][s] = x
                    
            # Update lower bound
            self.LB = objective_master

            # Update iteration
            i += 1

            # Save values for bounds 
            self.lower_bounds.append(self.LB)
            self.upper_bounds.append(self.UB)

            print("Iteration i={} : UB={}, LB={}".format(i, self.UB, self.LB))

        for t in [0,1,2]:
            for s in self.node[t]:
                _, _, y = self.subproblem(self.x, t, s)
                self.y[t][s] = y 

        # Display the results
        self.show_results(self.LB, self.x, self.y)

    def show_results(self, objective, x, y):
        print(f"The optimal objective function is {objective}")
        for t in self.stage:
            for s in self.node[t]:
                print(f"At node {t,s}, investment = {x[t][s]} and operations = {y[t][s]}")
        
benders_dec = Benders()
benders_dec.solve_problem()