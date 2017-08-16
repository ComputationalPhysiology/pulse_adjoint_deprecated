# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 10:38:52 2017

@author: sigurdll
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 13:50:17 2017

@author: sigurdll
"""

#Imports
import Tkinter as tk
import ttk
import vtk
import tkFileDialog
import tkMessageBox
import ntpath
def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

import dolfin as df

import os

import shutil

from mesh_generation.mesh_utils import load_geometry_from_h5
                          
from general_solver import general_solver, save_to_h5, load_from_h5

from closed_loop_inhomogenous import closed_loop

from demo_vtk import demo_mesh
from demo_vtk import demo_facetfunction
from demo_vtk import demo_cellfunction
from demo_vtk import demo_displacement
from demo_vtk import demo_scalarfunction

import sys

sys.path.append("/usr/lib/python2.7/dist-packages/vtk")

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from vtk.tk.vtkTkRenderWindowInteractor import vtkTkRenderWindowInteractor
from bullseye_plot import bullseye_plot
import numpy as np

N = tk.N
S = tk.S
E = tk.E
W = tk.W

#Using regular Tkinter widgets to create custom widgets.

#Modification: Easyer gridding, resizing, and style        
class CustOptionMenu(ttk.OptionMenu):
    def __init__(self, *args, **kwargs):
        coor = kwargs.pop("coor", None)
        ttk.OptionMenu.__init__(self, *args, **kwargs)
        self.s = ttk.Style()
        self.s.theme_use('clam')
        self.configure(style='TMenubutton')
        if coor is not None:
            self.grid(row = coor[0], column = coor[1], rowspan = coor[2], columnspan = coor[3],  sticky = N+S+E+W)

#Modification: Easyer gridding, resizing, and style 
class CustFrame(ttk.Frame):
    def __init__(self, *args, **kwargs):
        coor = kwargs.pop('coor', None)
        ttk.Frame.__init__(self, *args, **kwargs)
        self.s = ttk.Style()
        self.s.theme_use('clam')
        self.configure(style='TFrame')
        if coor is not None:
            self.grid(row = coor[0], column = coor[1], rowspan = coor[2], columnspan = coor[3],  sticky = N+S+E+W)
        
    def resize(self):
        for rows in range(self.grid_size()[1]):
            tk.Grid.rowconfigure(self, rows, weight = 1)
        for cols in range(self.grid_size()[0]):
            tk.Grid.columnconfigure(self, cols, weight = 1)

#Modification: Easyer placement and resizing
class CustToplevel(tk.Toplevel):
    def __init__(self, *args, **kwargs):
        place = kwargs.pop('place', None)
        tk.Toplevel.__init__(self, *args, **kwargs)
        if place is not None:
            self.geometry("%dx%d%+d%+d" % (place[0], place[1], place[2], place[3]))
    def resize(self):
        for rows in range(self.grid_size()[1]):
            tk.Grid.rowconfigure(self, rows, weight = 1)
        for cols in range(self.grid_size()[0]):
            tk.Grid.columnconfigure(self, cols, weight = 1)

#Modification: Easyer gridding, resizing, and style 
class CustLabel(ttk.Label):
    def __init__(self, *args, **kwargs):
        coor = kwargs.pop('coor', None)
        ttk.Label.__init__(self, *args, **kwargs)
        self.s = ttk.Style()
        self.s.theme_use('clam')
        self.configure(style='TLabel')
        if coor is not None:
            self.grid(row = coor[0], column = coor[1], rowspan = coor[2], columnspan = coor[3],  sticky = N+S+E+W)

#Modification: Easyer gridding, resizing, and style 
class CustTitle(ttk.Label):
    def __init__(self, *args, **kwargs):
        coor = kwargs.pop('coor', None)
        ttk.Label.__init__(self, *args, **kwargs)
        self.s = ttk.Style()
        self.s.theme_use('clam')
        self.configure(style='TLabel')
        self.configure(font=(10))
        if coor is not None:
            self.grid(row = coor[0], column = coor[1], rowspan = coor[2], columnspan = coor[3],  sticky = N+S+E+W)

#Modification: Easyer gridding, resizing, style and default value         
class CustEntry(ttk.Entry):
    def __init__(self, *args, **kwargs):
        coor = kwargs.pop('coor', None)
        def_val = kwargs.pop('def_val', None)
        ttk.Entry.__init__(self, *args, **kwargs)
        self.s = ttk.Style()
        self.s.theme_use('clam')
        self.configure(style='TEntry')
        if coor is not None:
            self.grid(row = coor[0], column = coor[1], rowspan = coor[2], columnspan = coor[3],  sticky = N+S+E+W)
        if def_val is not None:
            self.insert(0, def_val)

#Modification: Easyer gridding, resizing, and style         
class CustButton(ttk.Button):
    def __init__(self, *args, **kwargs):
        coor = kwargs.pop('coor', None)
        ttk.Button.__init__(self, *args, **kwargs)
        self.s = ttk.Style()
        self.s.theme_use('clam')
        self.configure(style='TButton')
        if coor is not None:
            self.grid(row = coor[0], column = coor[1], rowspan = coor[2], columnspan = coor[3],  sticky = N+S+E+W)

#Modification: Easyer gridding, resizing, and style       
class CustNotebook(ttk.Notebook):
    def __init__(self, *args, **kwargs):
        coor = kwargs.pop('coor', None)
        ttk.Notebook.__init__(self, *args, **kwargs)
        self.s = ttk.Style()
        self.s.theme_use('clam')
        self.configure(style='TNotebook')
        if coor is not None:
            self.grid(row = coor[0], column = coor[1], rowspan = coor[2], columnspan = coor[3],  sticky = N+S+E+W)
        
    def resize(self):
        for rows in range(self.grid_size()[1]):
            tk.Grid.rowconfigure(self, rows, weight = 1)
        for cols in range(self.grid_size()[0]):
            tk.Grid.columnconfigure(self, cols, weight = 1)

#Modification: Easyer griding, resizing and setting mouse interactor     
class CustRWI(vtkTkRenderWindowInteractor):
    def __init__(self, *args, **kwargs):
        coor = kwargs.pop('coor', None)
        vtkTkRenderWindowInteractor.__init__(self, *args, **kwargs)
        self.style = vtk.vtkInteractorStyleTrackballCamera()
        self.SetInteractorStyle(self.style)
        #self.Initialize()
        #self.Start()
        if coor is not None:
            self.grid(row = coor[0], column = coor[1], rowspan = coor[2], columnspan = coor[3],  sticky = N+S+E+W)
            
    def resize(self):
        for rows in range(self.grid_size()[1]):
            tk.Grid.rowconfigure(self, rows, weight = 1)
        for cols in range(self.grid_size()[0]):
            tk.Grid.columnconfigure(self, cols, weight = 1)

#Modification: Easyer gridding and resizing
class CustText(tk.Text):
    def __init__(self, *args, **kwargs):
        coor = kwargs.pop('coor', None)
        tk.Text.__init__(self, *args, **kwargs)
        if coor is not None:
            self.grid(row = coor[0], column = coor[1], rowspan = coor[2], columnspan = coor[3],  sticky = N+S+E+W)
    
    def resize(self):
        for rows in range(self.grid_size()[1]):
            tk.Grid.rowconfigure(self, rows, weight = 1)
        for cols in range(self.grid_size()[0]):
            tk.Grid.columnconfigure(self, cols, weight = 1)

#Modification: Easyer gridding
class CustFigCanvas(FigureCanvasTkAgg):
    def __init__(self, *args, **kwargs):
        coor = kwargs.pop('coor', None)
        FigureCanvasTkAgg.__init__(self, *args, **kwargs)
        if coor is not None:
            self.get_tk_widget().grid(row = coor[0], column = coor[1], rowspan = coor[2], columnspan = coor[3],  sticky = N+S+E+W)
        self.show()


import threading
import logging

#Creating a redirector for output to terminal. This is used to redirect both the logger used in the solver and regular print statments.
class Std_redirector(object):
    def __init__(self,widget):
        self.widget = widget

    def write(self,string):
        self.widget.insert(tk.END,string)
        self.widget.see(tk.END)

#Creating a frame for the redirected output.
class OutputFrame(CustFrame):
    def __init__(self, *args, **kwargs):
        CustFrame.__init__(self, *args, **kwargs)
        self.text = CustText(master = self, coor = [0,0,1,2])#, wrap = "word")
        self.text.resize()
        logging.basicConfig(stream=Std_redirector(self.text), level = logging.INFO)
        sys.stdout = Std_redirector(self.text)
        self.resize()

#Custom threading class
class CustThread(threading.Thread):
    def __init__(self, *args, **kwargs):
        threading.Thread.__init__(self, *args, **kwargs)
        self._stop_run = False
    def stop(self):
        return

#Toplevel window for Holzapfel Ogden parameters
class MatParHO(CustToplevel):
    def __init__(self, *args, **kwargs):
        CustToplevel.__init__(self, *args, **kwargs)
        #Labels and entries
        self.title = CustTitle(master = self, coor = [0,0,1,2], text = 'Holzapfel Ogden')
        self.label_a = CustLabel(master = self, coor = [1,0,1,1], text = 'a')
        self.entry_a = CustEntry(master = self, coor = [1,1,1,1], def_val = 2.28)
        self.label_a_f = CustLabel(master = self, coor = [2,0,1,1], text = 'a_f')
        self.entry_a_f = CustEntry(master = self, coor = [2,1,1,1], def_val = 1.685)
        self.label_b = CustLabel(master = self, coor = [3,0,1,1], text = 'b')
        self.entry_b = CustEntry(master = self, coor = [3,1,1,1], def_val = 9.726)
        self.label_b_f = CustLabel(master = self, coor = [4,0,1,1], text = 'b_f')
        self.entry_b_f = CustEntry(master = self, coor = [4,1,1,1], def_val = 15.779)
        #Two buttons; one for saving changes and one to exit the toplevel window without saving changes
        self.but_ok = CustButton(master = self, coor = [5,1,1,1], text = 'Ok', command = self.confirm_HO)
        self.but_cancel = CustButton(master = self, coor = [5,0,1,1], text = 'Cancel', command = self.withdraw)
        
        self.resize()
        self.withdraw()
        self.protocol("WM_DELETE_WINDOW", self.withdraw)
        
    def confirm_HO(self):
        self.mat_params={'a' : float(self.entry_a.get()),
                         'a_f' : float(self.entry_a_f.get()),
                         'b' : float(self.entry_b.get()),
                         'b_f' : float(self.entry_b_f.get())}
        self.withdraw()
        return
        
#Toplevel window for Guccione parameters    
class MatParG(CustToplevel):
    def __init__(self, *args, **kwargs):
        CustToplevel.__init__(self, *args, **kwargs)
        #Labels and entries
        self.title = CustTitle(master = self, coor = [0,0,1,2], text = 'Guccione')
        self.label_C = CustLabel(master = self, coor = [1,0,1,1], text = 'C')
        self.entry_C = CustEntry(master = self, coor = [1,1,1,1], def_val = 2.0)
        self.label_b_f = CustLabel(master = self, coor = [2,0,1,1], text = 'b_f')
        self.entry_b_f = CustEntry(master = self, coor = [2,1,1,1], def_val = 8.0)
        self.label_b_t= CustLabel(master = self, coor = [3,0,1,1], text = 'b_t')
        self.entry_b_t = CustEntry(master = self, coor = [3,1,1,1], def_val = 2.0)
        self.label_b_fs = CustLabel(master = self, coor = [4,0,1,1], text = 'b_fs')
        self.entry_b_fs = CustEntry(master = self, coor = [4,1,1,1], def_val = 4.0)
        #Two buttons; one for saving changes and one to exit the toplevel window without saving changes
        self.but_ok = CustButton(master = self, coor = [5,1,1,1], text = 'Ok', command = self.confirm_G)
        self.but_cancel = CustButton(master = self, coor = [5,0,1,1], text = 'Cancel', command = self.withdraw)
        
        self.resize()
        self.withdraw()
        self.protocol("WM_DELETE_WINDOW", self.withdraw)
        
    def confirm_G(self):
        self.mat_params={'C' : float(self.entry_C.get()),
                         'bf' : float(self.entry_b_f.get()),
                         'bfs' : float(self.entry_b_fs.get()),
                         'bt' : float(self.entry_b_t.get())}
        self.withdraw()
        return

#Toplevel window for Neo Hookean parameters 
class MatParNH(CustToplevel):
    def __init__(self, *args, **kwargs):
        CustToplevel.__init__(self, *args, **kwargs)
        #Labels and entries
        self.title = CustTitle(master = self, coor = [0,0,1,2], text = 'Neo Hookean')
        self.label_mu = CustLabel(master = self, coor = [1,0,1,1], text = 'mu')
        self.entry_mu = CustEntry(master = self, coor = [1,1,1,1], def_val = 15.0)
        #Two buttons; one for saving changes and one to exit the toplevel window without saving changes
        self.but_ok = CustButton(master = self, coor = [2,1,1,1], text = 'Ok', command = self.confirm_NH)
        self.but_cancel = CustButton(master = self, coor = [2,0,1,1], text = 'Cancel', command = self.withdraw)
        
        self.resize()
        self.withdraw()
        self.protocol("WM_DELETE_WINDOW", self.withdraw)
        
    def confirm_NH(self):
        self.mat_params={'mu' : float(self.entry_mu.get())}
        self.withdraw()
        return

#Toplevel window for advanced parameters         
class AdvancedParameters(CustToplevel):
    def __init__(self, *args, **kwargs):
        CustToplevel.__init__(self, *args, **kwargs)

        self.title = CustTitle(master = self, coor = [0,0,1,2], text = 'Advanced parameters')
        
        self.label_active_model = CustLabel(master = self, coor = [1,0,1,1], text = 'Active model')
        active_model_optlist=('Active strain', 'Active stress')
        self.var_active_model = tk.StringVar(self)
        self.optmenu_active_model = CustOptionMenu(self, self.var_active_model, 'Active strain', *active_model_optlist, coor = [1,1,1,1])
        self.active_model = 'active_stress'        
        
        self.label_mat_model = CustLabel(master = self, coor = [2,0,1,1], text = 'Material model')
        mat_optlist=('Holzapfel Ogden', 'Guccione', 'Neo Hookean')
        self.var_mat_model = tk.StringVar(self)
        self.optmenu_mat_model = CustOptionMenu(self, self.var_mat_model, 'Holzapfel Ogden', *mat_optlist, coor = [2,1,1,1])
        self.material_model = 'holzapfel_ogden'
        
        self.but_mat_opt = CustButton(master = self, coor = [3,0,1,2], text = 'Material parameters', command = self.create_mat_par)
        #====================Material Parameters==========================================
        #====================Holzapfel Ogden==============================================
        self.win_HO = MatParHO(master = self, place = [250,150,250,250])
        self.win_HO.mat_params = {'a' : float(self.win_HO.entry_a.get()),
                                  'a_f' : float(self.win_HO.entry_a_f.get()),
                                  'b' : float(self.win_HO.entry_b.get()),
                                  'b_f' : float(self.win_HO.entry_b_f.get())}
        #====================Guccione=====================================================
        self.win_G = MatParG(master = self, place = [250,150,250,250])
        self.win_G.mat_params = {'C' : float(self.win_G.entry_C.get()),
                                 'bf' : float(self.win_G.entry_b_f.get()),
                                 'bfs' : float(self.win_G.entry_b_fs.get()),
                                 'bt' : float(self.win_G.entry_b_t.get())}
        #====================Neo Hookean==================================================
        self.win_NH = MatParNH(master = self, place = [250,75,250,250])
        self.win_NH.mat_params = {'mu' : float(self.win_NH.entry_mu.get())}
        #==================================================================================        
        
        self.label_spring_surf = CustLabel(master = self, coor = [4,0,1,1], text = 'Spring connection surface')
        spring_optlist=('Base surface', 'Outside surface')
        self.var_spring_surf = tk.StringVar(self)
        self.optmenu_spring_surf = CustOptionMenu(self, self.var_spring_surf, 'Base surface', *spring_optlist, coor = [4,1,1,1])
        self.spring_area = 'BASE'
        
        self.label_spring_constant = CustLabel(master = self, coor = [5,0,1,1], text = 'Spring Constant')
        self.entry_spring_constant = CustEntry(master = self, coor = [5,1,1,1], def_val = 1.0)
        self.spring_constant = 1.0
        
        self.label_T_ref = CustLabel(master = self, coor = [6,0,1,1], text = 'T_ref')
        self.entry_T_ref = CustEntry(master = self, coor = [6,1,1,1], def_val = 0.1)
        self.T_ref = 0.1
        
        self.but_ok = CustButton(master = self, coor = [7,1,1,1], text = 'Ok', command = self.confirm_adv_par)
        self.but_cancel = CustButton(master = self, coor = [7,0,1,1], text = 'Cancel', command = self.closing_adv_par)
        
        self.advanced_parameters={'active_model' : self.active_model,
                                  'material_model' : self.material_model,
                                  'mat_params' : self.win_HO.mat_params,
                                  'spring_area' : self.spring_area,
                                  'spring_constant' : self.spring_constant,
                                  'T_ref' : self.T_ref}

        self.resize()
        self.withdraw()
        self.protocol("WM_DELETE_WINDOW", self.closing_adv_par)
    
    #Opening toplevel window for one of the material model parameters
    def create_mat_par(self):
        if self.var_mat_model.get() == 'Holzapfel Ogden':
            self.win_HO.deiconify()
            self.win_G.withdraw()
            self.win_NH.withdraw()
        elif self.var_mat_model.get() == 'Guccione':
            self.win_HO.withdraw()
            self.win_G.deiconify()
            self.win_NH.withdraw()
        elif self.var_mat_model.get() == 'Neo Hookean':
            self.win_HO.withdraw()
            self.win_G.withdraw()
            self.win_NH.deiconify()
        return
    
    #Confirming changes to advanced parameters 
    def confirm_adv_par(self):
        self.spring_constant = float(self.entry_spring_constant.get())
        self.T_ref = float(self.entry_T_ref.get())
        if self.var_mat_model.get() == 'Holzapfel Ogden':
            self.mat_params=self.win_HO.mat_params
        elif self.var_mat_model.get() == 'Guccione':
            self.mat_params=self.win_G.mat_params
        elif self.var_mat_model.get() == 'Neo Hookean':
            self.mat_params=self.win_NH.mat_params
            
        self.active_model = self.var_active_model.get()
        self.material_model = self.var_mat_model.get()
        self.spring_area = self.var_spring_surf.get()
        self.spring_constant = float(self.entry_spring_constant.get())
        self.T_ref = float(self.entry_T_ref.get())
        
        if self.active_model == 'Active stress':
            self.active_model = 'active_stress'
        elif self.active_model == 'Active strain':
            self.active_model = 'active_strain'

        if self.material_model == 'Holzapfel Ogden':
            self.material_model = 'holzapfel_ogden'
        elif self.material_model == 'Guccione':
            self.material_model = 'guccione'
        elif self.material_model == 'Neo Hookean':
            self.material_model = 'neo_hookean'
        
        if self.spring_area == 'Base surface':
            self.spring_area = 'BASE'
        elif self.spring_area == 'Outside surface':
            self.spring_area = 'EPI'
        
        self.advanced_parameters = {'active_model' : self.active_model,
                     'material_model' : self.material_model,
                     'mat_params' : self.mat_params,
                     'spring_area' : self.spring_area,
                     'spring_constant' : self.spring_constant,
                     'T_ref' : self.T_ref}
        
        self.win_HO.withdraw()
        self.win_G.withdraw()
        self.win_NH.withdraw()
        self.withdraw()
        
        return
        
    def closing_adv_par(self):
        self.win_HO.withdraw()
        self.win_G.withdraw()
        self.win_NH.withdraw()
        self.withdraw()
        return

#Class creating a frame with labels and entries for the four contraction regions of the heart. Used to create the left side of the bullseye plot.
class ContRegion(CustFrame):
    def __init__(self, title_str, region_list, *args, **kwargs):
        CustFrame.__init__(self, *args, **kwargs)
        self.title = CustTitle(master = self, coor = [0,0,1,2], text = title_str)
        self.label_list = []
        self.entry_list = []
        
        for i in range(0,len(region_list)):
            self.label_list.append(CustLabel(master = self, coor = [i+1,0,1,1], text = '{0}'.format(region_list[i])))
            self.entry_list.append(CustEntry(master = self, coor = [i+1,1,1,1], def_val = 1.0))
            self.entry_list[i].number = region_list[i]
        self.resize()

#Creating the bullseye plot that devides the left verticle into 17 sections.        
class Bullseye(CustToplevel):
    def __init__(self, *args, **kwargs):
        CustToplevel.__init__(self, *args, **kwargs)
        self.frame_regions = CustFrame(master = self, coor = [0,0,23,2], borderwidth = 5)
        self.main_title = CustTitle(master = self.frame_regions, coor = [0,0,1,2], text = 'Regional contraction')
        self.frame_base = ContRegion(title_str = 'Base region', region_list = ['1', '2', '3', '4', '5', '6'], master = self.frame_regions, coor = [1,0,7,2], borderwidth = 5)
        self.frame_base.resize()        
        self.frame_mid = ContRegion(title_str = 'Mid region', region_list = ['7', '8', '9', '10', '11', '12'], master = self.frame_regions, coor = [8,0,7,2], borderwidth = 5)
        self.frame_mid.resize()
        self.frame_apical = ContRegion(title_str = 'Apical region', region_list = ['13', '14', '15', '16'], master = self.frame_regions, coor = [15,0,5,2], borderwidth = 5)
        self.frame_apical.resize()        
        self.frame_apex = ContRegion(title_str = 'Apex region', region_list = ['17'], master = self.frame_regions, coor = [20,0,2,2], borderwidth = 5)
        self.frame_apex.resize()        

        self.but_ok = CustButton(master = self.frame_regions, coor = [23,1,1,1], text = 'Ok', command = self.confirm_reg_values)
        self.but_cancel = CustButton(master = self.frame_regions, coor = [23,0,1,1], text = 'Cancel', command = self.withdraw)
        self.frame_regions.resize()        
        
        self.frame_bullseye = CustFrame(master = self, coor = [0,2,23,15], borderwidth = 5)
        self.fig = Figure()
        self.subplot = self.fig.add_subplot(111, polar = True)
        
        self.data = np.array(range(17)) + 1
        bullseye_plot(self.subplot, self.data)
        
        self.canvas = CustFigCanvas(self.fig, master = self.frame_bullseye, coor =[0,0,1,1])
        
        self.base_value = []
        self.mid_value = []
        self.apical_value = []
        self.apex_value = []
        #Setting the events to trigger the highlighting of the different sections in the bullseye plot.
        for i in range(0,6):
            self.frame_base.entry_list[i].bind('<FocusIn>', self.mark_region)
            self.frame_base.entry_list[i].bind('<FocusOut>', self.unmark_region)
            self.base_value.append(float(self.frame_base.entry_list[i].get()))
        for i in range(0,6):
            self.frame_mid.entry_list[i].bind('<FocusIn>', self.mark_region)
            self.frame_mid.entry_list[i].bind('<FocusOut>', self.unmark_region)
            self.mid_value.append(float(self.frame_mid.entry_list[i].get()))
        for i in range(0,4):
            self.frame_apical.entry_list[i].bind('<FocusIn>', self.mark_region)
            self.frame_apical.entry_list[i].bind('<FocusOut>', self.unmark_region)
            self.apical_value.append(float(self.frame_apical.entry_list[i].get()))
        for i in range(0,1):
            self.frame_apex.entry_list[i].bind('<FocusIn>', self.mark_region)
            self.frame_apex.entry_list[i].bind('<FocusOut>', self.unmark_region)
            self.apex_value.append(float(self.frame_apex.entry_list[i].get()))
        
        self.gamma = {'gamma_base' : self.base_value,
                      'gamma_mid' : self.mid_value,
                      'gamma_apical' : self.apical_value,
                      'gamma_apex' : self.apex_value}
        
        self.frame_bullseye.resize()
        self.resize()
        self.withdraw()
        self.protocol("WM_DELETE_WINDOW", self.withdraw)
        

    def mark_region(self, event):
        #self.subplot.clear()
        bullseye_plot(self.subplot, self.data, segBold = [int(event.widget.number)])
        self.canvas.draw()
        return
    def unmark_region(self, event):
        self.subplot.clear()
        bullseye_plot(self.subplot, self.data)
        self.canvas.draw()
        return
        
    def confirm_reg_values(self):
        for i in range(0,6):
            self.base_value[i] = float(self.frame_base.entry_list[i].get())
            self.mid_value[i] = float(self.frame_mid.entry_list[i].get())
        for i in range(0,4):
            self.apical_value[i] = float(self.frame_apical.entry_list[i].get())
        for i in range(0,1):
            self.apex_value[i] = float(self.frame_apex.entry_list[i].get())
        
        self.gamma = {'gamma_base' : self.base_value,
                      'gamma_mid' : self.mid_value,
                      'gamma_apical' : self.apical_value,
                      'gamma_apex' : self.apex_value}
        self.withdraw()
        return

#Creating a toplevel window for the parameters specific to the closed loop solver.
class ParamsCL(CustToplevel):
    def __init__(self, *args, **kwargs):
        CustToplevel.__init__(self, *args, **kwargs)
        self.title_main = CustTitle(master = self, coor = [0,0,1,2], text = 'Closed loop parameters')
        
        self.frame_cl = CustFrame(master = self, coor = [1,0,5,2], borderwidth = 5)
        self.label_BCL = CustLabel(master = self.frame_cl, coor = [2,0,1,1], text = 'BCL (cycle length - t<BCL)')
        self.entry_BCL = CustEntry(master = self.frame_cl, coor = [2,1,1,1], def_val = 200.0)
        self.label_t = CustLabel(master = self.frame_cl, coor = [3,0,1,1], text = 't')
        self.entry_t = CustEntry(master = self.frame_cl, coor = [3,1,1,1], def_val = 0.0)
        self.label_dt = CustLabel(master = self.frame_cl, coor = [4,0,1,1], text = 'dt')
        self.entry_dt = CustEntry(master = self.frame_cl, coor = [4,1,1,1], def_val = 3.0)
        self.label_ED_vol = CustLabel(master = self.frame_cl, coor = [5,0,1,1], text = 'End-Diastolic volume')
        self.entry_ED_vol = CustEntry(master = self.frame_cl, coor = [5,1,1,1], def_val = 70.0)
        self.frame_cl.resize()
        
        self.frame_wk = CustFrame(master = self, coor = [6,0,11,2], borderwidth = 5)
        self.title_windkessel = CustTitle(master = self.frame_wk, coor = [0,0,1,2], text = 'Windkessel parameters')
        self.label_Cao = CustLabel(master = self.frame_wk, coor = [1,0,1,1], text = 'Aorta compliance (reduce)')
        self.entry_Cao = CustEntry(master = self.frame_wk, coor = [1,1,1,1], def_val = 10.0/1000.0)
        self.label_Cven = CustLabel(master = self.frame_wk, coor = [2,0,1,1], text = 'Venous compliance')
        self.entry_Cven = CustEntry(master = self.frame_wk, coor = [2,1,1,1], def_val = 400.0/1000.0)
        self.label_Vart0 = CustLabel(master = self.frame_wk, coor = [3,0,1,1], text = 'Dead volume (Vrat0')
        self.entry_Vart0 = CustEntry(master = self.frame_wk, coor = [3,1,1,1], def_val = 510.0)
        self.label_Vven0 = CustLabel(master = self.frame_wk, coor = [4,0,1,1], text = 'Dead volume (Vven0)')
        self.entry_Vven0 = CustEntry(master = self.frame_wk, coor = [4,1,1,1], def_val = 2800.0)
        self.label_Rao = CustLabel(master = self.frame_wk, coor = [5,0,1,1], text = 'Aortic resistance (Rao)')
        self.entry_Rao = CustEntry(master = self.frame_wk, coor = [5,1,1,1], def_val = 10.0*1000.0)
        self.label_Rven = CustLabel(master = self.frame_wk, coor = [6,0,1,1], text = 'Aortic resistance (Rven)')
        self.entry_Rven = CustEntry(master = self.frame_wk, coor = [6,1,1,1], def_val = 2.0*1000.0)
        self.label_Rper = CustLabel(master = self.frame_wk, coor = [7,0,1,1], text = 'Peripheral resistance (Rper)')
        self.entry_Rper = CustEntry(master = self.frame_wk, coor = [7,1,1,1], def_val = 10.0*1000.0)
        self.label_V_ven = CustLabel(master = self.frame_wk, coor = [8,0,1,1], text = 'Peripheral resistance (V_ven)')
        self.entry_V_ven = CustEntry(master = self.frame_wk, coor = [8,1,1,1], def_val = 3600.0)
        self.label_V_art = CustLabel(master = self.frame_wk, coor = [9,0,1,1], text = 'Peripheral resistance (V_art)')
        self.entry_V_art = CustEntry(master = self.frame_wk, coor = [9,1,1,1], def_val = 640.0)
        self.label_mesh_scaler = CustLabel(master = self.frame_wk, coor = [10,0,1,1], text = 'Scale factor for mesh')
        self.entry_mesh_scaler = CustEntry(master = self.frame_wk, coor = [10,1,1,1], def_val = 2.6)
        self.frame_wk.resize()
        
        self.frame_but = CustFrame(master = self, coor = [17,0,1,2], borderwidth = 5)
        self.but_ok = CustButton(master = self.frame_but, coor = [0,0,1,1], text = 'Ok', command = self.confirm_CL_params)
        self.but_cancel = CustButton(master = self.frame_but, coor = [0,1,1,1], text = 'Cancel', command = self.withdraw)
        self.frame_but.resize()

        self.resize()
        self.withdraw()
        self.protocol("WM_DELETE_WINDOW", self.withdraw)
        
        self.CL_params = {'BCL' : float(self.entry_BCL.get()),
                          't' : float(self.entry_t.get()),
                          'dt' : float(self.entry_dt.get()),
                          'ED_vol' : float(self.entry_ED_vol.get()),
                          'Cao' : float(self.entry_Cao.get()),
                          'Cven' : float(self.entry_Cven.get()),
                          'Vart0' : float(self.entry_Vart0.get()),
                          'Vven0' : float(self.entry_Vven0.get()),
                          'Rao' : float(self.entry_Rao.get()),
                          'Rven' : float(self.entry_Rven.get()),
                          'Rper' : float(self.entry_Rper.get()),
                          'V_ven' : float(self.entry_V_ven.get()),
                          'V_art' : float(self.entry_V_art.get()),
                          'mesh_scaler' : float(self.entry_mesh_scaler.get())}
        
    def confirm_CL_params(self):
        self.CL_params = {'BCL' : float(self.entry_BCL.get()),
                          't' : float(self.entry_t.get()),
                          'dt' : float(self.entry_dt.get()),
                          'ED_vol' : float(self.entry_ED_vol.get()),
                          'Cao' : float(self.entry_Cao.get()),
                          'Cven' : float(self.entry_Cven.get()),
                          'Vart0' : float(self.entry_Vart0.get()),
                          'Vven0' : float(self.entry_Vven0.get()),
                          'Rao' : float(self.entry_Rao.get()),
                          'Rven' : float(self.entry_Rven.get()),
                          'Rper' : float(self.entry_Rper.get()),
                          'V_ven' : float(self.entry_V_ven.get()),
                          'V_art' : float(self.entry_V_art.get()),
                          'mesh_scaler' : float(self.entry_mesh_scaler.get())}
        self.withdraw()
        return
        

#Creating a parameter frame that will be put into the main frame. Here all the toplevel windows for other types of parameters are put. There are also some general parameters put in the frame.       
class ParameterFrame(CustFrame):
    def __init__(self, *args, **kwargs):
        CustFrame.__init__(self, *args, **kwargs)
        self.title = CustTitle(master = self, coor = [0,0,1,2], text='Parameters')
        self.label_lv_pressure = CustLabel(master = self, coor = [1,0,1,1], text='Left ventricular pressure')
        self.entry_lv_pressure = CustEntry(master = self,  coor = [1,1,1,1], def_val = 5.0)
        self.label_rv_pressure = CustLabel(master = self, coor = [2,0,1,1], text='Right ventricular pressure')
        self.entry_rv_pressure = CustEntry(master = self, coor = [2,1,1,1], def_val = 3.0)
        self.label_cont_mult = CustLabel(master = self, coor = [3,0,1,1], text='Contraction multiplier')
        self.entry_cont_mult = CustEntry(master = self, coor = [3,1,1,1], def_val = 25.0)
        self.label_BC = CustLabel(master = self, coor = [4,0,1,1], text = 'Boundary condition')
        BC_optlist=('Fixed in one direction', 'Fixed in three directions')
        self.var_BC = tk.StringVar(self)
        self.optmenu_BC = CustOptionMenu(self, self.var_BC, 'Fixed in one direction', *BC_optlist, coor = [4,1,1,1])

        self.but_adv_param = CustButton(master = self, coor = [6,0,1,2], text = 'Advanced parameters', command = self.create_adv_par)
        self.but_bullseye = CustButton(master = self, coor = [5,1,1,1], text='Bullseye parameters', command = self.create_bullseye_win)        
        self.but_CL_params = CustButton(master = self, coor = [5,0,1,1], text = 'Closed loop parameters', command = self.create_CL_win)
        self.but_CL_params.state(['disabled'])
        
        self.CL_win = ParamsCL(master = self)
        self.bullseye_win = Bullseye(master = self)
        self.adv_par_win = AdvancedParameters(master = self)
                
        self.resize()    

    def create_CL_win(self):
        self.CL_win.deiconify()
        return

    def create_bullseye_win(self):
        self.BC_type = self.var_BC.get()
        if self.BC_type == 'Fixed in three directions':
            for i in range(0,6):
                self.bullseye_win.base_value[i] = 0.0
                self.bullseye_win.frame_base.entry_list[i].delete(0, tk.END)
                self.bullseye_win.frame_base.entry_list[i].insert(0, 0.0)
            self.bullseye_win.deiconify()
        elif self.BC_type == 'Fixed in one direction':
            for i in range(0,6):
                self.bullseye_win.base_value[i] = 1.0
                self.bullseye_win.frame_base.entry_list[i].delete(0, tk.END)
                self.bullseye_win.frame_base.entry_list[i].insert(0, 1.0)
            self.bullseye_win.deiconify()
        return
    def create_adv_par(self):
        self.adv_par_win.deiconify()
        return

#A class that makes a plotting frame for the vtk implementations in the GUI.
class PlotFrame(CustFrame):
    def __init__(self, mesh, plot_type, *args, **kwargs):
        CustFrame.__init__(self, *args, **kwargs)
        self.plot_type = plot_type
        self.mesh = mesh
        if self.plot_type == 'mesh':
            self.vtkmesh = demo_mesh(self.mesh)
        elif self.plot_type == 'cell':
            self.vtkmesh = demo_cellfunction(self.mesh)
        elif self.plot_type == 'mark':
            self.vtkmesh = demo_facetfunction(self.mesh)
        
        self.rendered = CustRWI(master = self, coor = [0,0,1,1], rw=self.vtkmesh.renWin)
        self.rendered.resize()
        self.resize()
        
#Creating the notebook for plotting when the Closed loop solver is chosen.
class CreateCLNotebook(CustFrame):
    def __init__(self, *args, **kwargs):
        CustFrame.__init__(self, *args, **kwargs)
        self.notebook_plot = CustNotebook(master = self, coor = [4,0,1,2])
        self.notebook_plot.resize()
        
        self.frame_fill_pre = CustFrame(master = self.notebook_plot, coor = [0,0,1,1])
        self.notebook_pre = CustNotebook(master = self.frame_fill_pre, coor = [0,0,1,1])
        self.frame_fill_pre.resize() 
        self.notebook_pre.resize()
        self.tab_pre = self.notebook_plot.add(self.frame_fill_pre, text='Pre processing')
        
        #Creating a frame, figure and subplot for the closed loop results.
        self.frame_closed_loop = CustFrame(master = self.notebook_plot, coor = [0,0,1,1])
        self.fig = plt.Figure(figsize = (5.25, 5.5))
        self.subplot = self.fig.add_subplot(111)
        self.subplot.set_ylabel("Pressure (kPa)")
        self.subplot.set_xlabel("Volume (ml)")
        self.subplot.set_xlim([0,75])
        self.subplot.set_ylim([0,15])


        self.canvas = CustFigCanvas(self.fig, master = self.frame_closed_loop, coor =[0,0,1,1])
        self.tab_closed_loop = self.notebook_plot.add(self.frame_closed_loop, text = 'Closed loop')
        
        #setting a default mesh to be used for the plotting when the GUI is opened.
        self.patient = load_geometry_from_h5('../pulse_adjoint/example_meshes/simple_ellipsoid.h5')
        self.current_mesh = self.patient.mesh
        self.mesh_name = 'simple_ellipsoid.h5'
        
        #Adding tabs with plots to the plotting notebook.
        self.tab_frame_list = []
        self.tab_list = []
        self.tab_name_list = ['Mesh', 'Mark', 'Cell', 'Initial', 'Pressure', 'Contraction', 'Initial', 'Pressure', 'Contraction']
        self.plot_type_list = ['mesh', 'mark', 'cell']
        for i in range(0,3):
            self.tab_frame_list.append(PlotFrame(mesh = self.current_mesh, plot_type = self.plot_type_list[i], master = self.notebook_pre, coor = [0,0,1,1], borderwidth = 5))
            tab = self.notebook_pre.add(self.tab_frame_list[i], text=self.tab_name_list[i])
            self.tab_list.append(tab)
            self.tab_frame_list[i].resize()
            
        self.frame_closed_loop.resize()
        self.resize()
    #Function to live update the closed loop plot while the solver is running.
    def animate(self, i):
        self.fig.clf()
        #self.subplot.cla()
        self.subplot = self.fig.add_subplot(111)
        self.subplot.set_ylabel("Pressure (kPa)")
        self.subplot.set_xlabel("Volume (ml)")
        self.subplot.set_xlim([0,75])
        self.subplot.set_ylim([0,15])
        self.lines = []
        self.dots = []
        #The function pulls data from the '.txt' file 'pv_data_plot.txt' and plots it.
        self.pullData = open('pv_data_plot.txt',"r").read()
        self.dataArray = self.pullData.split('\n')
        self.xar = [0,0]
        self.yar = [0,0]
        for eachLine in self.dataArray:
            if len(eachLine)>1:
                self.x, self.y = eachLine.split(',')
                self.xar.append(float(self.x))
                self.yar.append(float(self.y))
        
        self.subplot.plot(self.yar, self.xar, '-', linewidth = 5)
        self.dots.append(self.subplot.plot(self.yar[-1], self.xar[-1], 'ro', markersize = 10))

        return

#Creating the notebook for plotting when the regular solver is chosen.
class CreateRegNotebook(CustFrame):
    def __init__(self, *args, **kwargs):
        CustFrame.__init__(self, *args, **kwargs)
        self.notebook_plot = CustNotebook(master = self, coor = [4,0,1,2])
        self.notebook_plot.resize()
        
        self.frame_fill_pre = CustFrame(master = self.notebook_plot, coor = [0,0,1,1])
        self.notebook_pre = CustNotebook(master = self.frame_fill_pre, coor = [0,0,1,1])
        self.frame_fill_pre.resize() 
        self.notebook_pre.resize()
        self.tab_pre = self.notebook_plot.add(self.frame_fill_pre, text='Pre processing')
        
        self.frame_fill_post = CustFrame(master = self.notebook_plot, coor = [0,0,1,1])
        self.notebook_post = CustNotebook(master = self.frame_fill_post, coor = [0,0,1,1])
        self.frame_fill_post.resize()
        self.notebook_post.resize()
        self.tab_post=self.notebook_plot.add(self.frame_fill_post, text='Post processing')
        
        self.frame_fill_disp = CustFrame(master = self.notebook_post, coor = [0,0,1,1])
        self.notebook_disp = CustNotebook(master = self.frame_fill_disp, coor = [0,0,1,1])
        self.frame_fill_disp.resize()
        self.notebook_disp.resize()
        self.tab_disp = self.notebook_post.add(self.frame_fill_disp, text = 'Displacement')
        
        self.frame_fill_stress = CustFrame(master = self.notebook_post, coor = [0,0,1,1])
        self.notebook_stress = CustNotebook(master = self.frame_fill_stress, coor = [0,0,1,1])
        self.frame_fill_stress.resize()
        self.notebook_stress.resize()
        self.tab_stress = self.notebook_post.add(self.frame_fill_stress, text = 'Stress')
        
        self.patient = load_geometry_from_h5('../pulse_adjoint/example_meshes/simple_ellipsoid.h5')
        self.current_mesh = self.patient.mesh
        self.mesh_name = 'simple_ellipsoid.h5'

        self.tab_frame_list = []
        self.tab_list = []
        self.tab_name_list = ['Mesh', 'Mark', 'Cell', 'Initial', 'Pressure', 'Contraction', 'Initial', 'Pressure', 'Contraction']  
        self.plot_type_list = ['mesh', 'mark', 'cell']
        for i in range(0,3):
            self.tab_frame_list.append(PlotFrame(mesh = self.current_mesh, plot_type = self.plot_type_list[i], master = self.notebook_pre, coor = [0,0,1,1], borderwidth = 5))
            tab = self.notebook_pre.add(self.tab_frame_list[i], text=self.tab_name_list[i])
            self.tab_list.append(tab)
            self.tab_frame_list[i].resize()
            
        for i in range(0,3):
            self.tab_frame_list.append(PlotFrame(mesh = self.current_mesh, plot_type = 'mesh', master = self.notebook_disp, coor = [0,0,1,1], borderwidth = 5))
            tab = self.notebook_disp.add(self.tab_frame_list[i+3], text=self.tab_name_list[i+3])
            self.tab_list.append(tab)
            self.tab_frame_list[i+3].resize()
            
        for i in range(0,3):
            self.tab_frame_list.append(PlotFrame(mesh = self.current_mesh, plot_type = 'mesh', master = self.notebook_stress, coor = [0,0,1,1], borderwidth = 5))
            tab = self.notebook_stress.add(self.tab_frame_list[i+6], text=self.tab_name_list[i+6])
            self.tab_list.append(tab)
            self.tab_frame_list[i+6].resize()
        
        self.resize()


#Creating a mesh frame where all plotting and handling of mesh is done. This frame is later being put in the main frame.
class MeshFrame(CustFrame):
    def __init__(self, *args, **kwargs):
        CustFrame.__init__(self, *args, **kwargs)
        self.patient = load_geometry_from_h5('../pulse_adjoint/example_meshes/simple_ellipsoid.h5')
        self.current_mesh = self.patient.mesh
        self.mesh_name = 'simple_ellipsoid.h5'
        
        self.title = CustTitle(master = self, coor = [0,0,1,2], text = 'Mesh')
        self.but_load_mesh = CustButton(master = self, coor = [1,0,1,2], text = 'Load mesh', command = self.load_mesh_file)
        self.but_plot_mesh = CustButton(master = self, coor = [2,0,1,2], text = 'Plot mesh', command = self.plot_mesh_file)
        
        self.plot_notebook = CreateRegNotebook(master = self, coor = [4,0,1,2], borderwidth = 5)
        self.plot_notebook.resize()        
        self.resize()

    #Function to load mesh from file.
    def load_mesh_file(self):
        mesh_file = tkFileDialog.askopenfilename()
        self.mesh_name = mesh_file
        
        if self.mesh_name.endswith('.xdmf'):
            f= df.XDMFFile(df.mpi_comm_world(), mesh_file)
            mesh = df.Mesh()
            f.read(mesh)
            self.current_mesh = mesh
            
        elif self.mesh_name.endswith('.h5'):
            self.mesh_name = ntpath.basename(self.mesh_name)
            self.patient = load_geometry_from_h5(self.mesh_name)
            self.current_mesh = self.patient.mesh
            
        elif self.mesh_name.endswith('.xml'):
            self.mesh_name = ntpath.basename(self.mesh_name)
            self.current_mesh = df.Mesh(self.mesh_name)
            
        return
    #Function to plot the loaded mesh in the pre processing tabs of the notebook.    
    def plot_mesh_file(self):
        for i in range(0,3):
            if i == 0:
                self.plot_notebook.tab_frame_list[i].vtkmesh = demo_mesh(self.current_mesh)
            elif i == 1:
                self.plot_notebook.tab_frame_list[i].vtkmesh = demo_cellfunction(self.current_mesh)
            elif i == 2:
                self.plot_notebook.tab_frame_list[i].vtkmesh = demo_facetfunction(self.current_mesh)
                
            self.plot_notebook.tab_frame_list[i].rendered = CustRWI(master = self.plot_notebook.tab_frame_list[i], coor = [0,0,1,1], rw = self.plot_notebook.tab_frame_list[i].vtkmesh.renWin)
            self.plot_notebook.tab_frame_list[i].rendered.Render()
        return


#Creates the main frame which is put in the root main loop of the GUI.
class MainFrame(CustFrame):
    def __init__(self, *args, **kwargs):
        CustFrame.__init__(self, *args, **kwargs)
        if os.path.isfile('pv_data_plot.txt'):
            open('pv_data_plot.txt', 'w').close()
        self.notebook_type = 'reg'
        #====================Mesh==========================================================
        self.frame_mesh = MeshFrame(master = self, coor = [2,0,8,2], borderwidth = 5)
        self.frame_mesh.resize()
        #====================Parameters====================================================
        self.frame_par = ParameterFrame(master = self, coor = [0,2,3,2], borderwidth = 5)
        self.frame_par.resize()
        #====================Run===========================================================
        #self.frame_run = RunFrame(master = self, coor = [3,2,2,2], borderwidth = 5)
        #self.frame_run.resize()
        self.title_run = CustTitle(master = self, coor = [3,2,1,2], text='Run')
        #====================Text Output===================================================
        self.text_output = OutputFrame(master = self, coor = [4,2,3,2], borderwidth = 5)
        self.text_output.resize()
        #====================Solver Buttons=================================================
        self.frame_but = CustFrame(master = self, coor = [7,2,3,2], borderwidth = 5)
        self.label_save = CustLabel(master = self.frame_but, coor = [0,0,1,1], text='Choose folder to save simulation results')
        self.folder_name = tk.StringVar()
        self.entry_save = CustEntry(master = self.frame_but, coor = [1,0,1,1], textvariable = self.folder_name)
        self.folder_name.set('Enter folder path here')
        self.but_save = CustButton(master = self.frame_but, coor = [2,0,1,1], text='Choose folder', command = self.choose_folder)
        self.but_run = CustButton(master = self.frame_but, coor = [1,1,1,1], text='Run', command = self.run_simulation)
        self.but_plot_res = CustButton(master = self.frame_but, coor = [1,2,1,1], text='Plot results', command = self.plot_res)
        self.but_plot_res.state(['disabled'])
        self.but_run_closed_loop = CustButton(master = self.frame_but, coor = [2,1,1,2], text='Run closed loop', command = self.run_simulation)
        self.but_run_closed_loop.state(['disabled'])
        self.frame_but.resize()
        #====================Solver Choice=================================================
        self.frame_solver = CustFrame(master = self, coor = [0,0,2,2], borderwidth = 5)
        self.title_solver = CustTitle(master = self.frame_solver, coor = [0,0,1,2], text = 'Choose solver type')
        self.but_reg_solver = CustButton(master = self.frame_solver, coor = [1,0,1,1], text = 'Regular solver', command = self.switch_reg)
        self.but_reg_solver.state(['disabled'])
        self.but_closed_loop_solver = CustButton(master = self.frame_solver, coor = [1,1,1,1], text = 'Closed loop solver', command = self.switch_closed)
        self.frame_solver.resize()  
        
        self.resize()
    
    #Function to switch to regular solver from closed loop solver.
    def switch_reg(self):
        self.notebook_type = 'reg'
        #Deleting the plots contained in the notebook for plotting.
        for i in range(0,len(main_frame.frame_mesh.plot_notebook.tab_frame_list)):
            del main_frame.frame_mesh.plot_notebook.tab_frame_list[i].vtkmesh.renWin
        #Changing the notebook for plotting
        self.frame_mesh.plot_notebook = CreateRegNotebook(master = self.frame_mesh, coor = [4,0,1,2], borderwidth = 5)
        self.frame_mesh.plot_notebook.resize()
        #Enabling and diabling buttons related to the regular solver.
        self.but_reg_solver.state(['disabled'])
        self.but_closed_loop_solver.state(['!disabled'])
        self.but_run_closed_loop.state(['disabled'])
        self.but_run.state(['!disabled'])
        self.frame_par.but_CL_params.state(['disabled'])
        
        #Setting all parameters back to default values.
        self.frame_par.adv_par_win.win_HO.entry_a.delete(0, tk.END)
        self.frame_par.adv_par_win.win_HO.entry_a.insert(0,2.28)
        self.frame_par.adv_par_win.win_HO.entry_a_f.delete(0, tk.END)
        self.frame_par.adv_par_win.win_HO.entry_a_f.insert(0,1.685)
        self.frame_par.adv_par_win.win_HO.entry_b.delete(0, tk.END)
        self.frame_par.adv_par_win.win_HO.entry_b.insert(0,9.726)
        self.frame_par.adv_par_win.win_HO.entry_b_f.delete(0, tk.END)
        self.frame_par.adv_par_win.win_HO.entry_b_f.insert(0,15.779)
        self.frame_par.adv_par_win.win_HO.confirm_HO()       
        
        self.frame_par.adv_par_win.win_G.entry_C.delete(0, tk.END)
        self.frame_par.adv_par_win.win_G.entry_C.insert(0,2.0)
        self.frame_par.adv_par_win.win_G.entry_b_f.delete(0, tk.END)
        self.frame_par.adv_par_win.win_G.entry_b_f.insert(0,8.0)
        self.frame_par.adv_par_win.win_G.entry_b_t.delete(0, tk.END)
        self.frame_par.adv_par_win.win_G.entry_b_t.insert(0,2.0)
        self.frame_par.adv_par_win.win_G.entry_b_fs.delete(0, tk.END)
        self.frame_par.adv_par_win.win_G.entry_b_fs.insert(0,4.0)
        self.frame_par.adv_par_win.win_G.confirm_G()
        
        self.frame_par.adv_par_win.win_NH.entry_mu.delete(0, tk.END)
        self.frame_par.adv_par_win.win_NH.entry_mu.insert(0,15.0)
        self.frame_par.adv_par_win.win_NH.confirm_NH()
        
        self.frame_par.adv_par_win.var_active_model.set('Active stress')
        self.frame_par.adv_par_win.var_mat_model.set('Holzapfel Ogden')
        self.frame_par.adv_par_win.var_spring_surf.set('Base surface')
        self.frame_par.adv_par_win.entry_spring_constant.delete(0, tk.END)
        self.frame_par.adv_par_win.entry_spring_constant.insert(0,1.0)
        self.frame_par.adv_par_win.entry_T_ref.delete(0, tk.END)
        self.frame_par.adv_par_win.entry_T_ref.insert(0,0.1)
        self.frame_par.adv_par_win.confirm_adv_par()
        
        for i in range(0,6):
            self.frame_par.bullseye_win.frame_base.entry_list[i].delete(0, tk.END)
            self.frame_par.bullseye_win.frame_base.entry_list[i].insert(0,1.0)
            self.frame_par.bullseye_win.frame_mid.entry_list[i].delete(0, tk.END)
            self.frame_par.bullseye_win.frame_mid.entry_list[i].insert(0,1.0)
        for i in range(0,4):
            self.frame_par.bullseye_win.frame_apical.entry_list[i].delete(0, tk.END)
            self.frame_par.bullseye_win.frame_apical.entry_list[i].insert(0,1.0)
        for i in range(0,1):
            self.frame_par.bullseye_win.frame_apex.entry_list[i].delete(0, tk.END)
            self.frame_par.bullseye_win.frame_apex.entry_list[i].insert(0,1.0)
        self.frame_par.bullseye_win.confirm_reg_values()
        
        self.frame_par.entry_lv_pressure.delete(0, tk.END)
        self.frame_par.entry_lv_pressure.insert(0,5.0)
        self.frame_par.entry_rv_pressure.delete(0, tk.END)
        self.frame_par.entry_rv_pressure.insert(0,3.0)
        self.frame_par.entry_cont_mult.delete(0, tk.END)
        self.frame_par.entry_cont_mult.insert(0,25.0)
        self.frame_par.var_BC.set('Fixed in one direction')
        
        return
    
    def switch_closed(self):
        self.notebook_type = 'CL'
        #Deleting the plots contained in the notebook for plotting.
        for i in range(0,len(main_frame.frame_mesh.plot_notebook.tab_frame_list)):
            del main_frame.frame_mesh.plot_notebook.tab_frame_list[i].vtkmesh.renWin
        #Changing the notebok for plotting.
        self.frame_mesh.plot_notebook = CreateCLNotebook(master = self.frame_mesh, coor = [4,0,1,2], borderwidth = 5)
        self.frame_mesh.plot_notebook.resize()
        #Enabling and disabling buttons related to the closed loop solver.
        self.but_closed_loop_solver.state(['disabled'])
        self.but_reg_solver.state(['!disabled'])
        self.but_run.state(['disabled'])
        self.but_run_closed_loop.state(['!disabled'])
        self.but_plot_res.state(['disabled'])
        self.frame_par.but_CL_params.state(['!disabled'])
        
        #Setting default values for the closed loop solver
        self.frame_par.adv_par_win.var_mat_model.set('Guccione')
        self.frame_par.adv_par_win.win_G.entry_C.delete(0, tk.END)
        self.frame_par.adv_par_win.win_G.entry_C.insert(0,200.0)
        self.frame_par.adv_par_win.win_G.entry_b_f.delete(0, tk.END)
        self.frame_par.adv_par_win.win_G.entry_b_f.insert(0,90.0)
        self.frame_par.adv_par_win.win_G.entry_b_t.delete(0, tk.END)
        self.frame_par.adv_par_win.win_G.entry_b_t.insert(0,10.0)
        self.frame_par.adv_par_win.win_G.entry_b_fs.delete(0, tk.END)
        self.frame_par.adv_par_win.win_G.entry_b_fs.insert(0,40.0)
        self.frame_par.adv_par_win.win_G.confirm_G()
        
        self.frame_par.adv_par_win.var_active_model.set('Active stress')
        self.frame_par.adv_par_win.entry_T_ref.delete(0, tk.END)
        self.frame_par.adv_par_win.entry_T_ref.insert(0,70e3)
        self.frame_par.adv_par_win.confirm_adv_par()
        
        self.frame_par.adv_par_win.var_spring_surf.set('Base surface')
        self.frame_par.adv_par_win.entry_spring_constant.delete(0, tk.END)
        self.frame_par.adv_par_win.entry_spring_constant.insert(0,1.0)
        self.frame_par.adv_par_win.confirm_adv_par()
        
        gamma_base = [1,1,1,1,0.5,0.6]
        gamma_mid =  [1,1,1,1,0.5,0.6]
        gamma_apical =  [1,1,0.55,0.6]
        gamma_apex = [0.6]
        for i in range(0,6):
            self.frame_par.bullseye_win.frame_base.entry_list[i].delete(0, tk.END)
            self.frame_par.bullseye_win.frame_base.entry_list[i].insert(0,gamma_base[i])
            self.frame_par.bullseye_win.frame_mid.entry_list[i].delete(0, tk.END)
            self.frame_par.bullseye_win.frame_mid.entry_list[i].insert(0,gamma_mid[i])
        for i in range(0,4):
            self.frame_par.bullseye_win.frame_apical.entry_list[i].delete(0, tk.END)
            self.frame_par.bullseye_win.frame_apical.entry_list[i].insert(0,gamma_apical[i])
        for i in range(0,1):
            self.frame_par.bullseye_win.frame_apex.entry_list[i].delete(0, tk.END)
            self.frame_par.bullseye_win.frame_apex.entry_list[i].insert(0,gamma_apex[i])
        self.frame_par.bullseye_win.confirm_reg_values()
        
        self.frame_par.entry_lv_pressure.delete(0, tk.END)
        self.frame_par.entry_lv_pressure.insert(0,5.0)
        self.frame_par.entry_rv_pressure.delete(0, tk.END)
        self.frame_par.entry_rv_pressure.insert(0,3.0)
        self.frame_par.entry_cont_mult.delete(0, tk.END)
        self.frame_par.entry_cont_mult.insert(0,25.0)
        self.frame_par.var_BC.set('Fixed in one direction')
        
        return
   
    #Function to choose a path to save the results from the simulation.   
    def choose_folder(self):
        self.folder_path = tkFileDialog.askdirectory()
        if self.folder_path:
            self.folder_name.set(self.folder_path)
        return
    
    #Runs the simulation. Either closed loop or regular.    
    def run_simulation(self):
        #Getting the patient bject and mesh from the loaded file.
        self.patient = load_geometry_from_h5(self.frame_mesh.mesh_name)
        self.current_mesh = self.patient.mesh
        
        #Getting parameters from the parameter frame
        self.lv_pressure = float(self.frame_par.entry_lv_pressure.get())
        self.rv_pressure = float(self.frame_par.entry_rv_pressure.get())
        self.cont_mult = float(self.frame_par.entry_cont_mult.get())
        

        self.folder_path = self.entry_save.get()
        if self.folder_path == 'Enter folder path here':
            self.folder_path = os.getcwd()
        self.mesh_name = self.frame_mesh.mesh_name
        self.gamma = self.frame_par.bullseye_win.gamma
        self.BC_type = self.frame_par.var_BC.get()
        if self.BC_type == 'Fixed in one direction':
            self.BC_type = 'fix_x'
        elif BC_type == 'Fixed in three directions':
            self.BC_type = 'fixed'
            
        parameters={'patient' : self.patient,
                    'mesh' : self.current_mesh,
                    'folder_path' : self.folder_path,
                    'mesh_name' : self.mesh_name,
                    'lv_pressure' : self.lv_pressure,
                    'rv_pressure' : self.rv_pressure,
                    'contraction_multiplier' : self.cont_mult,
                    'BC_type' : self.BC_type,
                    'gamma' : self.gamma}
        
        CL_parameters = self.frame_par.CL_win.CL_params
                    
        advanced_parameters = self.frame_par.adv_par_win.advanced_parameters
        
        if self.mesh_name == 'biv_mesh.xdmf':
            solver_biv_ellipsoid(parameters, advanced_parameters)
        elif self.mesh_name == 'lv_mesh.xdmf':
            solver_lv_ellipsoid(parameters, advanced_parameters)
        elif self.mesh_name.endswith('.h5'):
            if self.notebook_type =='reg':
                self.thread1 = CustThread(target = general_solver, args=(parameters, advanced_parameters))
                self.thread1.start()
                self.but_plot_res.state(['!disabled'])
            elif self.notebook_type == 'CL':
                if os.path.isfile('pv_data_plot.txt'):
                    os.remove('pv_data_plot.txt')
                with open('pv_data_plot.txt', 'w') as f: pass
            
                self.ani = animation.FuncAnimation(self.frame_mesh.plot_notebook.fig, self.frame_mesh.plot_notebook.animate, interval=1000)
                #closed_loop(parameters, advanced_parameters, CL_parameters)
                thread2 = CustThread(target = closed_loop, args = (parameters, advanced_parameters, CL_parameters))
                thread2.start()
        
        return
        
    def plot_res(self):
        #Loading the results from file.
        u_list = load_from_h5('output_u.h5')
        u1 = u_list[0]
        u2 = u_list[1]
        u3 = u_list[2]
        sf_list = load_from_h5('output_sf.h5')
        sf1 = sf_list[0]
        sf2 = sf_list[1]
        sf3 = sf_list[2]
        
        #Plotting the results.
        for i in range(3,6):
            self.frame_mesh.plot_notebook.tab_frame_list[i].vtkmesh=demo_displacement(self.current_mesh, u_list[i-3])
            self.frame_mesh.plot_notebook.tab_frame_list[i].rendered = CustRWI(master = self.frame_mesh.plot_notebook.tab_frame_list[i], coor = [0,0,1,1], rw = self.frame_mesh.plot_notebook.tab_frame_list[i].vtkmesh.renWin)
            self.frame_mesh.plot_notebook.tab_frame_list[i].rendered.Render()
            self.frame_mesh.plot_notebook.tab_frame_list[i].rendered.resize()
            
        for i in range(6,9):
            self.frame_mesh.plot_notebook.tab_frame_list[i].vtkmesh=demo_scalarfunction(self.current_mesh, sf_list[i-6])
            self.frame_mesh.plot_notebook.tab_frame_list[i].rendered = CustRWI(master = self.frame_mesh.plot_notebook.tab_frame_list[i], coor = [0,0,1,1], rw = self.frame_mesh.plot_notebook.tab_frame_list[i].vtkmesh.renWin)
            self.frame_mesh.plot_notebook.tab_frame_list[i].rendered.Render()
            self.frame_mesh.plot_notebook.tab_frame_list[i].rendered.resize()            

        return


root = tk.Tk()
root.wm_title('Pulse Adjoint Heart Simulator')

main_frame = MainFrame(root, coor = [0,0,1,1])
main_frame.resize()

for rows in range(root.grid_size()[1]):
    tk.Grid.rowconfigure(root, rows, weight = 1)
for cols in range(root.grid_size()[0]):
    tk.Grid.columnconfigure(root, cols, weight = 1)

def destroy_all():
    for i in range(0,len(main_frame.frame_mesh.plot_notebook.tab_frame_list)):
        del main_frame.frame_mesh.plot_notebook.tab_frame_list[i].vtkmesh.renWin


    main_frame.destroy()
    root.destroy()
    import psutil
    for p in [p for p in psutil.process_iter() if p.pid == os.getpid()]: p.kill()
    sys.exit(0)
    
    return

root.protocol("WM_DELETE_WINDOW", destroy_all)  

root.tk.mainloop()
