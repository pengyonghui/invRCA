# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 22:03:01 2021

@author: yonghui
"""
import tkinter as tk
from tkinter import ttk
import numpy as np

import matplotlib.pyplot as plt  
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
# from matplotlib.backends.figure import Figure

# from t2nnls import T2NNLS
from fitTikhonovReg import T2NNLS
from rotateT2phase import rotate_T2phase
import LoadNMRData_rca
import pltRawData_rca
import pltLcurve 
from getLambdaFromRMSE import get_lambda_from_Lcurve
import pltRTD

class ControlFrame(tk.Frame):
	def __init__(self, parent=None):
		self.parent = parent
		tk.Frame.__init__(self,self.parent)

		self.container = tk.Frame(self.parent)
		self.container.pack(side="top", fill="both", expand=True)

		self.frames = {}
		self.fr_open_file = OpenFilePanel(self.container, self)
		self.fr_invert_setting = InversionSettingPanel(self.container, self)
		self.fr_logs_print_out = LogsPrintOut(self.container, self)
		self.fr_view = CanvasView(self.container, self) 
		# make class instance(or the controller)
		self.frames[OpenFilePanel.__name__] = self.fr_open_file
		self.frames[InversionSettingPanel.__name__] = self.fr_invert_setting
		self.frames[CanvasView.__name__] = self.fr_view
		self.frames[LogsPrintOut.__name__] = self.fr_logs_print_out

		self.fr_open_file.grid(row=0, column=0, pady=5, sticky="nsew") 
		self.fr_invert_setting.grid(row=1, column=0, sticky="nsew")
		self.fr_logs_print_out.grid(row=2, column=0, rowspan=2, pady=5,sticky="nsew")
		self.fr_view.grid(row=0, column=1, rowspan=4, columnspan=2, pady=5, sticky='nsew')

		self.container.columnconfigure(0, weight=1)
		self.container.columnconfigure((1,2), weight=2)
		#self.container.rowconfigure((0,1), weight=1)
		self.container.rowconfigure(2, weight=2)

		self.fr_open_file.open_csv_button['command'] = lambda:[self.open_csv_file(), 
															   self.plot_raw_RCA()]
		self.fr_invert_setting.start_button['command'] = lambda:[self.start_inversion(),
																 self.clear_canvas(),
																 self.plot_fitted_raw_RCA(),
																 self.plot_reg_curve(),
																 self.plot_T2()]
		self.fr_invert_setting.reset_button['command'] = lambda:[self.set_default_lambda_T2_par(),
																 self.clear_canvas()]

		self.invT2NNLS = InversionT2NNLS()
		self.openCSVFile = OpenCSVFile()
		# show default settings in each entry
		self.set_default_lambda_T2_par()
		self.array_raw_data = None
		self.dir_csv_file = None

	def open_csv_file(self):
		self.openCSVFile.open_csv_file()
		self.dir_csv_file = self.openCSVFile.get_dir_csv_file()
		self.fr_open_file.open_csv_entry.insert(0, self.dir_csv_file)
		self.array_raw_data = self.openCSVFile.get_array_raw_data()

	# plot raw RCA data
	def plot_raw_RCA(self):	
		self.fr_view.ax_raw.clear()
		pltRawData_rca.plt_raw_rca(self.array_raw_data, self.fr_view.ax_raw)
		self.fr_view.canvas_raw.draw()

	# start inversion using the input-settings	
	def start_inversion(self):		
		if self.array_raw_data is None: 
			tk.messagebox.showerror(message = "No RCA data was imported")
		else:
			self.set_manu_lambda_T2_par()
			self.invT2NNLS.initial_inversion(self.array_raw_data)
			self.invT2NNLS.get_optimize_lambda()

	# plot the fitted raw RCA data			
	def plot_fitted_raw_RCA(self):	
		self.fr_view.ax_raw.clear()
		pltRawData_rca.plt_raw_rca(self.array_raw_data, self.fr_view.ax_raw)	
		pltRawData_rca.plt_fitted_rca(self.invT2NNLS.optimize_dsyn, self.array_raw_data, self.fr_view.ax_raw)
		self.fr_view.canvas_raw.draw()

	# plot lambda vs residual norm	
	def plot_reg_curve(self):		
		pltLcurve.plt_lambda_rmse(self.invT2NNLS.coef_lambda, self.invT2NNLS.rmse, 
								 self.invT2NNLS.optimize_index, self.fr_view.ax_reg)
		self.fr_view.canvas_reg.draw()
		# plot residual norm vs model norm
		pltLcurve.plt_rmse_model_norm(self.invT2NNLS.rmse, self.invT2NNLS.model_norm, 
									 self.invT2NNLS.optimize_index,  self.fr_view.ax_model)
		self.fr_view.canvas_model.draw()

	# plot the T2-distribution
	"""pltT2dist(T2, m, ax)"""
	def plot_T2(self):
		pltRTD.pltT2dist(self.invT2NNLS.T2, self.invT2NNLS.optimize_m, self.fr_view.ax_T2)
		self.fr_view.canvas_T2.draw()

	# default value assigned to each entry
	# lambda_vars = [1, 10000, 20]
	# time_vars = [-4, 1, 160]
	def set_default_lambda_T2_par(self):
		for i in range(3):
			self.fr_invert_setting.lambda_vars[i].set([1, 10000, 20][i])
			self.fr_invert_setting.time_vars[i].set([-4, 1, 160][i])

	# set inversion parameters with values got from the input of each entry
	def set_manu_lambda_T2_par(self):
		for i in range(3):
			self.invT2NNLS.lambda_vars[i] = self.fr_invert_setting.lambda_vars[i].get() 
			self.invT2NNLS.time_vars[i] = self.fr_invert_setting.time_vars[i].get() 	

	def clear_canvas(self):
		# self.fr_view.ax_raw.clear()
		# self.fr_view.canvas_raw.draw()
		self.fr_view.ax_reg.clear()
		# self.fr_view.canvas_reg.draw()
		self.fr_view.ax_model.clear()
		# self.fr_view.canvas_model.draw()
		self.fr_view.ax_T2.clear()
		# self.fr_view.canvas_T2.draw()

class OpenFilePanel(tk.Frame):
	"""	Open the NMR data file
	"""
	def __init__(self, parent, controller):
		self.parent = parent
		tk.LabelFrame.__init__(self, self.parent, text = "Open", borderwidth=2, relief="ridge")
		self.controller = controller
		self.browse_label = tk.Label(self, text = "Select file to open:")

		self.open_csv_button = tk.Button(self, text="Load RCA data", relief= "raised", bg='lightgray')	
		# self.open_csv_button.bind("<Button>", self.open_plot_csv_file)
		self.open_csv_entry = tk.Entry(self)

		self.browse_label.grid(row=0, column=1, padx=5, pady=10, sticky='ew')
		self.open_csv_button.grid(row=1, column=0, padx=5, sticky='ew')
		self.open_csv_entry.grid(row=1, column=1, columnspan=3, padx=5, sticky='ew')

		self.columnconfigure(1, weight=2)
		self.columnconfigure(0, weight=1)

class OpenCSVFile():
	def __init__(self):
		self.dir_csv_file = None 
		self.array_raw_data = None

	def open_csv_file(self):
		self.dir_csv_file = tk.filedialog.askopenfilename(initialdir = "/D:",
													  title = "Open RCA file",
													  filetypes = (("csv files","*.csv"),("all files","*.*")))
		# browse the csv file
		if self.dir_csv_file.rsplit(".")[-1] !="csv": 
			return		
		self.array_raw_data = LoadNMRData_rca.read_csv(self.dir_csv_file)

	def get_array_raw_data(self):		
		return self.array_raw_data

	def get_dir_csv_file(self):
		return self.dir_csv_file

class InversionT2NNLS():
	def __init__(self):
		# initial inversion parameters		
		self.data = None
		self.time = None
		self.noise = None

		# predefined lambda values - min|max|length
		self.lambda_vars = [1, 10000, 20]
		self.coef_lambda = np.geomspace(self.lambda_vars[0],self.lambda_vars[1], self.lambda_vars[2])
		# predefined T2 values - min|max\length
		self.time_vars = [-4, 1, 160]
		self.T2 = np.logspace(self.time_vars[0], self.time_vars[1], num = self.time_vars[2]).transpose()
		# the size to clip the data	in T2NNLS	
		self.winsize = 100

		self.rmse = [0 for _ in range(self.lambda_vars[2])]
		self.model_norm = [0 for _ in range(self.lambda_vars[2])]
		self.optimize_index =  None
		self.optimize_rmse = None
		self.optimize_model_norm = None
		self.m = [0 for _ in range(self.lambda_vars[2])]
		self.r = [0 for _ in range(self.lambda_vars[2])]
		self.dsyn = [0 for _ in range(self.lambda_vars[2])]
		self.optimize_m = []
		self.optimize_dsyn = [] 
		self.optimize_r = [0,0]

	def initial_inversion(self, array_raw_data):
		if array_raw_data is not None:
			self.data, self.time, self.noise = rotate_T2phase(array_raw_data)			
		else:
			tk.messagebox.showerror(message = " 'failure during reading RCA' \n data is none or data is not a numpy array")

		for i in range(self.lambda_vars[2]):	
			(self.m[i], self.r[i], self.dsyn[i]) = T2NNLS(self.data, self.time, self.noise, self.T2, self.coef_lambda[i], self.winsize)
			self.rmse[i] = self.r[i][0]
			self.model_norm[i] = self.r[i][1]
		self.optimize_index = get_lambda_from_Lcurve(self.coef_lambda, self.rmse)

	def get_optimize_lambda(self):
		self.optimize_rmse = self.r[self.optimize_index][0]
		self.optimize_model_norm = self.r[self.optimize_index][1]
		self.optimize_m = self.m[self.optimize_index]
		self.optimize_r = self.r[self.optimize_index]
		self.optimize_dsyn = self.dsyn[self.optimize_index]
	
	# def optimize_inversion(self, array_raw_data):
	# 	(self.optimize_m, self.optimize_r, self.optimize_dsyn, rnorm) = T2NNLS(self.data, self.time, self.noise, 
	# 																			self.T2, self.coef_lambda[self.optimize_index], 100)
	# 	self.optimize_rmse = self.optimize_r[0]
	# 	self.optimize_model_norm = self.optimize_r[1]

class InversionSettingPanel(tk.Frame):
	def __init__(self, parent, controller): 
		self.parent = parent 
		tk.LabelFrame.__init__(self, self.parent,text = "Inversion Settings", borderwidth=2, relief="ridge")
		self.controller = controller

		self.reg_label = tk.Label(self, text = "Regularization")
		self.reg_combox = ttk.Combobox(self, values = ["L2", "L1","None"])
		self.reg_combox.current(0)

		self.reset_button = tk.Button(self, text="Reset Settings",relief= "raised")

		self.lambda_label = tk.Label(self, text = "Lambda min|max|#")
		self.lambda_vars = []
		self.lambda_entries = []
		for i in range(3):
			var = tk.IntVar()
			self.lambda_vars.append(var)			
			entry = tk.Entry(self, bg='white', textvariable=self.lambda_vars[i], justify=tk.CENTER)
			self.lambda_entries.append(entry)

		self.time_label = tk.Label(self, text = "Time-T2 min|max|#")
		self.time_vars = []
		self.time_entries = []
		for i in range(3):
			var = tk.IntVar()
			self.time_vars.append(var)			
			entry = tk.Entry(self, bg='white',textvariable=self.time_vars[i], justify=tk.CENTER)
			self.time_entries.append(entry)
		
		# start inversion button
		self.start_button = tk.Button(self, text="Start Inversion",relief= "raised", bg='lightgray')
		
		self.grid(row=0, column=0, sticky='nsew')		

		self.reg_label.grid(row=0, column=0,pady=10, sticky='nsew')
		self.reg_combox.grid(row=0, column=1,pady=10,sticky='ns')
		self.reset_button.grid(row=0, column=2, columnspan=2, pady=10, sticky='ns')
		self.lambda_label.grid(row=1, column=0,sticky='nsew')
		self.lambda_entries[0].grid(row=1, column=1,sticky='nsew')
		self.lambda_entries[1].grid(row=1, column=2,sticky='nsew')
		self.lambda_entries[2].grid(row=1, column=3,sticky='nsew')
		self.time_label.grid(row=2, column=0, pady=10,sticky='nsew')
		self.time_entries[0].grid(row=2, column=1,pady=10,sticky='nsew')   
		self.time_entries[1].grid(row=2, column=2,pady=10,sticky='nsew')  
		self.time_entries[2].grid(row=2, column=3,pady=10,sticky='nsew')  

		self.start_button.grid(row=3, column=0, columnspan = 4, sticky='nsew')

		self.columnconfigure((0), weight=1)

class LogsPrintOut(tk.Frame):
	def __init__(self, parent, controller): 
		self.parent = parent 
		tk.Frame.__init__(self, self.parent, borderwidth=2, relief="ridge")
		self.controller = controller

		self.logs_label = tk.Label(self,bg='white')
		self.logs_label.grid(row=0, column=0, sticky='nsew')
		self.columnconfigure(0, weight=1)
		self.rowconfigure(0, weight=1)

class CanvasView(tk.Frame):
	def __init__(self, parent, controller):  
		self.parent = parent
		tk.Frame.__init__(self, self.parent, borderwidth=2, relief="ridge")
		self.controller = controller

		self.fr_raw_notebooks = ttk.Notebook(self)
		self.fr_raw_data = ttk.Frame(self.fr_raw_notebooks)
		self.fr_T2 = ttk.Frame(self.fr_raw_notebooks)
		self.fr_raw_notebooks.add(self.fr_raw_data, text = "Raw data")
		self.fr_raw_notebooks.add(self.fr_T2, text = "T2 dist")
		self.fr_raw_notebooks.grid(row=0, column=0, columnspan=2, pady=5, sticky="nsew")
		
		self.fr_reg_notebooks = ttk.Notebook(self)
		self.fr_reg_curve = ttk.Frame(self.fr_reg_notebooks)
		self.fr_reg_notebooks.add(self.fr_reg_curve, text="L-curve")
		self.fr_reg_notebooks.grid(row=1, column=0, sticky="nsew")

		self.rowconfigure((0,1), weight=1)
		self.columnconfigure(0, weight=2)

	# plot raw data on top
		self.fig_raw = Figure(figsize=(8,3), dpi=100) 
		self.ax_raw = self.fig_raw.add_subplot(111)	
		self.canvas_raw = FigureCanvasTkAgg(self.fig_raw, self.fr_raw_data)
		# self.canvas_raw.get_tk_widget().grid(row=0, column=0,rowspan=2, sticky='nsew')     
		self.canvas_raw.get_tk_widget().pack(side="top", fill="both", expand=True)
		self.canvas_raw.draw()

	# plot T2 on top
		self.fig_T2 = Figure(figsize=(8,3), dpi=100) 
		self.ax_T2 = self.fig_T2.add_subplot(111)	
		self.canvas_T2 = FigureCanvasTkAgg(self.fig_T2, self.fr_T2)
		# self.canvas_T2.get_tk_widget().grid(row=0, column=0,rowspan=2, sticky='nsew')     
		self.canvas_T2.get_tk_widget().pack(side="top", fill="both", expand=True)
		self.canvas_T2.draw()

	# plot on lambda vs residual norm on bottom-left
		self.fig_reg = Figure(figsize=(4,3),dpi=100)
		self.ax_reg = self.fig_reg.add_subplot(111)		
		self.canvas_reg = FigureCanvasTkAgg(self.fig_reg, self.fr_reg_curve)
		# self.canvas_reg.get_tk_widget().grid(row=0, column=0, sticky='nsew')	
		self.canvas_reg.get_tk_widget().pack(side="left", fill="both", expand=True)									  
		self.canvas_reg.draw()

	# plot residual norm vs model norm on bottom-right
		self.fig_model = Figure(figsize=(4,3),dpi=100)
		self.ax_model = self.fig_model.add_subplot(111)
		self.canvas_model = FigureCanvasTkAgg(self.fig_model, self.fr_reg_curve)
		# self.canvas_model.get_tk_widget().grid(row=0, column=1, sticky='nsew') 
		self.canvas_model.get_tk_widget().pack(side="left", fill="both", expand=True)										  
		self.canvas_model.draw()

class MainApplication(tk.Tk):
	def __init__(self):
		tk.Tk.__init__(self)

		# put the window in the middle of the screen/monitor
		screenWidth = self.winfo_screenwidth()
		screenHeight = self.winfo_screenheight()
		w = 800
		h = 300
		x = (screenWidth - w) / 4 # x_position from the top-left corner
		y = (screenHeight - h) / 4 # y_distance from the top-left corner
		self.geometry("%dx%d+%d+%d" % (w,h,x,y))
		# self.geometry("400x300")
		self.title("pyNMR-T2 inverstion") # title on the top left


if __name__=="__main__":
	app = MainApplication()
	ControlFrame(app)
	app.mainloop()