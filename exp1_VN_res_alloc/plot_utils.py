import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
mpl.use('agg')

font = {'size': 40}
plt.rc('font', **font)
plt.rcParams['figure.figsize'] = [12, 9]
plt.rcParams['lines.linewidth'] = 4

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
plt.rcParams['text.usetex'] = True

################################# diff C #####################################
# plt.figure()
# plt.plot(np.loadtxt('logs/obj_err_B_DPP_C0.1.txt'), label=r'$C=0.1$', linestyle='-')
# plt.plot(np.loadtxt('logs/obj_err_B_DPP_C0.8.txt'), label=r'$C=0.8$', linestyle='-')
# plt.plot(np.loadtxt('logs/obj_err_B_DPP_C1.5.txt'), label=r'$C=1.5$', linestyle='-')
# plt.xlim(0, 1000)
# plt.yscale('log')
# plt.legend(fontsize=40)
# plt.xlabel(r'$\mathrm{iteration}$ $t$', fontsize=45)
# plt.ylabel(r'$\left\vert\sum_{i=1}^N f_i(\bar{x}_{i,t}) - \sum_{i=1}^N f_i(x_i^\star)\right\vert$', fontsize=45)
# plt.savefig('C_obj.pdf', bbox_inches='tight')
# plt.close()

# plt.figure()
# plt.plot(np.loadtxt('logs/cons_val_B_DPP_C0.1.txt'), label=r'$C=0.1$', linestyle='-')
# plt.plot(np.loadtxt('logs/cons_val_B_DPP_C0.8.txt'), label=r'$C=0.8$', linestyle='-')
# plt.plot(np.loadtxt('logs/cons_val_B_DPP_C1.5.txt'), label=r'$C=1.5$', linestyle='-')
# plt.axhline(y=0, color='black', linestyle='--', linewidth=1)  # add horizontal line at y=0
# plt.xlim(0, 1000)
# plt.legend(fontsize=40)
# plt.xlabel(r'$\mathrm{iteration}$ $t$', fontsize=45)
# plt.ylabel(r'$\sum_{i=1}^N g_i(\bar{x}_{i,t})$', fontsize=45)
# plt.savefig('C_cons.pdf', bbox_inches='tight')
# plt.close()

# ################################## comp algos ################################
# plt.figure()
# plt.plot(np.loadtxt('logs/obj_err_B_DPP_C0.27.txt'), label=r'$\mathrm{B-DPP}$', linestyle='-')
# plt.plot(np.loadtxt('logs/obj_err_C_SP_SG.txt'), label=r'$\mathrm{C-SP-SG\ [13]}$', linestyle='--')
# plt.plot(np.loadtxt('logs/obj_err_DPD_TV.txt'), label=r'$\mathrm{DPD-TV\ [17]}$', linestyle='--')
# plt.plot(np.loadtxt('logs/obj_err_Falsone.txt'), label=r'$\mathrm{Dual\ Subgradient\ [11]}$', linestyle='--')
# plt.xlim(0, 1000)
# plt.yscale('log')
# plt.legend(fontsize=40, loc='upper right', bbox_to_anchor=(1.02, 1.02))
# plt.xlabel(r'$\mathrm{iteration}$ $t$', fontsize=45)
# plt.ylabel(r'$\left\vert\sum_{i=1}^N f_i(\bar{x}_{i,t}) - \sum_{i=1}^N f_i(x_i^\star)\right\vert$', fontsize=45)
# plt.savefig('algo_obj.pdf', bbox_inches='tight')
# plt.close()

class MyFigure:
    '''One figure contains several lines.'''
    
    def __init__(self, filename, xlabel, ylabel, yscale='log'):
        self.filename = filename
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.yscale = yscale
        
        self.label_line_dict = dict()
    
    def add_line(self, label:str, log:np.ndarray):
        self.label_line_dict[label] = log
    
    def paint(self, MAX_ITER=1000):
        plt.figure()
        # plt.plot(np.loadtxt('logs/cons_val_B_DPP_C0.27.txt'), label=r'$\mathrm{B-DPP}$', linestyle='-')
        
        for label, line in self.label_line_dict.items():
            plt.plot(line, label=label, linestyle='--')
        # plt.plot(np.loadtxt('logs/cons_val_B_DPP_C1.5.txt'), label=r'$\mathrm{B-DPP}$', linestyle='--')
        # plt.plot(np.loadtxt('logs/cons_val_C_SP_SG.txt'), label=r'$\mathrm{C-SP-SG\ [13]}$', linestyle='--')
        # plt.plot(np.loadtxt('logs/cons_val_DPD_TV.txt'), label=r'$\mathrm{DPD-TV\ [17]}$', linestyle='--')
        # plt.plot(np.loadtxt('logs/cons_val_Falsone.txt'), label=r'$\mathrm{Dual\ Subgradient\ [11]}$', linestyle='--')
        plt.axhline(y=0, color='black', linestyle='--', linewidth=1)  # add horizontal line at y=0
        plt.xlim(0, MAX_ITER)
        # plt.ylim(-0.5, 12)
        plt.yscale(self.yscale)
        plt.legend(fontsize=40)
        plt.xlabel(self.xlabel, fontsize=45)
        plt.ylabel(self.ylabel, fontsize=45)
        plt.savefig(f'{self.filename}.png', bbox_inches='tight')
        plt.close()