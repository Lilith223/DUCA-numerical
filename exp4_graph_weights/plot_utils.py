import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, LogLocator,FormatStrFormatter
from matplotlib import ticker
# mpl.use('agg')

# plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 40})
plt.rc('text', usetex=True)

# plt.rcParams['figure.figsize'] = [12, 9]
plt.rcParams['figure.figsize'] = [8, 6]
# plt.rcParams['lines.linewidth'] = 4

# mpl.rcParams['pdf.fonttype'] = 42
# mpl.rcParams['ps.fonttype'] = 42
# mpl.rcParams['xtick.labelsize'] = 20
# mpl.rcParams['ytick.labelsize'] = 20
# mpl.rcParams['ytick.minor.visible'] = True
# mpl.rcParams['ytick.minor.width'] = 3


mpl.rcParams['text.latex.preamble'] = r'\usepackage{amssymb}'
mpl.rcParams["legend.loc"] = 'upper right'

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
        '''yscale: "log", "symlog", "linear" '''
        self.filename = filename
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.yscale = yscale
        
        self.label_line = dict()
        self.label_style = dict()
        self.label_alpha = dict()
    
    def add_line(self, label:str, log:np.ndarray, style='', alpha=1):
        self.label_line[label] = log
        self.label_style[label] = style
        self.label_alpha[label] = alpha
        
    def add_line_file(self, label:str, file:str, style='', alpha=1):
        log = np.loadtxt(file)
        self.add_line(label, log, style=style, alpha=alpha)
    
    def clear(self):
        self.label_line = dict()
        self.label_style = dict()
    
    def paint(self, MAX_ITER=1000, nonnegy=False):
        fig = mpl.figure.Figure(facecolor='white')
        ax = fig.subplots()   
        ax.set_xlim(0, MAX_ITER)
        # ax.set_ylim(1e-4, 5e0)
        ax.set_yscale(self.yscale)
        
        ax.minorticks_on()
        ax.yaxis.set_major_locator(LogLocator(numticks=7))
        ax.yaxis.set_minor_locator(LogLocator(subs='auto'))
        # ax.yaxis.set_minor_locator(plt.LogLocator(subs=np.arange(2, 10)))
        # ax.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=7))
        # ax.yaxis.set_minor_locator(ticker.)
        # plt.tick_params(axis='y', which='minor')
        # ax.yaxis.set_minor_locator(plt.LogLocator(subs=np.arange(2, 10)))

        # Customize tick appearance
        # ax.set_yticks(plt.LogLocator(subs=np.arange(2, 10)), minor=True)
        
        # ax.tick_params(which='both', width=2)
        # ax.tick_params(which='major', length=7)
        # ax.tick_params(which='minor', length=4, color='r')
        
        
        for label, line in self.label_line.items():
            if self.label_style[label] == '':
                ax.plot(line, label=label, linestyle='--', linewidth=1)
            else:
                # ax.plot(line, self.label_style[label], label=label, 
                #         marker=self.label_marker[label], markevery=200, linewidth=1)     
                ax.plot(line, self.label_style[label], label=label,
                        linewidth=2, alpha=self.label_alpha[label])        
        
        
        
        ax.legend(fontsize=20)
        ax.set_xlabel(self.xlabel, fontsize=20)
        ax.set_ylabel(self.ylabel, fontsize=20)
        ax.grid(True)
        fig.savefig(f'{self.filename}.png', bbox_inches='tight', transparent=False)
        plt.close()


class MyFigureThree:
    '''Three MyFigure s'''
    
    def __init__(self):
        '''yscale: "log", "symlog", "linear" '''
        
        self.obj_err_figure = MyFigure(filename='obj_err', 
                            xlabel=r'$\mathrm{iteration}$ $k$', 
                            ylabel='objective error',
                            yscale='log')
        self.cons_vio_figure = MyFigure(filename='cons_vio', 
                                    xlabel=r'$\mathrm{iteration}$ $k$', 
                                    ylabel='constraint violation',
                                    yscale='log')
        self.x_dis_figure = MyFigure(filename='x_distance', 
                            xlabel=r'$\mathrm{iteration}$ $k$', 
                            ylabel='x distance',
                            yscale='log')
        self.oe_cv_figure = MyFigure(filename='oe_cv', 
                            xlabel=r'$\mathrm{iteration}$ $k$', 
                            ylabel='obj. err. + cons. vio.',
                            yscale='log')
        

        
        

    def add_lines(self, graph_name:str, log_prefix:str):
        filename_oe = f'log/{graph_name}/{log_prefix}_oe.txt'
        filename_cv = f'log/{graph_name}/{log_prefix}_cv.txt'
        filename_xd = f'log/{graph_name}/{log_prefix}_xd.txt'
        filename_oe_cv = f'log/{graph_name}/{log_prefix}_oe_cv.txt'
        filename_oe_avg = f'log/{graph_name}/{log_prefix}_oe_avg.txt'
        filename_cv_avg = f'log/{graph_name}/{log_prefix}_cv_avg.txt'
        filename_xd_avg = f'log/{graph_name}/{log_prefix}_xd_avg.txt'
        filename_oe_cv_avg = f'log/{graph_name}/{log_prefix}_oe_cv_avg.txt'
        
        # self.obj_err_figure.add_line_file(log_prefix, filename_oe)
        # self.cons_vio_figure.add_line_file(log_prefix, filename_cv)
        # self.x_dis_figure.add_line_file(log_prefix, filename_xd)

        self.obj_err_figure.add_line_file(log_prefix, filename_oe)
        self.cons_vio_figure.add_line_file(log_prefix, filename_cv)
        self.x_dis_figure.add_line_file(log_prefix, filename_xd)
        self.oe_cv_figure.add_line_file(log_prefix, filename_oe_cv)
    
    def clear(self):
        self.obj_err_figure.clear()
        self.cons_vio_figure.clear()
        self.x_dis_figure.clear()
        self.oe_cv_figure.clear()
        
    def paint(self, MAX_ITER:int):
        self.obj_err_figure.paint(MAX_ITER=MAX_ITER)
        self.cons_vio_figure.paint(MAX_ITER=MAX_ITER, nonnegy=True)
        self.x_dis_figure.paint(MAX_ITER=MAX_ITER, nonnegy=True)
        self.oe_cv_figure.paint(MAX_ITER=MAX_ITER)
            
            
        
        