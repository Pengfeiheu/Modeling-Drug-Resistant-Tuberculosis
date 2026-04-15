import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import time
import os
import sys


try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    plt.style.use('ggplot')

plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'font.family': 'serif', 
    'mathtext.fontset': 'cm' 
})

# --- 0. Setup ---
torch.manual_seed(1234)
np.random.seed(1234)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running on device: {device}")

# --- 1. NN model ---
class FCN(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size):
        super(FCN, self).__init__()
        layers = []
        layers.append(nn.Linear(input_size, hidden_layers[0]))
        layers.append(nn.Tanh())
        
        for i in range(len(hidden_layers) - 1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            layers.append(nn.Tanh())
            
        layers.append(nn.Linear(hidden_layers[-1], output_size))
        layers.append(nn.Sigmoid()) 
        
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# --- 2. PINN model ---
class TB_PINN:
    def __init__(self, df, N_population):
        self.N = N_population
        self.df_raw = df
        
        # === 1. time series data ===
        num_points = len(df)
        self.dates = pd.date_range(start='2017-01-01', periods=num_points, freq='MS')
        
        self.t_max = df['t'].max()
        
        t_raw = df['t'].values.reshape(-1, 1)
        Cs_raw = df['Is_cum'].values.reshape(-1, 1)
        Cr_raw = df['Ir_cum'].values.reshape(-1, 1)
        
        self.Cs_raw_numpy = Cs_raw
        self.Cr_raw_numpy = Cr_raw
        
        self.t_data = torch.tensor(t_raw / self.t_max, dtype=torch.float32).to(device).requires_grad_(True)
        self.Cs_data = torch.tensor(Cs_raw / self.N, dtype=torch.float32).to(device)
        self.Cr_data = torch.tensor(Cr_raw / self.N, dtype=torch.float32).to(device)
        
        self.t_col = torch.linspace(0, 1, 1000).view(-1, 1).to(device).requires_grad_(True)
        
        # initial condition
        self.t0 = torch.tensor([[0.0]], dtype=torch.float32).to(device)
        Is_0 = df['Is_new'].iloc[0] / self.N
        Ir_0 = df['Ir_new'].iloc[0] / self.N
        R_0 = 1.267e5 / self.N
        Cs_0 = df['Is_cum'].iloc[0] / self.N
        Cr_0 = df['Ir_cum'].iloc[0] / self.N
        S_0 = 1.0 - (Is_0 + Ir_0 + R_0)
        
        self.y0 = torch.tensor([S_0, Is_0, Ir_0, R_0, Cs_0, Cr_0], dtype=torch.float32).to(device)

        # model
        self.model = FCN(1, [64, 64, 64], 6).to(device)
        

        self.log_beta1 = nn.Parameter(torch.tensor([-1.3], dtype=torch.float32, device=device).requires_grad_(True))
        self.log_beta2 = nn.Parameter(torch.tensor([-4.6], dtype=torch.float32, device=device).requires_grad_(True))
        self.log_alpha0 = nn.Parameter(torch.tensor([-4.0], dtype=torch.float32, device=device).requires_grad_(True))
        self.log_rho = nn.Parameter(torch.tensor([np.log(0.12)], dtype=torch.float32, device=device).requires_grad_(True))
        self.log_omega = nn.Parameter(torch.tensor([np.log(0.045)], dtype=torch.float32, device=device).requires_grad_(True))
        
        # 
        self.Lam = 3.213e4 / self.N
        self.mu = 0.0012
        self.gamma = 0.05
        self.r1 = 0.1583
        self.r2 = 0.0407
        self.d1 = 0.0083
        self.d2 = 0.0283

        self.trainable_params = list(self.model.parameters()) + [
            self.log_beta1, self.log_beta2, self.log_alpha0, self.log_rho, self.log_omega
        ]
        
        self.history = {
            'loss': [],
            'beta1': [], 'beta2': [], 'alpha0': [], 'rho': [], 'omega': [],
            'epoch': []
        }
        
        self.global_iter = 0

    def get_params(self):
        return (torch.exp(self.log_beta1), torch.exp(self.log_beta2), 
                torch.exp(self.log_alpha0), torch.exp(self.log_rho), 
                torch.exp(self.log_omega))

    def net_forward(self, t):
        return self.model(t)

    def calculate_loss(self):
        beta1, beta2, alpha0, rho, omega = self.get_params()
        
        y_pred = self.net_forward(self.t_col)
        S, Is, Ir, R, Cs, Cr = y_pred[:, 0:1], y_pred[:, 1:2], y_pred[:, 2:3], y_pred[:, 3:4], y_pred[:, 4:5], y_pred[:, 5:6]
        
        grads = [torch.autograd.grad(y_pred[:, i], self.t_col, torch.ones_like(y_pred[:, i]), create_graph=True)[0] for i in range(6)]
        dSdt, dIsdt, dIrdt, dRdt, dCsdt, dCrdt = [g / self.t_max for g in grads]
        
        I_total = Is + Ir
        alpha_dynamic = alpha0 + (rho * I_total) / (1 + omega * I_total)
        
        loss_1 = dSdt - (self.Lam - beta1*S*Is - beta2*S*Ir - self.mu*S + self.gamma*R)
        loss_2 = dIsdt - (beta1*S*Is - alpha_dynamic*Is - (self.mu + self.r1 + self.d1)*Is)
        loss_3 = dIrdt - (beta2*S*Ir + alpha_dynamic*Is - (self.mu + self.r2 + self.d2)*Ir)
        loss_4 = dRdt - (self.r1*Is + self.r2*Ir - self.mu*R - self.gamma*R)
        loss_5 = dCsdt - (beta1*S*Is)
        loss_6 = dCrdt - (beta2*S*Ir + alpha_dynamic*Is)
        
        loss_ode = torch.mean(loss_1**2 + loss_2**2 + loss_3**2 + loss_4**2 + loss_5**2 + loss_6**2)
        
        y_data_pred = self.net_forward(self.t_data)
        Cs_pred_data = y_data_pred[:, 4:5]
        Cr_pred_data = y_data_pred[:, 5:6]
        loss_data = torch.mean((Cs_pred_data - self.Cs_data)**2) + torch.mean((Cr_pred_data - self.Cr_data)**2)
        
        y0_pred = self.net_forward(self.t0)
        loss_ic = torch.mean((y0_pred - self.y0)**2)
        
        return loss_ode + 10.0 * loss_data + loss_ic

    def record_history(self, loss_val):
        b1, b2, a0, r, w = self.get_params()
        self.history['loss'].append(loss_val)
        self.history['beta1'].append(b1.item())
        self.history['beta2'].append(b2.item())
        self.history['alpha0'].append(a0.item())
        self.history['rho'].append(r.item())
        self.history['omega'].append(w.item())
        self.history['epoch'].append(self.global_iter)

    def train(self, adam_epochs=100000, lbfgs_max_iter=10000):
        start_time = time.time()
        print(f"--- Stage 1: Adam Optimization ({adam_epochs} epochs) ---")
        optimizer_adam = torch.optim.Adam(self.trainable_params, lr=1e-3)
        
        for epoch in range(adam_epochs):
            optimizer_adam.zero_grad()
            loss = self.calculate_loss()
            loss.backward()
            optimizer_adam.step()
            
            self.global_iter += 1
            if self.global_iter % 10 == 0:
                self.record_history(loss.item())
            if epoch % 500 == 0:
                print(f"Adam Epoch {epoch}: Loss = {loss.item():.2e}")
                
        print(f"\n--- Stage 2: L-BFGS Optimization (Max {lbfgs_max_iter} iter) ---")
        optimizer_lbfgs = torch.optim.LBFGS(self.trainable_params, lr=1.0, max_iter=lbfgs_max_iter,
                                            history_size=50, tolerance_grad=1e-7, tolerance_change=1.0 * np.finfo(float).eps,
                                            line_search_fn="strong_wolfe")

        def closure():
            optimizer_lbfgs.zero_grad()
            loss = self.calculate_loss()
            loss.backward()
            self.global_iter += 1
            if self.global_iter % 10 == 0:
                self.record_history(loss.item())
            if self.global_iter % 100 == 0:
                print(f"L-BFGS Iter {self.global_iter}: Loss = {loss.item():.2e}")
            return loss

        optimizer_lbfgs.step(closure)
        print(f"Training Finished. Total Time: {time.time() - start_time:.2f}s")

    def plot_training_process_stacked(self):
       
        epochs = self.history['epoch']
        
        params_map = [
            (r'$\beta_1$', self.history['beta1']),
            (r'$\beta_2$', self.history['beta2']),
            (r'$\alpha_0$', self.history['alpha0']),
            (r'$\rho$', self.history['rho']),
            (r'$\omega$', self.history['omega'])
        ]
        
        fig, axes = plt.subplots(5, 1, figsize=(10, 14), sharex=True)
        if not isinstance(axes, np.ndarray): 
            axes = [axes]
        
        for i, (latex_name, data) in enumerate(params_map):
            ax = axes[i]
            final_val = data[-1] if len(data) > 0 else 0
            
            ax.plot(epochs, data, color='#1f77b4', linewidth=2.5, alpha=0.9, label='Estimate')
            ax.axhline(y=final_val, color='#ff7f0e', linestyle='--', linewidth=2, 
                       label=f'Converged: {final_val:.4f}')
            
            ax.set_ylabel(latex_name, fontsize=16, fontweight='bold', rotation=0, labelpad=20)
            ax.legend(loc='upper right', frameon=True, fontsize=10, shadow=True)
            ax.grid(True, linestyle='--', alpha=0.5)
            
            ax.ticklabel_format(axis='y', style='plain', useOffset=False)
            
            if i < 4:
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel('Training Iterations', fontsize=14)

        plt.suptitle('Parameter Estimation Process', fontsize=18, fontweight='bold', y=0.96)
        plt.subplots_adjust(hspace=0.05, top=0.93)
        plt.show()

    def plot_fitting_results(self):
        """
        plot fitting results for S, Is, Ir, R and cumulative cases Cs, Cr
        """
        t_smooth = torch.linspace(0, 1, 1000).view(-1, 1).to(device)
        with torch.no_grad():
            y_pred = self.model(t_smooth)
        
        t_plot_days = t_smooth.cpu().numpy().flatten() * self.t_max
        total_days = (self.dates[-1] - self.dates[0]).days
        
        start_ts = self.dates[0]
        pred_dates = [start_ts + pd.Timedelta(days=d) for d in (t_plot_days / self.t_max * total_days)]
        
        predictions = y_pred.cpu().numpy() * self.N
        
        var_info = [
            {'name': r'Susceptible ($S$)', 'data': predictions[:, 0], 'scale': 1e6, 'color': '#2ca02c', 'style': '-'},
            {'name': r'Infected Sensitive ($I_s$)', 'data': predictions[:, 1], 'scale': 1e3, 'color': '#d62728', 'style': '-'},
            {'name': r'Infected Resistant ($I_r$)', 'data': predictions[:, 2], 'scale': 1e3, 'color': '#9467bd', 'style': '-'},
            {'name': r'Recovered ($R$)', 'data': predictions[:, 3], 'scale': 1e6, 'color': '#8c564b', 'style': '-'},
        ]
        
        date_fmt = mdates.DateFormatter('%Y-%m')
        x_start = pd.Timestamp('2017-01-01')
        x_end = pd.Timestamp('2018-12-01')
        
        for var in var_info:
            plt.figure(figsize=(10, 6))
            
            scaled_data = var['data'] / var['scale']
            
            if var['scale'] == 1e6:
                unit_label = r'($\times 10^6$)'
            elif var['scale'] == 1e3:
                unit_label = r'($\times 10^3$)'
            else:
                unit_label = ''
            
            plt.plot(pred_dates, scaled_data, color=var['color'], linewidth=3, 
                     linestyle=var['style'], label=f"Fitted {var['name']}")
            
            plt.title(f"Dynamics of {var['name']}", fontsize=16)
            plt.ylabel(f"Population {unit_label}", fontsize=14)
            plt.xlabel('Date', fontsize=14)
            
            ax = plt.gca()
            ax.set_xlim([x_start, x_end])
            ax.xaxis.set_major_formatter(date_fmt)
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            plt.xticks(rotation=30)
            
            plt.legend(loc='best', frameon=True, fontsize=12)
            plt.grid(True, linestyle=':', alpha=0.6)
            plt.tight_layout()
            plt.show()
        
        self.plot_cumulative_cases_comparison(pred_dates, predictions)

    def plot_cumulative_cases_comparison(self, pred_dates, predictions):
        date_fmt = mdates.DateFormatter('%Y-%m')
        x_start = pd.Timestamp('2017-01-01')
        x_end = pd.Timestamp('2018-12-01')
        
        # --- figure 1: DS-TB ---
        plt.figure(figsize=(10, 6)) 
        plt.plot(pred_dates, predictions[:, 4], color='#1f77b4', linewidth=3, label=r'Fitted $C_s$')
        plt.scatter(self.dates, self.Cs_raw_numpy, color='black', s=50, marker='o', 
                   facecolors='none', edgecolors='k', linewidth=1.5, label='Reported Data')
        
        plt.title(r'Cumulative Cases of Sensitive TB ($C_s$)', fontsize=16)
        plt.ylabel('Population', fontsize=14)
        plt.xlabel('Date', fontsize=14)
        plt.legend(loc='upper left', fontsize=12, frameon=True)
        plt.grid(True, linestyle=':', alpha=0.6)
        

        ax = plt.gca()
        ax.set_xlim([x_start, x_end])
        ax.xaxis.set_major_formatter(date_fmt)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.gcf().autofmt_xdate(rotation=30)
        
        plt.tight_layout()
        plt.show()
        
        # --- figure 2: DR-TB  ---
        plt.figure(figsize=(10, 6)) 
        plt.plot(pred_dates, predictions[:, 5], color='#d62728', linewidth=3, label=r'Fitted $C_r$')
        plt.scatter(self.dates, self.Cr_raw_numpy, color='black', s=50, marker='^', 
                   facecolors='none', edgecolors='k', linewidth=1.5, label='Reported Data')
        
        plt.title(r'Cumulative Cases of Resistant TB ($C_r$)', fontsize=16)
        plt.ylabel('Population', fontsize=14)
        plt.xlabel('Date', fontsize=14)
        plt.legend(loc='upper left', fontsize=12, frameon=True)
        plt.grid(True, linestyle=':', alpha=0.6)
        
        ax = plt.gca()
        ax.set_xlim([x_start, x_end])
        ax.xaxis.set_major_formatter(date_fmt)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.gcf().autofmt_xdate(rotation=30)
        
        plt.tight_layout()
        plt.show()

    def evaluate_and_save(self, filename='fitting_results.csv'):
        t_eval = torch.linspace(0, 1, 100).view(-1, 1).to(device)
        with torch.no_grad():
            y_eval = self.model(t_eval)
        
        predictions = y_eval.cpu().numpy() * self.N
        t_eval_days = t_eval.cpu().numpy().flatten() * self.t_max
        
        result_df = pd.DataFrame({
            't': t_eval_days,
            'S': predictions[:, 0],
            'Is': predictions[:, 1],
            'Ir': predictions[:, 2],
            'R': predictions[:, 3],
            'Cs': predictions[:, 4],
            'Cr': predictions[:, 5]
        })
        
        b1, b2, a0, r, w = self.get_params()
        
        result_df.to_csv(filename, index=False)
        
        print(f"\n{'='*60}")
        print("FINAL ESTIMATED PARAMETERS")
        print('='*60)
        print(f"Beta_1  (Transmission DS-TB) : {b1.item():.6f}")
        print(f"Beta_2  (Transmission DR-TB) : {b2.item():.6e}")
        print(f"Alpha_0 (Baseline Mutation)  : {a0.item():.6f}")
        print(f"Rho     (Saturation Coeff)   : {r.item():.6f}")
        print(f"Omega   (Saturation Param)   : {w.item():.6f}")
        print('='*60)
        
        return result_df



def generate_dummy_data_24_months(N=2.48e7):
    t = np.arange(24) 
    Cs = 1500 / (1 + np.exp(-0.3 * (t - 10))) * N * 0.001 
    Cr = 100 / (1 + np.exp(-0.25 * (t - 12))) * N * 0.0001
    Is_new = np.diff(Cs, prepend=Cs[0])
    Ir_new = np.diff(Cr, prepend=Cr[0])
    df = pd.DataFrame({'t': t, 'Is_cum': Cs, 'Ir_cum': Cr, 'Is_new': Is_new, 'Ir_new': Ir_new})
    return df

if __name__ == "__main__":
    N_pop = 2.48e7
    
    if os.path.exists('data.csv'):
        df = pd.read_csv('data.csv')
    else:
        print("Using dummy data (24 months)...")
        df = generate_dummy_data_24_months(N_pop)
    
    if len(df) != 24:
        print(f"Warning: Data length is {len(df)}, but expected 24.")
        
    solver = TB_PINN(df, N_pop)
    
    solver.train(adam_epochs=100000, lbfgs_max_iter=10000)
    
    print("\n[1/3] Plotting Parameter Estimation...")
    solver.plot_training_process_stacked()
    
    print("\n[2/3] Plotting Individual Dynamics...")
    solver.plot_fitting_results()
    
    print("\n[3/3] Saving Results...")
    solver.evaluate_and_save('fitting_results.csv')