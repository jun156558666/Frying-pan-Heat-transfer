"""
フライパン（底面＋持ち手）非定常熱伝導解析 — 上面視 & 断面図（t = 300 s）
  底面 : 2D Backward Euler FDM（バーナー中央加熱、q_s = 15000 W/m²、R_b = 50 mm）
  持ち手: 1D フィン Backward Euler（根元 BC = 底板上面縁部温度の時刻歴）
  初期条件: T = 20°C（293 K）
出力:
  top_view_distribution.png   — 上からの図（円形＋フィン合成カラーマップ）
  cross_section_distribution.png — y=0 断面図（パン板 T(x,y) ＋ フィン）
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import factorized
from scipy.interpolate import RegularGridInterpolator

plt.rcParams['font.family'] = 'Meiryo'

# ============================================================
# 0. パラメータ
# ============================================================
materials = {
    '銅':                  {'k': 390.0,  'rho': 7700, 'c': 390 },
    'アルミ (A5052P)':     {'k': 137.0,  'rho': 2860, 'c': 880 },
    '鉄（ねずみ鋳鉄）':    {'k':  45.0,  'rho': 7200, 'c': 510 },
    'ステンレス (SUS304)': {'k':  16.0,  'rho': 8000, 'c': 500 },
    'フェノール樹脂 (PF)': {'k': 0.2618, 'rho': 1400, 'c': 1900},
}
conditions = ['空焚き', '水入り']

T_amb_C   = 20.0;  T_amb   = T_amb_C   + 273.15
T_water_C = 90.0;  T_water = T_water_C + 273.15

# 底板
q_s  = 15000.0
R_b  = 0.05
L, H = 0.122, 0.003
Nx, Ny = 122, 31
dx, dy = L/(Nx-1), H/(Ny-1)
x_arr  = np.linspace(0, L, Nx)
y_arr  = np.linspace(0, H, Ny)
N2D    = Nx * Ny
nid    = lambda i, j: j * Nx + i
interior_idx = np.array([nid(i,j) for j in range(1,Ny-1) for i in range(1,Nx-1)])

# 持ち手（フィン）
L_h    = 0.207
d_h    = 0.025
P_fin  = np.pi * d_h
Ac_fin = np.pi * d_h**2 / 4
h_fin  = 10.0
Nf     = 104
dx_fin = L_h / (Nf - 1)
x_fin_arr    = np.linspace(0, L_h, Nf)
fin_interior = np.arange(1, Nf - 1)

# SolidWorks 面積・h
A_air_空焚き  = 133546.6e-6
A_water_area = 45522.56e-6
A_air_水入り  = 88024.04e-6
h_air = 20.0;  h_wtr = 200.0
A_2D  = L

# 時間設定
t_end = 300.0;  dt = 0.1;  Nt = int(t_end / dt)

def get_bc(cond):
    if cond == '空焚き':
        UA    = h_air * A_air_空焚き
        T_eff = T_amb
    else:
        UA_w  = h_wtr * A_water_area
        UA_a  = h_air * A_air_水入り
        UA    = UA_w + UA_a
        T_eff = (UA_w * T_water + UA_a * T_amb) / UA
    return UA / A_2D, T_eff

# ============================================================
# 1. 2D Backward Euler ソルバー
# ============================================================
def build_2D(k, rho, c, h_eff, T_eff):
    coeff = rho * c / dt
    A  = lil_matrix((N2D, N2D))
    b0 = np.zeros(N2D)
    for j in range(Ny):
        for i in range(Nx):
            n = nid(i, j)
            if j == 0:
                A[n,n] = 1.0;  A[n, nid(i,1)] = -1.0
                b0[n]  = (q_s*dy/k) if x_arr[i] <= R_b else 0.0
            elif j == Ny-1:
                A[n,n]            = k/dy + h_eff
                A[n, nid(i,Ny-2)] = -k/dy
                b0[n]             = h_eff * T_eff
            elif i == 0:
                A[n,n] = 1.0;  A[n, nid(1,j)] = -1.0
            elif i == Nx-1:
                A[n,n] = 1.0;  A[n, nid(Nx-2,j)] = -1.0
            else:
                A[n,n]           =  coeff + 2*k/dx**2 + 2*k/dy**2
                A[n, nid(i-1,j)] = -k/dx**2
                A[n, nid(i+1,j)] = -k/dx**2
                A[n, nid(i,j-1)] = -k/dy**2
                A[n, nid(i,j+1)] = -k/dy**2
    return factorized(A.tocsr()), b0, coeff

# ============================================================
# 2. フィン Backward Euler ソルバー
# ============================================================
def build_fin(k, rho, c):
    coeff = rho * c / dt
    hpac  = h_fin * P_fin / Ac_fin
    A  = lil_matrix((Nf, Nf))
    for i in range(Nf):
        if i == 0:
            A[i,i] = 1.0
        elif i == Nf-1:
            A[i,i] = 1.0;  A[i,i-1] = -1.0
        else:
            A[i,i]   =  coeff + hpac + 2*k/dx_fin**2
            A[i,i-1] = -k/dx_fin**2
            A[i,i+1] = -k/dx_fin**2
    return factorized(A.tocsr()), coeff, hpac

# ============================================================
# 3. 非定常計算（2D + フィン連成、t=300s）
# ============================================================
def run_transient(k, rho, c, h_eff, T_eff):
    solve_2D, b0_2D, coeff_2D = build_2D(k, rho, c, h_eff, T_eff)
    solve_fin, coeff_fin, hpac = build_fin(k, rho, c)

    T_2D  = np.full(N2D, T_amb)
    T_fin = np.full(Nf,  T_amb)

    for s in range(Nt):
        b = b0_2D.copy()
        b[interior_idx] += coeff_2D * T_2D[interior_idx]
        T_2D = solve_2D(b)

        T_base_s = T_2D[nid(Nx-1, Ny-1)]

        bf = np.zeros(Nf)
        bf[0]            = T_base_s
        bf[fin_interior] = coeff_fin * T_fin[fin_interior] + hpac * T_amb
        T_fin = solve_fin(bf)

    T_field_C = T_2D.reshape(Ny, Nx) - 273.15   # 全2D温度場 (Ny, Nx) [°C]
    T_top_C   = T_field_C[-1, :]                 # 上面プロファイル
    T_fin_C   = T_fin - 273.15                   # フィン温度
    return T_top_C, T_fin_C, T_field_C

# ============================================================
# 4. 計算（全条件まとめて実行・結果をキャッシュ）
# ============================================================
print(f"非定常解析  t = {t_end:.0f}s  dt = {dt}s  q_s = {q_s:.0f} W/m²  R_b = {R_b*1000:.0f} mm\n")
results = {}   # (cond, name) -> (T_top_C, T_fin_C, T_field_C)

for cond in conditions:
    h_eff, T_eff = get_bc(cond)
    for name, mat in materials.items():
        k, rho, c = mat['k'], mat['rho'], mat['c']
        print(f"  {name} / {cond} ...", end=' ', flush=True)
        T_top_C, T_fin_C, T_field_C = run_transient(k, rho, c, h_eff, T_eff)
        results[(cond, name)] = (T_top_C, T_fin_C, T_field_C)
        T_max = max(T_top_C.max(), T_fin_C.max())
        T_min = min(T_top_C.min(), float(T_fin_C[-1]))
        print(f"最高 {T_max:.1f}°C  最低 {T_min:.1f}°C")
    print()

# ============================================================
# 5. 上からの図（top_view_distribution.png）
# ============================================================
Nx_c, Ny_c = 900, 450
x_min_mm   = -L * 1000 * 1.08
x_max_mm   = (L + L_h) * 1000 * 1.06
y_lim_mm   =  L * 1000 * 1.08

x_comp = np.linspace(x_min_mm, x_max_mm, Nx_c)
y_comp = np.linspace(-y_lim_mm, y_lim_mm, Ny_c)
X_comp, Y_comp = np.meshgrid(x_comp, y_comp)
R_comp     = np.sqrt(X_comp**2 + Y_comp**2) / 1000
x_fin_comp = X_comp / 1000 - L
pan_mask    = R_comp <= L
handle_mask = ((x_fin_comp >= 0) & (x_fin_comp <= L_h)
               & (np.abs(Y_comp/1000) <= d_h/2))
pan_only    = pan_mask & ~handle_mask
EXTENT_TV   = [x_min_mm, x_max_mm, -y_lim_mm, y_lim_mm]

fig1, axes1 = plt.subplots(2, 5, figsize=(28, 11))
plt.subplots_adjust(hspace=0.50, wspace=0.30)

for row, cond in enumerate(conditions):
    for col, name in enumerate(materials):
        T_top_C, T_fin_C, _ = results[(cond, name)]
        T_tip_C  = float(T_fin_C[-1])
        T_all_max = max(T_top_C.max(), T_fin_C.max())
        T_all_min = min(T_top_C.min(), T_tip_C)

        T_comp = np.full((Ny_c, Nx_c), np.nan)
        T_comp[pan_only]    = np.interp(R_comp[pan_only],    x_arr,    T_top_C)
        T_comp[handle_mask] = np.interp(x_fin_comp[handle_mask], x_fin_arr, T_fin_C)

        ax = axes1[row, col]
        im = ax.imshow(T_comp, extent=EXTENT_TV, origin='lower',
                       cmap='jet', aspect='equal',
                       vmin=T_all_min, vmax=T_all_max)
        cbar = plt.colorbar(im, ax=ax, pad=0.02, shrink=0.72, aspect=18)
        cbar.set_label('°C', fontsize=7)
        cbar.ax.tick_params(labelsize=6)

        ax.add_patch(plt.Circle((0,0), R_b*1000, fill=False, color='white', ls='--', lw=1.2))
        ax.add_patch(plt.Circle((0,0), L*1000,   fill=False, color='white', ls='-', lw=0.8, alpha=0.6))
        hw = d_h/2*1000
        for sgn in [1,-1]:
            ax.plot([L*1000, (L+L_h)*1000], [sgn*hw, sgn*hw], 'w-', lw=0.8, alpha=0.6)

        ax.set_xlim(x_min_mm, x_max_mm); ax.set_ylim(-y_lim_mm, y_lim_mm)
        ax.set_xlabel('x [mm]', fontsize=7); ax.set_ylabel('y [mm]', fontsize=7)
        ax.tick_params(labelsize=6)
        ax.set_title(
            f'{name} | {cond}\n最高 {T_all_max:.1f}°C  最低 {T_all_min:.1f}°C',
            fontsize=8.5
        )

plt.suptitle(
    f'フライパン 上からの図  t = {t_end:.0f} s（非定常 Backward Euler）\n'
    f'q_s = {q_s:.0f} W/m²  R_b = {R_b*1000:.0f} mm  初期温度 = {T_amb_C:.0f}°C',
    fontsize=12
)
OUT1 = r'C:\Users\jun1029\claude code\大学\３年\固体力学\top_view_distribution.png'
plt.savefig(OUT1, dpi=150, bbox_inches='tight');  plt.close()
print(f"保存: {OUT1}")

# ============================================================
# 6. 断面図（cross_section_distribution.png）
#    x: -L ~ L+L_h [mm]（パン直径＋フィン長さ）
#    y:  0 ~ d_h/2 [mm]（上半分、対称なので）
#    パン板: T(x,y) の 2D カラーマップ（FDM 解）
#    フィン: T(x)  を y 方向一様として描画
# ============================================================
Nx_cs = 800
Ny_cs = 350
y_cs_max_mm = d_h/2 * 1000   # 12.5 mm（フィン半径）

x_cs_mm = np.linspace(-L*1000, (L+L_h)*1000, Nx_cs)
y_cs_mm = np.linspace(0, y_cs_max_mm, Ny_cs)          # 0 〜 12.5 mm
X_cs, Y_cs = np.meshgrid(x_cs_mm, y_cs_mm)
x_cs_m = X_cs / 1000
y_cs_m = Y_cs / 1000

# パン領域: |x| ≤ L, 0 ≤ y ≤ H
pan_cs = (np.abs(x_cs_m) <= L) & (y_cs_m >= 0) & (y_cs_m <= H)
# フィン領域: L ≤ x ≤ L+L_h, 0 ≤ y ≤ d_h/2
fin_cs = (x_cs_m >= L) & (x_cs_m <= L+L_h) & (y_cs_m <= d_h/2)

EXTENT_CS = [x_cs_mm[0], x_cs_mm[-1], y_cs_mm[0], y_cs_mm[-1]]

fig2, axes2 = plt.subplots(2, 5, figsize=(28, 9))
plt.subplots_adjust(hspace=0.55, wspace=0.30)

for row, cond in enumerate(conditions):
    h_eff, T_eff = get_bc(cond)
    for col, name in enumerate(materials):
        T_top_C, T_fin_C, T_field_C = results[(cond, name)]
        T_tip_C  = float(T_fin_C[-1])
        T_all_max = max(T_top_C.max(), T_fin_C.max())
        T_all_min = min(T_field_C.min(), T_tip_C)

        T_cs = np.full((Ny_cs, Nx_cs), np.nan)

        # パン板: RegularGridInterpolator で T(|x|, y) を補間
        interp_fn = RegularGridInterpolator(
            (y_arr, x_arr), T_field_C,
            method='linear', bounds_error=False, fill_value=np.nan
        )
        pts = np.stack([y_cs_m[pan_cs], np.abs(x_cs_m[pan_cs])], axis=1)
        T_cs[pan_cs] = interp_fn(pts)

        # フィン: x 方向のみ補間（y 断面は一様と仮定）
        x_fin_local = x_cs_m[fin_cs] - L
        T_cs[fin_cs] = np.interp(x_fin_local, x_fin_arr, T_fin_C)

        ax = axes2[row, col]
        im = ax.imshow(T_cs, extent=EXTENT_CS, origin='lower',
                       cmap='jet', aspect='auto',
                       vmin=T_all_min, vmax=T_all_max)
        cbar = plt.colorbar(im, ax=ax, pad=0.02, shrink=0.80, aspect=20)
        cbar.set_label('°C', fontsize=7)
        cbar.ax.tick_params(labelsize=6)

        # バーナー境界（x 方向）
        ax.axvline( R_b*1000, color='white', ls='--', lw=1.0, alpha=0.8)
        ax.axvline(-R_b*1000, color='white', ls='--', lw=1.0, alpha=0.8)
        # パン縁
        ax.axvline( L*1000, color='white', ls='-', lw=0.8, alpha=0.5)
        ax.axvline(-L*1000, color='white', ls='-', lw=0.8, alpha=0.5)
        # パン板上面（y = H）
        ax.axhline(H*1000, color='white', ls=':', lw=1.0,
                   xmin=0, xmax=(2*L*1000)/(x_cs_mm[-1]-x_cs_mm[0]))

        ax.set_xlabel('x [mm]（中心→縁→持ち手）', fontsize=7)
        ax.set_ylabel('y [mm]（厚さ / 半径）', fontsize=7)
        ax.tick_params(labelsize=6)
        ax.set_title(
            f'{name} | {cond}\n最高 {T_all_max:.1f}°C  最低 {T_all_min:.1f}°C',
            fontsize=8.5
        )

plt.suptitle(
    f'フライパン y=0 断面図  t = {t_end:.0f} s（非定常 Backward Euler）\n'
    f'q_s = {q_s:.0f} W/m²  R_b = {R_b*1000:.0f} mm  '
    f'パン板: H={H*1000:.0f}mm  フィン: d={d_h*1000:.0f}mm',
    fontsize=12
)
OUT2 = r'C:\Users\jun1029\claude code\大学\３年\固体力学\cross_section_distribution.png'
plt.savefig(OUT2, dpi=150, bbox_inches='tight');  plt.close()
print(f"保存: {OUT2}")
