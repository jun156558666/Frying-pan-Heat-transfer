"""
フライパン（底面＋持ち手）非定常熱伝導解析 — 加熱・冷却フェーズ【円筒座標版】
  加熱フェーズ: t = 0   ~ 300 s（バーナー加熱 q_s = 15000 W/m²）
  冷却フェーズ: t = 300 ~ 600 s（加熱なし、自然対流のみ）
  2D FDM: 円筒座標（r, y）Backward Euler
出力:
  top_view_300s.png  — 上面視 t=300s（加熱終了時）
  top_view_600s.png  — 上面視 t=600s（冷却終了時）
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import factorized

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

# SolidWorks 境界条件面積
A_air_空焚き = 133547e-6    # m² — 空焚き時の空冷面積
A_air_水入り = 88024e-6     # m² — 水入り時の空冷面積
A_water      = 45522.56e-6  # m² — 水冷面積

# 底板
q_s = 15000.0
A_bottom_SW = 25611.2e-6               # m² — SolidWorks 加熱底面面積
L, H = 0.122, 0.003
R_b  = np.sqrt(A_bottom_SW / np.pi)   # ≈ 0.09028 m（SolidWorks 加熱面積から算出）
A_top = np.pi * L**2                   # 円形上面面積（冷却 BC 集約用）

Nx, Ny = 122, 31
dx, dy = L/(Nx-1), H/(Ny-1)
x_arr  = np.linspace(0, L, Nx)
y_arr  = np.linspace(0, H, Ny)
N2D    = Nx * Ny
nid    = lambda i, j: j * Nx + i
# j=0, j=Ny-1: BC行　i=Nx-1: リム BC行　→ interior = それ以外
interior_idx = np.array([nid(i,j) for j in range(1,Ny-1) for i in range(0,Nx-1)])

# 持ち手（フィン）
L_h    = 0.180   # SolidWorks: 180 mm（旧値 207 mm）
d_h    = 0.025
P_fin  = np.pi * d_h
Ac_fin = np.pi * d_h**2 / 4
h_fin  = 20.0
Nf     = 104
dx_fin = L_h / (Nf - 1)
x_fin_arr    = np.linspace(0, L_h, Nf)
fin_interior = np.arange(1, Nf - 1)

h_air = 20.0;  h_wtr = 200.0

# 時間設定
t_heat  = 300.0
t_cool  = 300.0
dt      = 0.1
Nt_heat = int(t_heat / dt)   # 3000
Nt_cool = int(t_cool / dt)   # 3000
Nt      = Nt_heat + Nt_cool  # 6000


def rim_h_params(cond):
    """上面は直接 h_air/h_wtr を使い、リム面に残余 UA を集約した h_rim を返す"""
    A_handle = P_fin * L_h           # 持ち手側面積 = π×d_h×L_h
    A_rim    = 2 * np.pi * L * H    # リム面積
    if cond == '空焚き':
        UA_pan = h_air * (A_air_空焚き - A_handle)   # パン本体分 UA
        UA_top = h_air * A_top
        h_rim  = max(0.0, UA_pan - UA_top) / A_rim
        return h_air, T_amb, h_rim, T_amb
    else:  # 水入り
        UA_pan = h_wtr * A_water + h_air * (A_air_水入り - A_handle)
        UA_top = h_wtr * A_top
        h_rim  = max(0.0, UA_pan - UA_top) / A_rim
        return h_wtr, T_water, h_rim, T_amb


# ============================================================
# 1. 2D Backward Euler ソルバー（円筒座標 r-y）
#    上面: h_top（h_air or h_wtr）直接印加
#    リム: h_rim（SolidWorks 残余 UA をリム面積で割った値）
#    加熱: j=0, r≤R_b → Neumann(q_s) / r>R_b → 断熱
#    → A 行列は両フェーズで共通、b0 だけ変わる
# ============================================================
def build_2D(k, rho, c, cond):
    h_top, T_top_bc, h_rim, T_rim = rim_h_params(cond)
    coeff = rho * c / dt
    A = lil_matrix((N2D, N2D))
    b0_heat = np.zeros(N2D)
    b0_cool = np.zeros(N2D)

    for j in range(Ny):
        for i in range(Nx):
            n = nid(i, j)
            r = x_arr[i]

            if j == 0:   # 底面（断熱 BC、加熱時は b で q_s を印加）
                A[n,n] = 1.0;  A[n, nid(i,1)] = -1.0
                if x_arr[i] <= R_b:
                    b0_heat[n] = q_s * dy / k
                # b0_cool[n] = 0（断熱）

            elif j == Ny-1:   # 上面（直接 h_top Robin BC）
                A[n,n]            = k/dy + h_top
                A[n, nid(i,Ny-2)] = -k/dy
                b0_heat[n] = h_top * T_top_bc
                b0_cool[n] = h_top * T_top_bc

            elif i == 0:   # 中心軸 r=0（ghost node T_{-1}=T_1）
                A[n,n]           =  coeff + 4*k/dx**2 + 2*k/dy**2
                A[n, nid(1,j)]   = -4*k/dx**2
                A[n, nid(i,j-1)] = -k/dy**2
                A[n, nid(i,j+1)] = -k/dy**2

            elif i == Nx-1:   # 外周リム（Robin BC — 残余 UA）
                A[n,n]            = k/dx + h_rim
                A[n, nid(Nx-2,j)] = -k/dx
                b0_heat[n] = h_rim * T_rim
                b0_cool[n] = h_rim * T_rim

            else:   # 内部節点（円筒座標ステンシル）
                r_p = r + dx/2
                r_m = r - dx/2
                A[n,n]           =  coeff + k*(r_p+r_m)/(r*dx**2) + 2*k/dy**2
                A[n, nid(i+1,j)] = -k*r_p/(r*dx**2)
                A[n, nid(i-1,j)] = -k*r_m/(r*dx**2)
                A[n, nid(i,j-1)] = -k/dy**2
                A[n, nid(i,j+1)] = -k/dy**2

    return factorized(A.tocsr()), b0_heat, b0_cool, coeff


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
# 3. 非定常計算（加熱 300s → 冷却 300s）
#    戻り値: {300: (T_top, T_fin, T_max_all), 600: (...)}
#    T_max_all = 2D全節点 + フィンの最高温度（底面j=0含む）
# ============================================================
def run_transient(k, rho, c, cond):
    solve, b0_heat, b0_cool, coeff_2D = build_2D(k, rho, c, cond)
    solve_fin, coeff_fin, hpac = build_fin(k, rho, c)

    T_2D  = np.full(N2D, T_amb)
    T_fin = np.full(Nf,  T_amb)
    snap  = {}

    for s in range(Nt):
        b = (b0_heat if s < Nt_heat else b0_cool).copy()
        b[interior_idx] += coeff_2D * T_2D[interior_idx]
        T_2D = solve(b)

        T_base_s = T_2D[nid(Nx-1, Ny-1)]
        bf = np.zeros(Nf)
        bf[0]            = T_base_s
        bf[fin_interior] = coeff_fin * T_fin[fin_interior] + hpac * T_amb
        T_fin = solve_fin(bf)

        if s == Nt_heat - 1:   # t = 300 s
            T_top     = (T_2D.reshape(Ny, Nx) - 273.15)[-1, :]
            T_max_all = float(max((T_2D - 273.15).max(), (T_fin - 273.15).max()))
            snap[300] = (T_top.copy(), (T_fin - 273.15).copy(), T_max_all)

    T_top     = (T_2D.reshape(Ny, Nx) - 273.15)[-1, :]
    T_max_all = float(max((T_2D - 273.15).max(), (T_fin - 273.15).max()))
    snap[600] = (T_top.copy(), (T_fin - 273.15).copy(), T_max_all)
    return snap


# ============================================================
# 4. 全条件計算
# ============================================================
print(f"非定常解析（円筒座標）  加熱 {t_heat:.0f}s + 冷却 {t_cool:.0f}s  dt = {dt}s")
print(f"R_b = {R_b*1000:.1f} mm（SolidWorks 加熱面積 {A_bottom_SW*1e6:.1f} mm²）\n")
results = {}   # (cond, name) -> snap

for cond in conditions:
    for name, mat in materials.items():
        k, rho, c = mat['k'], mat['rho'], mat['c']
        print(f"  {name} / {cond} ...", end=' ', flush=True)
        snap = run_transient(k, rho, c, cond)
        results[(cond, name)] = snap
        for t_s, (T_top, T_fin, T_max_all) in snap.items():
            Tmax = T_max_all
            Tmin = min(T_top.min(), float(T_fin[-1]))
            print(f"t={t_s}s: 最高{Tmax:.1f}°C 最低{Tmin:.1f}°C", end='  ')
        print()
    print()


# ============================================================
# 5. 可視化共通設定
# ============================================================
Nx_c, Ny_c = 900, 450
x_min_mm = -L * 1000 * 1.08
x_max_mm = (L + L_h) * 1000 * 1.06
y_lim_mm =  L * 1000 * 1.08

x_comp = np.linspace(x_min_mm, x_max_mm, Nx_c)
y_comp = np.linspace(-y_lim_mm, y_lim_mm, Ny_c)
X_comp, Y_comp = np.meshgrid(x_comp, y_comp)
R_comp      = np.sqrt(X_comp**2 + Y_comp**2) / 1000
x_fin_comp  = X_comp / 1000 - L
pan_mask    = R_comp <= L
handle_mask = ((x_fin_comp >= 0) & (x_fin_comp <= L_h)
               & (np.abs(Y_comp/1000) <= d_h/2))
pan_only    = pan_mask & ~handle_mask
EXTENT_TV   = [x_min_mm, x_max_mm, -y_lim_mm, y_lim_mm]

BASE   = r'C:\Users\jun1029\claude code\大学\３年\固体力学'
PHASES = {300: '加熱終了（バーナー ON）', 600: '冷却終了（バーナー OFF）'}


# ============================================================
# 6. 上面視図
# ============================================================
def plot_top_view(t_snap):
    fig, axes = plt.subplots(2, 5, figsize=(28, 11))
    plt.subplots_adjust(hspace=0.50, wspace=0.30)
    for row, cond in enumerate(conditions):
        for col, name in enumerate(materials):
            T_top_C, T_fin_C, T_max_all = results[(cond, name)][t_snap]
            T_tip_C   = float(T_fin_C[-1])
            T_all_max = T_max_all
            T_all_min = min(T_top_C.min(), T_tip_C)

            T_comp = np.full((Ny_c, Nx_c), np.nan)
            T_comp[pan_only]    = np.interp(R_comp[pan_only],        x_arr,     T_top_C)
            T_comp[handle_mask] = np.interp(x_fin_comp[handle_mask], x_fin_arr, T_fin_C)

            ax = axes[row, col]
            im = ax.imshow(T_comp, extent=EXTENT_TV, origin='lower',
                           cmap='jet', aspect='equal',
                           vmin=T_all_min, vmax=T_all_max)
            cbar = plt.colorbar(im, ax=ax, pad=0.02, shrink=0.72, aspect=18)
            cbar.set_label('°C', fontsize=7)
            cbar.ax.tick_params(labelsize=6)
            ax.add_patch(plt.Circle((0,0), R_b*1000, fill=False, color='white', ls='--', lw=1.2))
            ax.add_patch(plt.Circle((0,0), L*1000,   fill=False, color='white', ls='-',  lw=0.8, alpha=0.6))
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
        f'フライパン 上からの図  t = {t_snap} s  {PHASES[t_snap]}\n'
        f'q_s = {q_s:.0f} W/m²  R_b = {R_b*1000:.1f} mm  初期温度 = {T_amb_C:.0f}°C',
        fontsize=12
    )
    out = fr'{BASE}\top_view_{t_snap}s.png'
    plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print(f"保存: {out}")


# ============================================================
# 7. 出力（t=300s, t=600s）
# ============================================================
for t_snap in [300, 600]:
    plot_top_view(t_snap)
